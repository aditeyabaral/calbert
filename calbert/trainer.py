import torch
import logging
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn.functional as F

from model import CalBERT
from dataset import CalBERTDataSet


class SiamesePreTrainer:
    def __init__(self, model: CalBERT, train_dataset: CalBERTDataSet, eval_dataset=None, eval_strategy='epoch',
                 save_strategy='epoch', learning_rate=0.01, epochs=20, distance_metric='cosine',
                 use_contrastive_loss=True, contrastive_loss_type='softmax', temperature=0.07, loss_margin=0.25,
                 batch_size=16, save_best_model=True, optimizer_class='adam', optimizer_path=None,
                 model_dir="./calbert", device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.distance_metric = distance_metric
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss_type = contrastive_loss_type
        self.temperature = temperature
        self.loss_margin = loss_margin
        self.batch_size = batch_size
        self.num_batches = len(self.train_dataset) // self.batch_size
        self.save_best_model = save_best_model  # TODO: implement this
        self.optimizer = None
        self.load_optimizer(optimizer_class, optimizer_path)
        self.model_dir = model_dir
        self.device = device
        # logging.debug(f"Created SiamesePreTrainer with {self.__dict__}")

    def load_optimizer(self, optimizer_class='adam', optimizer_path=None):
        logging.info(f"Loading optimizer {optimizer_class}")

        if optimizer_class == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_class == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_class == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_class == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_class == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_class == 'sparse_adam':
            self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer_class} not supported")

        if optimizer_path is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path))

    def parse_training_args(self, args):
        logging.info(f"Parsing training args: {args}")
        if 'learning_rate' in args:
            self.learning_rate = args['learning_rate']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_class' in args:
            self.load_optimizer(optimizer_class=args['optimizer_class'])
        if 'epochs' in args:
            self.epochs = args['epochs']
        if 'batch_size' in args:
            self.batch_size = args['batch_size']
        if 'distance_metric' in args:
            self.distance_metric = args['distance_metric']
        if 'use_contrastive_loss' in args:
            self.use_contrastive_loss = args['use_contrastive_loss']
        if 'loss_margin' in args:
            self.loss_margin = args['loss_margin']
        if 'eval_strategy' in args:
            self.eval_strategy = args['eval_strategy']
        if 'save_strategy' in args:
            self.save_strategy = args['save_strategy']
        if 'load_best_model' in args:
            self.save_best_model = args['save_best_model']
        if 'model_dir' in args:
            self.model_dir = args['model_dir']

    def calculate_contrastive_loss(self, distance, distance_matrix=None, labels=None):
        if not self.use_contrastive_loss:
            return distance.mean()

        if self.contrastive_loss_type == 'binary':
            loss = 0.5 * (labels * distance ** 2 + (1 - labels) * F.relu(self.loss_margin - distance) ** 2)
            loss = loss.mean()
        elif self.contrastive_loss_type == 'linear':
            positive_distance = distance * labels
            negative_distance = distance * (1 - labels)
            loss = (self.loss_margin - positive_distance) + (self.loss_margin - negative_distance)
            loss = loss.mean()
        elif self.contrastive_loss_type == 'softmax':
            distance_matrix /= self.temperature
            loss = - F.log_softmax(distance_matrix).sum()
        else:
            raise ValueError(f"Contrastive loss type {self.contrastive_loss_type} not supported")

        return loss

    def train(self, **kwargs):
        self.parse_training_args(kwargs)
        torch.cuda.empty_cache()
        self.model.zero_grad()
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            for i in tqdm(range(0, len(self.train_dataset), self.batch_size)):
                base_language_sentences, target_language_sentences, labels = self.train_dataset.get_batch(i,
                                                                                                          i + self.batch_size)
                logging.info(f"Training on batch {i // self.batch_size}")
                distance, distance_matrix = self.forward(base_language_sentences, target_language_sentences)
                training_loss = self.calculate_contrastive_loss(distance, distance_matrix, labels)
                logging.info(
                    f"Epoch {epoch} Batch {i // self.batch_size} Training Loss: {training_loss.item()}")
                eval_loss = torch.tensor(0)
                if self.eval_strategy == 'batch' and self.eval_dataset is not None:
                    eval_loss = self.evaluate()
                    logging.info(
                        f"Epoch {epoch} Batch {i // self.batch_size} Evaluation Loss: {eval_loss.item()}")
                if self.save_strategy == 'batch':
                    self.create_checkpoint(training_loss, eval_loss, epoch, i // self.batch_size)
                training_loss.backward()
                self.optimizer.step()
            logging.info(f"Epoch {epoch} Training Loss: {training_loss.item()}")
            if self.eval_strategy == 'epoch' and self.eval_dataset is not None:
                eval_loss = self.evaluate()
                logging.info(f"Epoch {epoch} Evaluation Loss: {eval_loss.item()}")
            if self.save_strategy == 'epoch':
                self.create_checkpoint(training_loss, eval_loss, epoch)

    def forward(self, base_language_sentences, target_language_sentences):
        base_language_input = self.model.batch_encode(base_language_sentences)
        base_language_input = self.model.batch_embed(base_language_input)
        base_language_input = self.model.pooling(base_language_input)
        base_language_input = base_language_input.squeeze(-1)

        target_language_input = self.model.batch_encode(target_language_sentences)
        target_language_input = self.model.batch_embed(target_language_input)
        target_language_input = self.model.pooling(target_language_input)
        target_language_input = target_language_input.squeeze(-1)

        distance, distance_matrix = self.model.embedding_distance(base_language_input, target_language_input,
                                                                  metric=self.distance_metric)
        return distance, distance_matrix

    def evaluate(self):
        base_language_sentences, target_language_sentences, labels = self.eval_dataset.get_batch(0,
                                                                                                 len(self.eval_dataset))
        base_language_sentences = torch.tensor(base_language_sentences).to(self.device)
        target_language_sentences = torch.tensor(target_language_sentences).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        distance, distance_matrix = self.forward(base_language_sentences, target_language_sentences)
        loss = self.calculate_contrastive_loss(distance, distance_matrix, labels)
        return loss

    def create_checkpoint(self, training_loss, eval_loss, epoch, batch=None):
        save_directory = Path(self.model_dir)
        if not save_directory.exists():
            save_directory.mkdir(parents=True)
        checkpoint_directory_name = f'checkpoint-{epoch}'
        if batch is not None:
            checkpoint_directory_name = f'{checkpoint_directory_name}-{batch}'
        checkpoint_directory = save_directory.joinpath(checkpoint_directory_name)
        checkpoint_directory.mkdir(parents=True)
        self.model.save(checkpoint_directory)

        data = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'learning_rate': self.learning_rate,
            'training_loss': training_loss.item(),
            'eval_loss': eval_loss.item() if self.eval_dataset is not None else None
        }
        if batch is not None:
            data['batch'] = batch
        self.save_trainer(checkpoint_directory, data)

    def save_trainer(self, path, data=None):
        save_directory = Path(path)
        if not save_directory.exists():
            raise ValueError(f"Path {path} does not exist")
        save_directory = save_directory.joinpath('trainer.pt')
        if data is None:
            data = {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'learning_rate': self.learning_rate,
            }
        torch.save(data, save_directory)

    def save_model(self, path):
        save_directory = Path(path)
        if not save_directory.exists():
            raise ValueError(f"Path {path} does not exist")
        self.model.save(save_directory)
