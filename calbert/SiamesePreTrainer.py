import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .CalBERT import CalBERT
from .CalBERTDataset import CalBERTDataset


class SiamesePreTrainer:
    def __init__(self, model: CalBERT, train_dataset: CalBERTDataset, eval_dataset: CalBERTDataset = None,
                 eval_strategy: str = 'epoch', save_strategy: str = 'epoch', learning_rate: float = 0.01,
                 epochs: int = 20, distance_metric: str = 'cosine', use_contrastive_loss: bool = True,
                 contrastive_loss_type: str = 'softmax', temperature: float = 0.07, loss_margin: float = 0.25,
                 batch_size: int = 16, save_best_model: bool = True, save_best_strategy: str = 'train',
                 optimizer_class: str = 'adam', optimizer_path: Union[str, Path] = None,
                 model_dir: Union[str, Path] = "./calbert", use_tensorboard: bool = False,
                 tensorboard_log_path: str = "./tensorboard_logs", device: str = 'cpu'):
        """
        Initialize the Trainer to perform CalBERT's Siamese Pre-training

        :param model: The CalBERT model to train
        :param train_dataset: The training dataset
        :param eval_dataset: The evaluation dataset
        :param eval_strategy: The evaluation strategy. Either 'epoch' or 'batch'
        :param save_strategy: The save strategy. Either 'epoch' or 'batch'
        :param learning_rate: The learning rate
        :param epochs: The number of epochs to train
        :param distance_metric: The distance metric to use. Either 'cosine', 'euclidean', or 'manhattan'
        :param use_contrastive_loss: Whether to use contrastive loss
        :param contrastive_loss_type: The type of contrastive loss to use. Either 'binary', 'margin', or 'softmax'
        :param temperature: The temperature for the softmax contrastive loss
        :param loss_margin: The margin for the contrastive loss
        :param batch_size: The batch size
        :param save_best_model: Whether to save and load the best model at the end of the training
        :param save_best_strategy: The strategy to save the best model. Either 'epoch' or 'batch'
        :param optimizer_class: The optimizer class to use. Either 'adam', 'sgd', 'adagrad', 'adadelta' or 'rmsprop'
        :param optimizer_path: The path to the optimizer state dict
        :param model_dir: The directory to save the model
        :param use_tensorboard: Whether to use tensorboard for logging
        :param tensorboard_log_path: The path to the tensorboard log directory
        :param device: The device to use
        """
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
        self.save_best_model = save_best_model
        self.save_best_strategy = save_best_strategy
        self.optimizer = None
        self.load_optimizer(optimizer_class, optimizer_path)
        self.model_dir = model_dir
        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_path = tensorboard_log_path
        self.writer = SummaryWriter(log_dir=self.tensorboard_log_path) if self.use_tensorboard else None
        self.device = device

    def load_optimizer(self, optimizer_class: str = 'adam', optimizer_path: Union[str, Path] = None) -> None:
        """
        Initializes and loads the optimizer

        :param optimizer_class: The optimizer class to use. Either 'adam', 'sgd', 'adagrad', 'adadelta' or 'rmsprop'
        :param optimizer_path: The path to the optimizer state dict
        :return: None
        """
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

    def parse_training_args(self, args: dict[str, Union[str, int, float, Path]]) -> None:
        """
        Parse the training arguments

        :param args: A key-value dictionary of the training arguments
        :return: None
        """
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

    def calculate_contrastive_loss(self, distance: torch.Tensor, distance_matrix: torch.Tensor = None,
                                   labels: torch.Tensor = None) -> torch.Tensor:
        """
        Computes and returns the contrastive loss for the given distance and labels

        :param distance: The distance between the two embeddings
        :param distance_matrix: The distance matrix between the two embeddings
        :param labels: The labels for the contrastive loss
        :return: The contrastive loss
        """
        if not self.use_contrastive_loss:
            return distance.mean()

        if self.contrastive_loss_type == 'binary':
            loss = 0.5 * (labels * distance ** 2 + (1 - labels) * F.relu(self.loss_margin - distance) ** 2)
            loss = loss.mean()
        elif self.contrastive_loss_type == 'margin':
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

    def train(self, **kwargs) -> None:
        """
        Perform the Siamese Pre-training

        :param kwargs: Keyword arguments for the pre-training. Uses the same arguments as the SiamesePreTrainer
        constructor
        :return: None
        """
        self.parse_training_args(kwargs)
        torch.cuda.empty_cache()
        self.model.zero_grad()
        self.model.train()

        training_loss = torch.tensor(0.0)
        eval_loss = torch.tensor(0.0)
        best_model = {
            'loss': float('inf'),
            'batch': -1,
            'epoch': -1,
            'strategy': self.save_best_strategy
        }

        for epoch in tqdm(range(self.epochs)):
            for i in tqdm(range(0, len(self.train_dataset), self.batch_size)):
                batch_index = i // self.batch_size
                base_language_sentences, target_language_sentences, labels = self.train_dataset. \
                    get_batch(i, i + self.batch_size)
                logging.info(f"Training on batch {batch_index}")
                distance, distance_matrix = self.forward(base_language_sentences, target_language_sentences)
                training_loss = self.calculate_contrastive_loss(distance, distance_matrix, labels)
                logging.info(f"Epoch {epoch} Batch {batch_index} Training Loss: {training_loss.item()}")

                if self.save_best_strategy == 'train' and training_loss.item() < best_model['loss']:
                    best_model['loss'] = training_loss.item()
                    best_model['batch'] = batch_index
                    best_model['epoch'] = epoch

                if self.eval_strategy == 'batch' and self.eval_dataset is not None:
                    eval_loss = self.evaluate(self.eval_dataset)
                    logging.info(f"Epoch {epoch} Batch {batch_index} Evaluation Loss: {eval_loss.item()}")
                    if self.save_best_strategy == 'eval' and eval_loss.item() < best_model['loss']:
                        best_model['loss'] = eval_loss.item()
                        best_model['batch'] = batch_index
                        best_model['epoch'] = epoch

                if self.save_strategy == 'batch':
                    self.create_checkpoint(training_loss, eval_loss, epoch, batch_index)

                if self.use_tensorboard:
                    self.writer.add_scalar('Training Loss', training_loss.item(),
                                           epoch * self.num_batches + batch_index)
                    if self.eval_strategy == 'batch' and self.eval_dataset is not None:
                        self.writer.add_scalar('Evaluation Loss', eval_loss.item(),
                                               epoch * self.num_batches + batch_index)
                    self.writer.flush()

                training_loss.backward()
                self.optimizer.step()
            logging.info(f"Epoch {epoch} Training Loss: {training_loss.item()}")

            if self.eval_strategy == 'epoch' and self.eval_dataset is not None:
                eval_loss = self.evaluate(self.eval_dataset)
                logging.info(f"Epoch {epoch} Evaluation Loss: {eval_loss.item()}")
                if self.save_best_strategy == 'eval' and eval_loss.item() < best_model['loss']:
                    best_model['loss'] = eval_loss.item()
                    best_model['batch'] = batch_index
                    best_model['epoch'] = epoch

            if self.save_strategy == 'epoch':
                self.create_checkpoint(training_loss, eval_loss, epoch)

            if self.use_tensorboard:
                self.writer.add_scalar('Training Loss', training_loss.item(), epoch * self.num_batches + batch_index)
                if self.eval_strategy == 'epoch' and self.eval_dataset is not None:
                    self.writer.add_scalar('Evaluation Loss', eval_loss.item(), epoch * self.num_batches + batch_index)
                self.writer.flush()

        if self.use_tensorboard:
            self.writer.close()

        if self.save_best_model:
            epoch = best_model['epoch']
            batch = best_model['batch']
            loss = best_model['loss']
            checkpoint_path = Path(self.model_dir).joinpath(f"checkpoint-{epoch}")
            if not checkpoint_path.exists():
                checkpoint_path = Path(self.model_dir).joinpath(f"checkpoint-{epoch}-{batch}")
            if not checkpoint_path.exists():
                raise ValueError(f"Could not find best model checkpoint {checkpoint_path}")
            self.model = CalBERT.load(checkpoint_path)
            self.model.save(Path(self.model_dir.joinpath(f"best-model-{loss}")))

    def forward(self, base_language_sentences: list, target_language_sentences: list) -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single forward pass of the model and computes the distance between the two embeddings

        :param base_language_sentences: The base language sentences
        :param target_language_sentences: The target language sentences
        :return: The distance between the two embeddings and the distance matrix
        """
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

    def evaluate(self, eval_dataset: CalBERTDataset = None) -> torch.Tensor:
        """
        Evaluates the model on the given dataset

        :param eval_dataset: The dataset to evaluate on. If None, the model will be evaluated on the eval dataset passed
        to the constructor
        :return: The evaluation loss
        """
        self.model.eval()
        if eval_dataset is None:
            if self.eval_dataset is None:
                raise ValueError("No eval dataset provided")
            eval_dataset = self.eval_dataset

        base_language_sentences, target_language_sentences, labels = eval_dataset.get_batch(0, len(eval_dataset))
        base_language_sentences = torch.tensor(base_language_sentences).to(self.device)
        target_language_sentences = torch.tensor(target_language_sentences).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        distance, distance_matrix = self.forward(base_language_sentences, target_language_sentences)
        loss = self.calculate_contrastive_loss(distance, distance_matrix, labels)
        return loss

    def create_checkpoint(self, training_loss: torch.Tensor, eval_loss: torch.Tensor, epoch: int,
                          batch: int = None) -> None:
        """
        Creates a checkpoint of the model in the provided model directory

        :param training_loss: The training loss of the model
        :param eval_loss: The evaluation loss of the model
        :param epoch: The current epoch during pre-training
        :param batch: The current batch during pre-training
        :return: None
        """
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
        }
        if self.eval_dataset is not None:
            data['eval_loss'] = eval_loss.item()
        if batch is not None:
            data['batch'] = batch
        self.save_trainer_config(checkpoint_directory, data)

    def save_trainer_config(self, path: Union[str, Path], data: dict = None) -> None:
        """
        Saves the Trainer object to the provided path

        :param path: The path to save the Trainer in
        :param data: The data to save to the Trainer
        :return: None
        """
        save_directory = Path(path)
        if not save_directory.exists():
            raise ValueError(f"Path {path} does not exist")
        save_directory = save_directory.joinpath('trainer.pt')
        data = dict() if data is None else data
        data['learning_rate'] = self.learning_rate
        data['optimizer_state_dict'] = self.optimizer.state_dict()
        # data['params'] = self.__dict__
        torch.save(data, save_directory)

    def load_checkpoint(self, checkpoint_directory: Path) -> None:
        """
        Loads a checkpoint from the provided directory

        :param checkpoint_directory: The directory containing the checkpoint
        :return: None
        """
        self.model = CalBERT.load(checkpoint_directory)
        self.load_trainer_config(checkpoint_directory)

    def load_trainer_config(self, path: Union[str, Path]) -> None:
        """
        Loads a Trainer configuration from the provided path

        :param path: The path to load the Trainer from
        :return: The SiamesePreTrainer object
        """
        save_directory = Path(path)
        if not save_directory.exists():
            raise ValueError(f"Path {path} does not exist")
        save_directory = save_directory.joinpath('trainer.pt')
        trainer_dict = torch.load(save_directory)
        self.learning_rate = trainer_dict['learning_rate']
        self.optimizer.load_state_dict(trainer_dict['optimizer_state_dict'])

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Saves the CalBERT model to the provided path by invoking the save method of the model

        :param path: The path to the directory save the model in
        :return: None
        """
        save_directory = Path(path)
        if not save_directory.exists():
            raise ValueError(f"Path {path} does not exist")
        self.model.save(save_directory)
