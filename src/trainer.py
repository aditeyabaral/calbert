import torch
from calbert import CalBERT
from dataset import CalBERTDataSet
import torch.nn.functional as F


class Trainer:
    def __init__(self, model: CalBERT, train_dataset: CalBERTDataSet, eval_dataset=None, eval_strategy='epoch',
                 save_strategy='epoch', learning_rate=0.01, epochs=20, loss='cosine', use_contrastive_loss=True,
                 loss_margin=0.25,
                 batch_size=16,
                 load_best_model=True, optimizer_class='adam', optimizer_path=None, model_dir="./calbert",
                 device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = loss
        self.loss_margin = loss_margin
        self.batch_size = batch_size
        self.load_best_model = load_best_model

        self.load_optimizer(optimizer_class, optimizer_path)
        self.model_dir = model_dir
        self.device = device

    def load_optimizer(self, optimizer_class='adam', optimizer_path=None):
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
        if 'learning_rate' in args:
            self.learning_rate = args['learning_rate']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_class' in args:
            self.load_optimizer(optimizer_class=args['optimizer_class'])
        if 'epochs' in args:
            self.epochs = args['epochs']
        if 'batch_size' in args:
            self.batch_size = args['batch_size']
        if 'loss' in args:
            self.loss = args['loss']
        if 'eval_strategy' in args:
            self.eval_strategy = args['eval_strategy']
        if 'save_strategy' in args:
            self.save_strategy = args['save_strategy']
        if 'load_best_model' in args:
            self.load_best_model = args['load_best_model']
        if 'model_dir' in args:
            self.model_dir = args['model_dir']

    def train(self, **kwargs):
        self.parse_training_args(kwargs)
        torch.cuda.empty_cache()
        self.model.zero_grad()
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(self.train_dataset), self.batch_size):
                base_language_sentences, target_language_sentences, labels = self.train_dataset.get_batch(i,
                                                                                                          self.batch_size)
                base_language_sentences = torch.tensor(base_language_sentences).to(self.device)
                target_language_sentences = torch.tensor(target_language_sentences).to(self.device)
                labels = torch.tensor(labels).to(self.device)
                distance = self.forward(base_language_sentences, target_language_sentences)
                loss = 0.5 * (labels * distance + (1 - labels) * F.relu(distance - self.loss_margin)).mean()
                if self.eval_strategy == 'epoch':

                loss.backward()
                self.optimizer.step()

    def forward(self, base_language_sentences, target_language_sentences):
        base_language_input = self.model.batch_encode(base_language_sentences)
        base_language_input = self.model.batch_embed(base_language_input)
        base_language_input = self.model.pooling(base_language_input)
        base_language_input = base_language_input.squeeze(-1)

        target_language_input = self.model.batch_encode(target_language_sentences)
        target_language_input = self.model.batch_embed(target_language_input)
        target_language_input = self.model.pooling(target_language_input)
        target_language_input = target_language_input.squeeze(-1)

        distance = self.model.embedding_distance(base_language_input, target_language_input, metric=self.loss)
        return distance

    def evaluate(self):
        base_language_sentences, target_language_sentences, labels = self.eval_dataset.get_batch(0, len(self.eval_dataset))
        base_language_sentences = torch.tensor(base_language_sentences).to(self.device)
        target_language_sentences = torch.tensor(target_language_sentences).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        distance = self.forward(base_language_sentences, target_language_sentences)

