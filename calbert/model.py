import torch
import logging
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CalBERT(nn.Module):
    def __init__(self, model_path, num_pooling_layers=1, pooling_method='mean', device='cpu'):
        super(CalBERT, self).__init__()
        self.device = device
        self.num_pooling_layers = num_pooling_layers
        self.pooling_method = pooling_method
        self.model_path = model_path
        self.transformers_model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.num_pooling_layers > 0:
            self.pool = nn.AdaptiveAvgPool2d(
                (self.transformers_model.config.hidden_size, 1)) if self.pooling_method == 'mean' else nn.AdaptiveMaxPool2d(
                (self.transformers_model.config.hidden_size, 1))

    def add_tokens_to_tokenizer(self, tokens):
        self.tokenizer.add_tokens(tokens)
        new_vocabulary_size = len(self.tokenizer)
        self.transformers_model.resize_token_embeddings(new_vocabulary_size)
        return new_vocabulary_size

    def encode(self, sentence):
        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt').to(self.device)
        return encoding

    def batch_encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        encodings = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt').to(self.device)
        return encodings

    def embed(self, encoding):
        embedding = self.transformers_model(**encoding).last_hidden_state
        return embedding

    def batch_embed(self, encodings):
        embeddings = self.transformers_model(**encodings).last_hidden_state
        return embeddings

    def sentence_embedding(self, sentence, pooling=False):
        encoding = self.encode(sentence)
        embedding = self.encoding_to_embedding(encoding)
        if pooling:
            embedding = self.pooling(embedding)
        return embedding

    def batch_sentence_embedding(self, sentences, pooling=False):
        if isinstance(sentences, str):
            sentences = [sentences]
        encodings = self.batch_encode(sentences)
        embeddings = self.batch_encoding_to_embedding(encodings)
        if pooling and self.num_pooling_layers > 0:
            embeddings = self.pooling(embeddings)
        return embeddings

    def pooling(self, weights):
        if self.num_pooling_layers > 0:
            for _ in range(self.num_pooling_layers):
                weights = self.pool(weights)
            return weights
        else:
            logging.warning('No pooling layers specified. Returning weights as is.')
            return weights

    def embedding_distance(self, embedding1, embedding2, metric='cosine'):
        if metric == 'cosine':
            distance = 1 - F.cosine_similarity(embedding1, embedding2, dim=1)
            embedding2 = torch.transpose(embedding2, 0, 1)
            distance_matrix = 1 - torch.matmul(embedding1, embedding2)
        elif metric == 'euclidean':
            distance = F.pairwise_distance(embedding1, embedding2)
            distance_matrix = list()
            for i in range(embedding1.shape[0]):
                for j in range(embedding2.shape[0]):
                    distance_matrix.append(F.pairwise_distance(embedding1[i], embedding2[j]))
            distance_matrix = torch.tensor(distance_matrix).to(self.device)
        elif metric == 'manhattan':
            distance = F.pairwise_distance(embedding1, embedding2, p=1)
            distance_matrix = list()
            for i in range(embedding1.shape[0]):
                for j in range(embedding2.shape[0]):
                    distance_matrix.append(F.pairwise_distance(embedding1[i], embedding2[j], p=1))
            distance_matrix = torch.tensor(distance_matrix).to(self.device)
        else:
            raise ValueError('Invalid metric')
        return distance, distance_matrix

    def embedding_similarity(self, embedding1, embedding2):
        similarity = F.cosine_similarity(embedding1, embedding2)
        embedding2 = torch.transpose(embedding2, 0, 1)
        similarity_matrix = torch.matmul(embedding1, embedding2)
        return similarity, similarity_matrix

    def distance(self, sentence1, sentence2, metric='cosine'):
        embedding1 = self.sentence_embedding(sentence1)
        embedding2 = self.sentence_embedding(sentence2)
        return self.embedding_distance(embedding1, embedding2, metric)

    def similarity(self, sentence1, sentence2):
        embedding1 = self.sentence_embedding(sentence1)
        embedding2 = self.sentence_embedding(sentence2)
        return self.embedding_similarity(embedding1, embedding2)

    def forward(self, sentences, pooling=False):
        return self.batch_sentence_embedding(sentences, pooling)

    def save(self, path, save_pretrained=True, save_tokenizer=True):
        """
        Saves the CalBERT Siamese Network model

        :param path: (str) The path to save the model and tokenizer weights to.
        :param save_pretrained: (bool) Whether to save the transformer separately.
        :param save_tokenizer: (bool) Whether to save the tokenizer for the transformer separately. Applicable only
        if save_pretrained is True.
        :return:   None
        """
        save_directory = Path(path)
        if not save_directory.is_dir():
            raise ValueError('Invalid path. Please provide a directory path.')
        if not save_directory.exists():
            save_directory.mkdir(parents=True)
        torch.save(self.state_dict(), save_directory.joinpath('calbert.pt'))
        if save_pretrained:
            self.save_pretrained(save_directory, save_tokenizer)

    def save_pretrained(self, path, save_tokenizer=True):
        """
        Invokes the base transformer save_pretrained method to save the model and tokenizer.

        :param path: (str) The path to save the model and tokenizer weights to.
        :param save_tokenizer: (bool) Whether to save the tokenizer.
        :return:    None
        """
        save_directory = Path(path)
        if not save_directory.is_dir():
            raise ValueError('Invalid path. Please provide a directory path.')
        if not save_directory.exists():
            save_directory.mkdir(parents=True)
        self.transformers_model.save_pretrained(save_directory)
        if save_tokenizer:
            self.tokenizer.save_pretrained(path)

    # @staticmethod
    # def load(path):
    #     # TODO: Implement loading of CalBERT model
    #     return nn.Module.load_state_dict(torch.load(path))
