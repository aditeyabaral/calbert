import json
import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CalBERT(nn.Module):
    def __init__(self, model_path: str, num_pooling_layers: int = 1, pooling_method: str = 'mean',
                 device: str = 'cpu'):
        """
        Initialize CalBERT model for Siamese Pre-training.

        :param model_path: Path to the Transformer model and Tokenizer to use for CalBERT.
        :param num_pooling_layers: Number of pooling layers to use.
        :param pooling_method:  Method to use for pooling, either 'mean' or 'max'.
        :param device: Device to use for the model.
        """
        super(CalBERT, self).__init__()
        self.device = device
        self.num_pooling_layers = num_pooling_layers
        self.pooling_method = pooling_method
        self.model_path = model_path
        self.transformers_model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.num_pooling_layers > 0:
            self.pool = nn.AdaptiveAvgPool2d(
                (self.transformers_model.config.hidden_size,
                 1)) if self.pooling_method == 'mean' else nn.AdaptiveMaxPool2d(
                (self.transformers_model.config.hidden_size, 1))

    def add_tokens_to_tokenizer(self, tokens: list[str]) -> int:
        """
        Add new tokens to the CalBERT Tokenizer.

        :param tokens: List of tokens to add to the Tokenizer.
        :return: New vocabulary size of the Tokenizer
        """
        self.tokenizer.add_tokens(tokens)
        new_vocabulary_size = len(self.tokenizer)
        self.transformers_model.resize_token_embeddings(new_vocabulary_size)
        return new_vocabulary_size

    def encode(self, sentence: str) -> dict[str:torch.Tensor]:
        """
        Encode a sentence using the CalBERT Tokenizer

        :param sentence: Sentence to encode.
        :return: Dictionary containing the input ids, attention mask and token type ids.
        """
        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt').to(self.device)
        return encoding

    def batch_encode(self, sentences: list[str]) -> dict[str:torch.Tensor]:
        """
        Encode a list of sentences using the CalBERT Tokenizer.

        :param sentences: List of sentences to encode.
        :return: Dictionary containing the input ids, attention mask and token type ids.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        encodings = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt').to(self.device)
        return encodings

    def embed(self, encoding: dict[str:torch.Tensor]) -> torch.Tensor:
        """
        Returns the embedding representation of an encoding.

        :param encoding: Dictionary containing the input ids, attention mask and token type ids.
        :return: Embedding representation of the sentence.
        """
        embedding = self.transformers_model(**encoding).last_hidden_state
        return embedding

    def batch_embed(self, encodings: dict[str:torch.Tensor]) -> torch.Tensor:
        """
        Returns the embedding representation of a batch of encodings.

        :param encodings: Dictionary containing the input ids, attention mask and token type ids.
        :return: Embedding representation of the batch of sentences.
        """
        embeddings = self.transformers_model(**encodings).last_hidden_state
        return embeddings

    def sentence_embedding(self, sentence: str, pooling: bool = False) -> torch.Tensor:
        """
        Returns the sentence embedding of a sentence.

        :param sentence: Sentence to embed.
        :param pooling: Whether to pool the embedding.
        :return: Sentence embedding.
        """
        encoding = self.encode(sentence)
        embedding = self.encoding_to_embedding(encoding)
        if pooling:
            embedding = self.pooling(embedding)
        return embedding

    def batch_sentence_embedding(self, sentences: list[str], pooling: bool = False) -> torch.Tensor:
        """
        Returns the sentence embedding of a batch of sentences.

        :param sentences: List of sentences to embed.
        :param pooling: Whether to pool the embedding.
        :return: Sentence embeddings of the batch of sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        encodings = self.batch_encode(sentences)
        embeddings = self.batch_encoding_to_embedding(encodings)
        if pooling and self.num_pooling_layers > 0:
            embeddings = self.pooling(embeddings)
        return embeddings

    def pooling(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Returns the pooled representation of a batch of weights.

        :param weights: Batch of weights to pool.
        :return: Pooled representation of the batch of weights.
        """
        if self.num_pooling_layers > 0:
            for _ in range(self.num_pooling_layers):
                weights = self.pool(weights)
            return weights
        else:
            logging.warning('No pooling layers specified. Returning weights as is.')
            return weights

    def embedding_distance(self, embedding1: torch.Tensor, embedding2: torch.Tensor, metric: str = 'cosine') -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the distance between two embeddings defined by the metric.

        :param embedding1: First embedding.
        :param embedding2: Second embedding.
        :param metric: Metric to use for distance. Can be 'cosine', 'euclidean' or 'manhattan'.
        :return: Distance between the embeddings and the distance matrix.
        """
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

    def embedding_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the similarity between two embeddings.

        :param embedding1: First embedding.
        :param embedding2: Second embedding.
        :return: Similarity between the embeddings and the similarity matrix.
        """
        similarity = F.cosine_similarity(embedding1, embedding2)
        embedding2 = torch.transpose(embedding2, 0, 1)
        similarity_matrix = torch.matmul(embedding1, embedding2)
        return similarity, similarity_matrix

    def distance(self, sentence1: str, sentence2: str, metric='cosine', pooling: bool = False) -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the distance between two sentences.

        :param sentence1: First sentence.
        :param sentence2: Second sentence.
        :param metric: Metric to use for distance. Can be 'cosine', 'euclidean' or 'manhattan'.
        :param pooling: Whether to pool the embedding.  If True, the embedding is pooled before calculating the
        distance.
        :return:
        """
        embedding1 = self.sentence_embedding(sentence1, pooling=pooling)
        embedding2 = self.sentence_embedding(sentence2, pooling=pooling)
        return self.embedding_distance(embedding1, embedding2, metric)

    def similarity(self, sentence1: str, sentence2: str, pooling: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the similarity between two sentences.

        :param sentence1: First sentence.
        :param sentence2: Second sentence.
        :param pooling: Whether to pool the embedding.  If True, the embedding is pooled before calculating the
        similarity.
        :return:
        """
        embedding1 = self.sentence_embedding(sentence1, pooling=pooling)
        embedding2 = self.sentence_embedding(sentence2, pooling=pooling)
        return self.embedding_similarity(embedding1, embedding2)

    def forward(self, sentences: list[str], pooling: bool = False) -> torch.Tensor:
        """
        Returns the sentence embedding of a batch of sentences.

        :param sentences: List of sentences to embed.
        :param pooling: Whether to pool the embedding.
        :return:
        """
        return self.batch_sentence_embedding(sentences, pooling)

    def save(self, path: Union[Path, str], save_pretrained: bool = True, save_tokenizer: bool = True) -> None:
        """
        Saves the CalBERT Siamese Network model

        :param path: The directory path in which to save the model.
        :param save_pretrained: Whether to save the Transformer separately.
        :param save_tokenizer: Whether to save the Tokenizer for the Transformer separately. Applicable only
        if save_pretrained is True.
        :return: None
        """
        save_directory = Path(path)
        if not save_directory.exists():
            save_directory.mkdir(parents=True)
        torch.save(self.state_dict(), save_directory.joinpath('calbert.pt'))
        if save_pretrained:
            self.save_pretrained(save_directory, save_tokenizer)

    def save_pretrained(self, path: Union[Path, str], save_tokenizer: bool = True) -> None:
        """
        Invokes the base Transformer save_pretrained method to save the model and Tokenizer.

        :param path: The directory path in which to save the Transformer and Tokenizer
        :param save_tokenizer: Whether to save the Tokenizer.
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

    @staticmethod
    def load(path: Union[Path, str], model_path: Union[str, None] = None) -> 'CalBERT':
        """
        Loads the CalBERT Siamese Network model.

        :param path: The path to the CalBERT model. If this is a directory, ensure that it contains the calbert.pt file
        and the config.json to load the Transformer. If this is a file, it should be the calbert.pt file.
        :param model_path: The path to the Transformer model. If None, the model is loaded from the path using the
        config.json.
        :return: The loaded CalBERT Siamese Network model.
        """
        path = Path(path)
        if path.is_dir():
            config_path = path.joinpath('config.json')
            if not config_path.exists():
                raise ValueError(
                    'Invalid path. If you are providing a directory, ensure the config.json file for the Transformer '
                    'exists.')
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_path = config['_name_or_path']
            path = path.joinpath('calbert.pt')
        elif model_path is None:
            raise ValueError(
                'Invalid Transformer model name or path. Please provide a valid argument to load the model.')
        else:
            pass
        model = CalBERT(model_path)
        model.load_state_dict(torch.load(path))
        return model
