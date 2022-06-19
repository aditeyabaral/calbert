import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CalBERT(nn.Module):
    def __init__(self, model_path, num_pooling_layers=0, pooling_method='mean', device='cpu'):
        super(CalBERT, self).__init__()
        self.device = device
        self.num_pooling_layers = num_pooling_layers
        self.pooling_method = pooling_method
        self.model_path = model_path
        self.transformers_model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pool = nn.AdaptiveMaxPool1d(1) if self.pooling_method == 'max' else nn.AdaptiveAvgPool1d(1)

    def add_tokens_to_transformer(self, tokens):
        for token in tokens:
            self.tokenizer.add_tokens(token)
        new_vocabulary_size = len(self.tokenizer)
        self.transformers_model.resize_token_embeddings(new_vocabulary_size)
        return new_vocabulary_size

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

    def encode(self, sentence):
        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt').to(self.device)
        return encoding

    def embed(self, encoding):
        embedding = self.transformers_model(**encoding).last_hidden_state
        return embedding

    def batch_embed(self, encodings):
        embeddings = self.transformers_model(**encodings).last_hidden_state
        return embeddings

    def sentence_embedding(self, sentence):
        encoding = self.encode(sentence)
        embedding = self.encoding_to_embedding(encoding)
        return embedding

    def batch_sentence_embedding(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        encodings = self.batch_encode(sentences)
        embeddings = self.batch_encoding_to_embedding(encodings)
        return embeddings

    def pooling(self, weights):
        for _ in range(self.num_pooling_layers):
            weights = self.pool(weights)
        return weights

    def embedding_distance(self, embedding1, embedding2, metric='cosine'):
        if metric == 'cosine':
            distance = 1 - F.cosine_similarity(embedding1, embedding2)
        elif metric == 'euclidean':
            distance = F.pairwise_distance(embedding1, embedding2)
        elif metric == 'manhattan':
            distance = F.pairwise_distance(embedding1, embedding2, p=1)
        else:
            raise ValueError('Invalid metric')
        return distance

    def embedding_similarity(self, embedding1, embedding2):
        return F.cosine_similarity(embedding1, embedding2)

    def distance(self, sentence1, sentence2, metric='cosine'):
        embedding1 = self.sentence_embedding(sentence1)
        embedding2 = self.sentence_embedding(sentence2)
        return self.embedding_distance(embedding1, embedding2, metric)

    def similarity(self, sentence1, sentence2):
        embedding1 = self.sentence_embedding(sentence1)
        embedding2 = self.sentence_embedding(sentence2)
        return self.embedding_similarity(embedding1, embedding2)

    def forward(self, sentences):
        return self.batch_sentence_embedding(sentences)

    def save(self, path):
        torch.save(self.state_dict(), path)
        self.transformers_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def load(path):
        return CalBERT().load_state_dict(torch.load(path))
