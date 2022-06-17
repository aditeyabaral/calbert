import torch
import logging
import transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CalBERT(nn.Module):
    def __init__(self, model_path, num_pooling_layers=0, pooling='mean', device='cpu'):
        super(CalBERT, self).__init__()
        self.device = device
        self.num_pooling_layers = num_pooling_layers
        self.pooling = pooling
        self.model_path = model_path
        self.transformers_model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pool = nn.AdaptiveMaxPool1d(1) if self.pooling == 'max' else nn.AdaptiveAvgPool1d(1)

    def get_sentence_encodings(self, sentences):
        # encode the sentences
        encodings = self.tokenizer.encode_plus(
            sentences,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt').to(self.device)
        return encodings

    def get_sentence_embeddings(self, encodings):
        # get the embeddings for the language sentences
        embeddings = self.transformers_model(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask']).last_hidden_state[0]

        # pass the embeddings through the pooling layers
        for _ in range(self.num_pooling_layers):
            embeddings = self.pool(embeddings)

        embeddings = embeddings.squeeze(1)
        return embeddings

    def get_contrastive_loss(self, base_language_embeddings, target_language_embeddings, labels=None, metric='cosine'):
        # get the distance between the embeddings
        if metric == 'cosine':
            distance = 1 - F.cosine_similarity(base_language_embeddings, target_language_embeddings)
        elif metric == 'euclidean':
            distance = F.pairwise_distance(base_language_embeddings, target_language_embeddings)
        elif metric == 'manhattan':
            distance = F.pairwise_distance(base_language_embeddings, target_language_embeddings, p=1)
        else:
            raise ValueError('Invalid metric')

        # get the loss
        if labels is None:
            return distance
        else:
            return 0.5 * (labels * distance + (1 - labels) * F.relu(distance - 0.5))

    def forward(self, base_language_sentences, target_language_sentences, labels=None, metric='cosine'):
        # encode base and target language sentences
        base_language_encodings = self.get_sentence_encodings(base_language_sentences)
        target_language_encodings = self.get_sentence_encodings(target_language_sentences)

        # get the embeddings for the encodings
        base_language_embeddings = self.get_sentence_embeddings(base_language_encodings)
        target_language_embeddings = self.get_sentence_embeddings(target_language_encodings)

        print(base_language_embeddings.shape)
        print(target_language_embeddings.shape)

        # get the contrastive loss
        contrastive_loss = self.get_contrastive_loss(base_language_embeddings, target_language_embeddings).to(
            self.device)
        print(contrastive_loss)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path):
        return CalBERT().load_state_dict(torch.load(path))


if __name__ == '__main__':
    model = CalBERT(
        'xlm-roberta-base',
        num_pooling_layers=1,
        pooling='mean',
        device='cuda'
    ).cuda()
    base_language_sentences = ['Hello, my name is Cal.', 'I am a student.']
    target_language_sentences = ['Hola, mi nombre es Cal.', 'Soy un estudiante.']
    model(base_language_sentences, target_language_sentences)
    # print(model)
