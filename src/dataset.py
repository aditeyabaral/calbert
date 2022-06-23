import torch
import random
from tqdm.auto import tqdm
from collections import Counter
from torch.utils.data import Dataset


class CalBERTDataSet(Dataset):
    def __init__(self, base_language_sentences, target_language_sentences, labels=None, negative_sampling=False,
                 negative_sampling_size=0.5, min_count=10,
                 negative_sampling_count=1, shuffle=True):
        self.base_language_sentences = base_language_sentences
        self.target_language_sentences = target_language_sentences
        self.labels = torch.tensor(labels) if labels is not None else torch.ones(len(base_language_sentences))
        assert len(self.base_language_sentences) == len(self.target_language_sentences) == len(self.labels)

        self.total_examples = len(base_language_sentences)
        self.negative_sampling = negative_sampling
        self.negative_sample_size = negative_sampling_size
        self.negative_sample_count = negative_sampling_count
        self.min_count = min_count
        random.seed(0)

        if self.negative_sampling:
            self.sample_negative_examples()

        if shuffle:
            items = list(zip(self.base_language_sentences, self.target_language_sentences, self.labels))
            random.shuffle(items)
            self.base_language_sentences, self.target_language_sentences, self.labels = zip(*items)
            self.base_language_sentences = list(self.base_language_sentences)
            self.target_language_sentences = list(self.target_language_sentences)
            self.labels = torch.tensor(self.labels)

        self.tokens = list()
        self.compute_vocabulary()

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        return self.base_language_sentences[idx], self.target_language_sentences[idx], self.labels[idx]

    def sample_negative_examples(self):
        current_examples = list(zip(self.base_language_sentences, self.target_language_sentences, self.labels))
        new_examples = list()
        for example in tqdm(random.sample(current_examples, int(self.negative_sample_size * self.total_examples))):
            translation = example[0]
            transliteration = example[1]
            label = example[2]
            new_examples.append((translation, transliteration, label))

            sampled = False
            random_sample_items = list()
            while not sampled:
                random_sample_index = random.sample(range(self.total_examples), self.negative_sample_count)
                random_sample_items = [current_examples[i] for i in random_sample_index]
                random_sample_target_language_sentences = [self.target_language_sentences[i] for i in
                                                           random_sample_index]
                if transliteration not in random_sample_target_language_sentences:
                    sampled = True

            for sample in random_sample_items:
                new_examples.append((sample[0], sample[1], 0.0))

            # sample translation items - not needed
            # sampled = False
            # random_sample_items = list()
            # while not sampled:
            #     random_sample_index = random.sample(range(self.total_examples), self.negative_sample_count)
            #     random_sample_items = [current_examples[i] for i in random_sample_index]
            #     random_sample_base_language_sentences = [self.base_language_sentences[i] for i in random_sample_index]
            #     if translation not in random_sample_base_language_sentences:
            #         sampled = True
            #
            # for sample in random_sample_items:
            #     new_examples.append((sample[0], sample[1], 0.0))

            random.shuffle(new_examples)
            self.base_language_sentences, self.target_language_sentences, self.labels = zip(*new_examples)
            self.total_examples = len(self.base_language_sentences)

    def compute_vocabulary(self, min_count=None):
        self.min_count = min_count if min_count is not None else self.min_count
        self.tokens = set(self.tokens)
        new_tokens = list()
        for sentence in tqdm(self.base_language_sentences):
            for word in sentence.split():
                new_tokens.append(word)
        for sentence in tqdm(self.target_language_sentences):
            for word in sentence.split():
                new_tokens.append(word)

        counter = Counter(new_tokens)
        filtered_new_tokens = [token for token, count in counter.items() if count >= self.min_count]
        self.tokens.update(filtered_new_tokens)
        self.tokens = list(self.tokens)
        return self.tokens

    def get_tokens(self):
        return self.tokens

    def get_batch(self, start, end):
        return self.base_language_sentences[start:end], self.target_language_sentences[start:end], self.labels[start:end]

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
