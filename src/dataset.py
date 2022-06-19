import torch
import random
from tqdm.auto import tqdm
from torch.utils.data import Dataset


class CalBERTDataSet(Dataset):
    def __init__(self, translations, transliterations, labels=None, sample_negative=False, negative_sample_size=0.5,
                 negative_sample_count=1, shuffle=True):
        self.translations = translations
        self.transliterations = transliterations
        self.labels = labels if labels is not None else torch.ones(len(translations))
        assert len(translations) == len(transliterations) == len(labels)

        self.total_examples = len(translations)
        self.sample_negative = sample_negative
        self.negative_sample_size = negative_sample_size
        self.negative_sample_count = negative_sample_count
        random.seed(0)

        if self.sample_negative:
            self.sample_negative_examples()

        if shuffle:
            self.translations, self.transliterations, self.labels = zip(
                *sorted(zip(self.translations, self.transliterations, self.labels)))

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        return self.translations[idx], self.transliterations[idx], self.labels[idx]

    def sample_negative_examples(self):
        current_examples = list(zip(self.translations, self.transliterations, self.labels))
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
                random_sample_transliterations = [self.transliterations[i] for i in random_sample_index]
                if transliteration not in random_sample_transliterations:
                    sampled = True

            for sample in random_sample_items:
                new_examples.append((sample[0], sample[1], 0.0))

            # sample translation items - not needed
            # sampled = False
            # random_sample_items = list()
            # while not sampled:
            #     random_sample_index = random.sample(range(self.total_examples), self.negative_sample_count)
            #     random_sample_items = [current_examples[i] for i in random_sample_index]
            #     random_sample_translations = [self.translations[i] for i in random_sample_index]
            #     if translation not in random_sample_translations:
            #         sampled = True
            #
            # for sample in random_sample_items:
            #     new_examples.append((sample[0], sample[1], 0.0))

            self.translations, self.transliterations, self.labels = zip(*new_examples)
            self.total_examples = len(self.translations)

    def tokens(self):
        # TODO: return the unique tokens of the dataset
        pass

    def vocabulary(self):
        # TODO: use spaCy tokenizer and store all the tokens in a list
        pass

    def get_batch(self, start, end):
        return self.translations[start:end], self.transliterations[start:end], self.labels[start:end]

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
