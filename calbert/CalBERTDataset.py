import random
from collections import Counter
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CalBERTDataset(Dataset):
    def __init__(self, base_language_sentences: list[str], target_language_sentences: list[str], labels: float = None,
                 negative_sampling: bool = False, negative_sampling_size: float = 0.5, negative_sampling_count: int = 1,
                 negative_sampling_type: str = 'target', min_count: int = 10, shuffle: bool = True):
        """
        Create a CalBERTDataset from a list of base language sentences and target language sentences.

        :param base_language_sentences: Sentences in the base language
        :param target_language_sentences: Sentences in the target (code-mixed) language
        :param labels: Labels (binary or similarity scores) indicating relationship between base and target sentences
        :param negative_sampling: Whether to perform negative sampling of examples
        :param negative_sampling_size: Percentage of dataset to use for negative sampling
        :param negative_sampling_count: Number of negative samples to sample per positive example
        :param negative_sampling_type: Whether to sample from the base language or the target language or both
        :param min_count: Minimum frequency of a token in the dataset to be included in the vocabulary
        :param shuffle: Whether to shuffle the dataset
        """
        self.base_language_sentences = base_language_sentences
        self.target_language_sentences = target_language_sentences
        self.labels = torch.tensor(labels) if labels is not None else torch.ones(len(base_language_sentences))
        assert len(self.base_language_sentences) == len(self.target_language_sentences) == len(self.labels)

        self.total_examples = len(base_language_sentences)
        self.negative_sampling = negative_sampling
        self.negative_sampling_size = negative_sampling_size
        self.negative_sampling_count = negative_sampling_count
        self.negative_sampling_type = negative_sampling_type
        self.min_count = min_count
        random.seed(0)

        if self.negative_sampling:
            self.sample_negative_examples(self.negative_sampling_type)

        if shuffle:
            items = list(zip(self.base_language_sentences, self.target_language_sentences, self.labels))
            random.shuffle(items)
            self.base_language_sentences, self.target_language_sentences, self.labels = zip(*items)
            self.base_language_sentences = list(self.base_language_sentences)
            self.target_language_sentences = list(self.target_language_sentences)
            self.labels = torch.tensor(self.labels)

        self.tokens = list()
        self.compute_vocabulary(self.min_count)

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.

        :return: Number of examples in the dataset
        """
        return self.total_examples

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        """
        Obtain the base language sentence, target language sentence, and label at the given index in the dataset.

        :param idx: Index of the example in the dataset
        :return: A tuple of base language sentence, target language sentence, and label at the given index
        """
        return self.base_language_sentences[idx], self.target_language_sentences[idx], self.labels[idx]

    def sample_negative_examples(self, sampling: str = 'target') -> None:
        """
        Sample negative examples from the dataset for each positive example.

        :param sampling: Whether to sample from the base language or the target language or both
        :return: None
        """
        current_examples = list(zip(self.base_language_sentences, self.target_language_sentences, self.labels))
        new_examples = list()
        for example in tqdm(random.sample(current_examples, int(self.negative_sampling_size * self.total_examples))):
            translation = example[0]
            transliteration = example[1]
            label = example[2]
            new_examples.append((translation, transliteration, label))

            if sampling == 'target' or sampling == 'both':
                sampled = False
                random_sample_items = list()
                while not sampled:
                    random_sample_index = random.sample(range(self.total_examples), self.negative_sampling_count)
                    random_sample_items = [current_examples[i] for i in random_sample_index]
                    random_sample_target_language_sentences = [self.target_language_sentences[i] for i in
                                                               random_sample_index]
                    if transliteration not in random_sample_target_language_sentences:
                        sampled = True
                for sample in random_sample_items:
                    new_examples.append((sample[0], sample[1], 0.0))

            if sampling == 'base' or sampling == 'both':
                sampled = False
                random_sample_items = list()
                while not sampled:
                    random_sample_index = random.sample(range(self.total_examples), self.negative_sampling_count)
                    random_sample_items = [current_examples[i] for i in random_sample_index]
                    random_sample_base_language_sentences = [self.base_language_sentences[i] for i in
                                                             random_sample_index]
                    if translation not in random_sample_base_language_sentences:
                        sampled = True
                for sample in random_sample_items:
                    new_examples.append((sample[0], sample[1], 0.0))

            random.shuffle(new_examples)
            self.base_language_sentences, self.target_language_sentences, self.labels = zip(*new_examples)
            self.total_examples = len(self.base_language_sentences)

    def compute_vocabulary(self, min_count: int = None) -> list[str]:
        """
        Compute the vocabulary of the dataset by finding tokens appearing atleast min_count times.

        :param min_count: Minimum frequency of a token in the dataset to be included in the vocabulary
        :return: List of tokens in the dataset appearing atleast min_count times
        """
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

    def get_tokens(self) -> list[str]:
        """
        Returns the vocabulary of the dataset computed by compute_vocabulary.

        :return: List of tokens in vocabulary
        """
        return self.tokens

    def get_batch(self, start: int, end: int) -> tuple[list[str], list[str], torch.Tensor]:
        """
        Returns a batch of examples from the dataset between the given start and end indices.

        :param start: Start index of the batch in the dataset
        :param end: End index of the batch in the dataset
        :return: A tuple of base language sentences, target language sentences, and labels between the given start and
        end indices
        """
        return self.base_language_sentences[start:end], self.target_language_sentences[start:end], self.labels[
                                                                                                   start:end]

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the dataset object to the given path.

        :param path: Path to save the dataset object
        :return: None
        """
        torch.save(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> 'CalBERTDataset':
        """
        Load a CalBertDataset object from the given path.

        :param path: Path to load the dataset object
        :return: CalBertDataset object
        """
        return torch.load(path)
