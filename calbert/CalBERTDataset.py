import logging
import random
from collections import Counter
from pathlib import Path
from typing import Union, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CalBERTDataset(Dataset):
    def __init__(self, base_language_sentences: List[str], target_language_sentences: List[str], labels: float = None,
                 negative_sampling: bool = False, negative_sampling_size: float = 0.5, negative_sampling_count: int = 1,
                 negative_sampling_type: str = 'target', min_count: int = 10, shuffle: bool = True):
        """Create a CalBERTDataset from a list of base language sentences and target language sentences.

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
        logging.info(f"Creating CalBERTDataset")
        logging.debug(f"Creating CalBERTDataset with args: {locals()}")

        self.base_language_sentences = base_language_sentences
        self.target_language_sentences = target_language_sentences

        if labels is not None:
            self.labels = torch.tensor(labels)
            logging.info(f"Labels provided")
        else:
            self.labels = torch.ones(len(base_language_sentences))
            logging.info(f"No labels provided, using 1.0 as similarity scores for examples")

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
            logging.info("Shuffling dataset")
            items = list(zip(self.base_language_sentences, self.target_language_sentences, self.labels))
            random.shuffle(items)
            self.base_language_sentences, self.target_language_sentences, self.labels = zip(*items)
            self.base_language_sentences = list(self.base_language_sentences)
            self.target_language_sentences = list(self.target_language_sentences)
            self.labels = torch.tensor(self.labels)

        self.tokens = list()
        self.compute_vocabulary(self.min_count)

    def __len__(self) -> int:
        """Returns the total number of examples in the dataset.

        :return: Number of examples in the dataset
        """
        return self.total_examples

    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        """Obtain the base language sentence, target language sentence, and label at the given index in the dataset.

        :param idx: Index of the example in the dataset
        :return: A tuple of base language sentence, target language sentence, and label at the given index
        """
        logging.debug(f"Obtaining batch at index {idx}")
        return self.base_language_sentences[idx], self.target_language_sentences[idx], self.labels[idx]

    def sample_negative_examples(self, sampling: str = 'target') -> None:
        """Sample negative examples from the dataset for each positive example.

        :param sampling: Whether to sample from the base language or the target language or both
        :return: None
        """
        logging.info(f"Shuffling dataset with negative sampling size {self.negative_sampling_size}, "
                     f"negative sampling count {self.negative_sampling_count}, "
                     f"negative sampling type {self.negative_sampling_type}"
                     )
        current_examples = list(zip(self.base_language_sentences, self.target_language_sentences, self.labels))
        new_examples = list()
        for example in tqdm(random.sample(current_examples, int(self.negative_sampling_size * self.total_examples))):
            translation = example[0]
            transliteration = example[1]

            if sampling == 'target' or sampling == 'both':
                logging.debug(f"Sampling negative examples from target language")
                sampled = False
                random_sample_items = list()
                while not sampled:
                    random_sample_index = random.sample(range(self.total_examples), self.negative_sampling_count)
                    random_sample_items = list()
                    random_sample_target_language_sentences = list()
                    for i in random_sample_index:
                        random_sample_items.append(current_examples[i])
                        random_sample_target_language_sentences.append(self.target_language_sentences[i])
                    if transliteration not in random_sample_target_language_sentences:
                        sampled = True
                for sample in random_sample_items:
                    new_examples.append((translation, sample[1], 0.0))

            if sampling == 'base' or sampling == 'both':
                logging.debug(f"Sampling negative examples from base language")
                sampled = False
                random_sample_items = list()
                while not sampled:
                    random_sample_index = random.sample(range(self.total_examples), self.negative_sampling_count)
                    random_sample_items = list()
                    random_sample_base_language_sentences = list()
                    for i in random_sample_index:
                        random_sample_items.append(current_examples[i])
                        random_sample_base_language_sentences.append(self.base_language_sentences[i])
                    if translation not in random_sample_base_language_sentences:
                        sampled = True
                for sample in random_sample_items:
                    new_examples.append((sample[0], transliteration, 0.0))

        logging.info("Adding sampled negative examples to dataset")
        current_examples += new_examples
        random.shuffle(current_examples)
        self.base_language_sentences, self.target_language_sentences, self.labels = zip(*current_examples)
        self.base_language_sentences = list(self.base_language_sentences)
        self.target_language_sentences = list(self.target_language_sentences)
        self.labels = torch.tensor(self.labels)
        self.total_examples = len(self.base_language_sentences)

    def compute_vocabulary(self, min_count: int = None) -> List[str]:
        """Compute the vocabulary of the dataset by finding tokens appearing atleast min_count times.

        :param min_count: Minimum frequency of a token in the dataset to be included in the vocabulary
        :return: List of tokens in the dataset appearing atleast min_count times
        """
        logging.info(f"Computing vocabulary with min_count {min_count}")
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

    def get_tokens(self) -> List[str]:
        """Returns the vocabulary of the dataset computed by compute_vocabulary.

        :return: List of tokens in vocabulary
        """
        return self.tokens

    def get_batch(self, start: int, end: int) -> Tuple[List[str], List[str], torch.Tensor]:
        """Returns a batch of examples from the dataset between the given start and end indices.

        :param start: Start index of the batch in the dataset
        :param end: End index of the batch in the dataset
        :return: A tuple of base language sentences, target language sentences, and labels between the given start and end indices
        """
        logging.debug(f"Obtaining batch from index {start} to {end}")
        return self.base_language_sentences[start:end], self.target_language_sentences[start:end], self.labels[
                                                                                                   start:end]

    def save(self, path: Union[str, Path]) -> None:
        """Save the dataset object to the given path.

        :param path: Path to save the dataset object
        :return: None
        """
        logging.info(f"Saving CalBERTDataset dataset to {path}")
        torch.save(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> 'CalBERTDataset':
        """Load a CalBertDataset object from the given path.

        :param path: Path to load the dataset object
        :return: CalBertDataset object
        """
        logging.info(f"Loading CalBERTDataset dataset from {path}")
        return torch.load(path)
