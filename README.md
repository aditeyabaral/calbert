# CalBERT - Code-mixed Adaptive Language representations using BERT

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/calbert-code-mixed-adaptive-language-1/sentiment-analysis-on-iitp-product-reviews)](https://paperswithcode.com/sota/sentiment-analysis-on-iitp-product-reviews?p=calbert-code-mixed-adaptive-language-1)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/calbert-code-mixed-adaptive-language-1/sentiment-analysis-on-sail-2017)](https://paperswithcode.com/sota/sentiment-analysis-on-sail-2017?p=calbert-code-mixed-adaptive-language-1)

This repository contains the source code
for [CalBERT - Code-mixed Adaptive Language representations using BERT](http://ceur-ws.org/Vol-3121/short3.pdf),
published at AAAI-MAKE 2022, Stanford University.

CalBERT can be used to adapt existing Transformer language representations into another similar language by minimising
the semantic space between equivalent sentences in those languages, thus allowing the Transformer to learn
representations for words across two languages. It relies on a novel pre-training architecture named Siamese Pre-training to learn task-agnostic and language-agnostic
representations. For more information, please refer to the paper.

This framework allows you to perform CalBERT's Siamese Pre-training to learn representations for your own data and can be used to obtain dense vector representations for words, sentences or paragraphs. The base models used to 
train CalBERT consist of BERT-based Transformer models such as BERT, RoBERTa, XLM, XLNet, DistilBERT, and so on. 
CalBERT achieves state-of-the-art results on the SAIL and IIT-P Product Reviews datasets. CalBERT is also one of the
only models able to learn code-mixed language representations without the need for traditional pre-training methods and 
is currently one of the few models available for Indian code-mixing such as Hinglish.

# Installation

We recommend `Python 3.9` or higher for CalBERT.

## Install PyTorch

Follow [PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details on how to install PyTorch 
with or without CUDA.

## Install CalBERT

### Install with pip
   ```bash
   pip install calbert
   ```

### Install from source
You can also clone the current version from the [repository](https://github.com/aditeyabaral/calbert) and then directly 
install the package.
   ```bash
   pip install -e .
   ```

# Getting Started

You can read the [docs](https://calbert.readthedocs.io/en/latest/) to learn more about how to train CalBERT for your own
use case.

The following example shows you how to use CalBERT to obtain sentence embeddings.

# Training

This framework allows you to also train your own CalBERT models on your own code-mixed data so you can learn
embeddings for your custom code-mixed languages. There are various options to choose from in order to get the best
embeddings for your language.

First, initialise a model with the base Transformer
```python
from calbert import CalBERT
model = CalBERT('bert-base-uncased')
```

Create a CalBERTDataset using your sentences
```python
from calbert import CalBERTDataset
base_language_sentences = [
   "I am going to Delhi today via flight",
   "This movie is awesome!"
]
target_language_sentences = [
   "Main aaj flight lekar Delhi ja raha hoon.",
   "Mujhe yeh movie bahut awesome lagi!"
]
dataset = CalBERTDataset(base_language_sentences, target_language_sentences)
```

Then create a trainer and train the model
```python
from calbert import SiamesePreTrainer
trainer = SiamesePreTrainer(model, dataset)
trainer.train()
```

# Performance

Our models achieve state-of-the-art results on the SAIL and IIT-P Product Reviews datasets.

More information will be added soon.

# Application and Uses

This framework can be used for:

- Computing code-mixed as well as plain sentence embeddings
- Obtaining semantic similarities between any two sentences
- Other textual tasks such as clustering, text summarization, semantic search and many more.

# Citing and Authors

If you find this repository useful, please cite our publication [CalBERT - Code-mixed Apaptive Language representations using BERT](http://ceur-ws.org/Vol-3121/short3.pdf).

```bibtex
@inproceedings{calbert-baral-et-al-2022,
  author    = {Aditeya Baral and
               Aronya Baksy and
               Ansh Sarkar and
               Deeksha D and
               Ashwini M. Joshi},
  editor    = {Andreas Martin and
               Knut Hinkelmann and
               Hans{-}Georg Fill and
               Aurona Gerber and
               Doug Lenat and
               Reinhard Stolle and
               Frank van Harmelen},
  title     = {CalBERT - Code-Mixed Adaptive Language Representations Using {BERT}},
  booktitle = {Proceedings of the {AAAI} 2022 Spring Symposium on Machine Learning
               and Knowledge Engineering for Hybrid Intelligence {(AAAI-MAKE} 2022),
               Stanford University, Palo Alto, California, USA, March 21-23, 2022},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {3121},
  publisher = {CEUR-WS.org},
  year      = {2022},
  url       = {http://ceur-ws.org/Vol-3121/short3.pdf},
  timestamp = {Fri, 22 Apr 2022 14:55:37 +0200}
}
```

# Contact

Please feel free to contact us by emailing us to report any issues or suggestions, or if you have any further
questions.

Contact: - [Aditeya Baral](https://aditeyabaral.github.io/), [aditeya.baral@gmail.com](mailto:aditeya.baral@gmail.com)

You can also contact the other maintainers listed below.

- [Aronya Baksy](mailto:abaksy@gmail.com)
- [Ansh Sarkar](mailto:anshsarkar1@gmail.com)
- [Deeksha D](mailto:deekshad132@gmail.com)
