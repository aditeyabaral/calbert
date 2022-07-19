.. CalBERT documentation master file, created by
   sphinx-quickstart on Tue Jul 19 19:08:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CalBERT
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   calbert

This repository contains the source code
for `CalBERT - Code-mixed Apaptive Language representations using BERT <http://ceur-ws.org/Vol-3121/short3.pdf>`_,
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

Installation
============

We recommend `Python 3.9` or higher for CalBERT.

PyTorch with CUDA
+++++++++++++++++

If you want to use a GPU/ CUDA, you must install PyTorch with the matching CUDA Version. Follow 
`PyTorch - Get Started <https://pytorch.org/get-started/locally/>`_ for further details how to install PyTorch with CUDA.


Install with pip
----------------

.. code:: bash

   pip install calbert

Install from source
-------------------
You can also clone the current version from the `repository <https://github.com/aditeyabaral/calbert>`_ and then directly 
install the package.

.. code:: bash

   pip install -e .

Getting Started
===============

Detailed documentation coming soon.

The following example shows you how to use CalBERT to obtain sentence embeddings.

Training
========

This framework allows you to also train your own CalBERT models on your own code-mixed data so you can learn
embeddings for your custom code-mixed languages. There are various options to choose from in order to get the best
embeddings for your language.

First, initialise a model with the base Transformer

.. code:: python

   from calbert import CalBERT
   model = CalBERT('bert-base-uncased')

Create a CalBERTDataset using your sentences

.. code:: python

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



Then create a trainer and train the model

.. code:: python

   from calbert import SiamesePreTrainer
   trainer = SiamesePreTrainer(model, dataset)
   trainer.train()

Performance
===========

Our models achieve state-of-the-art results on the SAIL and IIT-P Product Reviews datasets.

More information will be added soon.

Application and Uses
====================

This framework can be used for:

* Computing code-mixed as well as plain sentence embeddings
* Obtaining semantic similarities between any two sentences
* Other textual tasks such as clustering, text summarization, semantic search and many more.