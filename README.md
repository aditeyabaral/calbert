# CalBERT
Code-mixed Adaptive Language representations using BERT

CalBERT adapts existing Transformer representations for a language to another language by minimising the semantic space between equivalent sequences in those languages, thus allowing the Transformer to learn representations for the same tokens across two languages. 

CalBERT is language agnostic, and can be used to adapt any language to any other language. It is also task agnostic, and can be fine-tuned on any task.

# How to use CalBERT

CalBERT is primarily meant to be used on an existing pre-trained Transformer model. CalBERT adapts the embeddings of the langauge the Transformer was pre-trained in to another target language which consists of the base language.

## Environment setup

If you use `conda`, you can create an environment with the following command:

```sh
conda env create -f environment.yml
```

You can also use the `requirements.txt` file to create an environment with the following command:

```sh
conda create -n calbert -f requirements.txt
```

## Data Preparation

The following terms will be used extensively in the following sections:

1. **Base Language**: The single language the Transformer was pre-trained in.
2. **Target Language**: The code-mixed language for which the Transformer will be adapting representations. This language is a superset of the base language, since it builds on top of the base language.

Note that the script language used across both languages has to be the same, i.e. Roman(English) for both languages or French for both languages and so on.

### Dataset Format

CalBERT requires code-mixed data in the following format. 

1. The first column contains the sentence in the base language, such as English

2. The second column contains the original sentence in the target language -- the code-mixed language for which the Transformer is trying to adapt representations, such as Hinglish

Examples of such data is given below:

| Translation  | Transliteration  |
|--------------|------------------|
| I am going to the airport today | Main aaj airport jaa raha hoon |
| I really liked this movie | mujhe yeh movie bahut achhi lagi |

An example of such a dataset is placed in the `data/` directory, named `dataset.csv`.

### Dataset Creation

The `utils` folder contains scripts that can help you generate code-mixed datasets. The `create_dataset.py` script can be used to create a dataset in the format described above.

The input to the script is a file which contains newline delimited sentences in either of the following formats:

1. In a code-mixed language (like Hinglish)
2. One of the constituent languages of the code-mixed language *except* the base language (like Hindi).

If your input data is code-mixed, pass `True` for the ```--format``` flag. Else pass `False`.

```sh
usage: python create_dataset.py [-h] --input INPUT --output OUTPUT --target TARGET --base BASE --format FORMAT

Create dataset from text file. Ensure that the text file contains newline delimited sentences either in the target language for adaptation, or one of the constituent languages of the code-mixed language
*except* the base language

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input file
  --output OUTPUT, -o OUTPUT
                        Output file as CSV
  --target TARGET, -t TARGET
                        Language code of one of the constituent languages of the code-mixed language except the base language
  --base BASE, -b BASE  Base language code used to originally pre-train Transformer
  --format FORMAT, -f FORMAT
                        Input data format is code-mixed
```

Example:

```bash
python create_dataset.py
  --input data/input_code_mixed.txt
  --output data/dataset.csv
  --target hi
  --base en
  --format True
```


## Siamese Pre-Training

To perform CalBERT's siamese pre-training, you need to use the `siamese_pretraining.py` script inside `src/`. It takes in the following arguments, all of which are self-explanatory.

```bash
usage: python siamese_pretraining.py [-h] --model MODEL --dataset DATASET [--hub HUB] [--loss LOSS] [--batch_size BATCH_SIZE] [--evaluator EVALUATOR] [--evaluator_examples EVALUATOR_EXAMPLES] [--epochs EPOCHS]
                              [--sample_negative SAMPLE_NEGATIVE] [--sample_size SAMPLE_SIZE] [--username USERNAME] [--password PASSWORD] [--output OUTPUT] [--hub_name HUB_NAME]

Siamese pre-train an existing Transformer model

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Transformer model name/path to siamese pre-train
  --dataset DATASET, -d DATASET
                        Path to dataset in required format
  --hub HUB, -hf HUB    Push model to HuggingFace Hub
  --loss LOSS, -l LOSS  Loss function to use -- cosine, contrastive or online_contrastive
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --evaluator EVALUATOR, -v EVALUATOR
                        Evaluate as you train
  --evaluator_examples EVALUATOR_EXAMPLES, -ee EVALUATOR_EXAMPLES
                        Number of examples to evaluate
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs
  --sample_negative SAMPLE_NEGATIVE, -s SAMPLE_NEGATIVE
                        Sample negative examples
  --sample_size SAMPLE_SIZE, -ss SAMPLE_SIZE
                        Number of negative examples to sample
  --username USERNAME, -u USERNAME
                        Username for HuggingFace Hub
  --password PASSWORD, -p PASSWORD
                        Password for HuggingFace Hub
  --output OUTPUT, -o OUTPUT
                        Output directory path
  --hub_name HUB_NAME, -hn HUB_NAME
                        Name of the model in the HuggingFace Hub
```

Example:

```bash
python siamese_pretraining.py \
  --model xlm-roberta-base \
  --dataset data/dataset.csv \
  --hub False \
  --loss cosine \
  --batch_size 32 \
  --evaluator True \
  --evaluator_examples 1000 \
  --epochs 10 \
  --sample_negative True \
  --sample_size 2 \
  --output saved_models/calbert-xlm-roberta-base
```

The Siamese pre-trained CalBERT model will be saved into the specified output directory as `[model_name]_TRANSFORMER`. This model can now be fine-tuned for any given task.

## Pre-Training from Scratch

If you would like to pre-train your own Transformer from scratch on Masked-Language-Modelling before performing the Siamese Pre-training, you can use the scripts provided in `src/pretrain-transformer/`