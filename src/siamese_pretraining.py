import os
import platform
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, models, evaluation
from sentence_transformers.losses import CosineSimilarityLoss
from transformers import AutoModel

parser = argparse.ArgumentParser(description='Pre-train a Transformer model from scratch')
parser.add_argument('--model', '-m', type=str, help='Model to train', required=True)
parser.add_argument('--dataset', '-d', type=str, help='Path to dataset', required=True, default="../data/dataset.csv")
parser.add_argument('--hub', '-hf', type=bool, help='Push model to HuggingFace Hub', required=False, default=False)
parser.add_argument('--username', '-u', type=str, help='Username for HuggingFace Hub', required=False)
parser.add_argument('--password', '-p', type=str, help='Password for HuggingFace Hub', required=False)
args = parser.parse_args()

MODEL_NAME = args.model
DATASET_PATH = args.dataset
PUSH_TO_HUB = args.hub
USERNAME = args.username
PASSWORD = args.password

if PUSH_TO_HUB is not None and PUSH_TO_HUB:
  if USERNAME is None or PASSWORD is None:
    print("Please provide username and password for pushing to HuggingFace Hub!\nRun the script with python pretrain_transformer.py -h for help.")
    exit()
  else:
    print("Logging into HuggingFace Hub!")
    if platform.system() == "Linux":
      os.system(f"printf '{USERNAME}\{PASSWORD}' | transformers-cli login")
    else:
      print("Could not login to HuggingFace Hub automatically! Please enter credentials again")
      os.system("transformers-cli login")

df = pd.read_csv(DATASET_PATH)
translation = list(df["translation"].values)
transliteration = list(df["transliteration"].values)

train_examples = list()
for i in range(df.shape[0]):
  train_examples.append(InputExample(texts=[translation[i], transliteration[i]], label=1.0))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
evaluator = evaluation.EmbeddingSimilarityEvaluator(translation, transliteration, [1.0 for _ in range(df.shape[0])])

word_embedding_model = models.Transformer(MODEL_NAME)
word_embedding_model.training = True
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss = CosineSimilarityLoss(model)

# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100, evaluator=evaluator, evaluation_steps=1000, save_best_model=True)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=20, warmup_steps=100)

model.save(f"../models/sentencetransformer/sentencetransformer-{MODEL_NAME}")
word_embedding_model.save(f"../models/transformer/additionalpretrained-{MODEL_NAME}")

if PUSH_TO_HUB is not None and PUSH_TO_HUB:
  print("Pushing to HuggingFace Hub!")
  word_embedding_model_hub = AutoModel.from_pretrained(f"../models/transformer/additionalpretrained-{MODEL_NAME}")
  hub_model_name = MODEL_NAME.lower()
  if "aditeyabaral/" in hub_model_name:
    hub_model_name = hub_model_name[13:]
  
  word_embedding_model_hub.push_to_hub(f"additionalpretrained-{hub_model_name}")
  word_embedding_model.tokenizer.push_to_hub(f"additionalpretrained-{hub_model_name}")
  model.save_to_hub(f"sentencetransformer-{hub_model_name}")
  del model, word_embedding_model, word_embedding_model_hub, pooling_model