import os
import sys
import nltk
nltk.download("punkt")

import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, models, evaluation
from sentence_transformers.losses import CosineSimilarityLoss
from transformers import AutoModel

# os.system("transformers-cli login")
MODEL_TYPE = sys.argv[1]

df = pd.read_csv("../data/final.csv")
translation = list(df["translation"].values)
transliteration = list(df["transliteration"].values)

train_examples = list()
for i in range(df.shape[0]):
  train_examples.append(InputExample(texts=[translation[i], transliteration[i]], label=1.0))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
evaluator = evaluation.EmbeddingSimilarityEvaluator(translation, transliteration, [1.0 for _ in range(df.shape[0])])

def getTokens(sentences):
  tokens = list()
  for sent in sentences:
    words = word_tokenize(sent)
    tokens.extend(words)
  return list(set(tokens))

tokens_translation = getTokens(translation)
tokens_transliteration = getTokens(transliteration)
tokens = list(set(translation + transliteration))
print(f"Number of Tokens: {len(tokens)}")

word_embedding_model = models.Transformer(MODEL_TYPE)
word_embedding_model.training = True
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss = CosineSimilarityLoss(model)

# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100, evaluator=evaluator, evaluation_steps=1000, save_best_model=True)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=20, warmup_steps=100)

model.save(f"../models/sentencetransformer/sentencetransformer-{MODEL_TYPE}")
word_embedding_model.save(f"../models/transformer/additionalpretrained-{MODEL_TYPE}")

# word_embedding_model_hub = AutoModel.from_pretrained(f"../models/transformer/additionalpretrained-{MODEL_TYPE}")
# hub_model_name = MODEL_TYPE.lower()
# if "aditeyabaral/" in hub_model_name:
#   hub_model_name = hub_model_name[13:]
# 
# word_embedding_model_hub.push_to_hub(f"additionalpretrained-{hub_model_name}")
# word_embedding_model.tokenizer.push_to_hub(f"additionalpretrained-{hub_model_name}")
# model.save_to_hub(f"sentencetransformer-{hub_model_name}")
# del model, word_embedding_model, word_embedding_model_hub, pooling_model