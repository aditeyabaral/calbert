import nltk
nltk.download("punkt")

import os
import pandas as pd
from tokenizers import BertWordPieceTokenizer
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer

df = pd.read_csv("../dataset.csv")
print(df.shape)

translation = list(df["translation"].values)
transliteration = list(df["transliteration"].values)
combined = translation + transliteration

if not os.path.isdir("data"):
  os.makedirs("data")
  os.makedirs("data/translation")
  os.makedirs("data/transliteration")
  os.makedirs("data/combined")

for i in range(df.shape[0]):
  translation_text = translation[i].strip()
  transliteration_text = transliteration[i].strip()
  with open(f"data/translation/{i}.txt", 'w') as f1, open(f"data/combined/{i}{i}.txt", 'w') as f2:
    f1.write(translation_text)
    f2.write(translation_text)
  with open(f"data/transliteration/{i}.txt", 'w') as f1, open(f"data/combined/{i}{i}.txt", 'w') as f2:
    f1.write(transliteration_text)
    f2.write(transliteration_text)

translation_path = [f"data/translation/{f}" for f in os.listdir("data/translation")]
transliteration_path = [f"data/transliteration/{f}" for f in os.listdir("data/transliteration")]
combined_path = [f"data/combined/{f}" for f in os.listdir("data/combined")]

sent_tokenizer = SentencePieceBPETokenizer()
word_tokenizer = BertWordPieceTokenizer()
byte_tokenizer = ByteLevelBPETokenizer()

sent_tokenizer.train(files=transliteration_path, min_frequency=2, 
                                  special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'], show_progress=True)

word_tokenizer.train(files=transliteration_path, min_frequency=2, 
                                  special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], show_progress=True)

byte_tokenizer.train(files=transliteration_path, min_frequency=2, 
                                  special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'], show_progress=True)

os.mkdir("../models/tokenizer/sentenceTokenizer")
os.mkdir("../models/tokenizer/wordTokenizer")
os.mkdir("../models/tokenizer/BPETokenizer")

sent_tokenizer.save_model("../models/tokenizer/sentenceTokenizer")
word_tokenizer.save_model("../models/tokenizer/wordTokenizer")
byte_tokenizer.save_model("../models/tokenizer/BPETokenizer")

sent_tokenizer_comb = SentencePieceBPETokenizer()
word_tokenizer_comb = BertWordPieceTokenizer()
byte_tokenizer_comb = ByteLevelBPETokenizer()

sent_tokenizer_comb.train(files=combined_path, min_frequency=2, 
                                  special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'], show_progress=True)

word_tokenizer_comb.train(files=combined_path, min_frequency=2, 
                                  special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], show_progress=True)

byte_tokenizer_comb.train(files=combined_path, min_frequency=2, 
                                  special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'], show_progress=True)

os.mkdir("../models/tokenizer/sentenceTokenizer_combined")
os.mkdir("../models/tokenizer/wordTokenizer_combined")
os.mkdir("../models/tokenizer/BPETokenizer_combined")

sent_tokenizer_comb.save_model("../models/tokenizer/sentenceTokenizer_combined")
word_tokenizer_comb.save_model("../models/tokenizer/wordTokenizer_combined")
byte_tokenizer_comb.save_model("../models/tokenizer/BPETokenizer_combined")