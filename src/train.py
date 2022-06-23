import logging
import pandas as pd
from calbert import CalBERT
from trainer import SiamesePreTrainer
from dataset import CalBERTDataSet

logging.basicConfig(
    level=logging.NOTSET,
    filemode="w",
    filename="train.log",
    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s",
)

logging.info("Reading dataset")
df = pd.read_csv(r"../data/dataset.csv")
translation = list(df["translation"].values)[:20000]
logging.debug(f"Number of translations: {len(translation)}")
transliteration = list(df["transliteration"].values)[:20000]
logging.debug(f"Number of transliterations: {len(transliteration)}")
dataset = CalBERTDataSet(
    translation,
    transliteration,
    min_count=50,
    # negative_sampling=True,
    # negative_sampling_size=0.05
)
# dataset.save('../data/dataset.pickle')
# dataset = CalBERTDataSet.load('../data/dataset.pickle')
logging.debug(f"Dataset has {len(dataset)} examples")

model = CalBERT(
    model_path='xlm-roberta-base',
    num_pooling_layers=1
)
# logging.debug(model)
model.add_tokens_to_tokenizer(dataset.get_tokens())

trainer = SiamesePreTrainer(
    model=model,
    train_dataset=dataset,
    batch_size=32,
    learning_rate=0.9
)
trainer.train()


