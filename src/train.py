import logging
import argparse
import pandas as pd

from calbert import CalBERT, CalBERTDataset, SiamesePreTrainer

logging.basicConfig(
    level=logging.NOTSET,
    filemode="w",
    filename="./train.log",
    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s",
)

# Dataset args
parser = argparse.ArgumentParser("Pre-train a Transformer using CalBERT")
parser.add_argument("-dt", "--training-data", type=str, required=True, help="Path to the data file")
parser.add_argument("-ul", "--unlabeled", action='store_true', required=True, help="Whether the data is unlabeled")
parser.add_argument("-ns", "--negative-sampling", action='store_true', required=False, default=False,
                    help="Whether to use negative sampling")
parser.add_argument("-nss", "--negative-sampling-size", type=float, required=False, default=0.5,
                    help="Percentage of dataset to use for negative sampling")
parser.add_argument("-nsc", "--negative-sampling-count", type=int, required=False, default=1,
                    help="Number of negative samples to sample per positive example")
parser.add_argument("-nst", "--negative-sampling-type", type=str, required=False, default='target',
                    choices=['base', 'target', 'both'],
                    help="Whether to sample from the base language or the target language or both")
parser.add_argument("-mc", "--min-count", type=int, required=False, default=10,
                    help="Minimum frequency of a token in the dataset to be included in the vocabulary")
parser.add_argument("-s", "--shuffle", action='store_true', required=False, default=False,
                    help="Whether to shuffle the dataset")

# Model args
parser.add_argument("-m", "--model", type=str, required=True, help="Transformer model to pre-train")
parser.add_argument("-np", "--num-pooling-layers", type=int, required=False, default=1,
                    help="Number of pooling layers to use")
parser.add_argument("-pm", "--pooling-method", type=str, required=False, default="mean", choices=["mean", "max"],
                    help="Pooling method to use")

# Trainer args
parser.add_argument("-es", "--evaluation-strategy", type=str, required=False, default="epoch",
                    choices=["batch", "epoch"], help="Evaluation strategy to use")
parser.add_argument("-ss", "--save-strategy", type=str, required=False, default="epoch", choices=["epoch", "batch"],
                    help="Save strategy to use")
parser.add_argument("-lr", "--learning-rate", type=float, required=False, default=0.01, help="Learning rate")
parser.add_argument("-ep", "--epochs", type=int, required=False, default=20, help="Number of epochs")
parser.add_argument("-ds", "--distance-metric", type=str, required=False, default="cosine",
                    choices=["cosine", "euclidean", "manhattan"], help="Distance metric to use")
parser.add_argument("-ucl", "--use-contrastive-loss", action='store_true', required=False, default=True,
                    help="Whether to use contrastive loss")
parser.add_argument("-clt", "--contrastive-loss-type", type=str, required=False, default="softmax",
                    choices=["binary", "softmax", "linear"], help="Contrastive loss type to use")
parser.add_argument("-t", "--temperature", type=float, required=False, default=0.07,
                    help="Temperature for the softmax contrastive loss")
parser.add_argument("-lm", "--loss-margin", type=float, required=False, default=0.25,
                    help="Margin for the contrastive loss")
parser.add_argument("-bs", "--batch-size", type=int, required=False, default=32, help="Batch size")
parser.add_argument("-sbm", "--save-best-model", action='store_true', required=False, default=False,
                    help="Whether to save the best model at the end of pre-training")
parser.add_argument("-oc", "--optimizer-class", type=str, required=False, default="adam",
                    choices=["adam", "sgd", "adagrad", "adadelta", "rmsprop"], help="Optimizer class to use")
parser.add_argument("-op", "--optimizer-path", type=str, required=False, default=None,
                    help="Path to the optimizer state dict to load")
parser.add_argument("-dir", "--save-dir", type=str, required=False, default="./saved_models",
                    help="Directory to save the model")
parser.add_argument("-d", "--device", type=str, required=False, default="cuda", choices=["cuda", "cpu"],
                    help="Device to use")

args = parser.parse_args()

# Read dataset
with open(args.training_data, 'r') as training_data_file:
    train_df = pd.read_csv(training_data_file)
base_language_sentences = train_df['base_language_sentences'].tolist()
target_language_sentences = train_df['target_language_sentences'].tolist()
labels = train_df['label'].tolist() if not args.unlabeled else None
train_dataset = CalBERTDataset(
    base_language_sentences=base_language_sentences,
    target_language_sentences=target_language_sentences,
    labels=labels,
    negative_sampling=args.negative_sampling,
    negative_sampling_size=args.negative_sampling_size,
    negative_sampling_count=args.negative_sampling_count,
    negative_sampling_type=args.negative_sampling_type,
    min_count=args.min_count,
    shuffle=args.shuffle,
)

# Initialise CalBERT
model = CalBERT(
    model_path=args.model,
    num_pooling_layers=args.num_pooling_layers,
    pooling_method=args.pooling_method,
    device=args.device
)

# Initialise Trainer

trainer = SiamesePreTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=None,
    eval_strategy=args.evaluation_strategy,
    save_strategy=args.save_strategy,
    learning_rate=args.learning_rate,
    epochs=args.epochs,
    distance_metric=args.distance_metric,
    use_contrastive_loss=args.use_contrastive_loss,
    contrastive_loss_type=args.contrastive_loss_type,
    temperature=args.temperature,
    loss_margin=args.loss_margin,
    batch_size=args.batch_size,
    save_best_model=args.save_best_model,
    optimizer_class=args.optimizer_class,
    optimizer_path=args.optimizer_path,
    model_dir=args.save_dir,
    device=args.device
)

# Train
trainer.train()
