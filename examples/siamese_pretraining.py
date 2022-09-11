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
parser.add_argument("-ult", "--unlabeled-training", action='store_true', default=False,
                    help="Whether the training data is unlabeled")
parser.add_argument("-de", "--evaluation-data", type=str, required=False, help="Path to the evaluation data file")
parser.add_argument("-ule", "--unlabeled-evaluation", action='store_true', default=False,
                    help="Whether the evaluation data is unlabeled")
parser.add_argument("-ns", "--negative-sampling", action='store_true', required=False, default=False,
                    help="Whether to use negative sampling for training data")
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
parser.add_argument("-lm", "--loss-metric", type=str, required=False, default="simclr",
                    choices=["distance", "hinge", "cosine", "bce", "mae", "mse", "contrastive", "softmargin", "simclr", "kldiv"], help="Loss metric to use")
parser.add_argument("-t", "--temperature", type=float, required=False, default=0.07,
                    help="Temperature for the SimCLR contrastive loss")
parser.add_argument("-lmg", "--loss-margin", type=float, required=False, default=0.25,
                    help="Margin for the contrastive loss")
parser.add_argument("-bs", "--batch-size", type=int, required=False, default=4, help="Batch size")
parser.add_argument("-sbm", "--save-best-model", action='store_true', required=False, default=False,
                    help="Whether to save the best model at the end of pre-training")
parser.add_argument("-sbs", "--save-best-strategy", type=str, required=False, default="epoch",
                    choices=["epoch", "batch"], help="Save best model strategy to use")
parser.add_argument("-oc", "--optimizer-class", type=str, required=False, default="adam",
                    choices=["adam", "sgd", "adagrad", "adadelta", "rmsprop"], help="Optimizer class to use")
parser.add_argument("-op", "--optimizer-path", type=str, required=False, default=None,
                    help="Path to the optimizer state dict to load")
parser.add_argument("-dir", "--model-dir", type=str, required=False, default="./saved_models",
                    help="Directory to save the model")
parser.add_argument("-tb", "--tensorboard", action='store_true', required=False, default=False,
                    help="Whether to use tensorboard for logging")
parser.add_argument("-tb-dir", "--tensorboard-log-dir", type=str, required=False, default="./tensorboard",
                    help="Directory to save tensorboard logs")
parser.add_argument("-d", "--device", type=str, required=False, default="cuda", choices=["cuda", "cpu"],
                    help="Device to use")

args = parser.parse_args()
logging.debug(args)

# Read training dataset
logging.info(f"Reading training dataset from {args.training_data}")
with open(args.training_data, 'r', encoding="utf-8") as training_data_file:
    train_df = pd.read_csv(training_data_file)
base_language_sentences_train = train_df['base_language_sentences'].tolist()
base_language_sentences_train = list(map(str.strip, base_language_sentences_train))
target_language_sentences_train = train_df['target_language_sentences'].tolist()
target_language_sentences_train = list(map(str.strip, target_language_sentences_train))
labels = train_df['label'].tolist() if not args.unlabeled_training else None
train_dataset = CalBERTDataset(
    base_language_sentences=base_language_sentences_train,
    target_language_sentences=target_language_sentences_train,
    labels=labels,
    negative_sampling=args.negative_sampling,
    negative_sampling_size=args.negative_sampling_size,
    negative_sampling_count=args.negative_sampling_count,
    negative_sampling_type=args.negative_sampling_type,
    min_count=args.min_count,
    shuffle=args.shuffle,
)

# Read evaluation dataset
if args.evaluation_data is not None:
    logging.info(f"Reading evaluation dataset from {args.evaluation_data}")
    with open(args.evaluation_data, 'r', encoding="utf-8") as evaluation_data_file:
        eval_df = pd.read_csv(evaluation_data_file)
    base_language_sentences_eval = eval_df['base_language_sentences'].tolist()
    base_language_sentences_eval = list(map(str.strip, base_language_sentences_eval))
    target_language_sentences_eval = eval_df['target_language_sentences'].tolist()
    target_language_sentences_eval = list(map(str.strip, target_language_sentences_eval))
    labels = eval_df['label'].tolist() if not args.unlabeled_evaluation else None
    eval_dataset = CalBERTDataset(
        base_language_sentences=base_language_sentences_eval,
        target_language_sentences=target_language_sentences_eval,
        labels=labels,
        negative_sampling=args.unlabeled_evaluation,
        negative_sampling_size=args.negative_sampling_size,
        negative_sampling_count=args.negative_sampling_count,
        negative_sampling_type=args.negative_sampling_type,
        min_count=args.min_count,
        shuffle=args.shuffle,
    )
else:
    eval_dataset = None

# Initialise CalBERT
logging.info("Initialising CalBERT model")
model = CalBERT(
    model_path=args.model,
    num_pooling_layers=3,#args.num_pooling_layers,
    pooling_method=args.pooling_method,
    device=args.device
)

# Create new tokenizer
logging.info("Training a new tokenizer")
# model.train_new_tokenizer(base_language_sentences_train + target_language_sentences_train)
model.add_tokens_to_tokenizer(train_dataset.tokens)

# Initialise Trainer
logging.info("Initialising Siamese Pre-Trainer")
trainer = SiamesePreTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_strategy=args.evaluation_strategy,
    save_strategy=args.save_strategy,
    learning_rate=args.learning_rate,
    epochs=args.epochs,
    distance_metric=args.distance_metric,
    loss_metric = args.loss_metric,
    temperature=args.temperature,
    loss_margin=args.loss_margin,
    batch_size=args.batch_size,
    save_best_model=args.save_best_model,
    save_best_strategy=args.save_best_strategy,
    optimizer_class=args.optimizer_class,
    optimizer_path=args.optimizer_path,
    model_dir=args.model_dir,
    use_tensorboard=args.tensorboard,
    tensorboard_log_path=args.tensorboard_log_dir,
    device=args.device
)

# Train
logging.info("Starting training")
trainer.train()

# Save final model
logging.info("Saving final model")
trainer.save_model('./saved_models/best_model/')
