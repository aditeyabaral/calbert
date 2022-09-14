import torch
import json
import pytest
from calbert import CalBERT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None
def set_model(test_id):
    global model
    model_path = conf[f"test{test_id}"]["config"]["model_path"]
    num_pooling_layers = conf[f"test{test_id}"]["config"]["num_pooling_layers"]
    model = CalBERT(model_path, num_pooling_layers=num_pooling_layers, device=device)

with open("conf/calbert_tests.json") as f:
    conf = json.load(f)
    num_tests = len(conf)

@pytest.mark.parametrize("test_id", range(1, num_tests+1))
def test_model_device_type(test_id):
    set_model(test_id)
    assert model.device == device
    assert model.transformer_model.device.type == device

@pytest.mark.parametrize("test_id", range(1, num_tests+1))
def test_model_initialisation(test_id):
    set_model(test_id)
    assert model.num_pooling_layers == conf[f"test{test_id}"]["config"]["num_pooling_layers"]
    assert model.transformer_model.config._name_or_path == conf[f"test{test_id}"]["config"]["model_path"]
    assert model.transformer_model.config.hidden_size == conf[f"test{test_id}"]["data"]["hidden_size"]
    assert model.tokenizer.vocab_size == conf[f"test{test_id}"]["data"]["default_vocab_size"]
    assert len(list(model.pool.modules())[0]) == conf[f"test{test_id}"]["config"]["num_pooling_layers"]

@pytest.mark.parametrize("test_id", range(1, num_tests+1))
def test_add_tokens(test_id):
    set_model(test_id)
    tokens = ["calbert", "is", "awesome"]
    new_vocab_size = model.add_tokens_to_tokenizer(tokens)
    assert new_vocab_size == conf[f"test{test_id}"]["data"]["new_vocab_size"]
    assert model.transformer_model.config.hidden_size == conf[f"test{test_id}"]["data"]["hidden_size"]

@pytest.mark.parametrize("test_id", range(1, num_tests+1))
def test_train_new_tokenizer(test_id):
    set_model(test_id)
    sentences = [
        "CalBERT is awesome",
        "CalBERT - Code-mixed Apaptive Language representations using BERT",
        "CalBERT can be used to adapt existing Transformer language representations into another similar language by minimising the semantic space between equivalent sentences in those languages, thus allowing the Transformer to learn representations for words across two languages. It relies on a novel pre-training architecture named Siamese Pre-training to learn task-agnostic and language-agnostic representations."
    ]
    new_tokenizer_vocab = model.train_new_tokenizer(sentences)
    assert new_tokenizer_vocab == conf[f"test{test_id}"]["data"]["new_tokenizer_vocab_size"]

# def test_encode():
#     pass

# def test_batch_encode():
#     pass

# def test_embed():
#     pass

# def test_batch_embed():
#     pass

# def test_sentence_embedding():
#     pass

# def test_batch_sentence_embedding():
#     pass

# def test_pooling():
#     pass

# def test_embedding_distance():
#     pass

# def test_embedding_similarity():
#     pass

# def test_distance():
#     pass

# def test_similarity():
#     pass

# def test_forward():
#     pass

# def test_save():
#     pass

# def test_save_pretrained():
#     pass

# def test_load():
#     pass