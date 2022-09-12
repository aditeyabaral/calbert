import torch
from calbert import CalBERT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CalBERT(
    model_path="bert-base-cased",
    num_pooling_layers=2,
    device=device
)

def test_model_device_type():
    assert model.device == device
    assert model.transformer_model.device.type == device

def test_model_initialisation():
    assert model.num_pooling_layers == 2
    assert model.transformer_model.config._name_or_path == "bert-base-cased"
    assert model.transformer_model.config.hidden_size == 768
    assert model.tokenizer.vocab_size == 28996
    assert len(list(model.pool.modules())[0]) == 2

def test_add_tokens():
    tokens = ["calbert", "is", "awesome"]
    new_vocab_size = model.add_tokens_to_tokenizer(tokens)
    assert new_vocab_size == 28997
    assert model.transformer_model.config.hidden_size == 768