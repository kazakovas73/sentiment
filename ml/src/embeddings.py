from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModel
import torch


# load config file
config_path = Path(__file__).parent.parent / "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


tokenizer = AutoTokenizer.from_pretrained(config["model"])
model_hf = AutoModel.from_pretrained(config["model"])


def create_embeddings(texts):
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        embeddings = model_hf(**inputs).last_hidden_state[:, 0, :].numpy()
    return embeddings


def tokenize_dataset(dataset):
    dataset.map()