import pandas as pd
from datasets import Dataset
from pathlib import Path
from data import etl

class SentimentDataset(Dataset):
    def __init__(self, dataset):
        datapath = Path(__file__).parent.parent.parent / "data" / dataset
        df = pd.read_csv(datapath, header=None)
        self.texts, self.targets = etl(df)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            self.embeddings[idx], 
            self.targets[idx]
        }