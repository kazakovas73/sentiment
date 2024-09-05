import pandas as pd
from pathlib import Path
from embeddings import create_embeddings
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np

# load config file
config_path = Path(__file__).parent.parent / "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def etl(df: pd.DataFrame):
    print("\tprocessing...")
    df.drop(columns=[0,1], inplace=True)
    df.columns = ['label', 'text']
    labels = sorted(df.label.unique())
    label2id = dict(zip(labels, range(len(labels))))
    df['target'] = df.label.apply(lambda x: label2id[x])
    df.dropna(inplace=True)

    if config['cut_len']:
        df = df.head(config['cut_len'])

    data_loader = DataLoader(df.text.values.tolist(), batch_size=config['batch_size'], shuffle=False)
    embeddings = []
    for batch in tqdm(data_loader):
        batch_embeddings = create_embeddings(batch)
        embeddings.append(batch_embeddings)

    # texts = df.text.values
    target = df.target.values
    del df

    print("\tsuccess")
    return np.concatenate(embeddings, axis=0), target


def split_datasets(path: str):
    print("-- Splitting datasets")
    df_train = pd.read_csv(Path(path) / 'twitter_training.csv', header=None)
    df_valid = pd.read_csv(Path(path) / 'twitter_validation.csv', header=None)

    X_train, y_train = etl(df_train)
    X_valid, y_valid = etl(df_valid)

    print(X_train.shape)

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,random_state=42, test_size=0.2)

    return X_train, X_valid, y_train, y_valid
