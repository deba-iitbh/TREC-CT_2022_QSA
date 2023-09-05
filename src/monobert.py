import config
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import ClinicalTrials
from engine import train_eval


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE, sep=" ")
    train_df = df[df.qid < 60]
    train_df = train_df.sample(frac=1)
    eval_df = df[df.qid >= 60]
    eval_df = eval_df.sample(frac=1)

    train_ds = ClinicalTrials(
        q_ids=train_df.qid.values,
        doc_ids=train_df.doc_id.values,
        labels=train_df.rel.values,
    )
    eval_ds = ClinicalTrials(
        q_ids=eval_df.qid.values,
        doc_ids=eval_df.doc_id.values,
        labels=eval_df.rel.values,
    )
    label_weights = train_df.rel.value_counts() / train_df.shape[0]
    label_weights = torch.tensor(label_weights.values, dtype=torch.float32)

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE)
    eval_dl = DataLoader(eval_ds, batch_size=config.BATCH_SIZE)
    print("*****Training*****")
    train_eval(train_dl, eval_dl, label_weights)
