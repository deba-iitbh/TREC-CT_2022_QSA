import config
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ClinicalTrials
from engine import infer


if __name__ == "__main__":
    df = pd.read_csv(
        "../input/BM25_FQ_2022.txt",
        names=["qid", "Q0", "doc_id", "rank", "bm25_score", "run"],
        sep=" ",
    )

    ds = ClinicalTrials(
        q_ids=df.qid.values,
        doc_ids=df.doc_id.values,
    )

    dl = DataLoader(ds, batch_size=config.BATCH_SIZE)
    # predictions = infer(dl)
    # df.loc["bert_score"] = predictions
    # df.to_csv("../output/BlueKW22.csv", index=False)

    label, confs = infer(dl, num_labels=3)
    df["rel"] = label
    df["rel_score"] = confs
    print(df[["qid", "rel", "rel_score"]].head())
    df.to_csv("../output/ClinicalKW22.csv", index=False)
