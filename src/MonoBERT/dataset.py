import config
from torch.utils.data import Dataset
import pandas as pd
import json
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer


class ClinicalTrials(Dataset):
    """
    Handle Clinical doc and query pairs
    """

    def __init__(self, q_ids, doc_ids, labels=None):
        self.q_ids = q_ids
        self.doc_ids = doc_ids
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL)
        self.max_len = config.MAX_LEN
        self.queries = pd.read_csv(
            "../input/kwqueries2021.txt", sep="\t", index_col=0, names=["qid", "topic"]
        ).to_dict()["topic"]
        with open("../input/trec-ct-json/trec_map.json", "r") as f:
            self.docs = json.load(f)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        q_id = self.q_ids[idx]
        query = self.queries[q_id]
        doc_id = self.doc_ids[idx]
        doc = self.docs[doc_id]

        try:
            inputs = self.tokenizer(
                doc,
                query,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                truncation="only_first",
            )
        except:
            print(doc_id, doc, query)

        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0)

        res = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type_ids,
        }

        if self.labels is not None:
            label = self.labels[idx]
            res["labels"] = torch.tensor(label, dtype=torch.long)
        return res
