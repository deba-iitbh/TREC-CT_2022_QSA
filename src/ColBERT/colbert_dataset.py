import config
from torch.utils.data import Dataset
import pandas as pd
import json


class ClinicalTrials(Dataset):
    """
    Handle Clinical doc and query pairs
    """

    def __init__(self, q_ids, doc_ids, tokenizer):
        self.q_ids = q_ids
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self.max_len = config.MAX_LEN
        self.queries = pd.read_csv(
            "../input/kwqueries2022.txt", sep="\t", index_col=0, names=["qid", "topic"]
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

        query_input = self.tokenizer(
            query,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation="only_first",
        )
        doc_input = self.tokenizer(
            doc,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation="only_first",
        )

        for k in query_input:
            query_input[k] = query_input[k].squeeze(0)

        for k in doc_input:
            doc_input[k] = doc_input[k].squeeze(0)
        # query_input.input_ids += [103] * 8  # [MASK]
        # query_input.attention_mask += [1] * 8
        # query_input["input_ids"] = torch.LongTensor(query_input.input_ids).unsqueeze(0)
        # query_input["attention_mask"] = torch.LongTensor(
        # query_input.attention_mask
        # ).unsqueeze(0)

        return query_input, doc_input
