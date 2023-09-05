from elasticsearch import Elasticsearch, helpers
import numpy as np
from tqdm import trange, tqdm
from glob import glob
from timeit import default_timer as timer
import json


class InvertedIndex:
    def __init__(self, user, passwd, index_name):
        self.user = user
        self.passwd = passwd
        self.index_name = index_name
        self._initialise_index()
        self._create_index()

    def _close_index(self):
        self.client.indices.close(index=self.index_name)

    def _open_index(self):
        self.client.indices.open(index=self.index_name)

    def _initialise_index(self):
        self.client = Elasticsearch(
            "http://localhost:9200", basic_auth=(self.user, self.passwd)
        )

    def _delete_index(self):
        self._close_index()
        self.client.indices.delete(index=self.index_name)

    def _create_index(self):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name)

        self._close_index()
        # create the inverted index
        self.client.indices.put_settings(
            settings={
                "index": {
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "b": 0.75,
                            "k1": 1.2,
                            "discount_overlaps": True,
                        }
                    },
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"],
                            }
                        }
                    },
                }
            },
            index=self.index_name,
        )

        self.client.indices.put_mapping(
            index=self.index_name,
            properties={
                "contents": {"type": "text", "analyzer": "my_analyzer"},
            },
        )

        self._open_index()

    def _add_docs(self, path):
        # Get some documents
        with open(path, "r") as f:
            docs = json.load(f)

        # Prepare documents for bulk update
        actions = []
        for idx in trange(len(docs)):
            action = {
                "_index": self.index_name,
                "_id": docs[idx]["id"],
                "_source": {
                    "contents": docs[idx]["contents"],
                },
            }
            actions.append(action)

        # Use Elasticsearch's bulk API to insert documents in bulk, pushing in batches of 10,000
        batch_size = 10000
        for i in tqdm(
            range(0, len(actions), batch_size), total=len(actions) // batch_size
        ):
            batch_actions = actions[i : i + batch_size]
            helpers.bulk(self.client, batch_actions)

    def _search_index(self, match_text):
        results = self.client.search(
            index=self.index_name,
            query={
                "fuzzy": {
                    "contents": {
                        "value": match_text,
                        "fuzziness": "AUTO",
                        "transpositions": True,
                    }
                }
            },
            size=1000,
        )

        return results["hits"]


def search_el(DOCS_FILE, QUERY_FILE, WRITE_FILE):
    index = InvertedIndex("elastic", "123456", "ir")
    index._add_docs(DOCS_FILE)
    with open(QUERY_FILE, "r") as f:
        queries = f.readlines()

    start = timer()
    with open(WRITE_FILE, "w") as f:
        for idx, query in tqdm(enumerate(queries), total=len(queries)):
            results = index._search_index(query)
            # Find the minimum and maximum scores in the results
            min_score = min(hit["_score"] for hit in results["hits"])
            max_score = max(hit["_score"] for hit in results["hits"])
            for res in results["hits"]:
                original_score = res["_score"]
                normalized_score = (original_score - min_score) / (
                    max_score - min_score
                )
                f.write(f"{idx+1} Q0 {res['_id']} {idx+1} {normalized_score} BM25\n")

    print(f"Elapsed Time: {timer() - start:.3f}")
    index._delete_index()


if __name__ == "__main__":
    DOCS_FILE = "../input/trec-ct-json/trec21_content.json"
    QUERY_FILE = "../input/kwqueries2022.txt"
    WRITE_FILE = "../input/BM25_KW_2022.txt"

    index = InvertedIndex("elastic", "123456", "ir")
    index._delete_index()

    search_el(DOCS_FILE, QUERY_FILE, WRITE_FILE)
