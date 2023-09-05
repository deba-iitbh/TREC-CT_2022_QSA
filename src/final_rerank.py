import pandas as pd

def consolidate_df(bm25, monobert):
    bm25 = bm25.sort_values(by=["qid", "doc_id"], ignore_index=True)
    monobert = monobert.sort_values(
        by=["qid"m "doc_id"],
        ignore_index=True,
    )

    monobert["bert_score"] = bm25["score"]
    monobert["score"] = 0.4 * monobert["bm25_score"] + 0.6 * monobert["bert_score"]
    monobert["Q0"] = "Q0"
    monobert["run"] = "IR-Project"
    res = monobert[["qid", "Q0", "doc_id", "rank", "score", "run"]]
    return res

if __name__ == "__main__":
    bm25 = pd.read_csv(
        "../input/BM25_KW_2022.txt",
        names=["qid", "Q0", "doc_id", "rank", "score", "run"],
        sep=" ",
    )
    monobert = pd.read_csv("../output/MonoKW22.csv")
    final = consolidate_df(bm25,monobert)
    final.to_csv("../output/final22.txt", sep=" ", header=False, index=False)
