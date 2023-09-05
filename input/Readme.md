# Input Data

Can also be used by install pip package[ir-datasets](https://ir-datasets.com/clinicaltrials.html)

## Access Documents (375580)
```py
import ir_datasets
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2022")
for doc in dataset.docs_iter():
    doc # namedtuple<doc_id, title, condition, summary, detailed_description, eligibility>
```

## Access Queries (50)
```py
import ir_datasets
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2022")
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>
```
