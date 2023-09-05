import config
import torch
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from tqdm import tqdm
import torch.nn as nn
from transformers.optimization import AdamW


# Model Building and training
def train_eval(train_dl, val_dl, label_weights):
    # Load the 'scibert' model from HuggingFace's model hub
    model = BertForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", num_labels=2
    )
    device = config.DEVICE
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    label_weights = label_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=label_weights, reduction="mean")

    train_pgbar = tqdm(train_dl, total=len(train_dl), unit="batch")
    val_pgbar = tqdm(val_dl, total=len(val_dl), unit="batch")

    for epoch in range(config.EPOCHS):
        # Training loop
        model.train()
        loss = None
        for batch in train_pgbar:
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])
            # loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_val_batches = 0
            val_accuracy = 0
            for batch in val_pgbar:
                for k in batch.keys():
                    batch[k] = batch[k].to(device)
                outputs = model(**batch)
                val_loss += criterion(outputs.logits, batch["labels"])
                _, preds = torch.max(outputs.logits, dim=1)
                val_accuracy += torch.sum(preds == batch["labels"]).item()
                num_val_batches += 1

            # Print validation metrics
            print("Epoch:", epoch + 1)
            print("Training loss:", loss.item())
            print("Validation loss:", val_loss / num_val_batches)
            print("Validation accuracy:", val_accuracy / len(val_dl))

    # Save the trained model to disk
    model.save_pretrained("../models/finetuned_sciBERT")


def infer(infer_dl, num_labels):
    # Load the 'scibert' model from HuggingFace's model hub
    model = BertForSequenceClassification.from_pretrained(config.MODEL, num_labels=num_labels)
    device = config.DEVICE
    model.to(device)
    infer_pgbar = tqdm(infer_dl, total=len(infer_dl), unit="batch")
    model.eval()
    labels = []
    confs = []
    with torch.no_grad():
        for batch in infer_pgbar:
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            logits = torch.softmax(outputs.logits, dim=1)
            conf, label = torch.max(logits, dim=1)
            labels.extend(label.cpu().tolist())
            confs.extend(conf.cpu().tolist())
    return labels, confs
