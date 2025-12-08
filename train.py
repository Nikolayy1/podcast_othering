import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import set_seed

from datasets import Dataset, DatasetDict
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch
import joblib


def main():
    set_seed(42)
    # Load NISQ dataset
    df_nisq = pd.read_csv("./NISQ_dataset/final_train.csv", sep=";")

    # As said in the paper, providing the speaker information might improve performance
    df_nisq = df_nisq[
        ["index", "question", "question_speaker", "ctx_after1_speaker", "label"]
    ]

    # Expects only one input text, so we build our input here

    def build_text(row):
        # Speaker pseudo tokens
        spk_q = f"<SPK_Q:{str(row['question_speaker']).replace(' ', '_')}>"
        spk_after = f"<SPK_AFTER:{str(row['ctx_after1_speaker']).replace(' ', '_')}>"

        # Input sequence
        return f"{spk_q} {spk_after} {row['question']}"

    df_nisq["text"] = df_nisq.apply(build_text, axis=1)

    label_encoder = LabelEncoder()
    df_nisq["label_id"] = label_encoder.fit_transform(df_nisq["label"])
    num_labels = len(label_encoder.classes_)

    print("Label mapping:", dict(enumerate(label_encoder.classes_)))

    # 80/10/10 split

    train_df, temp_df = train_test_split(
        df_nisq, test_size=0.20, random_state=42, stratify=df_nisq["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    # Load model and tokenizer

    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # Collect unique speakers
    special_tokens = {"additional_special_tokens": []}

    for spk in df_nisq["question_speaker"].astype(str).unique():
        special_tokens["additional_special_tokens"].append(
            f"<SPK_Q:{spk.replace(' ', '_')}>"
        )

    for spk in df_nisq["ctx_after1_speaker"].astype(str).unique():
        special_tokens["additional_special_tokens"].append(
            f"<SPK_AFTER:{spk.replace(' ', '_')}>"
        )

    # Add tokens
    tokenizer.add_special_tokens(special_tokens)

    # Tokenize dataset

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=128
        )

    # 1. Tokenize
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # 2. Rename label column BEFORE column cleanup
    tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")

    # 3. Remove everything except model inputs + labels
    keep = ["input_ids", "attention_mask", "labels"]
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset["train"].column_names if col not in keep]
    )

    # 4. Set PyTorch format
    tokenized_dataset.set_format("torch")

    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./roberta_minimal",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        load_best_model_at_end=True,
        logging_steps=20,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_results = trainer.evaluate(tokenized_dataset["test"])
    print("Test results:", test_results)

    trainer.save_model("./roberta_minimal_2")
    tokenizer.save_pretrained("./roberta_minimal_2")
    
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
