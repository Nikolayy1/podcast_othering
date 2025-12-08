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

    # -----------------------------
    # Load and Clean Dataset
    # -----------------------------
    df_nisq = pd.read_csv("./NISQ_dataset/final_train.csv", sep=";")
    df_nisq = df_nisq[
        ["index", "question", "question_speaker", "ctx_after1_speaker", "label"]
    ]

    # -----------------------------
    # Define build_text FIRST (important!)
    # -----------------------------
    def build_text(row):
        spk_q = f"<SPK_Q> {row['question_speaker']}"
        spk_after = f"<SPK_AFTER> {row['ctx_after1_speaker']}"
        return f"{spk_q} {spk_after} {row['question']}"

    # Apply clean text transformation
    df_nisq["text"] = df_nisq.apply(build_text, axis=1)

    # Encode labels
    label_encoder = LabelEncoder()
    df_nisq["label_id"] = label_encoder.fit_transform(df_nisq["label"])
    num_labels = len(label_encoder.classes_)

    print("Label mapping:", dict(enumerate(label_encoder.classes_)))

    # -----------------------------
    # Split into train / val / test (80/10/10)
    # -----------------------------
    # Train / Val / Test split
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

    # -----------------------------
    # Tokenizer and Special Tokens
    # -----------------------------
    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # Add only role-level speaker markers
    special_tokens = {"additional_special_tokens": ["<SPK_Q>", "<SPK_AFTER>"]}
    tokenizer.add_special_tokens(special_tokens)

    print("Special tokens:", special_tokens["additional_special_tokens"])

    # Tokenize the dataset
    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=256
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Rename labels column
    tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")

    # Keep only model inputs
    keep = ["input_ids", "attention_mask", "labels"]
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset["train"].column_names if c not in keep]
    )

    tokenized_dataset.set_format("torch")

    # -----------------------------
    # Model
    # -----------------------------
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Adjust embedding size for added tokens
    model.resize_token_embeddings(len(tokenizer))
    
    

    # -----------------------------
    # Training Arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="./roberta_minimal",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        load_best_model_at_end=True,
        logging_steps=20,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp_16=True,
    )

    # -----------------------------
    # Metrics
    # -----------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -----------------------------
    # Evaluate
    # -----------------------------
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print("Test results:", test_results)

    # -----------------------------
    # Save model + tokenizer
    # -----------------------------
    trainer.save_model("./roberta_minimal_2")
    tokenizer.save_pretrained("./roberta_minimal_2")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
