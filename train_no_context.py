import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import set_seed
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

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
    # Load Dataset
    # -----------------------------
    df_nisq = pd.read_csv("./NISQ_dataset/final_train.csv", sep=";")

    # Keep only question + label
    df_nisq = df_nisq[["question", "label"]]

    # Use ONLY the question, no speakers, no context
    df_nisq["text"] = df_nisq["question"]

    # Encode labels
    label_encoder = LabelEncoder()
    df_nisq["label_id"] = label_encoder.fit_transform(df_nisq["label"])
    num_labels = len(label_encoder.classes_)

    print("Label mapping:", dict(enumerate(label_encoder.classes_)))

    # -----------------------------
    # Train / Val / Test split
    # -----------------------------
    train_df, temp_df = train_test_split(
        df_nisq, test_size=0.20, random_state=42, stratify=df_nisq["label_id"]
    )
    val_df, test_df = train_df, test_df = train_test_split(
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
    # Tokenizer
    # -----------------------------
    model_name = "roberta-large"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # Tokenization
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Rename labels
    tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")

    # Keep only model inputs
    keep = ["input_ids", "attention_mask", "labels"]
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset["train"].column_names if c not in keep]
    )

    tokenized_dataset.set_format("torch")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_nisq["label_id"]),
        y=df_nisq["label_id"],
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # -----------------------------
    # Model
    # -----------------------------
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # -----------------------------
    # Training Arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="./roberta_question_only",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
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
    trainer = WeightedTrainer(
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
    # Save artifacts
    # -----------------------------
    trainer.save_model("./roberta_question_only_final")
    tokenizer.save_pretrained("./roberta_question_only_final")
    joblib.dump(label_encoder, "label_encoder_no_context.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
