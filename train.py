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

    HOSTS = {
        "HOWARD KURTZ",
        "CANDY CROWLEY",
        "SOLEDAD OBRIEN",
        "CAMPBELL BROWN",
        "WOLF BLITZER",
        "JOHN KING",
        "DON LEMON",
        "CAROL COSTELLO",
        "PIERS MORGAN",
        "FREDRICKA WHITFIELD",
        "BRIANNA KEILAR",
        "ARWA DAMON",
        "DAN LOTHIAN",
        "ED HENRY",
        "DAVID SHUSTER",
        "JEFF ZELENY",
        "SUSAN CANDIOTTI",
        "T.J. HOLMES",
        "JOE JOHNS",
        "GLORIA BORGER",
        "RICHARD QUEST",
        "ELISE LABOTT",
        "KIMBERLY DOZIER",
        "JENNIFER GRAY",
        "WILL RIPLEY",
    }

    def speaker_role(name):
        n = name.strip().upper()
        if n in HOSTS:
            return "HOST"
        else:
            return "GUEST"

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df_nisq = pd.read_csv("./NISQ_dataset/final_train.csv", sep=";")

    df_nisq = df_nisq[
        ["index", "question", "question_speaker", "ctx_after1_speaker", "label"]
    ]

    def build_text(row):
        role_q = speaker_role(row["question_speaker"])
        role_after = speaker_role(row["ctx_after1_speaker"])

        spk_q = f"<SPK_Q:{role_q}>"
        spk_after = f"<SPK_AFTER:{role_after}>"

        return f"{spk_q} {spk_after} {row['question']}"

    df_nisq["text"] = df_nisq.apply(build_text, axis=1)

    # Encode labels
    label_encoder = LabelEncoder()
    df_nisq["label_id"] = label_encoder.fit_transform(df_nisq["label"])
    num_labels = len(label_encoder.classes_)

    print("Label mapping:", dict(enumerate(label_encoder.classes_)))

    # -----------------------------
    # Train / Val / Test split
    train_df, temp_df = train_test_split(
        df_nisq,
        test_size=0.20,
        random_state=42,
        stratify=df_nisq["label_id"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # half of 20% → 10%
        random_state=42,
        stratify=temp_df["label_id"],
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

    special_tokens = {
        "additional_special_tokens": [
            "<SPK_Q:HOST>",
            "<SPK_Q:GUEST>",
            "<SPK_AFTER:HOST>",
            "<SPK_AFTER:GUEST>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

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

    model.resize_token_embeddings(len(tokenizer))

    # -----------------------------
    # Training Arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="./roberta_with_speakers",
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
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
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
    trainer.save_model("./roberta_with_speakers_final")
    tokenizer.save_pretrained("./roberta_with_speakers_final")
    joblib.dump(label_encoder, "label_encoder_with_speakers.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
