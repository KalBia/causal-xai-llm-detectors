"""RoBERTa-base fine-tuned on M4 (main detector). Labels: 0 = human, 1 = AI-generated."""

import pandas as pd

from .base import Detector


class RobertaDetector(Detector):
    name = "roberta-base"

    def __init__(self, cfg: dict):
        self.cfg = cfg["detector"]
        self.all_cfg = cfg
        self.model = None  # public: Integrated Gradients needs model + tokenizer
        self.tokenizer = None

    def fit(self):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        d = self.cfg
        base = d.get("pretrained", "roberta-base")
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)

        def encode(split):
            df = pd.read_json(f"{self.all_cfg['data']['dir'].strip()}/{split}.jsonl", lines=True)
            enc = tok(df["text"].tolist(), truncation=True, max_length=d["max_length"])
            rows = zip(enc["input_ids"], enc["attention_mask"], df["label"], strict=True)
            return [{"input_ids": i, "attention_mask": m, "labels": int(y)} for i, m, y in rows]

        args = TrainingArguments(
            output_dir=d["checkpoint_dir"],
            num_train_epochs=d["epochs"],
            per_device_train_batch_size=d["batch_size"],
            per_device_eval_batch_size=64,
            learning_rate=d["lr"],
            eval_strategy="epoch",
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",  # best epoch = lowest validation loss
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=self.all_cfg.get("seed", 42),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=encode("train"),
            eval_dataset=encode("val"),
            data_collator=DataCollatorWithPadding(tok),
            processing_class=tok,
        )

        trainer.train()
        trainer.save_model(d["checkpoint_dir"])
        tok.save_pretrained(d["checkpoint_dir"])
        self.load()

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = self.cfg["checkpoint_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def predict_proba(self, texts):
        import torch

        if self.model is None:
            self.load()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self.tokenizer(
                    texts[i : i + 32],
                    truncation=True,
                    max_length=self.cfg["max_length"],
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
                probs += torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].tolist()
        return probs
