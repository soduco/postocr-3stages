import shutil
from pathlib import Path
import os

import numpy as np
import pandas as pd
from evaluate import load  # Hugging Face
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class DataFrameDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        full_sample = self.feature[idx]
        full_sample["labels"] = self.target[idx]
        return full_sample


class ErrorDetector:
    """
    Perform error detection (i.e. detect samples with a ground truth `y`
    different from the feature `X`).
    """

    def __init__(
        self,
        *,
        save_dir: Path = Path("./models/error_detector"),
        model_name: str = "camembert-base",
        seed: int = 0,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 32,
        warm_start: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Can specify `kwargs` used in
        [`TrainingArguments`](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments)
        of `.fit` function.
        """
        if not warm_start and save_dir.exists():
            shutil.rmtree(save_dir)

        self.save_dir = save_dir
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        ).to(np.float)
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.seed = seed
        self.kwargs = kwargs

        if device not in ["cpu", "cuda"]:
            raise ValueError("device argument should be 'cpu' or 'cuda'")

        self.device = device

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    def _preprocess_samples(self, X: pd.Series, y: pd.Series) -> DataFrameDataset:
        tokenized = [self.tokenizer(feature) for feature in X]
        return DataFrameDataset(tokenized, y)

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """
        Train detector to recognize errored samples.
        """
        if self.device == "cpu":
            self.kwargs.update({"no_cuda": True})
        else:
            self.kwargs.update({"auto_find_batch_size": True})

        args = TrainingArguments(
            f"{self.save_dir / self.model_name}-finetuned",
            save_strategy="epoch",
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            seed=self.seed,
            **self.kwargs,
        )

        trainer = Trainer(
            self.model,
            args,
            train_dataset=self._preprocess_samples(X, y),
            tokenizer=self.tokenizer,
        )
        trainer.train()

    def predict(self, X: pd.Series) -> np.array:
        """
        Predict errored samples.
        """
        encoding_list = [
            self.tokenizer(feature, padding=True, return_tensors="pt").to(self.device) for feature in X
        ]
        errored_samples_pred = [
            self.model(**encoding).logits.item() > 0.5 for encoding in encoding_list
        ]
        return np.array(errored_samples_pred)

    def score(
        self, X: pd.Series, y: pd.Series, *, metric_name: str = "accuracy"
    ) -> float:
        """
        Predict the errored samples and return the "accuracy" against the ground truth `y`.

        The `metric_name` used by default is the "accuracy". But any Hugging Face metric can be used.
        """
        metric = load(metric_name)

        y_pred = self.predict(X)
        return metric.compute(predictions=y_pred, references=y)
