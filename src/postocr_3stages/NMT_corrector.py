import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
import re

import numpy as np
import pandas as pd

from .utils import save_df_on_disk, load_from_disk, nmt_score


def format_kwargs(kwargs):
    return [f"--{key}={value}" for key, value in kwargs.items()]


def run_subprocess(command, **kwargs):
    formatted_kwargs = format_kwargs(kwargs)
    return subprocess.run(command + formatted_kwargs)


class NMTCorrector:
    """
    Perform correction of samples `X` to their corresponding ground truth `y` using
    a Neural Machine Translation (NMT) model.

    Any arguments of 'onmt_train' (e.g. gpu_ranks)can be pass in the `init()` function:
    https://opennmt.net/OpenNMT-py/options/train.html
    """

    def __init__(
        self,
        *,
        save_path: Path = Path("./models/nmt/"),
        learning_rate: float = 0.5,
        seed: int = 0,
        vocab_n_sample: int = 10_000,
        train_steps: int = 2_000,
        warm_start: bool = True,
        **kwargs,
    ):
        if not warm_start and save_path.exists():
            shutil.rmtree(save_path)

        self.save_path = save_path
        self.learning_rate = learning_rate
        self.seed = seed
        self.vocab_n_sample = vocab_n_sample
        self.train_steps = train_steps
        self.kwargs = kwargs

    def _write_config_file(self, X_path: Path, Y_path: Path, config_path: Path):
        with open(config_path, "w") as config_fp:
            config_fp.write(
                "\n".join(
                    [
                        "data:",
                        "    train:",
                        f"        path_src: {X_path}",
                        f"        path_tgt: {Y_path}",
                        "    valid:",
                        f"        path_src: {X_path}",
                        f"        path_tgt: {Y_path}\n",
                    ]
                )
            )

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """
        Train a NMT model to predict correction for samples `X` with their corresponding
        ground truth `y`.
        """
        with TemporaryDirectory() as dir_path:
            X_path = Path(dir_path) / "train.in"
            Y_path = Path(dir_path) / "train.out"
            config_path = Path(dir_path) / "config.yaml"

            save_df_on_disk(X, X_path)
            save_df_on_disk(y, Y_path)

            self._write_config_file(X_path, Y_path, config_path)

            # Build vocab
            save_data = self.save_path / "data"
            run_subprocess(
                ["onmt_build_vocab", "-overwrite"],
                config=config_path,
                n_sample=self.vocab_n_sample,
                save_data=save_data / "samples",
                src_vocab=save_data / "vocab.src",
                tgt_vocab=save_data / "vocab.tgt",
                seed=self.seed,
            )

            # Training
            run_subprocess(
                ["onmt_train", "-overwrite"],
                config=config_path,
                num_workers=0,
                save_model=self.save_path / "model/checkpoint",
                save_data=save_data / "samples",
                src_vocab=save_data / "vocab.src",
                tgt_vocab=save_data / "vocab.tgt",
                learning_rate=self.learning_rate,
                seed=self.seed,
                valid_steps=self.train_steps + 1,
                train_steps=self.train_steps,
                save_checkpoint_steps=self.train_steps // 5,
                **self.kwargs,
            )

    def _get_model_path(self):
        dir_path = self.save_path / "model"

        def get_num(path: Path):
            return int(path.stem.split("_")[-1])

        return max(Path(dir_path).glob("*.pt"), key=get_num)

    def predict(self, X: pd.Series, **kwargs) -> pd.Series:
        """
        Use the NMT model to predict correction for samples `X`.
        """
        if len(X) == 0:
            return pd.Series(dtype=str)

        with TemporaryDirectory() as dir_path:
            X_path = Path(dir_path) / "test.in"
            Y_path = Path(dir_path) / "test.out"

            save_df_on_disk(X, X_path)
            run_subprocess(
                ["onmt_translate"],
                model=self._get_model_path(),
                src=X_path,
                output=Y_path,
                **kwargs,
            )

            y_pred = load_from_disk(Y_path, template_df=X)

        return y_pred

    def score(
        self,
        X: pd.Series,
        y: pd.Series,
        *,
        averaging: str = "macro",
        per_sample_metric: bool = False,
    ) -> Union[float, pd.Series]:
        """
        Compute prediction on `X` using `self.predict` and return its
        Character-Error-Rate against the Ground Truth `y`.

        Note:
        - The `averaging` specifies how the Character-Error-Rate is computed.
          Implicitly, the "macro"-average weights evenly every sample, while
          "micro"-average considers sample length. Any other average will result in
          a RuntimeError.
        - The `per_sample_metric` enable computation of the CER at the sample level.
            In this case, the `averaging` option is not used.
        - The CER metric is a 'lower-is-better' unnormalized metric. Its uge inside
            the sklearn's API might need some small modification.
        """
        y_pred = self.predict(X)
        return nmt_score(
            y_pred,
            y,
            averaging=averaging,
            per_sample_metric=per_sample_metric,
        )
