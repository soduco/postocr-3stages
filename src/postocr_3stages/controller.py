import numpy as np
import pandas as pd
from evaluate import load  # Hugging Face
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import isri_tools


class LengthController:
    """
    Performs length post-processing.
    The `X` predictions is compared to `ref_samples` in order to only keep the
    the less-different samples (i.e. the non-aberating corrections).

    Aberating modification are revert to the original non-corrected `ref_samples`.
    """
    def __init__(
        self,
        *,
        high_cut_threshold: float = 1.70,
        low_cut_threshold: float = 0.90,
    ):
        self.high_cut_threshold = high_cut_threshold
        self.low_cut_threshold = low_cut_threshold

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.Series, *, ref_samples: pd.Series = None) -> pd.Series:
        """
        Performs length post-processing.
        The `X` predictions is compared to `ref_samples` in order to only keep the
        the less-different samples (i.e. the non-aberating corrections).

        Aberating modification are revert to the original non-corrected `ref_samples`.

        `ref_samples` is a necessary argument. Without it, `predict` function won't
        do anything.
        """
        if ref_samples is None:
            return X

        ratios = X.str.len() / ref_samples.str.len()
        mask = (ratios > self.high_cut_threshold) | (ratios < self.low_cut_threshold)

        y_pred = X.copy()
        y_pred.loc[mask] = ref_samples.loc[mask]

        return y_pred

    def score(self, X: pd.Series, y: pd.Series, *, averaging: str = "macro") -> float:
        """
        Compute prediction on `X` using `self.predict` and return its
        Character-Error-Rate against the Ground Truth `y`.

        Note:
        - The `averaging` specifies how the Character-Error-Rate is computed.
          Implicitly, the "macro"-average weights evenly every sample, while
          "micro"-average considers sample length. Any other average will result in
          a RuntimeError.
        """
        if averaging == "macro":
            metric = load("cer")
        elif averaging == "micro":
            metric = load("character")
        else:
            raise RuntimeError(
                f"{averaging} not found. Please use either 'macro' or 'micro'"
            )

        y_pred = self.predict(X)
        return metric.compute(predictions=y_pred, references=y)


def local_stats(ref: str, pred: str) -> tuple[int, int, int, int, int]:
    """Compute character-level stats

    Args:
        ref (str): Reference (target) string
        pred (str): Predicted (automatically transcripted) string

    Returns:
        tuple[int, int, int, int, int]: (ref len, total errors, insertions, substitutions, deletions)
    """
    stats = isri_tools.compute_accurary_stats(pred, ref)
    return (
        stats.characters,
        stats.errors,
        stats.total_ops.insertions,
        stats.total_ops.substitutions,
        stats.total_ops.deletions,
    )


def compute_keep_decision(row: pd.Series) -> bool:
    target = row["Ground Truth"]
    ocr_txt = row["Sample"]
    corrected = row["Predictions"]

    err_ocr = isri_tools.compute_accurary_stats(ocr_txt, target).total_ops.errors
    err_corr = isri_tools.compute_accurary_stats(corrected, target).total_ops.errors

    keep = err_corr <= err_ocr
    return keep


class InsertDeletionController(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def load(self, path):
        # FIXME use clf load/save
        pass

    def save(self, path):
        # FIXME use clf load/save
        pass

    def _compute_features(self, X):
        X_indicators = pd.DataFrame(
            [local_stats(ref=ref, pred=pred) for ref, pred in X],
            columns=[
                "characters",
                "errors",
                "insertions",
                "substitutions",
                "deletions",
            ],
        )

        for col in ("insertions", "substitutions", "deletions"):
            X_indicators.loc[:, f"{col}_rel"] = (
                X_indicators[col] / X_indicators["characters"]
            )

        # keep only useful indicators
        X_final = np.array(X_indicators.loc[:, ("insertions_rel", "deletions_rel")])

        return X_final

    def fit(self, X, y):
        """
        Train the validator.

        Arguments
        ---------
        X: list of pairs of strings (ocr transcription, autocorrected) of length n_samples
        y: np.array of shape (n_samples, ) and dtype bool (True if we should keep correction, i.e. CER(corr) <= CER(OCR), False otherwise)
        """
        # compute features
        X_final = self._compute_features(X)
        # check shapes
        X_final, y = check_X_y(X_final, y)
        # create and train classifier
        self.classifier_ = RandomForestClassifier(n_estimators=10)
        self.classifier_.fit(X_final, y)

        return self

    def predict(self, X):
        """Compute predictions (whether to keep corrected string or not)

        Args:
            X (list[tuple[str, str]]): List of pairs of (ocr transcription, autocorrected)

        Returns:
            np.ndarray of type bool: binary decision to keep individual samples
        """
        # Check if fit has been called
        check_is_fitted(self)
        # compute features
        X_final = self._compute_features(X)
        # Input validation
        X_final = check_array(X_final)
        # compute predictions
        predictions = self.classifier_.predict(X_final)
        return predictions
