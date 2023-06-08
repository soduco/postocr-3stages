from typing import Union

import numpy as np
import pandas as pd

from .error_detector import ErrorDetector
from .NMT_corrector import NMTCorrector
from .controller import LengthController, InsertDeletionController, compute_keep_decision
from .utils import nmt_score


class Pipeline:
    """
    Performs the whole correction pipeline: detection, correction and length control.
    """

    def __init__(
        self,
        *,
        error_detector: ErrorDetector = None,
        nmt_corrector: NMTCorrector = None,
        controller: "Controller" = None
    ):
        self.error_detector = error_detector
        self.nmt_corrector = nmt_corrector
        self.controller = controller


    def _fit_controller(self, X: pd.Series, y: pd.Series) -> None:
        if isinstance(self.controller, LengthController):
            self.controller.fit(X, y)

        elif isinstance(self.controller, InsertDeletionController):
            X_errored = self._predict_detector(X)
            X_pred = self._predict_corrector(X.loc[X_errored])

            tmp_df = pd.DataFrame({
                "Sample": X,
                "Predictions": X_pred,
                "Ground Truth": y,
            })

            X_controller = np.array(tmp_df.loc[:, ("Sample", "Predictions")])
            y_controller = tmp_df.apply(compute_keep_decision, axis=1)

            self.controller.fit(X_controller, y_controller)

        else:
            raise AttributeError(
                f"{type(self.controller)} is not a valid controller type. Please "
                "choose between LengthController and InsertDeletionController."
            )


    def fit(self, X: pd.Series, y: pd.Series) -> None:
        if self.error_detector is not None:
            y_errored = (X != y).astype(float)
            self.error_detector.fit(X, y_errored)

        if self.nmt_corrector is not None:
            self.nmt_corrector.fit(X, y)

        if self.controller is not None:
            self._fit_controller(X, y)

    def _predict_detector(self, X: pd.Series) -> pd.Series:
        if self.error_detector is not None:
            return self.error_detector.predict(X)

        # XXX: onmt only accept non-null sample
        return np.array(X.str.len() != 0)

    def _predict_corrector(self, X: pd.Series) -> pd.Series:
        if self.nmt_corrector is not None:
            return self.nmt_corrector.predict(X)

        return X.copy()

    # TODO: Would need a refactor along with LengthController.predict
    # to have a more coherent interface.
    # i.e. both Controller should return boolean
    def _predict_controller(self, y_pred: pd.Series, X: pd.Series) -> pd.Series:
        if isinstance(self.controller, LengthController):
            y_pred = self.controller.predict(y_pred, ref_samples=X)

        elif isinstance(self.controller, InsertDeletionController):
            X_controller = np.column_stack([
                np.array(X),
                np.array(y_pred),
            ])

            y_correct = self.controller.predict(X_controller)
            y_pred[~y_correct] = X[~y_correct]

        elif self.controller is not None:
            raise AttributeError(
                f"{type(self.controller)} is not a valid controller type. Please "
                "choose between LengthController and InsertDeletionController."
            )

        return y_pred


    def predict(self, X: pd.Series) -> pd.Series:
        X_errored = self._predict_detector(X)
        incomplete_y_pred = self._predict_corrector(X.loc[X_errored])
        incomplete_y_pred = self._predict_controller(incomplete_y_pred, X.loc[X_errored])

        y_pred = pd.concat([incomplete_y_pred, X.loc[~X_errored]]).sort_index()
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
        Character-Error-Rate (CER) against the Ground Truth `y`.

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


