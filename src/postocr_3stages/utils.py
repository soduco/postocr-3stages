from typing import Union

from pathlib import Path
import re

import pandas as pd
from evaluate import load  # Hugging Face


def save_df_on_disk(df: pd.Series, path: Path) -> None:
    """
    Convert the dataframe `df` or serie to the openNMT format and
    saves it on disk at the desired `path`.

    The following transformation are performed in order to fit the
    openNMT format:
    - All characters are separated by spaces in order to perform
    translation at character-level.
    - All real whitespace characters are represented by underscore ("_")
    on disk.
    - Each sample from the dataframe is represented on a single line.
    """
    pd.set_option("display.max_colwidth", None)

    raw_str_df = df.to_string(header=False, index=False)
    no_padding = re.sub("^ *", "", raw_str_df, flags=re.MULTILINE)
    no_white_space_str = no_padding.replace(" ", "_")
    char_split_str = " ".join(no_white_space_str)

    with open(path, "w") as f:
        f.write(char_split_str)


def load_from_disk(path: Path, *, template_df: pd.Series = None) -> pd.Series:
    """
    Inverse transformation of `save_df_on_disk`.

    Use `template_df` to set index. Do not set them, if `template_df` is None.
    """
    with open(path, "r") as f:
        raw_data = f.readlines()

    no_useless_newline = map(lambda line: line.rstrip("\n"), raw_data)
    no_char_split = map(lambda line: line.replace(" ", ""), no_useless_newline)
    whitespace = map(lambda line: line.replace("_", " "), no_char_split)
    normalised = map(lambda line: line.replace("\\n", "\n"), whitespace)

    if template_df is None:
        return pd.Series(normalised)

    return pd.Series(normalised, index=template_df.index)


def nmt_score(
    y_pred: pd.Series,
    y: pd.Series,
    *,
    averaging: str = "macro",
    per_sample_metric: bool = False,
) -> Union[float, pd.Series]:
    """
    Compute the Character-Error-Rate (CER) between the prediction `y_pred` and
    its the Ground Truth `y`.

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
    if averaging == "macro":
        metric = load("character")
    elif averaging == "micro":
        metric = load("cer")
    else:
        raise RuntimeError(
            f"{averaging} not found. Please use either 'macro' or 'micro'"
        )

    if not per_sample_metric:
        return metric.compute(predictions=y_pred, references=y)

    return pd.Series([
        metric.compute(predictions=[y_pred_sample], references=[y_sample])
        for (y_pred_sample, y_sample) in zip(y_pred, y)
    ])
