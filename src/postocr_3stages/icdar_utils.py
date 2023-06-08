from pathlib import Path
import pandas as pd


def get_df(
    root_directory: Path,
    *,
    align_input: bool = False,
    filter_aligned_error: bool = False,
) -> pd.DataFrame:
    """
    Create a DataFrame Pandas from the icdar2019 dataset.
    The dataset can be downloaded here:
    https://sites.google.com/view/icdar2019-postcorrectionocr/dataset?authuser=0

    The expected file-architecture of `root_directory` is:
    [root_directory]
    ├── [lang_type]
    │   ├── [subfolder]
    │   │   ├── 0.txt
    │   │   └── ...
    │   └── ...
    └── ...

    It can therefore be called like this
    ```python
    root_directory = Path("ICDAR2019_POCR_competition_evaluation_4M_without_Finnish")
    df = get_df(root_directory)
    ```

    Note:
    - The `align_input` argument specifies whether to use the aligned version.
    - The `filter_aligned_error` argument removes aligned sentences that contain
    alignment error (i.e. with different length for the ground truth and the sample)
    """
    sample_list = []
    gt_list = []

    for file_path in root_directory.glob("**/**/*.txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()

            sample = ""
            ground_truth = ""
            for line in lines:
                if "[OCR_aligned]" in line and align_input:
                    sample = line.replace("[OCR_aligned]", "").strip()
                elif "[OCR_toInput]" in line and not align_input:
                    sample = line.replace("[OCR_toInput]", "").strip()
                elif "[ GS_aligned]" in line:
                    ground_truth = line.replace("[ GS_aligned]", "").strip()

            if align_input and filter_aligned_error and len(sample) != len(ground_truth):
                continue

            sample_list.append(sample)
            gt_list.append(ground_truth)

    return pd.DataFrame({
        "Sample": sample_list,
        "Ground Truth": gt_list,
    })
