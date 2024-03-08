# `postocr-3stages` – A Python package for post-OCR correction in 3 stages

**🚧 Careful! Work in progress! 🚧**

This is the home of the `postocr-3stages` Python package.
It aims at offering a simple solution for post-OCR correction with a friendly, *scikit-learn*-like API.

[![](https://img.shields.io/badge/HAL-Presentation-red)](https://hal.science/hal-04495595) [![](https://img.shields.io/badge/PDF-Presentation-blue)](./documentation/20230609-SIFED-postocr.pdf) Slides (in French) presented at SIFED 2023


## 📦 Installation
For now, the module can be almost completely installed using pip, except for our prototype for fast computation of edits between OCR and corrected strings.
The installation protocol can be summarized as follows:

```shell
# create a virtual env "myvenv" and activate it
python -m venv myvenv 
. myvenv/bin/activate

# install postocr package
pip install postocr-3stages

# manually download and install our irsi-tools package, with Python bindings of the original ISRI code
# - dependency
pip install pybind11
# - download the archive and uncompress
wget https://github.com/soduco/paper-ner-bench-das22/archive/refs/heads/main.zip
unzip main.zip
# - build and install
cd paper-ner-bench-das22-main/src/ocr/
pip install .
```


## 🚀 Quick start

This package tries to mimic the interface of [sklearn](https://scikit-learn.org/stable/) for easier use and compatibility.

Here is a minimal example to understand how the library works. In this example we try to make the model learn to transform the string "x" into "y".
The `.score` functions compute the [Character Error Rate (CER)](https://huggingface.co/spaces/evaluate-metric/cer), the lower, the better.

- `"OCR"`: original transcription from some OCR system to correct
- `"Gold"`: target transcription

```python
import pandas as pd

from postocr_3stages.error_detector import ErrorDetector
from postocr_3stages.controller import LengthController
from postocr_3stages.NMT_corrector import NMTCorrector
from postocr_3stages.pipeline import Pipeline

x = pd.Series(["x"] * 100)
y = pd.Series(["y"] * 100)

pipeline = Pipeline(
    error_detector=ErrorDetector(),
    nmt_corrector=NMTCorrector(train_steps=10),
    controller=LengthController(),
)

pipeline.fit(x, y)
pipeline.score(x, y) # 0 of CER
pipeline.predict(pd.Series("x")) # predict "y"
```

For more examples, see our [demo notebook](notebooks/demo.ipynb).


## 🛠️ Prepare your data
For both training and inference, data should be passed as a [Pandas' Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) of raw text.
Here is a minimal example of what such series could look like:
```python
import pandas as pd

X = pd.Series(["hello i'm some ocred text"])
y = pd.Series(["hello i'm the correct text"])
```


## 🔧 API

The module is composed of 4 objects, with a similar methods (`fit()`, `predict()` and `score()`) for each of them, but with different parameters.

### `postocr_3stages.Pipeline`: High-level utility
This is the end-to-end utility for direct training and correction.

Methods:

- `fit(ocr: Pandas.Series[str], gold: Pandas.Series[str]) -> postocr_3stages.Pipeline`: Train the complete pipeline using pairs of (OCR, Gold) string samples.
- `score(ocr: Pandas.Series[str], gold: Pandas.Series[str]) -> float`: Compute the CER of the complete pipeline using pairs of (OCR, Gold) string samples.
- `predict(ocr: Pandas.Series[str]) -> Pandas.Series[str]`: Predict the corrected version of each string from the input.

### `postocr_3stages.Detector`: Error Detector
Direct access to the error detection module. Warning: it classifies whole sequences and does not return the exact position of errors. A string is either correct or erroneous.

Methods:

- `fit(ocr: Pandas.Series[str], gold: Pandas.Series[bool]) -> postocr_3stages.Detector`: Train the complete pipeline using pairs of (OCR, correct?) samples.
- `score(ocr: Pandas.Series[str], gold: Pandas.Series[bool]) -> float`: Compute the accuracy of the detector.
- `predict(ocr: Pandas.Series[str]) -> Pandas.Series[bool]`: Predict whether each string sample is correct or not (contains errors).

### `postocr_3stages.Corrector`: Error Corrector
Direct access to the error correction module. It tries to transform each string given as input to reduce the number of errors it contains. It should be called on erroneous strings only.

Methods:

- `fit(ocr: Pandas.Series[str], gold: Pandas.Series[str]) -> postocr_3stages.Corrector`: Train the complete pipeline using pairs of (OCR, Gold) string samples.
- `score(ocr: Pandas.Series[str], gold: Pandas.Series[str]) -> float`: Compute the CER of the complete pipeline using pairs of (OCR, Gold) string samples.
- `predict(ocr: Pandas.Series[str]) -> Pandas.Series[str]`: Predict the corrected version of each string from the input.

### `postocr_3stages.Verifier`: Correction Verifier
Direct access to the correction verification module. This module takes as input a `numpy.ndarray` which contains the indicators from which decision should be taken (in our case insertion and deletion rates) and predicts whether the correction which lead to these indicators should be kept or discarded.

Methods:

- `fit(X: np.ndarray[float], y: np.ndarray[bool]) -> postocr_3stages.Verifier`: Train the complete pipeline using pairs of (features, target) string samples.
- `score(X: np.ndarray[float], y: np.ndarray[bool]) -> float`: Compute the accuracy of the verifier.
- `predict(X: np.ndarray[float]) -> np.ndarray[bool]`: Predict whether the corrections which led to the indicators passed as input should be kept or discarded.


## 🐛 🐞 🦗 🪳 Bugs
There are some known (and many unknowns) problems in this work-in-progress implementation.

- Our Python bindings of the original IRSI tools may have memory some leaks.


## 📝 TODO

- [ ] Make `pip install` work out of the box
- [ ] Add options for GPU/CPU training
- [ ] Implement loading/saving properly
- [ ] Offer some choice for the correction model (try a transformer model)
- [ ] Offer some choice for the verification model (add a simple edit length filter)
- [ ] Add some support for self-supervised pretraining

