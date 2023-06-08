# `postocr-3stages` – A Python package for post-OCR correction in 3 stages

**🚧 Careful! Work in progress! 🚧**

This is the home of the `postocr-3stages` Python package.
It aims at offering a simple solution for post-OCR correction with a friendly, *scikit-learn*-like API.


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


## 🛠️ Prepare your data
- size requirements (number of elements, sample len)
- syntactic properties
- format


## 🚀 Quick start

The example below assumes a dataset is loaded as a Pandas Dataframe with the following two columns:

- `"OCR"`: original transcription from some OCR system to correct
- `"Gold"`: target transcription

```python
from postocr_3stages import Pipeline

train_dataset, test_dataset = # FIXME load your dataset

postocr = Pipeline()
postocr.fit(train_dataset["OCR"], train_dataset["Gold"])
# ... wait
print("Post-OCR CER:")
print(postocr.score(train_dataset["OCR"], train_dataset["Gold"]))

print("Sample correction")
print(postocr.predict(["Samp1e strimg to corect."]))
```

For more examples, see our [demo notebook](notebooks/demo.ipynb).


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

