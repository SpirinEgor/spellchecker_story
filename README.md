# Spellchecker Story

Homework assignment for the lecture about Grazie in HSE.
The task is to create simple GEC system with porting it to JS or JVM.

## Solution

For each word, GEC extracts possible candidates with estimated probabilities.
To prepare candidates, the following approach is used:
1. Retrieve possible candidates by [HunSpell](http://hunspell.github.io/)
2. For each pair `<word; candidate>` calculate features:
   1. Normalized Damerau-Levenshtein distance.
   2. Normalized Jaro-Winkler distance.
   3. Normalized length of the longest common subsequences.
3. Logistic regression predicts the probability of correct fix by these features.

### Normalized distance

If `l` is edit distance between `s1` and `s2`,
then normalized distance is `(1 - l) / max(|s1|, |s2|)`.

For LCS, normalized distance is `LCS / max(|s1|, |s2|)`.

### Training logistic regression

Dataset for training model is taken from [norvig](https://www.norvig.com/ngrams/spell-errors.txt).
For each correct spell:
- positive example retrieved from dataset
- negative example retrieved from hunspell's suggestion

## Validation

To validate model [aspell](http://aspell.net/test/cur/) dataset is used.

Metrics:
- `Accuracy@1`: part of examples where the correct candidate has maximal probability.
- `Accuracy@5`: part of examples where the correct candidate has maximal probability.

## Usage

This section describes how to use this GEC system.
To install python dependencies (needed for model training):
```shell
pip install -r requirements.txt
```

To download necessary data use [`download_data.sh`](download_data.sh) script.

### Train GEC

To train model and save artifacts in ONNX format,
use [`train`](src/main/python/train.py) script:
```shell
PYTHONPATH="." python src/main/python/train.py \
  --train $TRAIN_DATA_PATH \
  --ckpt $CKPT_OUTPUT_FOLDER \
  --test $OPTIONAL_TEST_DATA_PATH
```
This will train logistic regression on the given train data,
test it on test data, and save logistic regression in onnx format.

### Validate GEC

To validate GEC, use [`validate`](src/main/python/validate.py) script:
```shell
PYTHONPATH="." python src/main/python/validate.py \
  --model $PATH_TO_ONNX_MODEL
  --test $PATH_TO_TEST_DATA
```

Current [`log_reg.onnx`](checkpoints/log_reg.onnx) model achieves:
```
Accuracy@1: 43.51, accuracy@5: 74.22
```

### Inference in JVM or JS

TODO

## Thoughts

1. Basically, spellchecker depends only on dictionaries for HunSpell and train dataset.
There are already a [bunch](https://github.com/wooorm/dictionaries) of such dictionaries for many languages.
Therefore, adding new language requires only collecting data with known misspells.
2. Converting logistic regression to ONNX format doesn't have much meaning.
But the suggested solution may be extended with other models from sklearn.
And since all models have the same interface, it should be easy.
