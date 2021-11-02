# Spellchecker Story

[![Main](https://github.com/SpirinEgor/spellchecker_story/actions/workflows/main.yaml/badge.svg?branch=master)](https://github.com/SpirinEgor/spellchecker_story/actions/workflows/main.yaml)

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
3. Train classifier to predict the probability of correct fix by these features.

### Normalized distance

If `l` is edit distance between `s1` and `s2`,
then normalized distance is `(1 - l) / max(|s1|, |s2|)`.

For LCS, normalized distance is `LCS / max(|s1|, |s2|)`.

### Training classifier

Dataset for training model is taken from [norvig](https://www.norvig.com/ngrams/spell-errors.txt).
For each correct spell:
- positive example retrieved from dataset
- negative example retrieved from hunspell's suggestion

Currently, two models are supported: Logistic Regression and Random Forest Classifier.
To choose model use special [config class](src/main/python/models.py).

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
  --model $PATH_TO_ONNX_MODEL \
  --test $PATH_TO_TEST_DATA
```

|                          | Accuracy@1 | Accuracy@5 |
|--------------------------|------------|------------|
| Logistic Regression      | 49.18      | 73.49      |
| Random Forest Classifier | 48.99      | 73.49      |

[`Checkpoints`](checkpoints) folder contains weights for these model.

### Inference on JVM

[`SpellCheckerKt`](src/main/kotlin/spellchecker/SpellChecker.kt) contains GEC transfer from Python to Kotlin.
It contains only functionality for inference, and therefore, model should be trained before using it.

[`AppKt`](src/main/kotlin/spellchecker/App.kt) shows an example of usage this system.
As python validation, it is used to check model accuracy on a test data.

To run example use Gradle:
```shell
gradle run --args="$PATH_TO_TEST_DATA $PATH_TO_ONNX_MODEL"
```

|                          | Accuracy@1 | Accuracy@5 |
|--------------------------|------------|------------|
| Logistic Regression      | 48.99      | 73.67      |
| Random Forest Classifier | 46.62      | 73.86      |

There is a little bit differents from Python validation.
Perhaps, it is due to small difference in floating-point arithmetic
(both kotlin and python use native C hunspell implementation and distance algorithms are deterministic).

## Thoughts

1. Basically, spellchecker depends only on dictionaries for HunSpell and train dataset.
There are already a [bunch](https://github.com/wooorm/dictionaries) of such dictionaries for many languages.
Therefore, adding new language requires only collecting data with known misspells for this language.
2. Current solution belong to Mixed class of GEC systems.
Therefore, future work may include adding new features, based on rules or existing text algorithms,
along with ranking model improvements.
