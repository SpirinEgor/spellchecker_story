from collections import Counter
from typing import List, Tuple, Counter as TCounter, Optional

import textdistance
from hunspell import HunSpell
from numpy import ndarray, array, vstack, float32
from onnxconverter_common import FloatTensorType
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from src.main.python.models import Candidate


class SpellChecker:
    _onnx_initial_state = "input"

    def __init__(self, dic_path: str, aff_path: str, seed: Optional[int] = None):
        self.__hunspell = HunSpell(dic_path, aff_path)
        self.__seed = seed

        self._clr: Optional[LogisticRegression] = None

    @staticmethod
    def _build_candidate_features(candidate: str, word: str) -> ndarray:
        max_len = max(len(word), len(candidate))
        levenshtein = (1 - textdistance.damerau_levenshtein(word, candidate)) / max_len
        jaro_winkler = (1 - textdistance.jaro_winkler(word, candidate)) / max_len
        lcs = len(textdistance.lcsseq(word, candidate)) / max_len
        return array([levenshtein, jaro_winkler, lcs])

    def fit(self, incorrect_words: List[str], correct_words: List[str]):
        X_train_list, y_train_list = [], []
        for word, good_candidate in tqdm(zip(incorrect_words, correct_words), "Build features", len(incorrect_words)):
            X_train_list.append(self._build_candidate_features(good_candidate, word))
            y_train_list.append(1)

            bad_candidate = None
            for _bc in self.__hunspell.suggest(word):
                if _bc != good_candidate:
                    bad_candidate = _bc
                    break
            if bad_candidate is not None:
                X_train_list.append(self._build_candidate_features(bad_candidate, word))
                y_train_list.append(0)

        X_train = array(X_train_list)
        y_train = array(y_train_list)
        print(f"Total shape of train data: {X_train.shape}")
        self._clr = LogisticRegression(n_jobs=-1, random_state=self.__seed)
        self._clr.fit(X_train, y_train)

    def _prepare_candidates(self, word: str) -> Tuple[List[str], ndarray]:
        word_candidates = self.__hunspell.suggest(word)
        if len(word_candidates) == 0:
            return word_candidates, array([])
        # [n candidates; n features]
        features = vstack([self._build_candidate_features(it, word) for it in word_candidates])
        return word_candidates, features

    def __call__(self, word: str) -> List[Candidate]:
        if self._clr is None:
            raise RuntimeError(f"Fit spellchecker before using it")

        word_candidates, features = self._prepare_candidates(word)
        if len(word_candidates) == 0:
            return [Candidate(word, 0.0)]
        # [n candidates]
        scores = self._clr.predict_proba(features)[:, 1]
        return [Candidate(c, s) for c, s in zip(word_candidates, scores)]

    def to_onnx(self, output_path: str):
        if self._clr is None:
            raise RuntimeError(f"Fit spellchecker before dumping to onnx it")
        initial_type = [(self._onnx_initial_state, FloatTensorType([None, 3]))]
        onnx = convert_sklearn(self._clr, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx.SerializeToString())

    def onnx_run(self, sess: InferenceSession, word: str) -> List[Candidate]:
        word_candidates, features = self._prepare_candidates(word)
        if len(word_candidates) == 0:
            return [Candidate(word, 0.0)]

        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name

        pred_onnx = sess.run([label_name], {input_name: features.astype(float32)})[0]
        probabilities = [it[1] for it in pred_onnx]

        return [Candidate(c, p) for c, p in zip(word_candidates, probabilities)]
