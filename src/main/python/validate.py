from argparse import ArgumentParser
from typing import Tuple, Callable, List

from onnxruntime import InferenceSession
from tqdm import tqdm

from src.main.python.models import Config, Candidate
from src.main.python.spell_checker import SpellChecker


def validate(spellchecker: Callable, test_data_path: str) -> Tuple[float, float]:
    (total, acc_1, acc_5) = (0, 0, 0)
    with open(test_data_path) as test_data:
        for line in tqdm(test_data, "Validate"):
            incorrect, correct = line.strip().split("\t")
            candidates = spellchecker(incorrect)
            candidates = sorted(candidates, key=lambda c: c.probability, reverse=True)
            total += 1
            if candidates[0].word == correct:
                acc_1 += 1
            if correct in [it.word for it in candidates[:5]]:
                acc_5 += 1
    return acc_1 / total, acc_5 / total


def main(onnx_model_path: str, test_data_path: str):
    config = Config.default_config()
    spellchecker = SpellChecker(config.dic_path, config.aff_path, config.classifier, config.seed)

    session = InferenceSession(onnx_model_path)

    def candidates_retrieve(word: str) -> List[Candidate]:
        return spellchecker.onnx_run(session, word)

    acc_1, acc_5 = validate(candidates_retrieve, test_data_path)

    print(f"Accuracy@1: {round(acc_1 * 100, 2)}, accuracy@5: {round(acc_5 * 100, 2)}")


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("--model", type=str, help="Path to ONNX model")
    __arg_parser.add_argument("--test", type=str, help="Path to test data")

    __args = __arg_parser.parse_args()
    main(__args.model, __args.test)
