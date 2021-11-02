from argparse import ArgumentParser
from os import makedirs
from os.path import join
from typing import Optional

from tqdm.auto import tqdm

from src.main.python.models import Config
from src.main.python.spell_checker import SpellChecker
from src.main.python.validate import validate


def is_only_letters(word: str) -> bool:
    return all(["a" <= c <= "z" for c in word])


def main(train_data_path: str, checkpoint_path: str, test_data_path: Optional[str] = None):
    config = Config.default_config()
    spellchecker = SpellChecker(config.dic_path, config.aff_path, config.seed)

    incorrect_words, correct_words = [], []
    with open(train_data_path) as train_data:
        for line in tqdm(train_data, "Preprocessing"):
            correct, misspells = line.strip().split(":")
            correct = correct.strip()
            incorrect = misspells.strip().split(",")[0].strip()

            if not is_only_letters(correct) and not is_only_letters(incorrect):
                continue

            correct_words.append(correct)
            incorrect_words.append(incorrect)

    spellchecker.fit(incorrect_words, correct_words)

    if test_data_path is not None:
        # Accuracy@1: 43.51, accuracy@5: 74.22
        acc_1, acc_5 = validate(spellchecker, test_data_path)
        print(f"Accuracy@1: {round(acc_1 * 100, 2)}, accuracy@5: {round(acc_5 * 100, 2)}")

    makedirs(checkpoint_path, exist_ok=True)
    spellchecker.to_onnx(join(checkpoint_path, "log_reg.onnx"))


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("--train", type=str, required=True, help="Path to train data")
    __arg_parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint folder")
    __arg_parser.add_argument("--test", type=str, help="Path to test data", default=None)

    __args = __arg_parser.parse_args()
    main(__args.train, __args.ckpt, __args.test)
