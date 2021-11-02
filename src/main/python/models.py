from dataclasses import dataclass
from os.path import join


@dataclass
class Candidate:
    word: str
    probability: float


@dataclass
class Config:
    dic_path: str
    aff_path: str
    seed: int

    @staticmethod
    def default_config() -> "Config":
        data_folder = "data"
        dic_path = join(data_folder, "index.dic")
        aff_path = join(data_folder, "index.aff")
        return Config(dic_path, aff_path, 7)
