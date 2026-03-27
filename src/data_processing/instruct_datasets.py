from abc import ABC, abstractmethod
import pandas as pd


class InstructDataset(ABC):
    """
    Abstract base class for Instruct Datasets
    """

    def __init__(self, dataset_path: str):
        self.dataset = None
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str) -> None:
        self.dataset = pd.read_csv(dataset_path)

    def rename_columns(self, columns: dict[str, str]) -> None:
        self.dataset = self.dataset.rename(columns=columns)

    def drop_columns(self, columns: list[str]) -> None:
        drop_cols = [c for c in columns if c in self.dataset.columns]
        self.dataset = self.dataset.drop(columns=drop_cols)

    def drop_bad_rows(self, columns: list[str]) -> None:
        self.dataset = self.dataset.dropna(subset=columns)
        self.dataset = self.dataset.drop_duplicates(subset=columns)

    def create_instruction(self, instruction: str) -> None:
        self.dataset["instruction"] = instruction

    @abstractmethod
    def create_prompt(self) -> None:
        pass

    def get_dataset(self) -> pd.DataFrame:
        return self.dataset


# ✅ Mistral Model
class MistralInstructDataset(InstructDataset):
    def create_prompt(self):
        self.dataset["prompt"] = (
            "<s>[INST] "
            + self.dataset["instruction"]
            + " This is the question: "
            + self.dataset["input"]
            + " [/INST]\n "
            + self.dataset["output"]
            + "</s>"
        )


# ✅ Llama2 Model
class LlamaInstructDataset(InstructDataset):
    def create_prompt(self):
        self.dataset["prompt"] = (
            "[s][INST] "
            + self.dataset["instruction"]
            + " This is the question: "
            + self.dataset["input"]
            + " [/INST]\n "
            + self.dataset["output"]
            + "[/s]"
        )


# ✅ Llama 3 Model
class Llama3InstructDataset(InstructDataset):
    def create_prompt(self):
        self.dataset["prompt"] = (
            "<|start_header_id|>system<|end_header_id|> "
            + self.dataset["instruction"]
            + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|> This is the question: "
            + self.dataset["input"]
            + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|> "
            + self.dataset["output"]
            + "<|eot_id|>"
        )


# ✅ Gemma Model
class GemmaInstructDataset(InstructDataset):
    def create_prompt(self):
        self.dataset["prompt"] = (
            "<start_of_turn>user "
            + self.dataset["instruction"]
            + " This is the question: "
            + self.dataset["input"]
            + "<end_of_turn>\n "
            "<start_of_turn>model "
            + self.dataset["output"]
            + "<end_of_turn>model"
        )
