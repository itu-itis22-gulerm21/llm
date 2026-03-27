import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from instruct_datasets import (
    GemmaInstructDataset,
    MistralInstructDataset,
    LlamaInstructDataset,
    Llama3InstructDataset,
)

# ---- BASE PATH ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_data")

REMOVE_COLUMNS = ["source", "focus_area"]
RENAME_COLUMNS = {"question": "input", "answer": "output"}
INSTRUCTION = "Answer the question truthfully, you are a medical professional."

DATASETS_PATHS = [
    os.path.join(RAW_DATA_DIR, "medical_meadow_wikidoc.csv"),
    os.path.join(RAW_DATA_DIR, "medquad.csv"),
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_dataset(dataset_path: str, model: str) -> pd.DataFrame:
    """
    Process the instruct dataset to be in the format required by the model.
    """
    logger.info(f"Processing dataset: {dataset_path} for {model} instruct model.")
    if model == "gemma":
        dataset = GemmaInstructDataset(dataset_path)
    elif model == "mistral":
        dataset = MistralInstructDataset(dataset_path)
    elif model == "llama":
        dataset = LlamaInstructDataset(dataset_path)
    elif model == "llama3":
        dataset = Llama3InstructDataset(dataset_path)
    else:
        raise ValueError(f"Model {model} not supported!")

    dataset.drop_columns(REMOVE_COLUMNS)
    dataset.rename_columns(RENAME_COLUMNS)
    dataset.create_instruction(INSTRUCTION)
    dataset.drop_bad_rows(["input", "output"])
    dataset.create_prompt()
    return dataset.get_dataset()


def create_dataset_hf(dataset: pd.DataFrame) -> DatasetDict:
    dataset.reset_index(drop=True, inplace=True)
    return DatasetDict({"train": Dataset.from_pandas(dataset)})


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    mistral_datasets = []
    gemma_datasets = []
    llama_datasets = []
    llama3_datasets = []

    for dataset_path in DATASETS_PATHS:
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]

        mistral_dataset = process_dataset(dataset_path, "mistral")
        llama_dataset = process_dataset(dataset_path, "llama")
        llama3_dataset = process_dataset(dataset_path, "llama3")
        gemma_dataset = process_dataset(dataset_path, "gemma")

        mistral_datasets.append(mistral_dataset)
        llama_datasets.append(llama_dataset)
        llama3_datasets.append(llama3_dataset)
        gemma_datasets.append(gemma_dataset)

        mistral_hf = create_dataset_hf(mistral_dataset)
        llama_hf = create_dataset_hf(llama_dataset)
        llama3_hf = create_dataset_hf(llama3_dataset)
        gemma_hf = create_dataset_hf(gemma_dataset)

        llama3_hf.push_to_hub(f"llama3_{dataset_name}_instruct_dataset")

    # Merge
    mistral_dataset = create_dataset_hf(pd.concat(mistral_datasets, ignore_index=True))
    llama_dataset = create_dataset_hf(pd.concat(llama_datasets, ignore_index=True))
    llama3_dataset = create_dataset_hf(pd.concat(llama3_datasets, ignore_index=True))
    gemma_dataset = create_dataset_hf(pd.concat(gemma_datasets, ignore_index=True))

    llama3_dataset.save_to_disk(
        os.path.join(PROCESSED_DIR, "medical_llama3_instruct_dataset")
    )
    llama3_dataset.push_to_hub("medical_llama3_instruct_dataset")

    # Short versions
    mistral_short = pd.concat(
        [mistral_datasets[0].iloc[:1000], mistral_datasets[0].iloc[-5000:-4000]],
        ignore_index=True,
    )
    llama_short = pd.concat(
        [llama_datasets[0].iloc[:1000], llama_datasets[0].iloc[-5000:-4000]],
        ignore_index=True,
    )
    llama3_short = pd.concat(
        [llama3_datasets[0].iloc[:1000], llama3_datasets[0].iloc[-5000:-4000]],
        ignore_index=True,
    )
    gemma_short = pd.concat(
        [gemma_datasets[0].iloc[:1000], gemma_datasets[0].iloc[-5000:-4000]],
        ignore_index=True,
    )

    llama3_short = create_dataset_hf(llama3_short)
    llama3_short.save_to_disk(
        os.path.join(PROCESSED_DIR, "medical_llama3_instruct_dataset_short")
    )
    llama3_short.push_to_hub("medical_llama3_instruct_dataset_short")
