import json
import os
from itertools import chain
from pathlib import Path
from typing import Optional

import ray
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from language_detector import LanguageDetector
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator
from text_chunker import TextChunker

load_dotenv()

RAY_FUTURES_BATCH_SIZE = 2000
CONFIG_DIR_PATH = os.environ["CONFIG_DIR_PATH"]


class HFDatasetFilters(BaseModel):
    language: Optional[str] = None
    date: Optional[str] = None
    streaming: Optional[bool] = None
    split: str = "train"


class BertPretrainingDatasetConfig(BaseModel):
    dataset_prefix: str
    max_chars: int = (
        820  # choosen based from: https://sidsite.com/posts/bert-from-scratch/
    )
    dataset_name_or_path: str | Path
    hf_dataset_filters: Optional[HFDatasetFilters] = None
    do_language_detection: Optional[bool] = False
    filtered_in_languages: list[str] = ["en"]
    destination_path: Path

    @field_validator("destination_path", mode="before")
    @classmethod
    def create_destination_path(cls, destination_path: Path) -> Path:
        """Create the destination path if it does not exist

        Args:
            destination_path (Path): The destination path

        Returns:
            Path: The destination path
        """
        absolute_destination_path = (
            Path(os.path.abspath(__file__)).parent / destination_path
        )
        if not absolute_destination_path.exists():
            absolute_destination_path.mkdir(parents=True)
            logger.info(f"Created destination path: {absolute_destination_path}")
        return absolute_destination_path

    @field_validator("dataset_name_or_path", mode="before")
    @classmethod
    def cast_dataset_name_or_path(cls, dataset_name_or_path: str) -> str | Path:
        """Cast the dataset name or path to a Path object if it is a path

        Args:
            dataset_name_or_path (str): The dataset name or path as a string

        Returns:
            str | Path: The dataset name or path as a string or a Path object
        """
        if cls.is_path(dataset_name_or_path):
            dataset_path = Path(dataset_name_or_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
            if stem := dataset_path.stem == "arrow":
                raise ValueError(f"Dataset file format must be .arrow, not {stem}")
            return dataset_path
        return dataset_name_or_path

    @staticmethod
    def is_path(dataset_name_or_path) -> bool:
        """Check if the dataset name or path is a path

        Args:
            dataset_name_or_path (str): The dataset name or path as a string

        Returns:
            bool: True if the dataset name or path is a path, False otherwise
        """
        return os.path.sep in str(dataset_name_or_path) or os.path.exists(
            dataset_name_or_path
        )

    def is_batch_on_disk(self, batch_idx: int) -> bool:
        """Check if the batch is already on disk

        Args:
            batch_idx (int): The batch index

        Returns:
            bool: True if the batch is already on disk, False otherwise
        """
        return (
            self.destination_path / f"{self.dataset_prefix}_{batch_idx}.jsonl"
        ).exists()


class BertPretrainingDataset:
    """Prepare the dataset for BERT pretraining. By combining various
    unlabeled text data sources"""

    def __init__(self, bert_pretrating_dataset_config: BertPretrainingDatasetConfig):
        self._config = bert_pretrating_dataset_config
        # Initialize Ray
        try:
            ray.shutdown()
            ray.init(num_cpus=os.cpu_count())  # type: ignore
        except Exception as e:
            logger.error(f"Error initializing Ray: {e}")

    def prepare(self) -> Dataset:
        """Prepare the dataset for BERT pretraining by combining various
        unlabeled text data sources and exporting them as a single dataset

        Returns:
            Dataset: The dataset for BERT pretraining
        """
        if is_dataset_path := self._config.is_path(self._config.dataset_name_or_path):
            return Dataset.from_file(str(self._config.dataset_name_or_path))
        elif not is_dataset_path:
            serialized_hf_dataset_filters = (
                self._config.hf_dataset_filters.model_dump()
                if self._config.hf_dataset_filters
                else {}
            )
            return load_dataset(
                self._config.dataset_name_or_path,
                num_proc=os.cpu_count(),
                **serialized_hf_dataset_filters,
            )
        else:
            raise NotImplementedError(
                "Currently only supports loading from an arrow file or"
                " loading from the Hugging Face datasets library"
            )

    def _write_to_disk(self, batch: list[dict[str, str]], batch_idx: int) -> None:
        """Write the processed dataset batch to disk as a JSONL file

        Args:
            list[dict[str, str]]: The processed dataset
            batch_idx (int): The batch index
        """
        with open(
            self._config.destination_path
            / f"{self._config.dataset_prefix}_{batch_idx}.jsonl",
            "w",
        ) as f:
            for row in batch:
                f.write(json.dumps(row) + "\n")

    def process(self) -> None:
        """Process the dataset for BERT pretraining by filtering out
        non-English text and chunking text into smaller pieces for language
        model pretraining. This method uses Ray actors to parallelize the work.
        By creating a pool of actors equal to the number of CPUs on the machine.
        The dataset is processed in batches to avoid memory issues. The processed
        dataset is written to disk as a JSONL file
        """
        logger.info(f"Processing dataset: {self._config.dataset_prefix}")
        dataset = self.prepare()

        # create a pool of actors
        num_cpus: int = os.cpu_count() or 1  # type: ignore
        lang_detect_actor_pool = [LanguageDetector.remote() for _ in range(num_cpus)]  # type: ignore
        text_chunker_actor_pool = [TextChunker.remote() for _ in range(num_cpus)]  # type: ignore

        # distribute work among actors
        lang_detect_futures = []
        text_chunk_futures = []
        batch_idx = 0
        total_rows_processed = 0
        for idx, row in enumerate(dataset):
            lang_detect_actor = lang_detect_actor_pool[idx % num_cpus]
            lang_detect_futures.append(lang_detect_actor.detect_language.remote(row))
            if len(lang_detect_futures) >= RAY_FUTURES_BATCH_SIZE:
                # gather results
                results = [
                    row
                    for row in ray.get(lang_detect_futures)
                    if row["language"] in self._config.filtered_in_languages
                ]
                for result_idx, result_row in enumerate(results):
                    text_chunker_actor = text_chunker_actor_pool[result_idx % num_cpus]
                    text_chunk_futures.append(
                        text_chunker_actor.chunk_text.remote(
                            result_row, self._config.max_chars
                        )
                    )
                # flatten the list of lists
                chunked_results = chain.from_iterable(list(ray.get(text_chunk_futures)))
                lang_detect_futures = []
                text_chunk_futures = []
                total_rows_processed += len(results)
                self._write_to_disk(chunked_results, batch_idx)  # type: ignore
                batch_idx += 1
                logger.info(f"Processed {total_rows_processed} rows")


def load_configs() -> list[dict]:
    """Load the configuration files for the BERT pretraining dataset

    Returns:
        list[dict]: The list of configuration files
    """
    config_files = list(Path(CONFIG_DIR_PATH).glob("*.yaml"))
    return [
        OmegaConf.to_object(OmegaConf.load(str(config_file)))
        for config_file in config_files
    ]


if __name__ == "__main__":
    configs = load_configs()
    for config in configs:
        if "hf_dataset_filters" in config:
            config["hf_dataset_filters"] = HFDatasetFilters(
                **config["hf_dataset_filters"]
            )
        bert_pretraining_config = BertPretrainingDatasetConfig(**config)
        bert_pretraining_dataset = BertPretrainingDataset(bert_pretraining_config)
        bert_pretraining_dataset.process()
