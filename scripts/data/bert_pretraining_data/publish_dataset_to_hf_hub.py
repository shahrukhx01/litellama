import json
import os
import random
from glob import glob
from pathlib import Path

from datasets import load_dataset


def create_chunked_dataset() -> None:
    """Create chunked dataset from the preprocessed dataset by shuffling and saving chunks"""

    dataset_file_paths = list(
        glob(
            str(
                Path(os.path.abspath(__file__)).parent
                / "dataset/bert_pretraining_data/*.jsonl"
            )
        )
    )

    dataset = load_dataset(
        "json",
        data_files=dataset_file_paths,
        streaming=True,  # prevents memory overflow
    )

    dataset = dataset["train"]

    batch_size = 1_000_000
    buffer_size = 100_000  # size of buffer for shuffling
    chunks = []
    chunk_count = 0

    output_dir = Path("processed_chunks")
    output_dir.mkdir(exist_ok=True)

    def save_chunk(buffer, chunk_count):
        # shuffle the buffer before saving
        random.shuffle(buffer)
        chunk_path = output_dir / f"chunk_{chunk_count}.jsonl"
        with open(chunk_path, "w") as f:
            for record in buffer:
                f.write(json.dumps(record) + "\n")
        print(f"Saved {chunk_path}")
        return str(chunk_path)

    buffer = []
    shuffle_buffer = []

    for example in dataset:
        shuffle_buffer.append(example)

        if len(shuffle_buffer) >= buffer_size:
            idx = random.randint(0, len(shuffle_buffer) - 1)
            buffer.append(shuffle_buffer.pop(idx))

        if len(buffer) >= batch_size:
            chunks.append(save_chunk(buffer, chunk_count))
            buffer = []
            chunk_count += 1

    while shuffle_buffer:
        buffer.append(shuffle_buffer.pop(random.randint(0, len(shuffle_buffer) - 1)))
        if len(buffer) >= batch_size:
            chunks.append(save_chunk(buffer, chunk_count))
            buffer = []
            chunk_count += 1

    if buffer:
        chunks.append(save_chunk(buffer, chunk_count))


def publish_dataset_to_hub() -> None:
    """Publish the chunked dataset to the Hugging Face Hub"""
    output_dir = Path("processed_chunks")
    chunk_files = sorted(output_dir.glob("chunk_*.jsonl"))  # ensure sorted order

    data_files = {"train": [str(file) for file in chunk_files]}
    dataset = load_dataset("json", data_files=data_files, num_proc=os.cpu_count())
    dataset.push_to_hub("shahrukhx01/wikipedia-bookscorpus-en-preprocessed")


if __name__ == "__main__":
    """Create chunked dataset and publish to the Hugging Face Hub"""
    create_chunked_dataset()
    publish_dataset_to_hub()
