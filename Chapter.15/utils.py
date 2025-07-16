import os
import sys
import logging

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

import numpy as np

from torch.utils.data import Dataset
from transformers import HfArgumentParser, TrainingArguments


# Set logging configs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logging.getLogger("numexpr").setLevel(logging.WARNING)

def log(str):
    logging.info(str)

def log_rank(str, rank=0):
    if int(os.environ["RANK"]) is rank:
        log(str)


class DummyDataset(Dataset):
    def __init__(self, sequence_length: int = 8192, num_samples: int = 1000000) -> None:
        self.num_samples = num_samples
        self.sequence_length = sequence_length
    def __len__(self) -> int:
        return self.num_samples
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        x = torch.LongTensor(np.random.randint(low= 0, high= 1000, size=(self.sequence_length + 1)))
        return {"input_ids": x[:-1], "labels": x[1:]}

@dataclass
class BenchmarkArguments:
    """
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    path_to_model: str = field(
        metadata={"help": "Path to model config"}
    )
    attn: str = field(
        metadata={"help": "Attention implementation"}
    )
    sequence_length: Optional[int] = field(
        default=8192, metadata={"help": "Sequence length of the model"}
    )
    num_samples: Optional[int] = field(
        default=10000, metadata={"help": "Number of samples for the training dataset"}
    )
    model_precision: Optional[str] = field(
        default=None, metadata={"help": "Torch dtyper for the model parameters"}
    )

def report_memory():
    """Simple GPU memory report."""
    giga_bytes = 1024.0 * 1024.0 * 1024.0
    log_rank('> Max reserved GPU Memory: {:.2f}'.format(
        torch.cuda.max_memory_reserved() / giga_bytes))
    

def get_args():
    parser = HfArgumentParser((TrainingArguments, BenchmarkArguments))
    _training_arguments, _benchmark_arguments = parser.parse_args_into_dataclasses()
    return _training_arguments, _benchmark_arguments
