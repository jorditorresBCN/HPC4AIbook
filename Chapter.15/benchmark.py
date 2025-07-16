import os
import torch

from transformers import AutoModelForCausalLM, Trainer
from utils import DummyDataset, get_args, log_rank, report_memory

def main(training_arguments, benchmark_arguments):
    # DTYPE for model parameters
    TORCH_DTYPE = torch.bfloat16 if benchmark_arguments.model_precision == "bf16" else torch.float
    world_size = int(os.environ["WORLD_SIZE"])

    model = AutoModelForCausalLM.from_pretrained(benchmark_arguments.path_to_model, attn_implementation=benchmark_arguments.attn, torch_dtype=TORCH_DTYPE)
    train_dataset = DummyDataset(benchmark_arguments.sequence_length, benchmark_arguments.num_samples)
    trainer = Trainer(model=model, args=training_arguments, train_dataset=train_dataset)
    metrics = trainer.train().metrics
    log_rank(f"> Training throughput:     {training_arguments.max_steps * training_arguments.per_device_train_batch_size * benchmark_arguments.sequence_length/metrics['train_runtime']:.2f} Tokens/s/GPU")
    report_memory()

if __name__ == "__main__":
    _training_arguments, _benchmark_arguments = get_args()
    main(_training_arguments, _benchmark_arguments)
