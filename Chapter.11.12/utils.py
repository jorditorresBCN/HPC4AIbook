import argparse
import logging
import os
import sys
import torch

# Set logging configs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logging.getLogger("numexpr").setLevel(logging.WARNING)


def get_args():
    parser = argparse.ArgumentParser(description="SA-MIRI | Lab 8")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation per GPU")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for the DataLoaders")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--epochs_eval", type=int, default=5, help="How often we compute the accuracy on the validation set"
    )
    parser.add_argument("--iteration_logging", type=int, default=500, help="How often we log the loss during training")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["custom", "resnet50", "vit"],
        required=True,
        help="Current supported models: ['custom', 'resnet50', 'vit']",
    )
    parser.add_argument(
        "--intermidiate_dimensions",
        nargs="+",
        type=int,
        default=None,
        help="Intermediate dimension of the custom dense model. e.g.: --intermidiate_dimensions 512 256 128 64",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Load pretrained model weights",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Compile model with torch.compile",
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer type: SGD, Adam or AdamW")
    parser.add_argument("--learning_rate", type=float, default="5e-4", help="Learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument("--resolution", type=int, default=224, help="Resolution of the input images")

    _args, _ = parser.parse_known_args()
    return _args


def report_memory():
    """Simple GPU memory report."""
    giga_bytes = 1024.0 * 1024.0 * 1024.0
#    string = "GPU memory (GB)"
#    string += f" | allocated: {torch.cuda.memory_allocated() / giga_bytes:.3f}"
#    string += f" | max allocated: {torch.cuda.max_memory_allocated() / giga_bytes:.3f}"
#    string += f" | reserved: {torch.cuda.memory_reserved() / giga_bytes:.3f}"
#    string += f" | max reserved: {torch.cuda.max_memory_reserved() / giga_bytes:.3f}"

    string = "GPU memory reserved:   "
    string += f" {int(torch.cuda.memory_reserved() / giga_bytes) } GB"

    return string


def log(str):
    logging.info(str)


def log_rank(str, rank=0):
    if int(os.environ["RANK"]) is rank:
        log(str)


def log_config(args, device, use_amp, train_dl_length, valid_dl_length, number_of_parameters):
    config_str = ["############# Config #############"]
    config_str.extend([f"### {k}: {v}" for k, v in vars(args).items()])
    config_str.append(f"### Device: {device}")
    config_str.append(f"### AMP Enabled: {use_amp}")
    config_str.append(f"### Total number of parameters {number_of_parameters}")
    config_str.append(f"### Total training samples (batches): {train_dl_length*args.batch_size} ({train_dl_length})")
#    config_str.append(
#        f"### Total validation samples (batches): {valid_dl_length*args.eval_batch_size} ({valid_dl_length})"
#    )
    config_str.append("##################################")

    if torch.distributed.is_initialized():
        for str in config_str:
            log_rank(str)
    else:
        for str in config_str:
            log(str)
