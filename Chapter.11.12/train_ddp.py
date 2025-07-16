import os
import time

import torch
from sklearn.metrics import accuracy_score
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed import init_process_group, all_reduce, ReduceOp
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet, vision_transformer

from utils import get_args, log_config, log_rank, report_memory
from dataset import DatasetFolder, MyCustomCollator, RGBCollator
from custom_model import MyCustomModel

MAP_STR_TO_COLLATOR = {
    "custom": MyCustomCollator,
    "resnet50": RGBCollator,
    "vit": RGBCollator,
}

MAP_STR_TO_OPTIM = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}

MAP_STR_TO_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    None: torch.float32,
}


def main(args, world_size, rank):
    # Init configs
    device_id = rank % torch.cuda.device_count()
    use_amp = True if args.mixed_precision else False
    # Build Dataset, Collator & DataLoader
    train_ds = DatasetFolder(os.path.join(args.dataset, "train"))
    valid_ds = DatasetFolder(os.path.join(args.dataset, "val"))

    collator = MAP_STR_TO_COLLATOR[args.model_name](args.resolution)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, drop_last=True)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, drop_last=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=valid_sampler,
    )

    n_classes = len(train_ds.classes)
    # Build Model
    if args.model_name == "custom":
        model = MyCustomModel(n_classes=n_classes, resolution=args.resolution)
    elif args.model_name == "resnet50":
        weights = resnet.ResNet50_Weights.DEFAULT if args.pretrained else None
        model = resnet.resnet50(weights=weights, num_classes=n_classes)
    else:
        weights = vision_transformer.ViT_H_14_Weights.DEFAULT if args.pretrained else None
        model = vision_transformer.vit_h_14(weights=weights, num_classes=n_classes)

    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    # Compile model with torch.compile to improve performance
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    # Set Optimizer, Loss & Metrics
    optimizer = MAP_STR_TO_OPTIM[args.optimizer](model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    ###################### Training ######################
    log_config(args, "cuda", use_amp, len(train_dl), len(valid_dl), number_of_parameters)
    ft0 = time.time()  # Full training timer
    tl_t = []  # Training loop times
    vl_t = []  # Validation loop times
    running_loss = 0.0
    global_batch_size = args.batch_size * world_size
    eval_global_batch_size = args.eval_batch_size * world_size
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        log_rank(f"[EPOCH: {epoch}] Starting training loop...")
        # Training Loop
        tl0 = time.time()  # Training loop timer
        for iteration, (inputs, labels) in enumerate(train_dl, start=1):
            # Copy inputs to the GPU
            inputs = inputs.to(device_id)
            labels = labels.to(device_id)

            # zero the parameter gradients
            optimizer.zero_grad()
            # AMP Context manager
            with autocast(device_type="cuda", dtype=MAP_STR_TO_DTYPE[args.mixed_precision], enabled=use_amp):
                # Forward pass
                outputs = model(inputs)
                # Compute loss
                loss = criterion(outputs, labels)
                del outputs
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()

            running_loss += loss.item()
            if iteration % args.iteration_logging == 0:
                log_rank(
                    f"[EPOCH: {epoch}] Loss at iteration {iteration}: {running_loss / args.iteration_logging:.3f}"
                )
                running_loss = 0.0

        tl = time.time() - tl0
        tl_t.append(tl)
        log_rank(f"[EPOCH: {epoch}] Training throughput: {(global_batch_size * len(train_dl))/(tl):.3f} imgs/s")
        running_loss = 0.0  # Reset loss tracker

        # Validation Loop
        if epoch % args.epochs_eval == 0 or epoch == args.num_epochs:
            log_rank(f"[EPOCH: {epoch}] Starting validation loop...")
            vl0 = time.time()  # Validation loop timer

            predictions = []
            references = []
            # Disable Dropout layers and BatchNorm layers for evaluation
            model.eval()

            for inputs, labels in valid_dl:
                inputs = inputs.to(device_id)
                labels = labels.to(device_id)
                # AMP Context manager
                with autocast(device_type="cuda", dtype=MAP_STR_TO_DTYPE[args.mixed_precision], enabled=use_amp):
                    # Context-manager that disables gradient calculation. Reduces memory consumption and speeds up computations
                    with torch.no_grad():
                        outputs = model(inputs)
                    outputs = outputs.argmax(dim=-1)
                    predictions.extend(outputs.tolist())
                    references.extend(labels.tolist())

            accuracy = torch.tensor(accuracy_score(references, predictions), device=device_id)
            # Aggregate metric of all DP groups
            all_reduce(accuracy, op=ReduceOp.AVG)
            vl = time.time() - vl0
            vl_t.append(vl)
            log_rank(
                f"[EPOCH: {epoch}] Accuracy: {accuracy:.3f} | Validation throughput: {(eval_global_batch_size * len(valid_dl))/(vl):.3f} imgs/s"
            )

    full_training_time = time.time() - ft0
    report = report_memory()
    log_rank("Training finished!")
    log_rank(
#        f"Complete training Time: {full_training_time:.2f} | Training throughput: {((args.num_epochs - 1) * global_batch_size * len(train_dl))/(sum(tl_t[1:])):.3f} imgs/s | Validation throughput: {(len(vl_t) * eval_global_batch_size * len(valid_dl))/(sum(vl_t)):.3f} imgs/s"
#    )
        f"Complete training Time: {int(full_training_time)} s "
    )
    log_rank(
        f"Training throughput:    {int(((args.num_epochs - 1) * global_batch_size * len(train_dl))/(sum(tl_t[1:])))} imgs/s"
    )
    log_rank(report)
    return args


if __name__ == "__main__":
    _args = get_args()
    # Init Distributed setup
    init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    main(_args, world_size, rank)
