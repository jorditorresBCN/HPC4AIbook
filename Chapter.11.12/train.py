import os
import time

import torch
from sklearn.metrics import accuracy_score
from torch import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torchvision.models import resnet, vision_transformer

from utils import get_args, log, log_config, report_memory
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


def main(args):
    # Init configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True if args.mixed_precision else False
    # Build Dataset, Collator & DataLoader
    train_ds = DatasetFolder(os.path.join(args.dataset, "train"))
    valid_ds = DatasetFolder(os.path.join(args.dataset, "val"))

    collator = MAP_STR_TO_COLLATOR[args.model_name](args.resolution)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
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
    model.to(device)
    # Compile model with torch.compile to improve performance
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    # Set Optimizer, Loss & Metrics
    optimizer = MAP_STR_TO_OPTIM[args.optimizer](model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    ###################### Training ######################
    log_config(args, device, use_amp, len(train_dl), len(valid_dl), number_of_parameters)
    ft0 = time.time()  # Full training timer
    tl_t = []  # Training loop times
    vl_t = []  # Validation loop times
    running_loss = 0.0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        log(f"[EPOCH: {epoch}] Starting training loop...")
        # Training Loop
        tl0 = time.time()  # Training loop timer
        for iteration, (inputs, labels) in enumerate(train_dl, start=1):
            # Copy inputs to the GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

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
                log(f"[EPOCH: {epoch}] Loss at iteration {iteration}: {running_loss / args.iteration_logging:.3f}")
                running_loss = 0.0

        tl = time.time() - tl0
        tl_t.append(tl)
        log(f"[EPOCH: {epoch}] Training throughput: {(args.batch_size * len(train_dl))/(tl):.3f} imgs/s")
        running_loss = 0.0  # Reset loss tracker

        # Validation Loop
        if epoch % args.epochs_eval == 0 or epoch == args.num_epochs:
            log(f"[EPOCH: {epoch}] Starting validation loop...")
            vl0 = time.time()  # Validation loop timer

            predictions = []
            references = []
            # Disable Dropout layers and BatchNorm layers for evaluation
            model.eval()

            for inputs, labels in valid_dl:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # AMP Context manager
                with autocast(device_type="cuda", dtype=MAP_STR_TO_DTYPE[args.mixed_precision], enabled=use_amp):
                    # Context-manager that disables gradient calculation. Reduces memory consumption and speeds up computations
                    with torch.no_grad():
                        outputs = model(inputs)
                    outputs = outputs.argmax(dim=-1)
                    predictions.extend(outputs.tolist())
                    references.extend(labels.tolist())

            accuracy = accuracy_score(references, predictions)
            vl = time.time() - vl0
            vl_t.append(vl)
            log(
                f"[EPOCH: {epoch}] Accuracy: {accuracy:.3f} | Validation throughput: {(args.eval_batch_size * len(valid_dl))/(vl):.3f} imgs/s"
            )

    full_training_time = time.time() - ft0
    report = report_memory()
    log("Training finished!")
    log(
#        f"Complete training Time: {full_training_time:.2f} | Training throughput:    {((args.num_epochs - 1) * args.batch_size * len(train_dl))/(sum(tl_t[1:])):.3f} imgs/s | Validation throughput: {(len(vl_t) * args.eval_batch_size * len(valid_dl))/(sum(vl_t)):.3f} imgs/s"
        f"Complete training Time: {int(full_training_time)} s "
    )
    log(
        f"Training throughput:    {int(((args.num_epochs - 1) * args.batch_size * len(train_dl))/(sum(tl_t[1:])))} imgs/s"
    )
    log(report)
    return args


if __name__ == "__main__":
    _args = get_args()
    main(_args)
