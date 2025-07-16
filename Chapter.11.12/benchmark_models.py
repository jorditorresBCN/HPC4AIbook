import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vision_transformer
from custom_model import MyCustomModel
from time import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_dataloader(input_shape, num_classes, num_samples=256, batch_size=8, device="cpu", dtype=torch.float32):
    X = torch.randn((num_samples,) + input_shape, dtype=dtype)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, dtype):
    model.train()
    model.to(device=device, dtype=dtype)
    start = time()
    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return time() - start

if __name__ == "__main__":
    # Config
    batch_size = 8
    num_samples = 256
    resolution = 224
    n_classes = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"\nRunning training epoch benchmark on {device.upper()} with {dtype}\n")

    # Custom model
    custom_model = MyCustomModel(n_classes=n_classes, resolution=resolution)
    custom_input_shape = (resolution * resolution,)
    custom_loader = generate_dataloader(custom_input_shape, n_classes, num_samples, batch_size, device, dtype)
    custom_loss = nn.CrossEntropyLoss()
    custom_optimizer = torch.optim.SGD(custom_model.parameters(), lr=0.01)
    custom_time = train_one_epoch(custom_model, custom_loader, custom_loss, custom_optimizer, device, dtype)
    print(f"Custom model: {count_parameters(custom_model):,} params | Epoch time: {custom_time:.2f} s")

   # ViT-H/14 model
    vit_model = vision_transformer.vit_h_14(weights=None, num_classes=n_classes)
    vit_input_shape = (3, resolution, resolution)
    vit_loader = generate_dataloader(vit_input_shape, n_classes, num_samples, batch_size, device, dtype)
    vit_loss = nn.CrossEntropyLoss()
    vit_optimizer = torch.optim.SGD(vit_model.parameters(), lr=0.01)
    vit_time = train_one_epoch(vit_model, vit_loader, vit_loss, vit_optimizer, device, dtype)
    print(f"ViT-H/14 model: {count_parameters(vit_model):,} params | Epoch time: {vit_time:.2f} s")
