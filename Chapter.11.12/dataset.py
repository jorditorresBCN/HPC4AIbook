import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class DatasetFolder(Dataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        extension: str = ".jpeg",
    ) -> None:
        self.root = root
        self.extension = extension

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx=class_to_idx, extension=extension)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extension: str = ".jpeg",
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if path.lower().endswith(extension):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes

        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)

        return instances

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, label = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        return {"image": img, "label": label}

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class MyCustomCollator:
    resolution: int = 64

    def __call__(self, samples):
        # Convert RGB --> Gray scale & Resize
        inputs = [sample["image"].convert("L").resize((self.resolution, self.resolution)) for sample in samples]
        # Convert PIL image to torch.tensor
        inputs = [pil_to_tensor(sample).to(torch.float32) for sample in inputs]
        # Reshape properly before feeding the tensor into the model
        # TIP: We use `torch.stack` to create the batch dimension!
        inputs = torch.stack([sample.flatten() for sample in inputs])

        # Convert labels (int) to torch.tensor
        labels = torch.tensor([torch.tensor(sample["label"]) for sample in samples])

        return inputs, labels


@dataclass
class RGBCollator:
    resolution: int

    def __call__(self, samples):
        # Resize
        inputs = [sample["image"].resize((self.resolution, self.resolution)) for sample in samples]
        # Convert PIL image to torch.tensor
        inputs = [pil_to_tensor(sample).to(torch.float32) for sample in inputs]
        # Reshape properly before feeding the tensor into the model
        # TIP: We use `torch.stack` to create the batch dimension!
        inputs = torch.stack([sample for sample in inputs])

        # Convert labels (int) to torch.tensor
        labels = torch.tensor([torch.tensor(sample["label"]) for sample in samples])

        return inputs, labels
