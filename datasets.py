import numpy as np
import cv2
import typing as t
from typing_extensions import Literal
import h5py

from utils import preprocess_image

import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(
        self,
        image_paths: t.Sequence[str],
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
        verbose: bool = False,
    ):

        super().__init__()

        self.image_paths = image_paths
        self.transforms = transforms

        self._rng = np.random.RandomState(24)
        if verbose:
            print(f"Loaded {len(image_paths)} paths")

        self.__cache = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]
        image = cv2.imread(self.image_paths[index])
        image, coords, distances = preprocess_image(image, self.transforms)
        self.__cache[index] = (image, coords, distances)

        return image, coords, distances


class MNISTDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
        verbose: bool = False,
    ):

        super().__init__()

        self.dataset = dataset
        self.transforms = transforms

        self._rng = np.random.RandomState(24)
        if verbose:
            print(f"Loaded {len(dataset)} paths")

        self.__cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]

        data = self.dataset[index]
        image = torch.permute(data[0], (1, 2, 0)).numpy()

        image, coords, distances = preprocess_image(image, self.transforms, mnist=True)

        self.__cache[index] = (image, coords, distances)

        return image, coords, distances


class TableChairDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        data_split: Literal["train", "valid", "test"],
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
    ):

        super().__init__()

        self.h5_file_path = h5_file_path
        self.transforms = transforms
        self.data_split = data_split

        if data_split == "train":
            self.data_key = "train_images"
        elif data_split == "valid":
            self.data_key = "val_images"
        else:
            self.data_key = "test_images"

        with h5py.File(self.h5_file_path, "r") as h5_file:
            self._images = h5_file[self.data_key][:]

        self.__cache = {}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]
        image = self._images[index].astype(np.uint8) * 255
        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        image, coords, distances = preprocess_image(image, self.transforms)

        self.__cache[index] = (image, coords, distances)

        return image, coords, distances


class PetDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
        verbose: bool = False,
    ):

        super().__init__()

        self.dataset = dataset
        self.transforms = transforms

        self._rng = np.random.RandomState(24)
        if verbose:
            print(f"Loaded {len(dataset)} paths")

        self.__cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]

        data, target = self.dataset[index]
        target.expand(data.shape)
        x_masked = target * data
        image = torch.permute(x_masked, (1, 2, 0)).numpy()

        image, coords, distances = preprocess_image(image, self.transforms)

        self.__cache[index] = (image, coords, distances)

        return image, coords, distances
