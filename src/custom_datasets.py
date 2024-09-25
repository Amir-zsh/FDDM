from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path


class DCTScaledDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image and label at the specified index
        data = self.data[index]
        if len(data) == 2:
            image, label = data
        else:
            image = data
            label = torch.tensor(0)

        if self.transform is not None:
            image = self.transform(image)

        # Return the scaled image and label as a tuple
        return image, label


# Load the Fashion MNIST dataset
# transform = Compose([
#                ToTensor(),
#                Normalize((0.5,), (0.5,))
# ])


class DatasetFromDir(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        transform=None,
        exts=["jpg", "jpeg", "png", "tiff", "JPEG"],
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


class DatasetWithCache(DCTScaledDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_data = []
        self.use_cache = False

    def set_use_cache(self, use_cache):
        # if use_cache:
        #     self.cached_data = torch.stack(self.cached_data)
        # else:
        #     self.cached_data = []
        self.use_cache = use_cache

    def __getitem__(self, index):
        # time.sleep(1)
        if not self.use_cache:
            x = super().__getitem__(index)
            # self.cached_data.append(x)
            # print(f"added to cache, {len(self.cached_data)}")

        else:
            x = self.cached_data[index]

        return x
