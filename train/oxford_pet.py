import os
import torch
import shutil
import numpy as np
import numpy.random as random
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # image = np.array(Image.open(image_path).convert("RGB"))
        image = Image.open(image_path).convert("RGB")

        trimap = np.array(Image.open(mask_path))
        mask, valid_mask = self._preprocess_mask(trimap)

        mask = Image.fromarray(mask)
        valid_mask = Image.fromarray(valid_mask)
        trimap = Image.fromarray(trimap)
            
        if self.transform is not None:
            image, mask, valid_mask, trimap = self.transform( image, mask, valid_mask, trimap)

        return {
            "image": image,
            "mask": mask,
            "valid_mask": valid_mask,
            "trimap":trimap,
        }

    @staticmethod
    def _preprocess_mask(trimap):
        mask = np.zeros_like(trimap, dtype=np.float32)

        mask[trimap == 1] = 1.0
        mask[trimap == 2] = 0.0

        valid_mask = (trimap != 3).astype(np.float32)

        return mask, valid_mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]

# while not in test mode , do shuffle 
    # incase dataset sorted in breed

        if self.mode != "test":
            rng = np.random.default_rng(100)
            rng.shuffle(filenames)

            split_idx = int(len(filenames) * 0.9)

            if self.mode == "train":
                filenames = filenames[:split_idx]
            else:
                filenames = filenames[split_idx:]

        # if self.mode == "train":  # 90% for train
        #     filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        # elif self.mode == "valid":  # 10% for validation
        #     filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

class SegmentationTransform:
    def __init__(self, image_size=(256, 256), train=False):
        self.image_size = image_size
        self.train = train

    def __call__(self, image, mask, valid_mask,trimap):

        image = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.BILINEAR
        )(image)

        mask = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST
        )(mask)

        valid_mask = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST
        )(valid_mask)
        
        trimap = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST
        )(trimap)

        image = TF.to_tensor(image)

        mask = torch.from_numpy(
            np.array(mask, dtype=np.float32)
        ).unsqueeze(0)

        valid_mask = torch.from_numpy(
            np.array(valid_mask, dtype=np.float32)
        ).unsqueeze(0)
        trimap = torch.from_numpy(
            np.array(trimap, dtype=np.int64)
        ).unsqueeze(0)

        return image, mask, valid_mask,trimap


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


def load_dataset(data_path, mode):
    assert mode in {"train", "valid", "test"}

    transform = SegmentationTransform(
        image_size=(256, 256)
    )

    return OxfordPetDataset(
        root=data_path,
        mode=mode,
        transform=transform
    )

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory,'dataset')

images_dir = os.path.join(data_directory, "images")
annotations_dir = os.path.join(data_directory, "annotations")
images_tar = os.path.join(data_directory, "images.tar.gz")
annotations_tar = os.path.join(data_directory, "annotations.tar.gz")

if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
    if os.path.exists(images_tar):
        extract_archive(images_tar)
    if os.path.exists(annotations_tar):
        extract_archive(annotations_tar)