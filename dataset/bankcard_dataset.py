from pathlib import Path
import random
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from torch.utils.data.dataset import Dataset
import albumentations as aug
import cv2
class Augmentation(object):
    def __init__(self) -> None:
        self.seq = iaa.Sequential([
            iaa.Emboss(),
            iaa.pillike.EnhanceContrast(),
            iaa.pillike.EnhanceBrightness(),
            iaa.pillike.EnhanceSharpness(),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 20))),
            iaa.Crop(px=(0, 4)),
            iaa.Affine(
                rotate=(-1, 1),
                scale={"x": (0.8, 0.98), "y": (0.85, 1.0)},
                translate_percent={"x": (-0.01, 0.01), "y": (-0.06, 0.06)}),
            iaa.PerspectiveTransform(scale=(0, 0.03)),
            iaa.Resize((0.3, 1.0), 'nearest')
        ])

    def __call__(self, image):
        return self.seq(image=image) 

class Augmentation2(object):
    def __init__(self) -> None:
        self.seq = aug.Compose([
            aug.Emboss(),
            aug.Sharpen(),
            aug.GaussianBlur(sigma_limit=(0, 1.0)),
            aug.GaussNoise(),
            aug.RandomShadow(),
            aug.RandomSunFlare(src_radius=80),
            aug.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5)),
            aug.Downscale(scale_min=0.4, scale_max=0.8),
            aug.Affine(
                rotate=(-3, 3),
                fit_output=True,
                scale={"x": (0.5, 1.0), "y": (0.5, 1.0)},
                translate_percent={"x": (-0.04, 0.04), "y": (-0.06, 0.06)}
                ),
            aug.Perspective(scale=(0, 0.1), fit_output=True),
            aug.RandomScale(scale_limit=(0.3, 1.0), interpolation=cv2.INTER_NEAREST)
        ])

    def __call__(self, image):
        return self.seq(image=image)['image']

class BankCardDataset(Dataset):
    
    def __init__(self, root, min_size=4, max_size=5, augmentation=Augmentation2(), rgb=False):
        self.color_format = 'RBG' if rgb else 'L'
        self.augmentation = augmentation
        self.min_size = min_size
        self.max_size = max_size
        self.images = list(Path(root).glob('*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_count = random.randint(self.min_size, self.max_size)
        image_paths = np.hstack((self.images[index], np.random.choice(self.images, image_count-1)))
        label = ''.join([ip.name[:4] for ip in image_paths])
        label = label.replace('_', '')
        img = np.hstack([np.array(Image.open(str(ip))) for ip in image_paths])
        if self.augmentation is not None:
            img = self.augmentation(img)
        return (Image.fromarray(img).convert(self.color_format), label)
