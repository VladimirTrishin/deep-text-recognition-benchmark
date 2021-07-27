from pathlib import Path
import random
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from torch.utils.data.dataset import Dataset

class Augmentation(object):
    def __init__(self) -> None:
        self.seq = iaa.Sequential([
            iaa.Emboss(),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 20)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Crop(px=(0, 4)),
            iaa.Affine(
                rotate=(-1, 1),
                scale={"x": (0.8, 1.02), "y": (0.85, 1.0)},
                translate_percent={"x": (-0.01, 0.01), "y": (-0.06, 0.06)}),
            iaa.PerspectiveTransform(scale=(0, 0.03)),
            iaa.Resize((0.3, 1.0), 'nearest')
        ])

    def __call__(self, image):
        return self.seq(image=image) 

class BankCardDataset(Dataset):
    
    def __init__(self, root, min_size=4, max_size=5, augmentation=Augmentation(), rgb=False):
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
