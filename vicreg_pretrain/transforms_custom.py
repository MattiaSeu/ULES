import random
from PIL import Image, ImageOps


class RandomFlipWithReturn:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.mirror(img), True
        return img, False


class RandomCropWithCoord:

    def __init__(self, size, p=0.5):
        self.p = p
        self.size = size

    def __call__(self, img):
        if random.random() < self.p:
            x = random.randint(0, img.size[0] - self.size[0])
            y = random.randint(0, img.size[1] - self.size[1])
            cropped_img = img.crop((x, y, x + self.size[0], y + self.size[1]))
            return cropped_img, x, y
        return img, -1, -1  # not to mix up with when the random crop starts from [0, 0]
