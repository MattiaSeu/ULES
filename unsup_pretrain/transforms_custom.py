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


class RandomSizeCropWithCoord:

    def __init__(self, min_max_ratio=[0.5, 0.8], p=0.5):
        self.p = p
        self.min_max_size=min_max_ratio

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            ratio_mod = random.uniform(self.min_max_size[0], self.min_max_size[1])
            crop_width = round(width*ratio_mod)
            crop_height = round(height*ratio_mod)

            max_x = width - crop_width
            max_y = height - crop_height

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            cropped_img = img.crop((x, y, x + crop_width, y + crop_height))
            return cropped_img, x, y, crop_width, crop_height
        return img, -1, -1, -1, -1  # not to mix up with when the random crop starts from [0, 0]
