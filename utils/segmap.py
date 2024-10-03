import numpy as np
import torch

ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']
# why i choose 20 classes
# https://stackoverflow.com/a/64242989

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)

colors = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(len(colors)), colors))


def encode_segmap(mask):
    # remove unwanted classes and rectify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask




def decode_segmap(temp):
    # convert gray scale to color
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def kitti_encode(color_kitti_labels):
    kitti_colors_dict = {
        (128, 128, 128): 11,
        (128, 0, 0): 1,
        (128, 64, 128): 2,
        (0, 0, 192): 3,
        (64, 64, 128): 4,
        (128, 128, 0): 5,
        (192, 192, 128): 6,
        (64, 0, 128): 7,
        (192, 128, 128): 8,
        (64, 64, 0): 9,
        (0, 128, 192): 10,
        (255, 0, 0): 0
    }

    temp = color_kitti_labels.cpu().numpy()
    # for rgb in temp.shape[-1]
    #     r[temp == l] = label_colours[l][0]
    #     g[temp == l] = label_colours[l][1]
    #     b[temp == l] = label_colours[l][2]
    rgb_array = temp.permute(1, 2, 0).numpy()

    gray = np.zeros(temp.shape[:2], dtype=np.uint8)

    # gray = torch.zeros(temp.shape[:2])
    def rgb_to_gray(rgb):
        # TODO: fix the unresolved reference or get rid of the function entirely
        gray = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for color, gray_value in rgb_to_gray_mapping.items():
            mask = np.all(rgb == np.array(color), axis=-1)
            gray[mask] = gray_value
        return gray

    for color, gray_value in kitti_colors_dict.items():
        mask = np.all(temp[0] == (np.array(color)), axis=-1)
        gray[mask] = gray_value

    # grayscale_kitti_labels = np.zeros((temp.shape[0], temp.shape[1], 1))
    # grayscale_kitti_labels[:, :, 0] =
    return gray


def kitti_decode(gray_encode):
    kitti_colors_list = [
        [255, 0, 0],  # ignore
        [128, 0, 0],  # building
        [128, 64, 128],  # road
        [0, 0, 192],  # sidewalk
        [64, 64, 128],  # fence
        [128, 128, 0],  # vegetation
        [192, 192, 128],  # pole
        [64, 0, 128],  # car
        [192, 128, 128],  # sign
        [64, 64, 0],  # pedestrian
        [0, 128, 192],  # cyclist
        [128, 128, 128],  # sky
    ]

    label_colours = dict(zip(range(12), kitti_colors_list))

    temp = gray_encode.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(12):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def multi_mat_decode(encoded_input):
    multi_mat_colors_list = [
        [44, 160, 44],  # asphalt
        [31, 119, 180],  # concrete
        [255, 127, 14],  # metal
        [214, 39, 40],  # road marking
        [140, 86, 75],  # fabric, leather
        [127, 127, 127],  # glass
        [188, 189, 34],  # plaster
        [255, 152, 150],  # plastic
        [23, 190, 207],  # rubber
        [174, 199, 232],  # sand
        [196, 156, 148],  # gravel
        [197, 176, 213],  # ceramic
        [247, 182, 210],  # cobblestone
        [199, 199, 199],  # brick
        [219, 219, 141],  # grass
        [158, 218, 229],  # wood
        [57, 59, 121],  # leaf
        [107, 110, 207],  # water
        [156, 158, 222],  # human body
        [99, 121, 57]  # sky
    ]
    label_colours = dict(zip(range(20), multi_mat_colors_list))

    temp = encoded_input.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(20):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def visnir_decode(encoded_input):

    visnir_colors_list = [
        [ 192,192,192],  #       asphalt
        [ 105,105,105],  #       gravel
        [ 160, 82, 45],  #        soil
        [ 244,164, 96],  #        sand
        [  60,179,113],  #        bush
        [  34,139, 34],  #       forest
        [ 154,205, 50],  #      low grass
        [   0,128,  0],  #      high grass
        [   0,100,  0],  #  misc. vegetation
        [   0,250,154],  #      tree crown
        [ 139, 69, 19],  #      tree trunk
        [   1, 51, 73],  #      building
        [ 190,153,153],  #       fence
        [   0,132,111],  #        wall
        [   0,  0,142],  #        car
        [   0, 60,100],  #        bus
        [ 135,206,250],  #        sky
        [ 128,  0,128],  #    misc. object
        [ 153,153,153],  #        pole
        [ 255,255,  0],  #     traffic sign
        [ 220, 20, 60],  #      person
        [ 255,182,193],  #      animal
        [ 220,220,220],  #    ego vehicle
        [ 0,0,0      ]  #      undefined

    ]
    label_colours = dict(zip(range(len(visnir_colors_list)), visnir_colors_list))

    temp = encoded_input.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(20):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
