from semantic_kitti_api.auxiliary.laserscan import LaserScan
import os
import matplotlib.pyplot as plt
rootdir = "/home/matt/data/kitti_sem/train/dataset/sequences/"


for subdir, dirs, files in os.walk(rootdir, topdown=False):
    for file in files:
        scan = LaserScan(project=True)
        scan.open_scan(subdir + "/" + file)
        rv_image = scan.proj_range
        fig, axs = plt.subplots(1, figsize=(64 // 8, 1024 // 8))
        pos = axs.imshow(rv_image, interpolation='none')
        plt.show()

