import os
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches

import cv2
from PIL import Image

# pylint: disable=no-member
from evaluation import constants

matplotlib.use('Agg')

RUN_PATH = 'run'


class Plotter:
    def __init__(self, run_path, image):
        self.run_path = run_path
        os.mkdir(self.run_path)
        self.image = image

    def set_image(self, image):
        self.image = image

    def save_img(self):
        im = Image.fromarray(self.image)
        path = os.path.join(self.run_path, constants.ORIGINAL_IMG_FILE_NAME)
        im.save(path)

    def plot_image(self, title):
        plt.figure()
        plt.imshow(self.image)
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_any_image(self, img, title):
        plt.figure()
        plt.imshow(img)
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)

    def plot_point_img(self, img, points, title):
        plt.figure()
        plt.imshow(img)
        plt.scatter(points[:, 0], points[:, 1])
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)

    def plot_test_point(self, point, title):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.image)
        plt.scatter(point[0], point[1], s=100, c='red', marker='x')
        plt.title(f"{title} Point")

        plt.tight_layout()

        path = os.path.join(self.run_path, f"{title}_point_result.jpg")
        plt.savefig(path)

    def plot_key_points(self, key_point_list, titles):
        plt.figure(figsize=(12, 8))

        if len(key_point_list) == 1:
            key_points = key_point_list[0]
            plt.imshow(self.image)
            plt.scatter(key_points[:, 0],
                        key_points[:, 1],
                        s=50,
                        c='red',
                        marker='x')
            plt.title('Predicted Key Point')

        else:
            for i in range(3):
                key_points = key_point_list[i]
                plt.subplot(1, 3, i + 1)
                plt.imshow(self.image)
                plt.scatter(key_points[:, 0],
                            key_points[:, 1],
                            s=50,
                            c='red',
                            marker='x')
                plt.title(f'Predicted Key Point {titles[i]}')

        plt.tight_layout()

        path = os.path.join(self.run_path, "key_point_results.jpg")
        plt.savefig(path)

    def plot_heatmaps(self, heatmaps, titles):
        plt.figure(figsize=(12, 8))

        if heatmaps.shape[0] == 1:
            heatmap_plot = plt.imshow(heatmaps[0],
                                      cmap=plt.cm.viridis,
                                      vmin=0,
                                      vmax=1)
            plt.colorbar(heatmap_plot, shrink=0.5)
            plt.title('Predicted Heatmap')

        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                                ncols=3,
                                                figsize=(15, 5))
            plt.subplots_adjust(wspace=0.2,
                                hspace=0.1,
                                left=0.1,
                                right=1.0,
                                top=0.9,
                                bottom=0.1)
            axis = [ax1, ax2, ax3]
            for i in range(3):
                im = axis[i].imshow(heatmaps[i],
                                    cmap=plt.cm.viridis,
                                    vmin=0,
                                    vmax=1)
                axis[i].set_title(f'Predicted Heatmap {titles[i]}')
            fig.colorbar(im, ax=axis, shrink=0.8)

        # plt.tight_layout()
        path = os.path.join(self.run_path, "heatmaps_results.jpg")
        plt.savefig(path)
        # plt.show()