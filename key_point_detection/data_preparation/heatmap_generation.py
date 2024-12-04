import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch

IMAGE_DIR = 'images'
LABEL_DIR = 'labels'

# jsonファイルからデータを取得する関数
def get_data_from_json(json_path):
    # jsonファイルを読み込む
    with open(json_path) as f:
        data = json.load(f)
    # jsonファイルのデータを返す
    return data

# アノテーションデータを解析する関数
def get_annotations(data):
    # アノテーションデータを格納するリストを初期化
    annotation_list = []

    # 全てのデータに対して繰り返し処理
    for data_point in data:
        # キーポイントのアノテーションを格納する辞書を初期化
        key_point_annotation = {}

        # Get image name. We have image name in format :
        # /data/upload/1/222ae49e-1_cropped_000001_jpg.rf.c7410b0b01b2bc3a6cdff656618a3015.jpg
        # get rid of everything before the '-'
        # 画像名を取得
        idx = data_point['data']['img'].find('-') + 1
        # 画像名を辞書に追加
        key_point_annotation['img_name'] = data_point['data']['img'][idx:]

        # スタートノッチ，エンドノッチ，中間ノッチの座標を格納するリストを初期化
        key_point_annotation['start'] = []
        key_point_annotation['end'] = []
        key_point_annotation['middle'] = []

        # データポイントの中からresultを取得
        for annotation in data_point['annotations'][0]['result']:
            # Start Notchの場合
            if annotation['value']['keypointlabels'][0] == 'start':
                key_point_annotation['start'].append(
                    {k: annotation['value'][k]
                     for k in ('x', 'y')})

            # End Notchの場合
            if annotation['value']['keypointlabels'][0] == 'end':
                key_point_annotation['end'].append(
                    {k: annotation['value'][k]
                     for k in ('x', 'y')})

            # middle Notchの場合
            if annotation['value']['keypointlabels'][0] == 'middle':
                key_point_annotation['middle'].append(
                    {k: annotation['value'][k]
                     for k in ('x', 'y')})

        # アノテーションリストにアノテーションを追加
        annotation_list.append(key_point_annotation)

    return annotation_list


def add_gaussian_to_heatmap(heatmap, x, y, sigma=8, visualize=False):
    """
    Generates a 2D Gaussian point at location x,y in tensor t.

    x should be in range (-1, 1) to match the output of fastai's PointScaler.

    sigma is the standard deviation of the generated 2D Gaussian.
    """
    h, w = heatmap.shape

    # Heatmap pixel per output pixel

    tmp_size = sigma * 3

    # Top-left
    x1, y1 = int(x - tmp_size), int(y - tmp_size)

    # Bottom right
    x2, y2 = int(x + tmp_size + 1), int(y + tmp_size + 1)

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    center = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    gaussian = torch.tensor(
        np.exp(-((tx - center)**2 + (ty - center)**2) / (2 * sigma**2)))

    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)

    scale = 255 if visualize else 1

    # add the gaussian to the heatmap by taking max of values
    # Safe-guarding the slicing by ensuring the ranges are non-negative
    if (g_x_min < g_x_max) and (g_y_min < g_y_max):
        # heatmap[img_y_min:img_y_max, img_x_min:img_x_max] = \
        #     np.maximum(scale * gaussian[g_y_min:g_y_max, g_x_min:g_x_max],
        #                heatmap[img_y_min:img_y_max, img_x_min:img_x_max])
        gaussian_np = gaussian.numpy()  # torch tensorをnumpy配列に変換
        heatmap[img_y_min:img_y_max, img_x_min:img_x_max] = \
            np.maximum(scale * gaussian_np[g_y_min:g_y_max, g_x_min:g_x_max],
                    heatmap[img_y_min:img_y_max, img_x_min:img_x_max])

    return heatmap


# key_point_list is list of dicts with 'x' and 'y' coordinate, 1 for each keypoint
# outputs heatmap
def generate_heatmap(key_point_list, img=None, size=(448, 448)):
    # Initialize empty heatmap
    if img is not None:
        heatmap = img
        visualize = True
    else:
        heatmap = torch.zeros(size)
        visualize = False

    # For each key point add gaussian kernel centered around it to heatmap

    for key_point in key_point_list:
        heatmap = add_gaussian_to_heatmap(heatmap,
                                          key_point['x'] * size[0] / 100,
                                          key_point['y'] * size[1] / 100,
                                          visualize=visualize)

    return heatmap.numpy()


def heatmap_from_key_points(annotation, size):
    size = (size, size)
    heatmap_start = generate_heatmap(annotation['start'], size=size)
    heatmap_middle = generate_heatmap(annotation['middle'], size=size)
    heatmap_end = generate_heatmap(annotation['end'], size=size)

    heatmap = np.stack((heatmap_start, heatmap_middle, heatmap_end), axis=0)
    return heatmap


def main():
    args = read_args()
    json_path = args.annotation
    base_path = args.directory

    img_directory = os.path.join(base_path, IMAGE_DIR)
    label_directory = os.path.join(base_path, LABEL_DIR)

    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    data = get_data_from_json(json_path)
    annotations = get_annotations(data)

    fig = plt.figure(figsize=(10, 15))
    fig.subplots_adjust(left=0.05,
                        right=0.95,
                        bottom=0.05,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)
    rows = int(len(annotations) / 20) + 1
    columns = 2

    plot_idx = 1
    for i, annotation in enumerate(annotations):
        heatmap = heatmap_from_key_points(annotation, args.size)
        if i % 13 == 0:
            img_path = os.path.join(img_directory, annotation['img_name'])
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            heatmap_show = generate_heatmap(annotation['middle'],
                                            torch.tensor(gray))
            fig.add_subplot(rows, columns, plot_idx)
            plt.axis('off')
            plt.imshow(heatmap_show)
            plot_idx += 1

        # filename = annotation['img_name'][:-4] + '.npy'
        filename = os.path.splitext(annotation['img_name'])[0] + '.npy'
        path = os.path.join(label_directory, filename)
        np.save(path, heatmap)
        i += 1
    plt.show()


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation',
                        type=str,
                        required=True,
                        help="json file of annotations")
    parser.add_argument('--directory',
                        type=str,
                        required=True,
                        help="Directory where images are and labels go")
    parser.add_argument('--size',
                        type=int,
                        required=True,
                        help="Heatmap format is size x size")

    return parser.parse_args()


if __name__ == '__main__':
    main()
