import os
import time
import json
import argparse

import numpy as np
from PIL import Image
import cv2

from key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from plots import RUN_PATH, Plotter
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_keypoint_coordinate(image, key_point_model_path, run_path, eval_mode, json_data, cpu=True):

    result = {}

    image_open = Image.open(image).convert("RGB") # 画像をRGB形式で開く
    image_open = np.asarray(image_open) # 画像をNumPy配列に変換

    resized_img = cv2.resize(image_open,dsize=(448, 448),interpolation=cv2.INTER_CUBIC)

    # 画像プロット用のインスタンスを生成
    plotter = Plotter(run_path, resized_img) 

    if eval_mode:
        plotter.save_img()

    # キーポイントデータを取得するためのモデルを初期化
    key_point_inferencer = KeyPointInference(key_point_model_path, cpu=cpu)
    # ヒートマップを取得
    heatmaps = key_point_inferencer.predict_heatmaps(resized_img)
    # キーポイントの検出
    key_point_list = detect_key_points(heatmaps)

    titles = []

    for i, keypoint in enumerate(json_data["keypoints"]):
        keypoint_name = keypoint["name"]
        keypoint_coordinate = key_point_list[i].tolist()
        result[keypoint_name] = keypoint_coordinate
        titles.append(keypoint_name)

    if eval_mode:
        plotter.plot_heatmaps(heatmaps, titles)
        plotter.plot_key_points(key_point_list, titles)

    return result

def imagepath_to_coordinates(image_path, key_point_model_path, json_path, cpu):
    result = {}
    image_open = Image.open(image_path).convert("RGB")
    image_open = np.asarray(image_open)


    resized_img = cv2.resize(image_open,dsize=(448, 448),interpolation=cv2.INTER_CUBIC)

    # キーポイントデータを取得するためのモデルを初期化
    key_point_inferencer = KeyPointInference(key_point_model_path, cpu)
    # ヒートマップを取得
    heatmaps = key_point_inferencer.predict_heatmaps(resized_img)
    # キーポイントの検出
    key_point_list = detect_key_points(heatmaps)
    json_data = load_json(json_path)
    for i, keypoint in enumerate(json_data["keypoints"]):
        keypoint_name = keypoint["name"]
        keypoint_coordinate = key_point_list[i].tolist()
        result[keypoint_name] = keypoint_coordinate

    return result

def main():

    parser = argparse.ArgumentParser(description="Keypoint detection pipeline")
    parser.add_argument("--input", required=True, help="Path to the input image or directory")
    parser.add_argument("--key_point_model", required=True, help="Path to the keypoint detection model")
    parser.add_argument("--base_path", required=True, help="Base path to save outputs")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file with keypoint names")

    args = parser.parse_args()

    input_path = args.input # 入力画像のパス
    key_point_model = args.key_point_model # キーポイント検出のモデルのパス
    base_path = args.base_path # outputの保存先
    eval_mode = args.eval # 評価モードの有無
    json_path = args.json_path # jsonファイルのパス

    time_str = time.strftime("%Y%m%d%H%M%S") # 処理の開始時間を出力
    # 処理の開始時間に基づいてoutputの保存先を設定し、フォルダを作成
    base_path = os.path.join(base_path, RUN_PATH + '_' + time_str) 
    os.makedirs(base_path)

    # 結果保存フォルダを作成
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    json_data = load_json(json_path) # jsonデータを読み込む

    # 画像に対するパイプラインを実行
    if os.path.isfile(input_path): # もし画像がファイルの場合
        image_name = os.path.basename(input_path) # 画像のパスを取得
        run_path = os.path.join(base_path, image_name) # outputの保存先を設定

        result = get_keypoint_coordinate(input_path, key_point_model, run_path, eval_mode, json_data)

        # JSONファイルとして保存
        json_file_path = os.path.join(results_dir, f"{image_name}.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

    elif os.path.isdir(input_path): # 入力されたのがディレクトリだった場合
        for image_name in os.listdir(input_path):
            img_path = os.path.join(input_path, image_name)
            run_path = os.path.join(base_path, image_name)

            result = get_keypoint_coordinate(input_path, key_point_model, run_path, eval_mode, json_data)

            # JSONファイルとして保存
            json_file_path = os.path.join(results_dir, f"{image_name}.json")
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(result, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()