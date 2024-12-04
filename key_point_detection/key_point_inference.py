from PIL import Image

from key_point_detection.key_point_extraction import full_key_point_extraction
from key_point_detection.model import load_model
from key_point_detection.key_point_dataset import custom_transforms

# キーポイント推論を行うクラス
class KeyPointInference:
    def __init__(self, model_path, cpu=False):
        # モデルパスを指定してロード
        self.model = load_model(model_path, cpu)

    def predict_heatmaps(self, image):
        # 入力画像をPIL形式に変換
        img = Image.fromarray(image)
        # 画像の前処理を実行
        image_t = custom_transforms(train=False, image=img)
        # バッチ次元を追加
        image_t = image_t.unsqueeze(0)

        # モデルを使用して画像からヒートマップを推論
        heatmaps = self.model(image_t)

        # 推論結果をdetachしてNumPy配列に変換
        heatmaps = heatmaps.detach().squeeze(0).numpy()

        return heatmaps # ヒートマップを返す

# ヒートマップからキーポイントを抽出する関数
def detect_key_points(heatmaps):
    # ヒートマップと閾値を指定してキーポイントを抽出
    key_point_list = full_key_point_extraction(heatmaps, 0.6)

    # 抽出されたキーポイントリストを返す
    return key_point_list
