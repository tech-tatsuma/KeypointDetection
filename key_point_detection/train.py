import argparse
import os
import time
import sys
import logging
import json

import torch
import optuna

from torch import nn, optim
from torch.utils.data import DataLoader

# Append path of parent directory to system to import all modules correctly
parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

# pylint: disable=wrong-import-position
from key_point_dataset import RUN_PATH, KeypointImageDataSet, \
    TRAIN_PATH, IMG_PATH, LABEL_PATH
from key_point_validator import KeyPointVal
from model import ENCODER_MODEL_NAME, Encoder, Decoder, EncoderDecoder, \
    INPUT_SIZE, N_HEATMAPS, N_CHANNELS

BATCH_SIZE = 8

# キーポイント検出の訓練を行うクラス
class KeyPointTrain:
    def __init__(self, base_path, debug):

        self.debug = debug # デバッグモードの有無

        # 画像とアノテーションデータのパスを設定
        image_folder = os.path.join(base_path, TRAIN_PATH, IMG_PATH)
        annotation_folder = os.path.join(base_path, TRAIN_PATH, LABEL_PATH)

        # 特徴抽出器（エンコーダ）の設定
        self.feature_extractor = Encoder(pretrained=True)

        # 訓練データセットの設定
        self.train_dataset = KeypointImageDataSet(
            img_dir=image_folder,
            annotations_dir=annotation_folder,
            train=True,
            val=False)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4)

        # デコーダの生成
        self.decoder = self._create_decoder()

        # 損失関数の定義(二値交差エントロピー損失)
        self.criterion = nn.BCELoss()

        # エンコーダとデコーダから成る完全なモデルの設定
        self.full_model = EncoderDecoder(self.feature_extractor, self.decoder)

        # 損失を記録するための辞書
        self.loss = {}

    # デコーダを生成するための関数
    def _create_decoder(self):
        n_feature_channels = self.feature_extractor.get_number_output_channels(
        )
        if self.debug:
            print(f"Number of feature channels is {n_feature_channels}")
        return Decoder(n_feature_channels, N_CHANNELS, INPUT_SIZE, N_HEATMAPS)

    # モデルの訓練を行う関数
    def train(self, num_epochs, learning_rate):

        # GPUが使える場合はGPUを使用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.debug:
            print(f"Using {device} device")

        # モデルをGPUに転送
        self.full_model.to(device)

        # 最適化アルゴリズムの設定
        optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50)

        # 訓練ループ
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, annotations in self.train_dataloader:
                # Forward pass
                inputs, annotations = inputs.to(device), annotations.to(device) # データをデバイスに転送
                outputs = self.full_model(inputs) # 順伝播
                loss = self.criterion(outputs, annotations) # 損失の計算

                # Backward pass and optimization
                optimizer.zero_grad() # 勾配をゼロで初期化
                loss.backward() # 逆伝播

                optimizer.step() # パラメータ更新

                running_loss += loss.item() # 損失を加算

            # エポックごとの平均損失を計算
            loss = running_loss / len(self.train_dataloader)
            self.loss[epoch + 1] = loss

            # 学習率の更新とログの出力
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(loss)
            after_lr = optimizer.param_groups[0]["lr"]
            loss_msg = f"Epoch {epoch + 1}: Loss = {loss}, lr {before_lr} -> {after_lr}"
            print(loss_msg)
            logging.info(loss_msg)

        print('Finished Training')

    # 訓練された完全なモデルを返す関数
    def get_full_model(self):
        return self.full_model

# def objective(trial, base_path, debug):
#     num_epochs = 50
#     learning_rate = round(trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),6)
#     print("learning_rate: ", learning_rate)

#     trainer = KeyPointTrain(base_path, debug)
#     trainer.train(num_epochs, learning_rate)
#     validation_loss = trainer.loss[num_epochs] # 仮に最終エポックの損失を評価スコアとする

#     return validation_loss

def main():
    # 引数を読み込む
    args = read_args()

    # 訓練のためのハイパーパラメーターを設定
    num_epochs = args.epochs
    # learning_rate = args.learning_rate
    learning_rate = round(0.009400290931734527,6)

    base_path = args.data # データの基本パス
    val = args.val # 検証を行うかどうか
    debug = args.debug # デバッグモードの有無

    # 再現性のためシード値を固定
    torch.manual_seed(0)

    # 実行ディレクトリの設定
    time_str = time.strftime("%Y%m%d-%H%M%S")
    run_path = os.path.join(base_path, RUN_PATH + '_' + time_str)
    os.makedirs(run_path, exist_ok=True)

    # ロガーをセットアップ
    log_path = os.path.join(run_path, "run.log")
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    # 訓練クラスの初期化
    if debug:
        print("initializing trainer")

    trainer = KeyPointTrain(base_path, debug)
    if debug:
        print("initialized trainer successfully")

    # モデルの訓練
    logging.info("Start training")
    if debug:
        print("start training")
    trainer.train(num_epochs, learning_rate)
    logging.info("Finished training")

    # 訓練されたモデルを取得
    model = trainer.get_full_model()

    # モデルの保存
    model_path = os.path.join(run_path, f"model_{time_str}.pt")
    torch.save(model.state_dict(), model_path)

    # 損失データの保存
    loss_path = os.path.join(run_path, "loss.json")
    loss_json = json.dumps(trainer.loss, indent=4)
    with open(loss_path, "w") as outfile:
        outfile.write(loss_json)

    # パラメータの保存
    params = {
        'encoder': ENCODER_MODEL_NAME,
        'number of decoder channels': N_CHANNELS,
        'initial learning rate': learning_rate,
        'epochs': num_epochs,
        'batch size': BATCH_SIZE
    }

    param_file_path = os.path.join(run_path, "paramaters.txt")
    write_parameter_file(param_file_path, params)

    if val:
        validator = KeyPointVal(model, base_path, time_str)
        validator.validate()

# def main():
#     args = read_args()

#     base_path = args.data
#     debug = args.debug

#     torch.manual_seed(0)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(lambda trial: objective(trial, base_path, debug), n_trials=20)

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Best trial:")
#     trial = study.best_trial

#     print("    Value: ", trial.value)
#     print("    Params: ")
#     for key, value in trial.params.items():
#         print("      {}: {}".format(key, value))

# パラメータファイルを書き込む関数
def write_parameter_file(filename, params):
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

# コマンドライン引数を解析する関数
def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument('--learning_rate',
                        type=float,
                        required=False,
                        default=3e-4,
                        help="Learning rate for training")
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help="Base path of data")
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main()
