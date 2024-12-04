# Key Point Detection

事前にトレーニングされたモデルで特徴を抽出し、その特徴をデコーダネットワークの入力として使用し、重みを学習する。このデコーダはキーポイントのヒートマップを出力する。このヒートマップからキーポイントを抽出する。

## Training setup

### Dependencies
PyTorchとscikit-learnだけが必要。メインのREADMEのように仮想環境を設定すればこのスクリプトも実行可能となる。

### Dataset setup

すでにゲージに切り取られた画像のデータをトレーニングする。これはパイプライン内のノッチ検出器モデルへの入力であるからである。これには、ノートブック`data_preparation/crop_resize_images_keypoint_training.ipynb`を使用して、ゲージ画像を切り取ってサイズ変更することができる。

これらの画像を取得したら、label-studio <https://labelstud.io/>でラベルを付けることができる。

label-studioからJSONファイルを抽出した後、ヒートマップ生成ファイルを実行できる。そのためには、以下のコマンドを実行する。

```shell
python heatmap_generation.py --annotation annotation.json --directory direcotory_path --size SIZE
```
このため、ディレクトリパスには、jsonファイルに注釈がつけられた全ての画像を保持するフォルダimagesが含まれている必要がある。スクリプトは、全ての注釈が保存される新しいサブフォルダラベルを作成する。最終的な構造は次のようになる。
```
--direcotory_path
    --images
    --labels
```

引数SIZEはリサイズされた画像をサイズです。入力画像とヒートマップはSIZExSIZEになります。ここでは、初期値の448を選択する。モデルが変更されるとこのパラメータも変更できる。

次にラベル付きデータを次のように構造化する。
```
--training_data
    --train
        --images
        --labels
    --val
        --images
        --labels
```
### Training and Validation

To train your model you can run the training script with the following command:
モデルを訓練するためには、次のコマンドで実行できる。
```shell
python train.py --epochs epochs --learning_rate initial_learning_rate --data training_data --val --debug
```

`--data`については、前のステップで設定したフォルダ`training_data`を指定する。valフラグを設定すると訓練後すぐにvalidateが実行され、定性的な結果が得られる。

次のスクリプトを実行してモデルを検証することができる。

```shell
python key_point_validator.py --model_path path/to/model.pt --data data
```

ここでのデータは、依然と同じベースディレクトリ`training_data`である。

## Technical details

### Encoder

特徴を抽出するために、ビジュアルトランスフォーマーモデルDinoV2を使用する。<https://github.com/facebookresearch/dinov2>

## Decoder

現時点では、デコーダには非常にシンプルなモデルを用意している。このモデルは、抽出された特徴に対して1x1畳み込みを行い、それらをバイナリーアップサンプリングします。
### Training
現時点では、バイナリクロスエントロピー損失を持つAdamとのトレーニングを使用している。また、学習率スケジューラも使用している。トレーニングには、データ拡張、特にランダムクロップ、ランダムローテーション、ランダムジッターも使用している。

## Keypoint extraction
キーポイントを検出するために[Mean-Shift](https://en.wikipedia.org/wiki/Mean_shift)を使用する。具体的には、[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)の実装。

ここでは、正しいカットオフ閾値を選択することが重要である。また、ヒートマップを最初に正規化して、値が0から1の間になるようにすることも重要である。閾値以下のポイントはクラスタリングには考慮されない。閾値を下げると計算時間が増加する。私たちは0.5の閾値を選択する。

## Evaluation Predicted Key Points

視覚検査以外のキーポイントを比較・評価するために、私たちは3つの異なるメトリックを計算する。
1. 平均距離:各予測点は、最小ユークリッド距離によって各真点に割り当てられる。これらの最小距離は平均化される。
2. PCK:正しく予測された真のキーポイントの割合:
少なくとも1つの予測ポイントが近くにある真のポイントの割合。
3. 対応しない予測点の割合:
真の点に近くない予測点の割合。
