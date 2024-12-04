import numpy as np
from sklearn.cluster import MeanShift, KMeans
from scipy.spatial.distance import cdist

MEAN_DIST_KEY = "mean distance of predicted and true"
PCK_KEY = "Percentage of true where at least one predicted is close"
NON_ASSIGNED_KEY = "Percentage non assigned predicted points"

# ヒートマップからキーポイントを抽出する関数
def full_key_point_extraction(heatmaps, threshold=0.5, bandwidth=20):
    key_point_list = [] # 抽出されたキーポイントを格納するリスト
    for i in range(heatmaps.shape[0]): # ヒートマップの枚数分ループ
        # 中間点
        if i == 1:
            cluster_centers = extract_key_points(heatmaps[i], threshold,
                                                 bandwidth)
            key_point_list.append(cluster_centers)
        # 開始点と終了点
        else:
            cluster_center = extract_start_end_points(heatmaps[i], threshold)
            key_point_list.append(cluster_center)
    return key_point_list

# 開始点と終了点を抽出する関数
def extract_start_end_points(heatmap, threshold):
    max_heat = np.max(heatmap)
    if max_heat == 0:  # ヒートマップの最大値が0の場合
        return np.array([[0, 0]])  # 適当なデフォルトの座標を返すか、他の処理を実施
    # ヒートマップを0から1の範囲に正規化
    heatmap = heatmap / np.max(heatmap)

    coords = np.argwhere(heatmap > threshold) # 閾値以上の座標を抽出
    if coords.size == 0:  # 閾値以上の点がない場合
        return np.array([[0, 0]])  # 適当なデフォルトの座標を返す
    coords[:, [1, 0]] = coords[:, [0, 1]] # 座標を入れ替える

    kmeans = KMeans(n_clusters=1, n_init=3)# KMeansクラスタリングを用いて１つのクラスタを見つける
    kmeans.fit(coords)

    cluster_center = kmeans.cluster_centers_ # クラスタの中心を取得

    return cluster_center

# キーポイントを抽出する関数
def extract_key_points(heatmap, threshold, bandwidth):
    max_heat = np.max(heatmap)
    if max_heat == 0:
        return np.array([[0, 0]])
    # ヒートマップを0から1の範囲に正規化
    heatmap = heatmap / np.max(heatmap)

    # 閾値以上の座標を抽出
    coords = np.argwhere(heatmap > threshold)
    if coords.size == 0:
        return np.array([[0, 0]])
    # 座標を入れ替える
    coords[:, [1, 0]] = coords[:, [0, 1]]

    # Meanshiftクラスタリングを使用
    ms = MeanShift(bandwidth=bandwidth, n_jobs=-1)
    ms.fit(coords)

    # クラスタの中心を取得
    cluster_centers = ms.cluster_centers_

    return cluster_centers

# キーポイントの評価メトリクスを計算する関数
def key_point_metrics(predicted, ground_truth, threshold=10):
    """
    Gives back three different metrics to evaluate the predicted keypoints.
    For mean_distance each prediction is assigned to the true keypoint
    with smallest distance to it and then these distances are averaged
    For p_non_assigned we have the percentage of predicted key points
    that are not close to any true keypoint and therefore are non_assigned.
    For pck we have the percentage of true key points,
    where at least one predicted key point is close to it.

    For both p_non_assigned and pck,
    two key_points being close means that their distance is smaller than the threshold.
    :param predicted:
    :param ground_truth:
    :param threshold:
    :return:
    """
    distances = cdist(predicted, ground_truth) # 予測点と真の点の距離行列を計算

    cor_pred_indices = np.argmin(
        distances, axis=1)  # 予測に最も近い真の値の点のインデックス
    cor_true_indices = np.argmin(
        distances, axis=0)  # 芯の値の点に最も近い予測のインデックス

    # 予測点と対応する真の点との距離を計算
    corresponding_truth = ground_truth[cor_pred_indices]

    # calculate the Euclidean distances between predicted points and corresponding groundtruths
    pred_distances = np.linalg.norm(predicted[:len(corresponding_truth)] -
                                    corresponding_truth,
                                    axis=1)
    mean_distance = np.mean(pred_distances) # 平均距離を計算

    non_assigned = np.sum(pred_distances > threshold) # 閾値より大きい距離の点の数
    p_non_assigned = non_assigned / len(predicted) # 割合を計算

    # 真の点と対応する予測店との距離を計算
    corresponding_pred = predicted[cor_true_indices]

    gt_distances = np.linalg.norm(ground_truth[:len(corresponding_pred)] -
                                  corresponding_pred,
                                  axis=1)
    correct = np.sum(gt_distances <= threshold) # 閾値以下の距離の点の数
    pck = correct / len(
        ground_truth
    )  # 正確な予測の場合を計算

    results_dict = {
        MEAN_DIST_KEY: mean_distance,
        PCK_KEY: pck,
        NON_ASSIGNED_KEY: p_non_assigned
    }
    return results_dict
