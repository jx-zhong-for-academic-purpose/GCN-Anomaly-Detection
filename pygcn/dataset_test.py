import os
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def plot_gt(anomaly_score, vid):
    plt.plot(anomaly_score)
    plt.title(vid)
    plt.xlabel("frame number")
    plt.ylabel("anomaly score")
    plt.ylim([0, 1.1])
    plt.show()


def get_auc(ab_score):
    test_gt_file = "/home/zjx/data/UCF_Crimes/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
    test_frames_folder = "/home/zjx/data/UCF_Crimes/Video_frames/"
    all_gt = []
    all_score = []

    with open(test_gt_file, "r") as f:
        ground_truth = f.readlines()
        for gt in ground_truth:
            tmp = gt.split()
            vid = tmp[0].replace(".mp4", "")
            # if "Normal" == tmp[1]: continue
            frame_path = os.path.join(test_frames_folder, vid)
            frame_cnt = len(os.listdir(frame_path))
            anomaly_score = np.zeros(frame_cnt)
            if -1 != int(tmp[2]):
                b = int(tmp[2]) - 1
                e = int(tmp[3])
                anomaly_score[b: e] = 1

            if -1 != int(tmp[4]):
                b = int(tmp[4]) - 1
                e = int(tmp[5])
                anomaly_score[b: e] = 1

            # show_flag = False
            # for i in range(1, 18):
            #    if anomaly_score[-i] == 1:
            #        show_flag = True
            # if not show_flag:
            #    continue
            cur_ab = ab_score[vid]
            ab_same_size = np.zeros_like(anomaly_score)
            print len(ab_same_size), len(cur_ab)
            for i in range(len(ab_same_size)):
                ab_same_size[i] = cur_ab[i]

            all_gt.extend(anomaly_score.tolist())
            all_score.extend(ab_same_size.tolist())

    return roc_auc_score(all_gt, all_score)
    # print("AUC: %.6f" % roc_auc_score(all_gt, all_score))


if __name__ == '__main__':
    feature_path = "/home/zjx/data/UCF_Crimes/C3D_features/c3d_fc6_features.hdf5"
    ucf_crime_test = UCFCrimeTest(feature_path)
    test_loader = DataLoader(dataset=ucf_crime_test)

    for data in test_loader:
        feat, is_normal, anomaly_score, vid = data
