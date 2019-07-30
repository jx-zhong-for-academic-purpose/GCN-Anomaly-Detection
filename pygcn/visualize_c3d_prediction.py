import os
import numpy as np
import collections
import array
from matplotlib import pyplot as plt
import cv2

test_videos = "/home/lnn/data/UCF_Crimes/TestVideos/"

vid2gt = {}
vid2abnormality = {}


def softmax(raw_score):
    exp_s = np.exp(raw_score - raw_score.max(axis=-1)[..., None])
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


def read_binary_blob(filename):
    #
    # Read binary blob file from C3D
    # INPUT
    # filename    : input filename.
    #
    # OUTPUT
    # s           : a 1x5 matrix indicates the size of the blob
    #               which is [num channel length height width].
    # blob        : a 5-D tensor size num x channel x length x height x width
    #               containing the blob data.
    # read_status : a scalar value = 1 if sucessfully read, 0 otherwise.


    # precision is set to 'single', used by C3D

    # open file and read size and data buffer
    # [s, c] = fread(f, [1 5], 'int32');
    read_status = 1
    blob = collections.namedtuple('Blob', ['size', 'data'])

    f = open(filename, 'rb')
    s = array.array("i")  # int32
    s.fromfile(f, 5)

    if len(s) == 5:
        m = s[0] * s[1] * s[2] * s[3] * s[4]

        # [data, c] = fread(f, [1 m], precision)
        data_aux = array.array("f")
        data_aux.fromfile(f, m)
        data = np.array(data_aux.tolist())

        if len(data) != m:
            read_status = 0;

    else:
        read_status = 0;

    # If failed to read, set empty output and return
    if not read_status:
        s = []
        blob_data = []
        b = blob(s, blob_data)
        return s, b, read_status

    # reshape the data buffer to blob
    # note that MATLAB use column order, while C3D uses row-order
    # blob = zeros(s(1), s(2), s(3), s(4), s(5), Float);
    blob_data = np.zeros((s[0], s[1], s[2], s[3], s[4]), np.float32)
    off = 0
    image_size = s[3] * s[4]
    for n in range(0, s[0]):
        for c in range(0, s[1]):
            for l in range(0, s[2]):
                # print n, c, l, off, off+image_size
                tmp = data[np.array(range(off, off + image_size))];
                blob_data[n][c][l][:][:] = tmp.reshape(s[3], -1);
                off = off + image_size;
    b = blob(s, blob_data)
    f.close()
    return s, b, read_status


def plot_gt_vs_ans():
    global test_videos
    ab_score = {}
    import pickle
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    with open(videos_pkl, 'rb') as f:
        test_videos = pickle.load(f)
    for vid in test_videos:
        with np.load("/home/lnn/data/UCF_Crimes/flow2100/%s_flow.npz" % vid, 'r') as f:
            # with np.load("/home/lnn/data/UCF_Crimes/rgb4500/%s_rgb.npz" % vid, 'r') as f:
            boundary = f["scores"].mean(axis=1)
            boundary = [softmax(b) for b in boundary]
            abnormality = np.zeros(len(boundary))
            for i in range(len(boundary)):
                abnormality[i] = boundary[i][1]

            # abnormality = cv2.blur(abnormality, (1, 7)).flatten()

            ab_score[vid] = abnormality

        '''
        import h5py
        mat_path = "/home/zjx/workspace/abnormality-detection-Lu/data/testing_result/res_err_per_frame_regionalRes_%s.mat" % vid
        with h5py.File(mat_path, 'r') as f:
            abnormality = f["res_err"][:].flatten()
        '''
        pos_thr = 0.98
        neg_thr = 0.03
        min_cnt = 4
        max_cnt = 120
        pos = list(np.where(abnormality > pos_thr)[0])
        neg = list(np.where(abnormality < neg_thr)[0])
        if len(pos) < min_cnt:
            pos = np.argsort(1 - abnormality)[0: min_cnt].tolist()
        elif len(pos) > max_cnt:
            pos = np.argsort(1 - abnormality)[0: max_cnt].tolist()
        if len(neg) < min_cnt:
            neg = np.argsort(abnormality)[0: min_cnt].tolist()
        elif len(neg) > max_cnt:
            neg = np.argsort(abnormality)[0: max_cnt].tolist()

        mask = list(set(pos + neg))

        continue
        gt = test_videos[vid]
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        x1 = range(len(gt))
        ax1.set_ylim([0, 1.1])
        ax1.set_xlim([0, len(x1)])
        ax1.plot(x1, gt, label="Ground Truth", color='g')

        ax2 = fig.add_subplot(212)  # ax1.twinx()
        x2 = range(len(abnormality))
        ax2.set_ylim([0, 1.1])  # 0.05 + 0.21])
        ax2.set_xlim([0, len(x2)])
        ax2.plot(x2, abnormality, label="C3D Prediction", color='r')
        ax2.scatter(mask, abnormality[mask])
        # ax2.set_ylabel("Abnormality Score")

        plt.title(vid)
        plt.legend(loc="best")
        plt.show()
    from utils import evaluate_result
    evaluate_result(ab_score)


from sklearn.metrics import roc_auc_score


def evaluate_best_hyper_param(input):
    (ab_score, pos_thr, neg_thr, min_cnt, max_cnt) = input
    gt_list = []
    ans_list = []
    for v in ab_score:
        abnormality = ab_score[v]
        gt = test_videos[v]
        pos = list(np.where(abnormality > pos_thr)[0])
        neg = list(np.where(abnormality < neg_thr)[0])
        if len(pos) < min_cnt:
            pos = np.argsort(1 - abnormality)[0: min_cnt].tolist()
        elif len(pos) > max_cnt:
            pos = np.argsort(1 - abnormality)[0: max_cnt].tolist()
        if len(neg) < min_cnt:
            neg = np.argsort(abnormality)[0: min_cnt].tolist()
        elif len(neg) > max_cnt:
            neg = np.argsort(abnormality)[0: max_cnt].tolist()
        mask = list(set(pos + neg))
        ratio = float(len(gt)) / len(abnormality)
        for m in mask:
            ans_list.append(abnormality[m])
            b = int(0.5 + ratio * m)
            e = int(0.5 + ratio * (m + 1))
            gt_list.append(1 if (sum(gt[b:e]) / float(e - b)) > 0.5 else 0)
    return pos_thr, neg_thr, min_cnt, max_cnt, roc_auc_score(gt_list, ans_list)


def find_best_param_for_topk():
    global test_videos
    ab_score = {}
    import pickle
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    with open(videos_pkl, 'rb') as f:
        test_videos = pickle.load(f)
    for vid in test_videos:
        with np.load("/home/lnn/data/UCF_Crimes/kinetics_rgb4600/%s_rgb.npz" % vid, 'r') as f:
            # with np.load("/home/lnn/data/UCF_Crimes/rgb4500/%s_rgb.npz" % vid, 'r') as f:
            tmp = f["scores"].mean(axis=1)
            abnormality = 1.0 / (1 + np.exp(-tmp)).flatten()
            ab_score[vid] = abnormality
    # print evaluate_best_hyper_param((ab_score,0.8,0.2,16,128))
    # pos_thr, neg_thr, min_cnt, max_cnt
    input_list = []
    for i in np.arange(0.7, 1, 0.01):
        for j in np.arange(0.0, 0.4, 0.01):
            for k in range(1, 16):
                for l in range(1, 32):
                    input_list.append((ab_score, i, j, 4 * k, 8 * l))
    print len(input_list)
    from multiprocessing import Pool
    p = Pool(16)
    ans = p.map(evaluate_best_hyper_param, input_list)
    max_ret = -1
    best_param = None
    for a in ans:
        pos_thr, neg_thr, min_cnt, max_cnt, ret = a
        if max_ret < ret:
            max_ret = ret
            best_param = pos_thr, neg_thr, min_cnt, max_cnt
    print max_ret, best_param


if __name__ == '__main__':
    find_best_param_for_topk()
    exit(0)

    feat, abnormality = [], []
    import pickle

    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    with open(videos_pkl, 'rb') as f:
        test_videos = pickle.load(f)
    for vid in test_videos:
        with np.load("/home/lnn/data/UCF_Crimes/rgb4500_feat/%s_rgb.npz" % vid, 'r') as f:
            tmp = f["scores"].mean(axis=1)
            tmp.resize((tmp.shape[0], tmp.shape[1]))
            feat.append(tmp)
        with np.load("/home/lnn/data/UCF_Crimes/test_pred_groundtruth/%s_rgb.npz" % vid, 'r') as f:
            tmp = f["scores"].mean(axis=1)[:, 1]
            abnormality.append((tmp, vid))

    feat = np.concatenate(feat)
    #abnormality = np.concatenate(abnormality)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=18, n_jobs=16)
    y_pred = kmeans.fit_predict(feat)

    vid2cluster_ab = dict()

    offset = 0
    for ab in abnormality:
        (it, vid) = ab
        vid2cluster_ab[vid] = (y_pred[offset: offset + len(it)], it)
        offset += len(it)
        continue
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(y_pred[offset: offset + len(it)])
        ax2.plot(it)
        plt.title(vid)
        plt.show()


    def get_idf():
        cluster2idf_normal = dict()
        normal_video_count = 0
        for vid in vid2cluster_ab:
            if "Normal" in vid:
                normal_video_count += 1
                (cluster, ab_score) = vid2cluster_ab[vid]
                for c in cluster:
                    if not cluster2idf_normal.has_key(c):
                        cluster2idf_normal[c] = 0
                    cluster2idf_normal[c] += 1
        for c in range(18):
            if not cluster2idf_normal.has_key(c):
                cluster2idf_normal[c] = 0
            cluster2idf_normal[c] = np.log(normal_video_count / float(1 + cluster2idf_normal[c]))

        return cluster2idf_normal

    vid2abscore = dict()
    cluster2idf_normal = get_idf()
    for vid in vid2cluster_ab:
        (cluster, ab_score) = vid2cluster_ab[vid]
        c, tf = np.unique(cluster, return_counts=True)
        cluster2tf = dict(zip(c, tf))
        for c in cluster2tf:
            cluster2tf[c] /= float(len(cluster))
        tf_idf = []
        for c in cluster:
            tf_idf.append(cluster2tf[c] * cluster2idf_normal[c])
        if "Normal" not in vid:
            thr_arr = np.sort(np.unique(tf_idf))
            if len(thr_arr) > 3:
                thr = thr_arr[-4]
            elif len(thr_arr) > 2:
                thr = thr_arr[-3]
            elif len(thr_arr) > 1:
                thr = thr_arr[-2]
            else:
                thr = thr_arr[-1]
            vid2abscore[vid] = np.where(tf_idf >= thr, 1, 0)
        continue
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        #print vid, tf_idf
        ax1.plot(tf_idf, color='r')
        ax2.plot(ab_score, color='g')
        plt.title(vid)
        plt.show()

    from utils import evaluate_result
    evaluate_result(vid2abscore)
