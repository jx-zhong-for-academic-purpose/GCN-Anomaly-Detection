from torch.utils.data.dataloader import default_collate

import numpy as np
from random import choice
import os
import pickle
from math import exp


def compress_feature(input_feat, output_size, global_size):
    # sample vertex
    if input_feat.shape[0] > output_size:
        step = input_feat.shape[0] // global_size
        start_point = np.random.randint(step)
        sample_index = np.linspace(start_point, input_feat.shape[0], global_size + 1, endpoint=False,
                                   dtype=int).tolist()
        local_size = output_size - global_size
        local_center = choice(sample_index)
        for i in range(local_center - local_size // 2, local_center + local_size // 2 + 1):
            if i < 0 or i >= input_feat.shape[0] or i in sample_index:
                continue
            sample_index.append(i)
    else:
        sample_index = np.arange(input_feat.shape[0], dtype=int).tolist()
    output_dimension = len(sample_index)

    # establish the adjacent matrix A^tilde
    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            if i == j:
                adj[i][j] = 1.0
            else:
                adj[i][j] = 1.0 / abs(sample_index[i] - sample_index[j])
    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]

    return output_feat.astype(np.float32), adj_hat.astype(np.float32)


def graph_generator(raw_feat, output_size=32000, global_size=16000):  # raw_feat.shape: (l,4096)
    # L2-normalization
    feat = raw_feat / np.linalg.norm(raw_feat, ord=2, axis=-1).reshape(-1, 1)
    # Compress into 32 segments
    return compress_feature(feat, output_size, global_size)


from random import randrange


def test_sampling(input_feat, raw_pred, raw_uncertainty, param):
    (max_cnt) = param
    sample_index = []
    for i in range(len(input_feat)):
        sample_index.append(i)
    if len(sample_index) > max_cnt:
        cut_begin = randrange(len(sample_index) - max_cnt)
        assert (len(sample_index) >= cut_begin + max_cnt)
        sample_index = sample_index[cut_begin: cut_begin + max_cnt]
    sample_index.sort()
    output_dimension = len(sample_index)
    # establish the adjacent matrix A^tilde

    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            adj[i][j] = exp(-abs(sample_index[i] - sample_index[j]))
    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]
    return output_feat.astype(np.float32), adj_hat.astype(np.float32), np.array(sample_index)


def uniform_sampling(input_feat, raw_pred, raw_uncertainty, param):
    # interval=4, pos_threshold=0.8, neg_threshold=0.2,
    # min_cnt=2, max_cnt=64, reserved_thr=0.2):
    (interval, pos_threshold, neg_threshold, min_cnt, max_cnt, reserved_thr) = param
    local_samples = interval * 2
    labeled_index = list()
    for i in range(len(raw_pred)):
        if raw_pred[i] > pos_threshold or raw_pred[i] < neg_threshold:
            labeled_index.append(i)
        elif abs(0.5 - raw_pred[i]) > reserved_thr:
            b = i - interval / 2
            e = i + interval / 2 + 1
            flag = True
            for j in range(b, e):
                if not 0 <= j < len(raw_pred):
                    continue
                if abs(0.5 - raw_pred[j]) > abs(0.5 - raw_pred[i]):
                    flag = False
            if flag:
                labeled_index.append(i)
    if len(labeled_index) / 2.0 < min_cnt:
        pos = np.argsort(raw_pred)[0: min_cnt].tolist()
        neg = np.argsort(raw_pred)[-1: -min_cnt].tolist()
        labeled_index = list(set(pos + neg))
    elif len(labeled_index) / 2.0 > max_cnt:
        cut_begin = randrange(len(labeled_index) - 2 * max_cnt)
        assert (len(labeled_index) >= cut_begin + max_cnt * 2)
        labeled_index = labeled_index[cut_begin: cut_begin + max_cnt * 2]

    labeled_index.sort()
    sample_index = set()
    for i in labeled_index:
        b = i - local_samples / 2
        e = i + 1 + local_samples / 2
        for j in range(b, e):
            if 0 <= j < len(raw_pred):
                sample_index.add(j)
    sample_index = list(sample_index)
    sample_index.sort()
    labeled_index_in_the_graph = []
    for i in range(len(sample_index)):
        set_labeled_index = set(labeled_index)
        if sample_index[i] in set_labeled_index:
            labeled_index_in_the_graph.append(i)
    output_dimension = len(sample_index)
    # establish the adjacent matrix A^tilde

    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            adj[i][j] = exp(-abs(sample_index[i] - sample_index[j]))
            continue
            if i == j:
                adj[i][j] = 2.0 ** (-1.0)
            else:
                adj[i][j] = 2.0 ** (-abs(sample_index[i] - sample_index[j]))
    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]
    return output_feat.astype(np.float32), adj_hat.astype(np.float32), \
           np.array(labeled_index_in_the_graph), np.array(labeled_index)


def sample_top_k(input_feat, raw_pred, pos_threshold=0.8, neg_threshold=0.2,
                 min_cnt=64, max_cnt=120, local_samples=8):
    pos = list(np.where(raw_pred > pos_threshold))
    neg = list(np.where(raw_pred < neg_threshold))
    if len(pos) < min_cnt:
        pos = np.argsort(raw_pred)[0: min_cnt].tolist()
    elif len(pos) > max_cnt:
        pos = np.argsort(raw_pred)[-1: -max_cnt].tolist()
    if len(neg) < min_cnt:
        neg = np.argsort(raw_pred)[0: min_cnt].tolist()
    elif len(neg) > max_cnt:
        neg = np.argsort(raw_pred)[-1: -max_cnt].tolist()

    labeled_index = list(set(pos + neg))

    labeled_index.sort()
    sample_index = set()
    for i in labeled_index:
        b = i - local_samples / 2
        e = i + 1 + local_samples / 2
        for j in range(b, e):
            if 0 <= j < len(raw_pred):
                sample_index.add(j)
    sample_index = list(sample_index)
    sample_index.sort()
    labeled_index_in_the_graph = []
    for i in range(len(sample_index)):
        set_labeled_index = set(labeled_index)
        if sample_index[i] in set_labeled_index:
            labeled_index_in_the_graph.append(i)
    output_dimension = len(sample_index)
    # establish the adjacent matrix A^tilde

    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            if i == j:
                adj[i][j] = 2.0 ** (-1.0)
            else:
                adj[i][j] = 2.0 ** (-abs(sample_index[i] - sample_index[j]))
    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]
    return output_feat.astype(np.float32), adj_hat.astype(np.float32), labeled_index_in_the_graph, labeled_index


import random


def soft_uniform_sampling(input_feat, raw_pred, raw_uncertainty, param):
    # interval=4, pos_threshold=0.8, neg_threshold=0.2,
    # min_cnt=2, max_cnt=64, reserved_thr=0.2):

    (interval, pos_threshold, neg_threshold, min_cnt, max_cnt, reserved_thr) = param
    local_samples = interval * 2
    labeled_index = list()
    threshold = np.sort(raw_uncertainty)[int(0.3 * len(raw_uncertainty))]

    for i in range(len(raw_pred)):
        if raw_uncertainty[i] <= threshold:
            labeled_index.append(i)

    '''
    for i in range(len(raw_pred)):
        if raw_pred[i] > pos_threshold or raw_pred[i] < neg_threshold:
            labeled_index.append(i)
        elif abs(0.5 - raw_pred[i]) > reserved_thr:
            b = i - interval / 2
            e = i + interval / 2 + 1
            flag = True
            for j in range(b, e):
                if not 0 <= j < len(raw_pred):
                    continue
                if abs(0.5 - raw_pred[j]) > abs(0.5 - raw_pred[i]):
                    flag = False
            if flag:
                labeled_index.append(i)
    '''
    if len(labeled_index) / 2.0 < min_cnt:
        pos = np.argsort(raw_pred)[0: min_cnt].tolist()
        neg = np.argsort(raw_pred)[-1: -min_cnt].tolist()
        labeled_index = list(set(pos + neg))
    elif len(labeled_index) / 2.0 > max_cnt:
        cut_begin = randrange(len(labeled_index) - 2 * max_cnt)
        assert (len(labeled_index) >= cut_begin + max_cnt * 2)
        labeled_index = labeled_index[cut_begin: cut_begin + max_cnt * 2]

    labeled_index.sort()
    sample_index = set()
    for i in labeled_index:
        b = i - local_samples / 2
        e = i + 1 + local_samples / 2
        for j in range(b, e):
            if 0 <= j < len(raw_pred):
                sample_index.add(j)
    sample_index = list(sample_index)
    sample_index.sort()
    labeled_index_in_the_graph = []
    for i in range(len(sample_index)):
        set_labeled_index = set(labeled_index)
        if sample_index[i] in set_labeled_index:
            labeled_index_in_the_graph.append(i)
    output_dimension = len(sample_index)
    # establish the adjacent matrix A^tilde

    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            adj[i][j] = exp(-abs(sample_index[i] - sample_index[j]))
    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]
    return output_feat.astype(np.float32), adj_hat.astype(np.float32), \
           np.array(labeled_index_in_the_graph), np.array(labeled_index)


def build_test_graph(input_feat, raw_pred):
    return sample_top_k(input_feat, raw_pred, top_k=256, local_samples=8)


def collate_video(data_list):  # avoid tensor size mis-match problem
    length_list = [len(it[0][0]) for it in data_list]
    most_length = max(length_list, key=length_list.count)
    batch = [it for it in data_list if most_length == len(it[0][0])]
    return default_collate(batch)


def make_gt():
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    with open(videos_pkl, 'rb') as f:
        videos = pickle.load(f)
    src = "/home/lnn/workspace/UCF_Crimes/_iter_5000_c3d_ave_0/"
    dst = "/home/lnn/data/UCF_Crimes/test_pred_groundtruth/"
    for v in videos:
        src_file = os.path.join(src, v + "_c3d.npz")
        dst_file = os.path.join(dst, v + "_c3d.npz")
        f = np.load(src_file)
        begin_idx = f["begin_idx"].copy()
        scores = f["scores"].copy()
        gt = videos[v]
        ratio = float(len(gt)) / float(len(scores))
        for i in range(len(scores)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            ans = np.mean(gt[b: e])
            ab_score = 1 if ans >= 0.5 else 0
            scores[i, :, 0] = ab_score
        np.savez(dst_file, scores=scores, begin_idx=begin_idx)


from sklearn.metrics import roc_auc_score

def evaluate_result(vid2abnormality):
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    gt = []
    ans = []
    with open(videos_pkl, 'rb') as f:
        videos = pickle.load(f)
    for vid in videos:
        if not vid2abnormality.has_key(vid):
            print("The video %s is excluded on the result!" % vid)
            continue
        cur_ab = np.array(vid2abnormality[vid])
        cur_gt = np.array(videos[vid])
        ratio = float(len(cur_gt)) / float(len(cur_ab))
        cur_ans = np.zeros_like(cur_gt)
        for i in range(len(cur_ab)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            cur_ans[b: e] = cur_ab[i]
        gt.extend(cur_gt.tolist())
        ans.extend(cur_ans.tolist())
    ret = roc_auc_score(gt, ans)
    print("Test AUC@ROC: %.4f" % ret)
    return ret


def min_max_norm(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


from torch.autograd import Variable
from torch.nn.functional import sigmoid


def test_gcn_model(test_loader, model, gpu_id=0, min_max_norm_flag=False, omit_normal_flag=False):
    vid2ans = {}
    vid2oldans = {}
    for data in test_loader:
        (feat, adj, sample_index), old_pred, vid = data
        if omit_normal_flag and "Normal" in vid[0]:
            continue
        feat, adj, old_pred = Variable(feat), Variable(adj), Variable(old_pred)
        sample_index = sample_index.cpu().numpy().flatten()

        if gpu_id != -1:
            feat = feat.cuda(gpu_id)
            adj = adj.cuda(gpu_id)
            old_pred = old_pred.cuda(gpu_id)

        graph_pred = sigmoid(model(feat, adj)).data.cpu().numpy().flatten()
        new_pred = old_pred.data.cpu().numpy().flatten().copy()
        vid2oldans[vid[0]] = new_pred.copy()

        if "Normal" not in vid[0]:
            new_pred[sample_index] = graph_pred
        vid2ans[vid[0]] = min_max_norm(new_pred) if min_max_norm_flag else new_pred

    return vid2ans, vid2oldans


if __name__ == '__main__':
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    vid2ans = dict()
    vid2abnormality = dict()
    with open(videos_pkl, 'rb') as f:
        test_videos = pickle.load(f)
    normal_cnt = 0
    false_alarm_cnt = 0
    for vid in test_videos:
        with np.load("/home/lnn/data/UCF_Crimes/test_pred_groundtruth/%s_flow.npz" % vid, 'r') as f:
            gt = f["scores"].mean(axis=1)
            vid2abnormality[vid] = gt.flatten()
        with np.load("/home/lnn/workspace/UCF_Crimes/20181218/_iter_3400/%s_flow.npz" % vid, 'r') as f:
        #with np.load("/home/lnn/workspace/UCF_Crimes/c3d_features_2/_iter_750/%s_c3d.npz" % vid, 'r') as f:
        #with np.load("/home/lnn/workspace/UCF_Crimes/kinetics_2_rgb_high_conf/_iter_1000/%s_rgb.npz" % vid, 'r') as f:
            tmp = f["scores"].mean(axis=1)
            ans = 1.0 / (1 + np.exp(-tmp)).flatten()
            vid2ans[vid] = ans #min_max_norm(ans) if "Normal" not in vid else ans
        if "Normal" in vid:
            normal_cnt += len(ans)
            for a in ans:
                if a > 0.5:
                    false_alarm_cnt += 1
        continue

        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        x1 = range(len(gt))
        ax1.set_ylim((0, 1.1))
        # ax1.set_xlim([0, len(x1)])
        ax1.plot(gt, label="Ground Truth", color='g')
        # plt.legend(loc="best")1.

        ax2 = fig.add_subplot(212)  # ax1.twinx()
        x2 = range(len(ans))
        ax2.set_ylim((0, 1.1))  # 0.05 + 0.21])
        # ax2.set_xlim([0, len(x2)])
        ax2.plot(ans, label="Prediction", color='r')
        # ax2.set_ylabel("Abnormality Score")
        plt.title(vid)
        plt.legend(loc="best")
        #plt.show()
        plt.savefig("/dev/shm/flow_fig/%s_%d.png" % (vid, 7642))

    evaluate_result(vid2ans)
    print false_alarm_cnt * 100.0 / normal_cnt
