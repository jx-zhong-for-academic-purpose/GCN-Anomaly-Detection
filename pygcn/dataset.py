from torch.utils.data import Dataset

import pickle
import numpy as np
import os


class UCFCrime(Dataset):
    def __init__(self, videos_pkl, prediction_folder, feature_folder, modality,
                 normalized=True, graph_generator=None, graph_generator_param=None):
        self.__vid__ = []
        self.__feat__ = []
        self.__pred__ = []
        self.__graph_func__ = graph_generator
        self.__graph_param__ = graph_generator_param

        def softmax(raw_score):
            exp_s = np.exp(raw_score - raw_score.max(axis=-1)[..., None])
            sum_s = exp_s.sum(axis=-1)
            return exp_s / sum_s[..., None]

        with open(videos_pkl, 'rb') as f:
            videos = pickle.load(f)
        for v in videos:
            self.__vid__.append(v)
            feat_path = os.path.join(feature_folder, "%s_%s.npz" % (v, modality))
            pred_path = os.path.join(prediction_folder, "%s_%s.npz" % (v, modality))
            with np.load(feat_path, 'r') as f:
                tmp = f["scores"].mean(axis=1)
                tmp.resize((tmp.shape[0], tmp.shape[1]))
                self.__feat__.append(np.array(tmp))
            with np.load(pred_path, 'r') as f:
                tmp = f["scores"].mean(axis=1)
                if normalized:
                    self.__pred__.append(1.0 / (1 + np.exp(-tmp)).flatten())
                else:
                    self.__pred__.append(np.array(tmp))
        assert len(self.__vid__) == len(self.__feat__) == len(self.__pred__)

    def __getitem__(self, index):
        feat = self.__feat__[index]
        pred = self.__pred__[index]
        vid = self.__vid__[index]
        if self.__graph_func__:
            feat = self.__graph_func__(feat, pred, self.__graph_param__)
        return feat, pred, vid

    def __len__(self):
        return len(self.__vid__)

from random import randint
class UCFCrimeSlow(Dataset):
    def __init__(self, videos_pkl, prediction_folder, feature_folder, modality,
                 normalized=True, graph_generator=None, graph_generator_param=None, random_crop=True):
        self.__vid__ = []
        self.__feat__ = []
        self.__pred__ = []
        self.__graph_func__ = graph_generator
        self.__graph_param__ = graph_generator_param
        self.__normalize__ = normalized
        self.__prediction_folder__ = prediction_folder
        self.__feature_folder__ = feature_folder
        self.__modality__ = modality
        with open(videos_pkl, 'rb') as f:
            videos = pickle.load(f)
        for v in videos:
            self.__vid__.append(v)
        self.__random_crop__ = random_crop

    def __getitem__(self, index):
        vid = self.__vid__[index]
        feat_path = os.path.join(self.__feature_folder__, "%s_%s.npz" % (vid, self.__modality__))
        pred_path = os.path.join(self.__prediction_folder__, "%s_%s.npz" % (vid, self.__modality__))
        with np.load(feat_path, 'r') as f:
            if self.__random_crop__:
                tmp = f["scores"][:,randint(0, f["scores"].shape[1] - 1),:]
            else:
                tmp = f["scores"].mean(axis=1)
            feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1]))
        with np.load(pred_path, 'r') as f:
            tmp = f["scores"].mean(axis=1).flatten()
            uncertainty = f["scores"].var(axis=1).flatten()
            if self.__normalize__:
                pred = 1.0 / (1 + np.exp(-tmp))
            else:
                pred = np.array(tmp)
        if self.__graph_func__:
            feat = self.__graph_func__(feat, pred, uncertainty, self.__graph_param__)
        #print feat[0].shape, feat[1].shape, feat[2].shape, feat[3].shape, pred.shape
        return feat, pred, vid

    def __len__(self):
        return len(self.__vid__)



from utils import graph_generator
from torch.utils.data import DataLoader

if __name__ == '__main__':
    feature_folder = "../../../data/UCF_Crimes/flow2100_feat/"
    prediction_folder = "../../../data/UCF_Crimes/flow2100/"
    modality = "flow"
    ucf_crime_train = UCFCrime(prediction_folder, feature_folder, modality, graph_generator)
    train_loader = DataLoader(dataset=ucf_crime_train, batch_size=1, shuffle=True, num_workers=8)
