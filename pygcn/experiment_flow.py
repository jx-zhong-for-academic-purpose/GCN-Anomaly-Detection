from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.nn import SmoothL1Loss
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from models import NoiseFilter
from dataset import UCFCrime, UCFCrimeSlow
from utils import soft_uniform_sampling, test_sampling, test_gcn_model, min_max_norm
from utils import build_test_graph
from utils import uniform_sampling

from torch import nn


class SigmoidMAELoss(nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = L1Loss()

    def forward(self, pred, target):
        return self.__l1_loss__(self.__sigmoid__(pred), target)


class SigmoidCrossEntropyLoss(nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp))


from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


class ClusteringLoss(nn.Module):
    def __init__(self, n_clusters, n_z, alpha=1.0):
        super(ClusteringLoss, self).__init__()
        self.alpha = alpha
        # cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.kl_divergence = nn.KLDivLoss()

    def forward(self, z):
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


def get_centroids_of_kmean(data, k):
    kmeans = KMeans(k, n_jobs=16)
    y_pred = kmeans.fit_predict(data)
    return y_pred


def train_gcn(param):
    torch.cuda.empty_cache()
    videos_pkl_train = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_train.pkl"
    videos_pkl_test = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    feature_folder = "/home/lnn/workspace/UCF_Crimes/kinetics_flow5000_feat/"
    prediction_folder = "/home/lnn/workspace/UCF_Crimes/kinetics_flow5000/"
    test_pred_gt_folder = "/home/lnn/data/UCF_Crimes/test_pred_groundtruth/"
    modality = "flow"
    gpu_id = 1
    iter_size = 32
    ucf_crime_train = UCFCrimeSlow(videos_pkl_train, prediction_folder, feature_folder, modality,
                               graph_generator=soft_uniform_sampling, graph_generator_param=param)
    train_loader = DataLoader(dataset=ucf_crime_train, batch_size=1, shuffle=True, num_workers=16)
    model = NoiseFilter(nfeat=1024, nclass=1)
    criterion_supervised = SigmoidCrossEntropyLoss()
    criterion_unsupervised = SigmoidMAELoss()
    if gpu_id != -1:
        model = model.cuda(gpu_id)
        criterion_supervised = criterion_supervised.cuda(gpu_id)
        criterion_unsupervised = criterion_unsupervised.cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    opt_scheduler = optim.lr_scheduler.StepLR(optimizer, 16, 0.1)
    iter_count = 0
    avg_loss_train = 0
    alpha = 0.5
    vid2mean_pred = {}
    #model.load_state_dict(torch.load("flow_9_0.6.pth"))
    for epoch in range(20):
        model.train()
        opt_scheduler.step()
        for step, data in enumerate(train_loader):
            (feat, adj, labeled_index_in_the_graph, labeled_index), pred, vid = data
            feat, adj, pred = Variable(feat), Variable(adj), Variable(pred)

            if not vid2mean_pred.has_key(vid[0]):
                vid2mean_pred[vid[0]] = pred.data.cpu().numpy().flatten().copy()
            mean_pred = Variable(torch.from_numpy(vid2mean_pred[vid[0]]), requires_grad=False)

            if gpu_id != -1:
                feat = feat.cuda(gpu_id)
                adj = adj.cuda(gpu_id)
                pred = pred.cuda(gpu_id)
                mean_pred = mean_pred.cuda(gpu_id)

            if iter_count % iter_size == 0:
                optimizer.zero_grad()

            output = model(feat, adj)
            labeled_index_in_the_graph = np.array(labeled_index_in_the_graph).flatten()
            labeled_index = np.array(labeled_index).flatten()
            sample_index = get_sample_index(labeled_index, pred)

            if "Normal" in vid[0]:
                loss_train = criterion_supervised(output.view(1, -1),
                                                  pred.view(1, -1)[:, range(output.shape[1])])
            else:
                '''
                loss_train = criterion_supervised(output.view(1, -1)[:, labeled_index_in_the_graph],
                                                  pred.view(1, -1)[:, labeled_index])
                '''
                loss_train = criterion_supervised(output.view(1, -1)[:, labeled_index_in_the_graph],
                                                  pred.view(1, -1)[:, labeled_index])+ \
                             criterion_unsupervised(output.view(1, -1),
                                                    mean_pred.view(1, -1)[:, sample_index])


            avg_loss_train += loss_train
            iter_count += 1
            loss_train.backward()

            #torch.nn.utils.clip_grad_norm(model.parameters(), 40)

            mean_pred_current = mean_pred.data.cpu().numpy().copy().flatten()
            mean_pred_current[sample_index] = sigmoid(output).data.cpu().numpy().copy().flatten()
            vid2mean_pred[vid[0]] = alpha * vid2mean_pred[vid[0]] + (1 - alpha) * mean_pred_current

            if (iter_count + 1) % iter_size == 0:
                print("Train loss: %.4f" % (avg_loss_train / iter_size))
                avg_loss_train = 0
                optimizer.step()

        torch.save(model.state_dict(), "flow_%d.pth" % epoch)
        # iter_count += 1610
        # model.load_state_dict(torch.load("%d.pth" % iter_count))

        '''
        x = range(len(gt))
        plt.scatter(x, gt, color='g')
        plt.scatter(x, ans, color='r')
        plt.show()
        '''
        print("Epoch %d done !" % epoch)


def get_sample_index(labeled_index, pred):
    local_samples = 8
    sample_index = set()
    for i in labeled_index:
        b = i - local_samples / 2
        e = i + 1 + local_samples / 2
        for j in range(b, e):
            if 0 <= j < len(pred.data.cpu().numpy().flatten()):
                sample_index.add(j)
    sample_index = list(sample_index)
    sample_index.sort()
    return sample_index


if __name__ == '__main__':
    '''
    param_list = []
    for pos_threshold in np.arange(0.5, 1, 0.1):
        for neg_threshold in np.arange(0.05, 0.5, 0.1):
            for reserved_thr in np.arange(-0.01, 0.45, 0.1):
                interval = 4
                min_cnt = 2
                max_cnt = 64
                param = (interval, pos_threshold, neg_threshold, min_cnt, max_cnt, reserved_thr)
                param_list.append(param)
    print len(param_list)

    param_list = [(4,0.9,0.15,2,64,0.2)]
    best_result, best_param = -1, None
    import sys
    for p in param_list:
        result = train_gcn(p)
        print "Param: ", p
        sys.stdout.flush()
        if best_result < result:
            best_result = result
            best_param = p
    print best_result, best_param
    '''
    param = (4, 0.7, 0.1, 2, 1500, -0.1)
    train_gcn(param)
