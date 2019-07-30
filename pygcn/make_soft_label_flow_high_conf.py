from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import sigmoid
import torch
import numpy as np
from models import NoiseFilter
from dataset import UCFCrime, UCFCrimeSlow
from pygcn.experiment_c3d import get_sample_index
from utils import uniform_sampling, test_sampling, test_gcn_model, soft_uniform_sampling, min_max_norm
from os import path
from matplotlib import pyplot as plt


videos_pkl_train = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_train.pkl"
feature_folder = "/home/lnn/workspace/UCF_Crimes/kinetics_flow5000_feat/"
prediction_folder = "/home/lnn/workspace/UCF_Crimes/kinetics_flow5000/"
gcn_model_path = "/home/lnn/workspace/pygcn/pygcn/flow_18_ablation_only_feat_adj1.pth"
modality = "flow"
gpu_id = 1
output_folder = "/home/lnn/workspace/pygcn/output_flow_high_conf_new1/"

if __name__ == '__main__':
    param = (4, 0.7, 0.1, 2, 1600, -0.1)
    ucf_crime_train = UCFCrimeSlow(videos_pkl_train, prediction_folder, feature_folder, modality,
                                   graph_generator=soft_uniform_sampling, graph_generator_param=param, random_crop=False)
    train_loader = DataLoader(dataset=ucf_crime_train, num_workers=16)
    model = NoiseFilter(nfeat=1024, nclass=1)

    if gpu_id != -1:
        model = model.cuda(gpu_id)

    model.load_state_dict(torch.load(gcn_model_path))
    model.eval()
    vid2ans = {}
    for step, data in enumerate(train_loader):
        (feat, adj, labeled_index_in_the_graph, labeled_index), pred, vid = data
        feat, adj, pred = Variable(feat), Variable(adj), Variable(pred)

        if gpu_id != -1:
            feat = feat.cuda(gpu_id)
            adj = adj.cuda(gpu_id)
            pred = pred.cuda(gpu_id)

        output = model(feat, adj).data.cpu().numpy().flatten()
        labeled_index_in_the_graph = np.array(labeled_index_in_the_graph).flatten()
        labeled_index = np.array(labeled_index).flatten()
        sample_index = get_sample_index(labeled_index, pred)
        new_pred = pred.data.cpu().numpy().flatten().copy()
        if "Normal" not in vid[0]:
            new_pred[sample_index] = output
        vid2ans[vid[0]] = min_max_norm(new_pred)

    for v in vid2ans:
        output = vid2ans[v]
        output_txt = path.join(output_folder, "%s.txt" % v)
        np.savetxt(output_txt, output, fmt='%.18f', delimiter='\n')
        print "Done: %s" % v