from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

import numpy as np
from sklearn.metrics import roc_auc_score

from utils import graph_generator
from utils import collate_video
from models import NoiseFilter
from dataset import UCFCrime
from dataset_test import UCFCrimeTest

gpu_id = 1
iter_size = 16
frames_per_feat = 1
if __name__ == '__main__':
    feature_path = "/home/zjx/data/UCF_Crimes/C3D_features/c3d_fc6_features.hdf5"
    ucf_crime = UCFCrime(feature_path, graph_generator)
    ucf_crime_test = UCFCrimeTest(feature_path, graph_generator)
    train_loader = DataLoader(dataset=ucf_crime, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_video)
    test_loader = DataLoader(dataset=ucf_crime_test, num_workers=4, collate_fn=collate_video)

    model = NoiseFilter(nfeat=4096, nclass=2)
    criterion = CrossEntropyLoss()

    if gpu_id != -1:
        model = model.cuda(gpu_id)
        criterion = criterion.cuda(gpu_id)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    iter_count = 0
    avg_loss_train = 0

    for epoch in range(8):
        model.train()
        for step, data in enumerate(train_loader):
            (feat, adj), is_normal, vid = data
            feat, adj, is_normal = Variable(feat), Variable(adj), Variable(is_normal)

            if gpu_id != -1:
                feat = feat.cuda(gpu_id)
                adj = adj.cuda(gpu_id)
                is_normal = is_normal.cuda(gpu_id)

            if iter_count % iter_size == 0:
                optimizer.zero_grad()

            output = model(feat, adj)
            loss_train = criterion(output, is_normal)
            avg_loss_train += loss_train
            iter_count += 1
            loss_train.backward()

            if (iter_count + 1) % iter_size == 0:
                print("Train loss: %.4f" % (avg_loss_train / iter_size))
                avg_loss_train = 0
                optimizer.step()

        model.eval()
        gt = []
        ans = []
        for test_video in test_loader:
            (feat, adj), is_normal, anomaly_score, vid = test_video
            if len(anomaly_score) > 32000:
                continue
            feat, adj, is_normal = Variable(feat), Variable(adj), Variable(is_normal)

            if gpu_id != -1:
                feat = feat.cuda(gpu_id)
                adj = adj.cuda(gpu_id)
                is_normal = is_normal.cuda(gpu_id)
            pred = softmax(model(feat, adj), dim=2)
            cur_gt = anomaly_score.numpy().flatten().tolist()
            ans_seg = pred.cpu().data.numpy().reshape(-1, 2)[:, 0].tolist()
            cur_answer = np.zeros(len(cur_gt)).tolist()
            for i in range(len(ans_seg)):
                for j in range(frames_per_feat):
                    cur_answer[i * frames_per_feat + j] = ans_seg[i]

            gt.extend(cur_gt)
            ans.extend(cur_answer)

        print("Test AUC@ROC: %.4f" % roc_auc_score(gt, ans))
        print("Epoch %d done !" % epoch)
