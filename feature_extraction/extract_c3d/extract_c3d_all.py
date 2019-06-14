# !/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
import cv2
import numpy as np
from multiprocessing import Pool, current_process

gpu_list = [0, 1, 2, 3]
worker_cnt = 36

score_name = "fc-action-ucf_crimes"
rgb_prefix = "img_"
video_folder = "../ucf_crimes_rgb/"
modality = "c3d"
deploy_file = "./ucf_crimes/c3d_deploy.prototxt"

caffe_path = "../caffe_c3d/"
sys.path.append(os.path.join(caffe_path, "python"))
from pyActionRecog.action_caffe import CaffeNet

step = 16
dense_sample = True
output_folder = "../c3d_features/"
caffemodel = "./models/c3d_iter_1000.caffemodel"


def build_net():
    global net
    gpu_id = gpu_list[current_process()._identity[0] % len(gpu_list)]
    net = CaffeNet(deploy_file, caffemodel, gpu_id)


def eval_video(video_frame_path):
    global net
    vid = os.path.basename(video_frame_path)
    print("video {} doing".format(vid))
    all_files = os.listdir(video_frame_path)
    frame_cnt = len(all_files)
    if modality == "c3d":
        stack_depth = 16
    else:
        raise ValueError(modality)
    output_file = os.path.join(os.path.join(output_folder, os.path.basename(caffemodel).replace(".caffemodel","")), vid + "_c3d" + ".npz")
    if os.path.isfile(output_file):
        print("{} exists!".format(output_file))
        return
    frame_ticks = range(1, frame_cnt + 1, step)
    frame_scores = []
    for tick in frame_ticks:
        if modality == "c3d":
            if dense_sample:
                frames = []
                for i in range(0, step, stack_depth):
                    frame_idx = [min(frame_cnt, tick + i + offset) for offset in range(stack_depth)]
                    for idx in frame_idx:
                        name = "{}{:06d}.jpg".format(rgb_prefix, idx)
                        frames.append(cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR))
                scores = net.predict_single_c3d_rgb_stack(frames, score_name, frame_size=(171,128))
            else:
                print("Sparse sampling has yet to be done.")
        frame_scores.append(scores)
    np.savez(output_file, scores=frame_scores, begin_idx=frame_ticks)
    print("video {} done".format(vid))


if __name__ == '__main__':
    video_name_list = os.listdir(video_folder)
    video_path_list = [os.path.join(video_folder, it) for it in video_name_list]
    pool = Pool(processes=worker_cnt, initializer=build_net)
    pool.map(eval_video, video_path_list)

