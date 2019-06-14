# !/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
import cv2
import numpy as np
from multiprocessing import Pool, current_process

gpu_list = [0, 1, 2, 3]
worker_cnt = 56

score_name = "fc-action-ucf_crimes"
flow_x_prefix = "flow_x_"
flow_y_prefix = "flow_y_"
rgb_prefix = "img_"
video_folder = "../ucf_crimes_rgb/"
modality = "rgb"
deploy_file = "./ucf_crimes/bn_inception_rgb_deploy.prototxt"

caffe_path = "../caffe_tsn"
sys.path.append(os.path.join(caffe_path, "python"))
from pyActionRecog.action_caffe import CaffeNet
step = 5
dense_sample = True
output_folder = "../rgb_features"
caffemodel = "./models/_iter_400.caffemodel"


def build_net():
    global net
    gpu_id = gpu_list[current_process()._identity[0] % len(gpu_list)]
    net = CaffeNet(deploy_file, caffemodel, gpu_id)


def eval_video(video_frame_path):
    global net
    vid = os.path.basename(video_frame_path)
    print("video {} doing".format(vid))
    all_files = os.listdir(video_frame_path)
    frame_cnt = len(all_files) // 3
    if modality == "rgb":
        stack_depth = 1
    elif modality == "flow":
        stack_depth = 5
        frame_cnt -= 1
    else:
        raise ValueError(modality)
    output_file = os.path.join(os.path.join(output_folder, os.path.basename(caffemodel).replace(".caffemodel","")), vid + "_" + modality + ".npz")
    if os.path.isfile(output_file):
        print("{} exists!".format(output_file))
        return
    frame_ticks = range(1, frame_cnt + 1, step)
    frame_scores = []
    for tick in frame_ticks:
        if modality == "rgb":
            if dense_sample:
                scores = []
                for i in range(0, step, stack_depth):
                    if i + tick > frame_cnt:
                        continue                    
                    name = "{}{:06d}.jpg".format(rgb_prefix, tick + i)
                    frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
                    scores.append(net.predict_single_frame([frame, ], score_name, frame_size=(340, 256)))
                scores = np.array(scores).mean(axis=0)
            else:
                name = "{}{:06d}.jpg".format(rgb_prefix, tick)
                frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
                scores = net.predict_single_frame([frame, ], score_name, frame_size=(340, 256))
        if modality == "flow":
            if dense_sample:
                scores = []
                for i in range(0, step, stack_depth):
                    frame_idx = [min(frame_cnt, tick + i + offset) for offset in xrange(stack_depth)]
                    flow_stack = []
                    for idx in frame_idx:
                        x_name = "{}{:06d}.jpg".format(flow_x_prefix, idx)
                        y_name = "{}{:06d}.jpg".format(flow_y_prefix, idx)
                        flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                        flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
                    scores.append(net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256)))
                scores = np.array(scores).mean(axis=0)
            else:
                frame_idx = [min(frame_cnt, tick + offset) for offset in xrange(stack_depth)]
                flow_stack = []
                for idx in frame_idx:
                    x_name = "{}{:06d}.jpg".format(flow_x_prefix, idx)
                    y_name = "{}{:06d}.jpg".format(flow_y_prefix, idx)
                    flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                    flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
                scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
        frame_scores.append(scores)
    np.savez(output_file, scores=frame_scores, begin_idx=frame_ticks)
    print("video {} done".format(vid))


if __name__ == '__main__':    
    video_name_list = os.listdir(video_folder)
    os.mkdir(os.path.join(output_folder, os.path.basename(caffemodel).replace(".caffemodel","")))    
    video_path_list = [os.path.join(video_folder, it) for it in video_name_list]
    pool = Pool(processes=worker_cnt, initializer=build_net)
    pool.map(eval_video, video_path_list)
