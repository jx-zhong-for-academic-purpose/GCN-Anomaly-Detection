#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np


class Mat2info(object):
    def __init__(self, mat_file="/home/zjx/workspace/cvpr2018/matlab/groundtruth/validation_set.mat"):
        self.__video2fps = dict()
        self.__video2classes = dict()
        self.__need_temp_det = dict()
        self.__video_name_list = list()
        self.__det_classes_list = list()

        from scipy.io import loadmat
        data = loadmat(mat_file)
        if data.has_key("validation_videos"):
            data = data["validation_videos"]
        elif data.has_key("test_videos"):
            data = data["test_videos"]
        else:
            raise "Not test_videos or validation_videos!"

        assert len(data["video_name"][0]) == len(data["frame_rate_FPS"][0])
        video_name = data["video_name"].tolist()
        frame_rate_FPS = data["frame_rate_FPS"].tolist()
        for i in xrange(len(data["video_name"][0])):
            self.__video2fps[str(video_name[0][i][0])] = frame_rate_FPS[0][i][0][0]

        assert len(data["video_name"][0]) == len(data["primary_action_index"][0])
        assert len(data["video_name"][0]) == len(data["secondary_actions_indices"][0])
        video_name = data["video_name"].tolist()
        primary_action_idx = data["primary_action_index"].tolist()
        secondary_action_idx = data["secondary_actions_indices"].tolist()

        for i in xrange(len(data["video_name"][0])):
            action_idx = [primary_action_idx[0][i][0][0]]
            action_idx.extend([x[0] for x in secondary_action_idx[0][i]])
            self.__video2classes[str(video_name[0][i][0])] = action_idx

        det = np.loadtxt("/home/zjx/workspace/cvpr2018/matlab/groundtruth/detclasslist.txt", dtype=str)
        det_set = set()
        for it in det:
            det_set.add(int(it[0]))
        self.__det_classes_list = list(det_set)
        for i in xrange(len(data["video_name"][0])):
            cls = self.__video2classes[str(video_name[0][i][0])]
            self.__video_name_list.append(str(video_name[0][i][0]))
            self.__need_temp_det[str(video_name[0][i][0])] = False
            for it in cls:
                if int(it) in det_set:
                    self.__need_temp_det[str(video_name[0][i][0])] = True

    def need_temp_det(self, videoname):
        return self.__need_temp_det[videoname]

    def get_fps_from_video(self, videoname):
        return self.__video2fps[videoname]

    def get_matlab_classes_from_video(self, videoname):
        return self.__video2classes[videoname]

    def get_video_list(self):
        return self.__video_name_list

    def get_det_classes_list(self):
        return self.__det_classes_list



if __name__ == "__main__":
    print Mat2info().get_fps_from_video("video_validation_0000304.mpeg")
    print Mat2info("../matlab/test_set_final.mat").get_fps_from_video("video_test_0000304")
