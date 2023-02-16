import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
import lanedet.utils.openlane_metric as openlane_metric
import cv2
from tqdm import tqdm
import logging
import json

LIST_FILE = {
    'train': 'seg_label/list/training_gt.txt',
    'val': 'seg_label/list/validation_gt.txt',
    'test': 'seg_label/list/test_gt.txt',
} 

@DATASETS.register_module
class OpenLane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading OpenLane annotations...')
        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def load_annotation(self, line):
        max_lane = 14
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        # if len(line) > 2:
        #     exist_list = [int(l) for l in line[2:]]
        #     infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'json'  # remove sufix jpg and add lines.txt
        anno_path = anno_path.replace('img', 'lane3d_1000_v1.2/lane3d_1000')
        with open(anno_path, 'r') as anno_file:
            # data = [list(map(float, line.split())) for line in anno_file.readlines()]
            data = json.load(anno_file)
        leng = min(max_lane, len(data['lane_lines']))
        lanes = []
        for i in range(leng):
            l = [(int(x), int(y)) for x, y in zip(data['lane_lines'][i]['uv'][0], data['lane_lines'][i]['uv'][1]) if x >= 0]
            if (len(l)>1):
                lanes.append(l)
        # lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
        #          for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        # lanes = sorted(lanes, key=lambda x: x[0])  # sort by x
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.array(list(self.cfg.sample_y))[::-1] / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        result = openlane_metric.eval_predictions(output_basedir, self.data_root, self.list_path, official=True)
        self.logger.info(result)
        return result['F1']
