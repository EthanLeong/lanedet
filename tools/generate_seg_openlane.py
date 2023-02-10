import json
import numpy as np
import cv2
import os
import argparse

TRAIN_SET = ['training.txt']
VAL_SET = ['validation.txt']
TEST_SET = ['test.txt']

def gen_label_for_json(args, image_set):
    H, W = 1280, 1920
    SEG_WIDTH = 30
    save_dir = args.savedir
    max_lane = 14

    os.makedirs(os.path.join(args.root, args.savedir, "list"), exist_ok=True)
    list_f = open(os.path.join(args.root, args.savedir, "list", "{}_gt.txt".format(image_set)), "w")

    json_path = os.path.join(args.root, args.savedir, "{}.txt".format(image_set))
    with open(json_path) as f:
        for line in f:
            anno_path = os.path.join('lane3d_1000_v1.2', 'lane3d_1000', line.replace('.jpg', '.json'))
            with open(anno_path.replace('\n', ''), 'r') as anno_f:
                label = json.load(anno_f)
            # ---------- clean and sort lanes -------------
            lanes = []
            _lanes = []
            slope = [] # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
            if len(label['lane_lines']) < max_lane:
                leng = len(label['lane_lines'])
            else:
                leng = max_lane
            for i in range(leng):
                l = [(int(x), int(y)) for x, y in zip(label['lane_lines'][i]['uv'][0], label['lane_lines'][i]['uv'][1]) if x >= 0]
                if (len(l)>1):
                    _lanes.append(l)
                    slope.append(np.arctan2(l[-1][1] - l[0][1], l[0][0] - l[-1][0]) / np.pi * 180)
            # print('1', len(_lanes))
            # print('1', slope)
            _lanes = [_lanes[i] for i in np.argsort(slope)]
            slope = [slope[i] for i in np.argsort(slope)]
            # print('2', len(_lanes))
            # print('2', slope)
            idx = [None for i in range(max_lane)]
            for i in range(len(slope)):
                if slope[i] <= -90:
                    idx[6] = i
                    idx[5] = i-1 if i > 0 else None
                    idx[4] = i-2 if i > 1 else None
                    idx[3] = i-3 if i > 2 else None
                    idx[2] = i-4 if i > 3 else None
                    idx[1] = i-5 if i > 4 else None
                    idx[0] = i-6 if i > 5 else None
                else:
                    idx[7] = i
                    idx[8] = i+1 if i+1 < len(slope) else None
                    idx[9] = i+2 if i+2 < len(slope) else None
                    idx[10] = i+3 if i+3 < len(slope) else None
                    idx[11] = i+4 if i+4 < len(slope) else None
                    idx[12] = i+5 if i+5 < len(slope) else None
                    idx[13] = i+6 if i+6 < len(slope) else None
                    break
            # print('idx', idx)
            for i in range(max_lane):
                lanes.append([] if idx[i] is None else _lanes[idx[i]])
            # print('3', len(lanes))

            # ---------------------------------------------

            img_path = os.path.join('img', label['file_path'])
            # seg_img = cv2.imread(img_path)
            seg_img = np.zeros((H, W, 3))
            list_str = []  # str to be written to list.txt
            for i in range(len(lanes)):
                coords = lanes[i]
                if len(coords) < 4:
                    list_str.append('0')
                    continue
                for j in range(len(coords)-1):
                    cv2.line(seg_img, coords[j], coords[j+1], (i+1, i+1, i+1), SEG_WIDTH//2)
                list_str.append('1')

            seg_path = img_path.split("/")
            seg_path, img_name = os.path.join(args.root, args.savedir, seg_path[1], seg_path[2]), seg_path[3]
            os.makedirs(seg_path, exist_ok=True)
            seg_path = os.path.join(seg_path, img_name[:-3]+"png")
            # cv2.imwrite(seg_path, seg_img)

            seg_path = "/".join([args.savedir, *img_path.split("/")[1:3], img_name[:-3]+"png"])
            if seg_path[0] != '/':
                seg_path = '/' + seg_path
            if img_path[0] != '/':
                img_path = '/' + img_path
            list_str.insert(0, seg_path)
            list_str.insert(0, img_path)
            list_str = " ".join(list_str) + "\n"
            list_f.write(list_str)


def generate_json_file(save_dir, json_file, image_set):
    with open(os.path.join(save_dir, json_file), "w") as outfile:
        for json_name in (image_set):
            with open(os.path.join(args.root, "list_1000", json_name)) as infile:
                for line in infile:
                    outfile.write(line)

def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)
                               
    print("generating train_val set...")
    generate_json_file(save_dir, "training.txt", TRAIN_SET)
    gen_label_for_json(args, 'training')
    
    print("generating val set...")
    generate_json_file(save_dir, "validation.txt", VAL_SET)
    gen_label_for_json(args, 'validation')
                               
    # print("generating test set...")
    # generate_json_file(save_dir, "test.txt", TEST_SET)
    # gen_label_for_json(args, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the OpenLane dataset')
    parser.add_argument('--savedir', type=str, default='seg_label', help='The root of the OpenLane dataset')
    args = parser.parse_args()

    generate_label(args)
