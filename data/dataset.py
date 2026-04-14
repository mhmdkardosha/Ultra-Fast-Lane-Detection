import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=50, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
    

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts, is_binary_mask = self._get_index(label)
        # get the coordinates of lanes at row anchors



        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)
            if is_binary_mask:
                # Binary masks (0/255) are converted to background/foreground labels.
                seg_label = (seg_label > 0).long()

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _is_binary_lane_mask(self, label_arr):
        uniq = np.unique(label_arr)
        return np.all(np.isin(uniq, [0, 255])) or np.all(np.isin(uniq, [0, 1]))

    def _extract_binary_lane_points(self, label_arr, sample_tmp):
        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        h = label_arr.shape[0]
        for i, r in enumerate(sample_tmp):
            y = int(round(r))
            y = min(max(y, 0), h - 1)
            row_pos = np.where(label_arr[y] > 0)[0]

            for lane_slot in range(self.num_lanes):
                all_idx[lane_slot, i, 0] = r
                all_idx[lane_slot, i, 1] = -1

            if len(row_pos) == 0:
                continue

            split_idx = np.where(np.diff(row_pos) > 1)[0] + 1
            segments = np.split(row_pos, split_idx)
            centers = []
            widths = []
            for seg in segments:
                if len(seg) == 0:
                    continue
                centers.append(float(np.mean(seg)))
                widths.append(len(seg))

            if len(centers) > self.num_lanes:
                keep_idx = np.argsort(widths)[-self.num_lanes:]
                keep_idx = sorted(keep_idx, key=lambda idx: centers[idx])
                centers = [centers[idx] for idx in keep_idx]
            else:
                centers = sorted(centers)

            for lane_slot, center in enumerate(centers[:self.num_lanes]):
                all_idx[lane_slot, i, 1] = center

        return all_idx

    def _get_index(self, label):
        w, h = label.size
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))
        else:
            sample_tmp = self.row_anchor

        label_arr = np.asarray(label)
        is_binary_mask = self._is_binary_lane_mask(label_arr)

        if is_binary_mask:
            all_idx = self._extract_binary_lane_points(label_arr, sample_tmp)
        else:
            all_idx = np.zeros((self.num_lanes,len(sample_tmp),2))
            for i,r in enumerate(sample_tmp):
                y = int(round(r))
                y = min(max(y, 0), label_arr.shape[0] - 1)
                label_r = label_arr[y]
                for lane_idx in range(1, self.num_lanes + 1):
                    pos = np.where(label_r == lane_idx)[0]
                    if len(pos) == 0:
                        all_idx[lane_idx - 1, i, 0] = r
                        all_idx[lane_idx - 1, i, 1] = -1
                        continue
                    pos = np.mean(pos)
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i,:,1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i,valid,:]
            # get all valid lane points
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp, is_binary_mask
