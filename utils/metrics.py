import numpy as np
import torch
import time,pdb

def converter(data):
    if isinstance(data,torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()
def fast_hist(label_pred, label_true,num_classes):
    # Ignore labels outside [0, num_classes), e.g. 255 as ignore label.
    valid = (label_true >= 0) & (label_true < num_classes) & (label_pred >= 0) & (label_pred < num_classes)
    label_true = label_true[valid].astype(int)
    label_pred = label_pred[valid].astype(int)
    hist = np.bincount(num_classes * label_true + label_pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

class Metric_mIoU():
    def __init__(self,class_num):
        self.class_num = class_num
        self.hist = np.zeros((self.class_num,self.class_num))
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)

        self.hist += fast_hist(predict,target,self.class_num)

    def reset(self):
        self.hist = np.zeros((self.class_num,self.class_num))

    def _iou_per_class(self):
        intersection = np.diag(self.hist)
        union = np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) - intersection
        iou = np.full(self.class_num, np.nan, dtype=np.float64)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        return iou

    def get_miou(self):
        iou = self._iou_per_class()
        valid_iou = iou[~np.isnan(iou)]
        if valid_iou.size == 0:
            return 0.0
        return float(np.mean(valid_iou))

    def get_lane_iou(self):
        iou = self._iou_per_class()[1:]
        valid_iou = iou[~np.isnan(iou)]
        if valid_iou.size == 0:
            return 0.0
        return float(np.mean(valid_iou))

    def get_acc(self):
        class_total = self.hist.sum(axis=1)
        acc = np.full(self.class_num, np.nan, dtype=np.float64)
        valid = class_total > 0
        acc[valid] = np.diag(self.hist)[valid] / class_total[valid]
        valid_acc = acc[~np.isnan(acc)]
        if valid_acc.size == 0:
            return 0.0
        return float(np.mean(valid_acc))
        
    def get(self):
        return self.get_miou()

class Metric_LaneIoU(Metric_mIoU):
    def get(self):
        return self.get_lane_iou()
class MultiLabelAcc():
    def __init__(self):
        self.cnt = 0
        self.correct = 0
    def reset(self):
        self.cnt = 0
        self.correct = 0
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)
        self.cnt += len(predict)
        self.correct += np.sum(predict==target)
    def get_acc(self):
        return self.correct * 1.0 / self.cnt
    def get(self):
        return self.get_acc()
class AccTopk():
    def __init__(self,background_classes,k):
        self.background_classes = background_classes
        self.k = k
        self.cnt = 0
        self.top5_correct = 0
    def reset(self):
        self.cnt = 0
        self.top5_correct = 0
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)
        self.cnt += len(predict)
        background_idx = (predict == self.background_classes) + (target == self.background_classes)
        self.top5_correct += np.sum(predict[background_idx] == target[background_idx])
        not_background_idx = np.logical_not(background_idx)
        self.top5_correct += np.sum(np.absolute(predict[not_background_idx]-target[not_background_idx])<self.k)
    def get(self):
        return self.top5_correct * 1.0 / self.cnt



def update_metrics(metric_dict, pair_data):
    for i in range(len(metric_dict['name'])):
        metric_op = metric_dict['op'][i]
        data_src = metric_dict['data_src'][i]
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])


def reset_metrics(metric_dict):
    for op in metric_dict['op']:
        op.reset()


if __name__ == '__main__':

    # p = np.random.randint(5, size=(800, 800))
    # t = np.zeros((800, 800))
    # me = Metric_mIoU(5)
    # me.update(p,p)
    # me.update(p,t)
    # me.update(p,p)
    # me.update(p,t)
    # print(me.get_miou())
    # print(me.get_acc())
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    b = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 0])
    me = AccTopk(0,5)
    me.update(b,a)
    print(me.get())