import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# 评价指标
class metric():
    def __init__(self, one_hot=True):
        """
        :param one_hot: 是否为one-hot类型
        """
        self.one_hot = one_hot
        print('evaluation metric contain iou,precision,recall,f1-score')

    def iou_cal(self, pred, mask, ignore_map=None):
        """
        计算交并比
        :param pred: 预测结果
        :param mask: 标签结果
        :param ignore_map: 忽略的部分
        """
        # 标签里包含的类别，包含的每种标签数字仅保留一个
        unique_labels = np.unique(mask)
        # 类别数量
        num_unique_labels = len(unique_labels)

        # 交集intersection 并集union
        I, U = np.zeros(num_unique_labels), np.zeros(num_unique_labels)
        # 权重
        alpha = np.zeros(num_unique_labels)
        # 计算像素点数量
        sum_all = 1
        for sum_shape in mask.shape:
            sum_all *= sum_shape
        # 遍历每类标签
        for index, val in enumerate(unique_labels):
            # 取出预测标签中当前标签val的区域
            pred_i = pred == val
            label_i = mask == val
            # 计算像素数量
            sum_i = np.sum(label_i)
            # 去除忽略部分
            if ignore_map is not None:
                mask_i = ignore_map == 255
                pred_i = np.logical_and(pred_i, mask_i)
                label_i = np.logical_and(label_i, mask_i)
            # 权重为当前类别占的比例
            alpha[index] = float(sum_i) / sum_all
            # 交集与并集
            I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
            U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
        self.alpha = alpha
        # 每类的交并比
        self.iou = I / U
        # 平均交并比
        self.miou = np.mean(self.iou)
        # 加权交并比
        self.wiou = np.sum(self.alpha * self.iou)

    def precision_recall_f1_cal(self, pred, mask):
        """
        计算准确率召回率及f1分数
        :param pred: 预测
        :param mask: 标签
        """
        pred = pred.reshape(-1)
        mask = mask.reshape(-1)
        self.precision, self.recall, self.f1score, self.support = precision_recall_fscore_support(mask, pred, average='macro')

    def __call__(self, pred, mask, ignore_map=None, need_iou=True, need_prf1=False):
        # 计算二类别的iou等指标
        # 将预测与标签转0-1表示
        if self.one_hot:
            pred = np.argmax(pred, axis=1)
            mask = np.argmax(mask, axis=1)
        else:
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
        if need_iou:
            self.iou_cal(pred, mask, ignore_map)
        if need_prf1:
            self.precision_recall_f1_cal(pred, mask)
