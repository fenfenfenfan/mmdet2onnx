import matplotlib.pyplot as plt
import os


class plotter():
    """
    Chart tool
    :param num_class: The number of classes
    :param save_dir: The path to save the file
    """
    def __init__(self, num_class, save_dir):
        self.save_dir = save_dir
        self.num_class = num_class
        self.reset_cate()
        self.larger_than_half = 0
        self.less_than_half = 0

    def reset_cate(self):
        self.cate_num = [0 for _ in range(self.num_class+1)]

    def update_cate(self, labels:list):
        for i in labels:
            self.cate_num[i]+=1

    def draw_categorical_distribution(self, training=True):
        """
        绘制类的分布图
        :param training: 是否为训练集的类分布
        """
        plt.bar(range(self.num_class+1), self.cate_num)
        plt.ylabel("number")
        plt.xlabel("category") 

        plt.title("Categorical Distribution")
        if training:
            plt.savefig(os.path.join(self.save_dir, 'train_categorical_distribution.jpg'))
        else:
            plt.savefig(os.path.join(self.save_dir, 'val_categorical_distribution.jpg'))
        plt.clf()
        plt.cla()

    # 统计位于图像上半区的bbox的比例
    def update_bbox(self, h, ratio, boxes):
        for box in boxes:
            if box[3]<h*ratio:
                self.less_than_half += 1
            else:
                self.larger_than_half += 1

    def draw_bbox_lower_bound(self, training=True):
        """
        绘制bounding box的分布,用于比较位于图像上半部分与下半部分bounding box的数量
        """
        plt.bar(['higher than specified height','lower than specified height'], [self.less_than_half, self.larger_than_half])
        plt.ylabel("number")
        plt.xlabel("bbox distribution") 

        plt.title("bbox Distribution")
        if training:
            plt.savefig(os.path.join(self.save_dir, 'train_bbox_distribution.jpg'))
        else:
            plt.savefig(os.path.join(self.save_dir, 'val_bbox_distribution.jpg'))
        plt.clf()
        plt.cla()
