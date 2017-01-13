import numpy as np
from matplotlib import pyplot as plt
import struct

label_file = "TrainingSet/train-labels.idx1-ubyte"
image_file = "TrainingSet/train-images.idx3-ubyte"


class TrainingSet:
    count = 0
    width = 0
    height = 0
    pics = []

    def __init__(self):
        fimg = open(image_file, "rb")
        header = fimg.read(16)
        header = struct.unpack(">IIII", header)
        self.count = header[1]
        self.width = header[2]
        self.height = header[3]
        pics = np.fromfile(fimg, dtype=np.ubyte, count=self.width * self.height * self.count)
        fimg.close()
        self.pics = pics.reshape(self.count, self.width, self.height)

    def get_pic(self, index):
        return self.pics[index]


# W[10*width*height] dot img[width*height] + bias[10]
class Classifier:
    self.class_count = 10

    def __init__(self, training_set):
        self.W = np.random.rand(training_set.width * training_set.height * class_count)
        print self.W

    def score(self, img_idx):



x = TrainingSet()
plt.gray()
for i in range(x.count):
    pic = x.get_pic(i)
    pic = pic / 255.0
    print pic
    plt.imshow(pic)
    plt.show()
