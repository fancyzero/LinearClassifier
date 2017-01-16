import numpy as np
from matplotlib import pyplot as plt
import struct

label_file = "TrainingSet/train-labels.idx1-ubyte"
image_file = "TrainingSet/train-images.idx3-ubyte"


class TrainingSet:
    count = 0
    width = 0
    height = 0
    class_count = 10

    def __init__(self):
        f = open(image_file, "rb")
        header = f.read(16)
        header = struct.unpack(">IIII", header)
        self.count = header[1]
        self.width = header[2]
        self.height = header[3]
        self.pics = np.fromfile(f, dtype=np.ubyte, count=self.width * self.height * self.count)
        f.close()
        self.pics = self.pics.reshape(self.count, self.width, self.height)
        f = open(label_file, "rb")
        f.read(8)
        self.labels = np.fromfile(f, dtype=np.ubyte, count=self.count)
        print "loaded " + str(self.count)

    def get_pic(self, index):
        return self.pics[index]

    def get_label(self, index):
        return self.labels[index]

xx = np.ndarray((60000,10,748))


exit()
# W[10*width*height] dot img[width*height] + bias[10]

ts = TrainingSet()
bias = np.zeros(ts.class_count)
w = np.random.rand(ts.width * ts.height * ts.class_count).reshape(ts.class_count, ts.width * ts.height)


def score(idx):
    return w.dot(ts.get_pic(idx).flatten()) + bias


def loss(scores, idx):
    margins = np.maximum(0,scores-scores[idx]+1)
    margins[idx] = 0
    return np.sum(margins)


print ts.get_label(0)
plt.gray()

s = score(0)
print s, loss(s,0)
