import numpy as np
from matplotlib import pyplot as plt
import struct

label_file = "TrainingSet/train-labels.idx1-ubyte"
image_file = "TrainingSet/train-images.idx3-ubyte"


class TrainingSet:
    examples = 0
    width = 0
    height = 0
    classes = 10

    def __init__(self):
        f = open(image_file, "rb")
        header = f.read(16)
        header = struct.unpack(">IIII", header)
        self.examples = header[1]
        self.width = header[2]
        self.height = header[3]
        self.pics = np.fromfile(f, dtype=np.ubyte, count=self.width * self.height * self.examples)
        f.close()
        self.pics = self.pics.reshape(self.examples, self.width * self.height)
        f = open(label_file, "rb")
        f.read(8)
        self.labels = np.fromfile(f, dtype=np.ubyte, count=self.examples)
        print "loaded " + str(self.examples)


# W[10*width*height] dot img[width*height] + bias[10]
ts = TrainingSet()
b = np.zeros(ts.classes)
w = np.random.rand(ts.width * ts.height * ts.classes).reshape(ts.classes, ts.width * ts.height) * 0.001


def regularization_l2(strength):
    return (w ** 2).dot(np.ones(ts.width * ts.height)) * strength


def score(x, w):  # x is the example
    return w.dot(x) + b


def loss_softmax(s, label):
    return -np.log10(s[label] / np.sum(np.exp(s)))


def L_i(s, i):  # i is the label
    return loss_softmax(s, i)


def num_gradient(step):
    gradient = w
    for i in range(ts.classes):
        for j in range(ts.width * ts.height):
            print i,j
            w_h = w
            w_h[i, j] += step
            gradient[i, j] = ((L(w_h) - L(w)) / step)[i]
        print gradient



def L(w):  # full loss
    loss = 0
    for i in range(ts.examples):
        s = score(ts.pics[i], w)
        li = L_i(s, ts.labels[i]) + regularization_l2(1)
        loss += li
    return loss / ts.examples


num_gradient(0.001)

plt.gray()
