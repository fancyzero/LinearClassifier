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


