import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import classifier as cf
from matplotlib import cm

fig, ax = plt.subplots()
plt.axhline(0, color='red')
plt.axvline(0, color='red')

classes = [0, 1, 2]
lines = []
for i in classes:
    l, = plt.plot([], [], 'k-', color=cm.rainbow(i / 3.0))
    lines.append(l)

points = np.random.rand(20).reshape(10, 2) - 0.5
labels = np.random.randint(0, 3, 10)
colors = map(lambda l: cm.rainbow(l / 3.0), labels)

scat = plt.scatter(points[:, 0], points[:, 1], animated=True, color=colors)

examples = points
num_classes = len(classes)
example_size = len(examples[0])

w = np.random.rand(num_classes * example_size).reshape(num_classes, example_size)
b = np.zeros(num_classes)
cif = cf.Classifier(labels, examples, num_classes, w, b, 1)


def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    x = lines
    x.append(scat)
    return x


def update(frame):
    global w
    global b
    print "before"
    print w
    gradient = cif.num_gradient(w, b, 100)
    for wl, l, bl in zip(w, lines, b):
        slope = wl[1] / wl[0]
        l.set_data([1000, -1000], [slope * 1000 + bl, slope * -1000 + bl])
    x = lines
    x.append(scat)
    w += gradient * -0.001
    print "after"
    print w
    return x


ani = FuncAnimation(fig, update, frames=100,  init_func=init, blit=True)

plt.show()
