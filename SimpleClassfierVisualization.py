import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import classifier as cf
from matplotlib import cm

fig, ax = plt.subplots()
plt.axhline(0, color='red')
plt.axvline(0, color='red')
ax.set_xlim(-1, 1)
ax.set_ylim(1, -1)
classes = [0, 1, 2]
lines = []
for i in classes:
    l, = plt.plot([], [], 'k-', color=cm.rainbow(i / 3.0))
    lines.append(l)

points = np.random.rand(18).reshape(9, 2) - 0.5
points[0] = [0.5, 0.4]
points[1] = [0.8, 0.3]
points[2] = [0.3, 0.8]
points[3] = [-0.4, 0.3]
points[4] = [-0.3, 0.7]
points[5] = [-0.7, 0.2]
points[6] = [0.7, -0.4]
points[7] = [0.5, -0.6]
points[8] = [-0.4, -0.5]

labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
colors = map(lambda l: cm.rainbow(l / 3.0), labels)

scat = plt.scatter(points[:, 0], points[:, 1], animated=False, color=colors)

examples = np.append(points, np.ones([len(points), 1]), 1)
num_classes = len(classes)
example_size = len(examples[0])

w = np.random.rand(num_classes * (example_size)).reshape(num_classes, example_size)
w = np.array([[1, 2, 0], [2, -4, 0.5], [3, -1, -.5]])

cif = cf.Classifier(labels, examples, num_classes, 0.1)


def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_aspect("equal")
    return lines + [scat, ]


def update(frame):
    global w
    step = 0.1
    gradient = cif.num_gradient(w)
    for wl, l in zip(w, lines):
        slope = wl[0] / wl[1]
        l.set_data([1, -1], [-slope * 1 - wl[2] / wl[1], -slope * -1 - wl[2] / wl[1]])
        w += gradient * -step
    return lines + [scat, ]

'''
h = 0.00001
w = np.array([[1, 2, 0], [2, -4, 0.5], [3, -1, -.5]])
a = cif.full_loss(w)
w = np.array([[1, 2, 0], [2 + h, -4, 0.5], [3, -1, -.5]])
b = cif.full_loss(w)
print "{0},{1}={2}".format(a, b, (b - a) / h)
'''
ani = FuncAnimation(fig, update, interval=1000, frames=None, init_func=init, blit=False)
update(None)
plt.show()
