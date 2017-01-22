import numpy as np


class Classifier:
    def __init__(self, labels, examples, num_classes, w, b, regularization_strength=1):
        self.labels = labels
        self.examples = examples
        self.num_classes = num_classes
        self.current_w = w
        self.current_b = b
        self.regularization_strength = regularization_strength

    def regularization_l2(self, w, strength):
        return (w ** 2).dot(np.ones(w.shape[1])) * strength

    def score(self, w, b, x):
        return w.dot(x) + b

    def loss_softmax(self, s, label):
        return -np.log10(np.exp(s[label]) / np.sum(np.exp(s)))

    def loss_i(self, s, i):  # loss for score "s" against label "i"
        return self.loss_softmax(s, i)

    def update(self):
        pass

    def num_gradient(self, w, b, step):
        gradient = np.zeros(w.shape)
        for i in range(self.num_classes):
            for j in range(w.shape[1]):
                w_h = np.copy(w)
                w_h[i, j] += step
                loss_w_h = self.full_loss(w_h, b)
                loss = self.full_loss(w, b)
                loss_gradient = ((loss_w_h - loss) / step)
                gradient[i, j] = loss_gradient[i]
        return gradient

    def full_loss(self, w, b):
        loss = np.zeros(self.num_classes)
        for i in range(len(self.examples)):
            s = self.score(w, b, self.examples[i])
            li = self.loss_i(s, self.labels[i]) + self.regularization_l2(w, self.regularization_strength)
            loss += li
        return loss / len(self.examples)
