import numpy as np


class Classifier:
    def __init__(self, labels, examples, num_classes, regularization_strength=0.1):
        self.labels = labels
        self.examples = examples
        self.num_classes = num_classes
        self.regularization_strength = regularization_strength

    def regularization_l2(self, w, strength):
        weights = w[:, :2]
        return np.sum(weights ** 2) * strength

    def score(self, w, x):
        return w.dot(x)

    def loss_softmax(self, s, label):
        return -np.log(np.exp(s[label]) / np.sum(np.exp(s)))

    def loss_i(self, s, i):  # loss for score "s" against label "i"
        return self.loss_softmax(s, i)

    def analistic_gradient(self, w):
        pass

    def numercical_gradient(self, w, h=0.00001):
        gradient = np.zeros(w.shape)
        loss = self.full_loss(w)
        it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            i = it.multi_index
            old_value = w[i]
            w[i] += h
            loss_w_h = self.full_loss(w)
            w[i] = old_value
            loss_gradient = ((loss_w_h - loss) / h)
            gradient[i] = loss_gradient + 0.5*self.regularization_strength*w[i]
            it.iternext()
        return gradient

    def full_loss(self, w):
        loss = 0
        for i in range(len(self.examples)):
            s = self.score(w, self.examples[i])
            li = self.loss_i(s, self.labels[i])
            loss += li
        #print "mean loss {1} \nreg loss {0}".format(self.regularization_l2(w, self.regularization_strength),loss / len(self.examples))
        return loss / len(self.examples)# + self.regularization_l2(w, self.regularization_strength)
