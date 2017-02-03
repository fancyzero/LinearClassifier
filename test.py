import numpy as np

# here is some test of back propagation

p1 = np.random.rand()
p2 = np.random.rand()
p3 = np.random.rand()

def softmax(s):
    return -np.log(np.exp(s[0])/np.sum(np.exp(s)))

delta = 0.00001
a = [p1,p2,p3]
ah = [p1+delta,p2,p3]
print (softmax(ah) - softmax(a)) / delta


x = p1
a = [p2,p3]
a = np.sum(np.exp(a))
x = np.exp(x)

print x * a/(a+x)**2 * - np.sum(a+x)/x  # back propagation of softmax



a = np.array([[1,2,0],[2,-4,0.5],[3,-1,-0.5]])
ah = np.array([[1+delta,2,0],[2,-4,0.5],[3,-1,-0.5]])
b = [0.5,0.4,1]

print (softmax(ah.dot(b)) - softmax(a.dot(b)))/delta
