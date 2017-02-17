import cPickle
import copy
import gzip
import os
import unittest
import urllib

import numpy as np

EPSILON = 1e-100



def sigmoid(x):
    return np.exp(- np.logaddexp(0, -x))

def sigmoid_der(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    D, N = x.shape
    exps = np.exp(x)
    norm = np.broadcast_to(np.sum(exps, axis=1), (N, D)).T
    return exps / norm

def softmax_jacobian(x):
    S = softmax(x)
    D, N = S.shape
    S = S.reshape((D, N, 1))
    ST = S.transpose((0, 2, 1))
    return - np.matmul(S, ST) + S * np.eye(N)

def loss(y, t):
    return np.sum(-t * np.log(y), axis=1)

def loss_der(y, t):
    der = - t / y
    return der

class TestPrimitives(unittest.TestCase):

    def setUp(self):
        np.random.seed(654654)

    def test_softmax(self):
        x = np.random.uniform(size=(10, 100))

        # apply softmax explicitely
        sexp = np.empty_like(x)
        for i in range(x.shape[0]):
            sexp[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))

        # compare with function
        sfun = softmax(x)

        self.assertTrue(np.allclose(sexp, sfun))

    def test_softmax_jacobian(self):

        x = np.random.uniform(size=(3, 4))
        analy = softmax_jacobian(x)
        
        # finite difference approx
        approx = np.empty_like(analy)
        xc = np.array(x, copy=True).astype(np.complex128)
        for i, j, k in np.ndindex(approx.shape):
            # approx_i,j,k = d_(y_i)_j / d_(y_i)_k
            xc[i,k] = complex(xc[i, k], EPSILON)
            approx[i, j, k] = np.imag(softmax(xc)[i,j] / EPSILON)
            xc[i,k] = np.real(xc[i,k])

        self.assertTrue(np.allclose(analy, approx))

        
class MLP(object):

    def __init__(self, Ni, Nh, No):
        self.Ni = Ni
        self.Nh = Nh
        self.No = No
        
        self.W = np.random.normal(0, np.sqrt(Ni), (Nh, Ni))
        self.b = np.zeros(Nh)
        self.V = np.random.normal(0, np.sqrt(Nh), (No, Nh))
        self.c = np.zeros(No)

    def fprop(self, X):
        self.ND = X.shape[0]
        self.X = X
        self.A1 = np.dot(self.X, self.W.T) + self.b
        self.H = sigmoid(self.A1)
        self.A2 = np.dot(self.H, self.V.T) + self.c
        self.Y = softmax(self.A2)
        return self.Y

    def bprop(self, Y, T):
        self.tgt = T
        self.dLdy = np.atleast_3d(loss_der(Y, T))
        self.dyda2 = softmax_jacobian(self.A2)
        self.dLda2 = np.matmul(self.dyda2, self.dLdy)
        self.dLdc = self.dLda2
        self.dLdV = np.matmul(self.dLda2, np.atleast_3d(self.H).transpose((0,2,1)))
        self.dLdh = np.matmul(self.V.T, self.dLda2)
        self.dLda1 = self.dLdh * np.atleast_3d(sigmoid_der(self.A1))
        self.dLdb = self.dLda1
        self.dLdW = np.matmul(self.dLda1, np.atleast_3d(self.X).transpose((0,2,1)))

    def train(self, X, Y, nbepochs, vsplit):

        tmax = int(X.shape[0] * (1 - vsplit))
        vX = X[tmax:]
        vY = Y[tmax:]

        last_vloss = float('inf')
        patience = 0

        for epoch in range(nbepochs):
            losses = []
            for i in range(0, tmax, 32):
                xbatch = X[i:i+32]
                ybatch = Y[i:i+32]

                y = self.fprop(xbatch)
                self.bprop(y, ybatch)

                losses.append(np.mean(loss(y, ybatch), axis=0))
    
                self.c -= 0.01 * np.mean(self.dLdc, axis=0).reshape(self.c.shape)
                self.V -= 0.01 * np.mean(self.dLdV, axis=0)
                self.b -= 0.01 * np.mean(self.dLdb, axis=0).reshape(self.b.shape)
                self.W -= 0.01 * np.mean(self.dLdW, axis=0)

            # epoch is done, compute validation loss
            vloss = np.mean(loss(self.fprop(vX), vY))
            print 'epoch {}: training loss={}, validation loss={}'.format(epoch, np.mean(losses), vloss)

            if vloss > last_vloss:
                print '** validation loss increased'
                if patience < 2:
                    patience += 1
                else:
                    print '** Early stopping!'
                    break
            last_vloss = vloss
            

    def loss_from_y(self, y):
        return loss(y, self.tgt)

    def loss_from_a2(self, a2):
        return self.loss_from_y(softmax(a2))

    def loss_from_c(self, c):
        return self.loss_from_a2(np.dot(self.H, self.V.T) + c)

    def loss_from_V(self, V):
        return self.loss_from_a2(np.dot(self.H, V.T) + self.c)

    def loss_from_h(self, H):
        return self.loss_from_a2(np.dot(H, self.V.T) + self.c)

    def loss_from_a1(self, a1):
        return self.loss_from_h(sigmoid(a1))

    def loss_from_b(self, b):
        return self.loss_from_a1(np.dot(self.X, self.W.T) + b)

    def loss_from_W(self, W):
        return self.loss_from_a1(np.dot(self.X, W.T) + self.b)

class TestMLPGradients(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)

        self.mlp = MLP(10, 5, 2)
        self.x = np.random.uniform(size=(10,10))
        self.tgt = np.random.randint(2, size=(10,2))
        self.tgt[:,1] = 1 - self.tgt[:,0]
        self.mlp.fprop(self.x)
        self.mlp.bprop(self.mlp.Y, self.tgt)

    def test_dLdc(self):

        dLdc = np.empty_like(self.mlp.dLdc)
    
        for i,j,k in np.ndindex(*dLdc.shape):
            c = np.array(self.mlp.c, copy=True).astype(np.complex128)
            c[j] = complex(c[j], EPSILON)
            dLdc[i,j,k] = np.imag(self.mlp.loss_from_c(c)[i] / EPSILON)
              
        self.assertTrue(np.allclose(dLdc, self.mlp.dLdc))

    def test_dLdV(self):

        dLdV = np.zeros_like(self.mlp.dLdV)
        
        for i,j,k in np.ndindex(*dLdV.shape):
            V = np.array(self.mlp.V, copy=True).astype(np.complex128)
            V[j,k] = complex(V[j, k], EPSILON)
            dLdV[i,j,k] =  np.imag(self.mlp.loss_from_V(V)[i] / EPSILON)

        self.assertTrue(np.allclose(dLdV, self.mlp.dLdV))
        
    def test_dLdb(self):

        dLdb = np.empty_like(self.mlp.dLdb)
    
        for i,j,k in np.ndindex(*dLdb.shape):
            b = np.array(self.mlp.b, copy=True).astype(np.complex128)
            b[j] = complex(b[j], EPSILON)
            dLdb[i,j,k] = np.imag(self.mlp.loss_from_b(b)[i] / EPSILON)
              
        self.assertTrue(np.allclose(dLdb, self.mlp.dLdb))

    def test_dLdW(self):

        dLdW = np.zeros_like(self.mlp.dLdW)
        
        for i,j,k in np.ndindex(*dLdW.shape):
            W = np.array(self.mlp.W, copy=True).astype(np.complex128)
            W[j,k] = complex(W[j, k], EPSILON)
            dLdW[i,j,k] =  np.imag(self.mlp.loss_from_W(W)[i] / EPSILON)

        self.assertTrue(np.allclose(dLdW, self.mlp.dLdW))


def one_hot(dset):
    rowsel = np.arange(dset.shape[0])
    encoded = np.zeros((dset.shape[0], 10))
    encoded[rowsel, dset] = 1

    return encoded
    

def load_mnist():
    if not os.path.exists('mnist.pkl.gz'):
        urllib.urlretrieve(
            url='http://deeplearning.net/data/mnist/mnist.pkl.gz',
            filename='mnist.pkl.gz'
        )

    with gzip.open('mnist.pkl.gz', 'rb') as dataf:
        train_set, valid_set, test_set = cPickle.load(dataf)

    train_set = train_set[0], one_hot(train_set[1])
    valid_set = valid_set[0], valid_set[1]
    test_set = test_set[0], test_set[1]

    return train_set, valid_set, test_set


def train():

    train_set, valid_set, test_set = load_mnist()
    mlp = MLP(train_set[0].shape[1], 100, 10)

    mean = np.mean(train_set[0], axis=0)
    std = np.std(train_set[0], axis=0)
    std[np.where(std == 0)] = 1
    
    mlp.train(
        X=(train_set[0] - mean) / std,
        Y=train_set[1],
        vsplit=0.1,
        nbepochs=1000
    )

    pred = mlp.fprop((valid_set[0] - mean) / std)
    good = np.count_nonzero(np.argmax(pred, axis=1) == valid_set[1])

    print '*** accuracy: {}'.format(float(good) / valid_set[1].shape[0])

     
if __name__ == '__main__':
    # unittest.main()
    train()
        
