import copy
import unittest

import numpy as np

EPSILON = 1e-100

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_der(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    return np.exp(x) / (np.sum(np.exp(x)))

def softmax_jacobian(x):
    S = softmax(x)
    return - np.dot(S, S.T) + np.diag(S[:,0])

def loss(y, t):
    return np.sum(-t * np.log(y))

def loss_der(y, t):
    der = - t / y
    return der


class MLP(object):

    def __init__(self, Ni, Nh, No):
        self.W = np.random.normal(0, np.sqrt(Ni), (Nh, Ni))
        self.b = np.zeros(Nh).reshape((Nh, 1))
        self.V = np.random.normal(0, np.sqrt(Nh), (No, Nh))
        self.c = np.zeros(No).reshape((No, 1))

    def fprop(self, X):
        self.X = X
        self.A1 = np.dot(self.W, self.X.T) + self.b
        self.H = sigmoid(self.A1)
        self.A2 = np.dot(self.V, self.H) + self.c
        self.Y = softmax(self.A2)
        return self.Y.T

    def bprop(self, Y, T):
        self.tgt = T
        self.dLdy = loss_der(Y, T)
        print 'LOSSDER: {}'.format(self.dLdy)
        self.dyda2 = softmax_jacobian(self.A2)
        self.dLda2 = np.dot(self.dyda2.T, self.dLdy)
        self.dLdc = self.dLda2
        self.dLdV = np.dot(self.dLda2, self.H.T)
        self.dLdh = np.dot(np.dot(self.V.T, self.dyda2), self.dLdy)
        self.dLda1 = self.dLdh * sigmoid_der(self.A1)
        self.dLdb = self.dLda1
        self.dLdW = np.dot(self.dLda1, self.X)


    def loss_from_y(self, y):
        return loss(y, self.tgt)

    def loss_from_a2(self, a2):
        return self.loss_from_y(softmax(a2))

    def loss_from_c(self, c):
        return self.loss_from_a2(np.dot(self.V, self.h) + c)

    def loss_from_V(self, V):
        return self.loss_from_a2(np.dot(V, self.h) + self.c)

    def loss_from_h(self, h):
        return self.loss_from_a2(np.dot(self.V, h) + self.c)

    def loss_from_a1(self, a1):
        return self.loss_from_h(sigmoid(a1))

    def loss_from_b(self, b):
        return self.loss_from_a1(np.dot(self.W, self.x.T) + b)

    def loss_from_W(self, W):
        return self.loss_from_a1(np.dot(W, self.x.T) + self.b)

    def loss_from_x(self, x):
        return self.loss_from_a1(np.dot(self.W, x) + self.b)

class TestPrimitives(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(1, sigmoid(float('inf')))
        self.assertEqual(0.5, sigmoid(0))
        self.assertEqual(0, sigmoid(-float('inf')))

    def test_sigmoid_der(self):
        x = 1.2
        der = sigmoid_der(x)

        approx = np.imag(sigmoid(complex(x, EPSILON)) / EPSILON)

        self.assertTrue(np.isclose(der, approx))

    def test_softmax(self):
        x = np.array([0.1, 1])
        s = softmax(x)
        self.assertEqual(x.shape, s.shape)
        self.assertEqual(np.argmax(s), 1)

    def test_softmax_jacobian(self):
        x = np.array([0.1, 1, 0.6, 0.5]).reshape((4,1))
        sd = softmax_jacobian(x)
        self.assertEqual(sd.shape, (x.shape[0], x.shape[0]))

        approx = np.zeros_like(sd)
        for i, j in np.ndindex(sd.shape):
            xe = np.array(x, copy=True).astype(np.complex128)
            xe[i] = complex(xe[i], EPSILON)
            approx[i,j] = np.imag(softmax(xe)[j] / EPSILON)

        self.assertTrue(np.allclose(sd, approx))

    def test_loss_der(self):
        y = 0.62
        t = 1
        der = loss_der(y, t)
        approx = np.imag(loss(complex(y, EPSILON), t) / EPSILON)
        self.assertTrue(np.isclose(approx, der))


class TestMLPGradients(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.mlp = MLP(10, 5, 2)
        self.x = [
            [0.1, 1.2, 0.3, 1.02, 0.5, 0.5, 0.1, 1.35, 10, -12],
        ]
        self.x = np.array(self.x).reshape((1,10))
        self.tgt = np.array([0, 1])
        self.mlp.fprop(self.x)
        self.mlp.bprop(self.mlp.y, self.tgt)

    def test_dLdy(self):

        dLdy = np.zeros_like(self.mlp.dLdy)
        for i in np.ndindex(*dLdy.shape):
            y = np.array(self.mlp.y, copy=True).astype(np.complex128)
            y[i] = complex(y[i], EPSILON)
            dLdy[i] = np.imag(self.mlp.loss_from_y(y) / EPSILON)

        self.assertTrue(np.allclose(dLdy, self.mlp.dLdy))

    def test_dLda2(self):

        dLda2 = np.zeros_like(self.mlp.dLda2)
        for idx in np.ndindex(*dLda2.shape):
            a2 = np.array(self.mlp.a2, copy=True).astype(np.complex128)
            a2[idx] = complex(a2[idx], EPSILON)
            dLda2[idx] = np.imag(self.mlp.loss_from_a2(a2) / EPSILON)
      
        self.assertTrue(np.allclose(dLda2, self.mlp.dLda2))


    def test_dLdc(self):

        dLdc = np.zeros(2)
        for i in np.ndindex(dLdc.shape):
            c = np.array(self.mlp.c, copy=True).reshape((2,)).astype(np.complex128)
            c[i] = complex(c[i], EPSILON)
            dLdc[i] =  np.imag(self.mlp.loss_from_c(c.reshape((2,1))) / EPSILON)

        self.assertTrue(np.allclose(dLdc, self.mlp.dLdc[:,0]))

    def test_dLdV(self):

        dLdV = np.zeros_like(self.mlp.dLdV)
        for i,j in np.ndindex(*dLdV.shape):
            V = np.array(self.mlp.V, copy=True).astype(np.complex128)
            V[i,j] = complex(V[i,j], EPSILON)
            dLdV[i,j] =  np.imag(self.mlp.loss_from_V(V) / EPSILON)

        self.assertTrue(np.allclose(dLdV, self.mlp.dLdV))

    def test_dLdh(self):

        dLdh = np.zeros_like(self.mlp.dLdh)
        for i in np.ndindex(*dLdh.shape):
            h = np.array(self.mlp.h, copy=True).astype(np.complex128)
            h[i] = complex(h[i], EPSILON)
            dLdh[i] =  np.imag(self.mlp.loss_from_h(h) / EPSILON)

        self.assertTrue(np.allclose(dLdh, self.mlp.dLdh))

    def test_dLda1(self):

        dLda1 = np.zeros_like(self.mlp.dLda1)
        for idx in np.ndindex(*dLda1.shape):
            a1 = np.array(self.mlp.a1, copy=True).astype(np.complex128)
            a1[idx] = complex(a1[idx], EPSILON)
            dLda1[idx] = np.imag(self.mlp.loss_from_a1(a1) / EPSILON)
      
        self.assertTrue(np.allclose(dLda1, self.mlp.dLda1))

    def test_dLdb(self):

        dLdb = np.zeros(5)
        for i in np.ndindex(dLdb.shape):
            b = np.array(self.mlp.b, copy=True).reshape((5,)).astype(np.complex128)
            b[i] = complex(b[i], EPSILON)
            dLdb[i] =  np.imag(self.mlp.loss_from_b(b.reshape((5,1))) / EPSILON)

        self.assertTrue(np.allclose(dLdb, self.mlp.dLdb[:,0]))

    def test_dLdW(self):

        dLdW = np.zeros_like(self.mlp.dLdW)
        for i,j in np.ndindex(*dLdW.shape):
            W = np.array(self.mlp.W, copy=True).astype(np.complex128)
            W[i,j] = complex(W[i,j], EPSILON)
            dLdW[i,j] =  np.imag(self.mlp.loss_from_W(W) / EPSILON)

        self.assertTrue(np.allclose(dLdW, self.mlp.dLdW))

        
if __name__ == '__main__':
    unittest.main()
        
