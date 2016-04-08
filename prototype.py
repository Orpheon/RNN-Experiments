from __future__ import division, print_function

import numpy as np
import math
import constants

class Neuron(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.memory = 0
        self.dW = [[0 for j in range(len(inputs)+1)] for i in range(4)]
        variance = math.sqrt(2.0/len(inputs))
        self.weights = [np.random.randn(len(inputs)+1)*variance for i in range(4)]

    def trainstep(self, X):
        # http://i.imgur.com/M4cePPv.png
        X += [1]

        neuron = [0, 0, 0, 0]
        neuron[0] = sum([X[i]*self.weights[0][i] for i in range(len(inputs)+1)])
        neuron[1] = sum([X[i]*self.weights[1][i] for i in range(len(inputs)+1)])
        neuron[2] = sum([X[i]*self.weights[2][i] for i in range(len(inputs)+1)])
        neuron[3] = sum([X[i]*self.weights[3][i] for i in range(len(inputs)+1)])

        for i in range(4):
            neuron[i] = max(neuron[i]*constants.LEAK_COEFFICIENT, neuron[i])

        M = neuron[2]*self.memory
        S = neuron[0]*neuron[1] + M
        T = neuron[3]*S

        # Derivative calc for backprop later
        dX = [0, 0, 0, 0]
        dX[0] = neuron[1]*neuron[3]
        dX[1] = neuron[0]*neuron[3]
        dX[2] = self.memory*neuron[3]
        dX[3] = S

        # Apply derivative of activation function
        for i in range(4):
            if dX[i] < 0:
                dX[i] *= -constants.LEAK_COEFFICIENT

        # Apply multiplication with inputs to get derivative w.r.t. weights
        for i in range(4):
            for j in range(len(inputs)+1):
                self.dW[i][j] = dX[i] * X[j]

        self.memory = S

        return T


    def backprop(self, dext):
        for i in range(4):
            for j in range(len(inputs)+1):
                self.dW[i][j] *= dext
