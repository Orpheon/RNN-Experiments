from __future__ import division, print_function

import numpy as np
import math
import constants

def dactivation(x):
    if x >= 0:
        return 1
    return -constants.LEAK_COEFFICIENT
batch_dactivation = np.vectorize(dactivation, otypes=[np.float])

# Numpy error settings to error when shit happens
#np.seterr(all='raise', under='warn')

class NeuronLayer(object):
    def __init__(self, n_inputs, layersize):
        # Initialize memory as zeros
        self.memory = np.zeros(layersize, dtype=np.float)

        # Initialize random weights so that neuron has output variance sqrt(2)
        variance = math.sqrt(2.0/n_inputs)
        self.weights = np.reshape(np.random.randn((n_inputs+1)*layersize*4)*variance, (layersize*4, n_inputs+1))

        self.n_inputs = n_inputs
        self.layersize = layersize

    def trainstep(self, X):
        # X is the output from the last layer, and as such is a vector of length self.n_inputs
        if np.isnan(X[0]):
            import sys
            sys.exit(1)
        # First add an extra 1 at the end to represent the bias
        X = np.append(X, 1)
        # Multiply it with the weights of this layer. The result is a vector of length 4*self.layersize
        wX = X.dot(self.weights.T)
        # Apply leaky ReLU on the first input, and clip all the others (they are supposed to be modulators)
        l = wX.size/4
        wX[:l] = np.maximum(constants.LEAK_COEFFICIENT*wX[:l], wX[:l])
        wX[l:] = np.maximum(0, np.minimum(1, wX[l:]))
        # http://i.imgur.com/M4cePPv.png
        # Combine the memory and the third input
        M = wX[2*l:3*l] * self.memory
        # Combine that with the two first inputs
        S = wX[:l]*wX[l:2*l] + M
        # Multiply the result with the last gate
        T = S * wX[3*l:]
        # We're done, but cache some of the between-results for later backprop calculation
        self.wX = wX
        self.S = S
        self.X = X
        self.oldmem = self.memory

        # Store memory for next time
        self.memory =  S.__copy__()

        return T

    def backprop(self, dext):
        # dext is a vector of size self.layersize
        # First tile it to allow for the 4 inputs of each neuron, so it becomes of length 4*self.layersize
        dext = np.tile(dext, 4)

        # Calculate the derivative of each neuron input with respect to the end
        dwX = np.zeros_like(self.wX)
        l = dwX.size/4
        dwX[:l] = self.wX[l:2*l]*self.wX[3*l:]
        dwX[1*l:2*l] = self.wX[:l]*self.wX[3*l:]
        dwX[2*l:3*l] = self.oldmem*self.wX[3*l:]
        if np.max(self.oldmem) > 1:
            print(np.max(self.oldmem))
        dwX[3*l:] = self.S
        dwX[:l] *= batch_dactivation(dwX[:l])

        # Multiply this (chain rule) with the output derivatives
        dwX *= dext

        # Calculate the derivative of the weights for ourselves and of the inputs for the next layer
        # (numpy doesn't allow transpose of 1d vectors, so have to do it with a reshape)
        dW = np.mat(dwX).T.dot(np.mat(self.X))
        dX = dwX.dot(self.weights)

        # Add on L2 regularization
        dW += self.weights*constants.REG_COEFFICIENT/len(self.weights)
        # Clip the gradient to prevent explosions
        # FIXME: Shouldn't be necessary
        #dW = np.maximum(-constants.GRADIENT_CLIP, np.minimum(constants.GRADIENT_CLIP, dW))
        # Update the weights
        self.weights -= dW * constants.LEARNING_RATE

        # Return the derivatives w.r.t. the input for the next layer
        # The last element of dX is the bias which after this point is irrelevant
        return dX[:-1]


class NeuralNetwork(object):
    def __init__(self):
        # Custom alphabet because we don't need the whole ascii dataset
        # First add all the lowercase letters, then add the common symbols, and then add a placeholder for a capitalization value
        #self.alphabet = [chr(x) for x in range(97, 123)] + [chr(x) for x in range(32, 65)] + [None]
        self.alphabet = ["a", "b", "c"]

        # len(alphabet) inputs and outputs (both are interpreted as a choice in an array), NETWORK_DEPTH-1 hidden layers (output layer counts as a layer)
        self.net = [NeuronLayer(len(self.alphabet), constants.LAYER_SIZE)]
        self.net += [NeuronLayer(constants.LAYER_SIZE, constants.LAYER_SIZE) for x in range(constants.NETWORK_DEPTH-1)]
        self.net += [NeuronLayer(constants.LAYER_SIZE, len(self.alphabet))]

    def trainstep(self, inputchar, trueresult):
        # inputchar is an ascii character
        if inputchar not in self.alphabet:
            # Wew
            print("ERROR: Character {0} not in alphabet! Skipping.".format(inputchar))
            return None
        if trueresult not in self.alphabet:
            print("ERROR: Character {0} not in alphabet! Skipping.".format(trueresult))
            return None

        # Construct our input vector
        index = self.alphabet.index(inputchar)
        tmpvec = np.zeros(len(self.alphabet))
        tmpvec[index] = 1.0

        # Feed that to the network
        for layer in self.net:
            tmpvec = layer.trainstep(tmpvec)

        # Grab our prediction
        prediction = self.alphabet[np.argmax(tmpvec)]

        # Since we're training, we want to know how accurate this is
        index = self.alphabet.index(trueresult)
        # Using SVM classifier
        # In other words, subtract (tmpvec[correct_answer] - 1), clip everything below 0 to 0, and then sum everything but the correct answer
        tmpvec += 1.0 - np.repeat(tmpvec[index], tmpvec.size)
        tmpvec[index] = 0

        # L2 reg
        regloss = 0
        for layer in self.net:
            regloss += np.sum(np.square(layer.weights)) / len(layer.weights)
        regloss /= len(self.net)

        vecloss = np.sum(np.maximum(0, tmpvec))
        loss = vecloss + constants.REG_COEFFICIENT*regloss
        # Normalize anything above 0 to 1
        tmpvec = np.maximum(0, batch_dactivation(tmpvec))
        tmpvec[index] = -vecloss

        # Backpropagate it through the network, which will update itself
        for layer in reversed(self.net):
            tmpvec = layer.backprop(tmpvec)

        return loss, vecloss, regloss, prediction


n = NeuralNetwork()

'''
text = "Locus iste a Deo factus est, inaestimabile sacramentum, irreprehensibilis est.".lower()
iteration = 0
while True:
    l = []
    for i, c in enumerate(text):
        if c != ".":
            l.append(n.trainstep(c, text[i]))
    print("\nIteration: {0}".format(iteration))
    print("Average loss: {0}".format(sum([x[0] for x in l])/len(l)))
    print("Average vloss: {0}".format(sum([x[1] for x in l])/len(l)))
    print("Average rloss: {0}".format(sum([x[2] for x in l])/len(l)))
    print([x[3] for x in l])
    iteration += 1'''

i = 0
while True:
    loss, vloss, rloss, pred = n.trainstep("a", "c")
    print("Iteration", i)
    print("Loss", loss)
    print("Vloss", vloss)
    print("Rloss", rloss)
    i += 1
