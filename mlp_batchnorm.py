"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

>>> activation = Identity()
>>> activation(3)
3
>>> activation.forward(3)
3
"""

import numpy as np
import os



class Activation(object):
    """ Interface for activation functions (non-linearities).

        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ Identity function (already implemented).
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Implement the sigmoid non-linearity """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1.0 / (1.0 + np.exp(np.negative(x)))
        return self.state

    def derivative(self):
        return self.state * (1.0 - self.state)


class Tanh(Activation):
    """ Implement the tanh non-linearity """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(np.negative(x))) / (np.exp(x) + np.exp(np.negative(x)))
        return self.state

    def derivative(self):
        return 1.0 - np.square(self.state)


class ReLU(Activation):
    """ Implement the ReLU non-linearity """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(x, 0)
        return self.state

    def derivative(self):
        return 1.0 * (self.state > 0)


# CRITERION


class Criterion(object):
    """ Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y

        x = x - np.amax(x, axis=1, keepdims=True)
        x = np.exp(x)
        self.sm = x / np.sum(x, axis=1, keepdims=True)
        self.loss = np.sum(np.negative(np.log(self.sm) * y), axis=1)

        return self.loss

    def derivative(self):
        derivative = self.sm - self.labels
        return derivative


class BatchNorm(object):
    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))
        
        # for test
        self.fan_in = fan_in

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        #assert(x.shape[1] == self.fan_in)
        self.x = x
        if eval == False:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.running_mean = self.alpha * self.running_mean + (1.0 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1.0 - self.alpha) * self.var

            self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta
        else:
            self.out = self.gamma * x / np.sqrt(self.running_var + self.eps) + (self.beta
                       - self.gamma * self.running_mean / np.sqrt(self.running_var + self.eps))

        return self.out

    def backward(self, delta):
        #assert(delta.shape[1] == self.fan_in)
        m = delta.shape[0]

        dnorm = delta * self.gamma
        self.dbeta = np.sum(delta, axis=0)
        self.dgamma = np.sum(delta * self.norm, axis=0)

        dvar = np.sum(dnorm * (self.x - self.mean) * 
                      -0.5 * np.power(self.var + self.eps, -3/2), axis=0)
        dmean = (-np.sum(dnorm / np.sqrt(self.var + self.eps), axis=0) - 
                 2.0 * dvar * np.sum(self.x - self.mean, axis=0) / m)
        dx = dnorm / np.sqrt(self.var + self.eps) + 2.0 * dvar * (self.x - self.mean) / m + dmean / m
        #assert(dx.shape[0] == m)
        #assert(dx.shape[1] == self.fan_in)

        return dx


def random_normal_weight_init(d0, d1):
    return np.random.normal(0, 1, (d0, d1))

def zeros_bias_init(d):
    return np.zeros((d,))


class MLP(object):
    """ A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes
        # For each weight, shape is (input, output)
        self.W = []
        self.dW = []
        # For each bias, shape is (1, output)
        self.b = []
        self.db = []
        self.last_dW = []
        self.last_db = []

        for i in range(self.nlayers):
            if i == 0:
                d0 = input_size
            else:
                d0 = hiddens[i-1]

            if i == len(hiddens):
                d1 = output_size
            else:
                d1 = hiddens[i]

            self.W.append(weight_init_fn(d0,d1))
            self.dW.append(np.zeros((d0,d1)))
            self.last_dW.append(np.zeros((d0,d1)))
            self.b.append(bias_init_fn(d1))
            self.db.append(np.zeros((d1,)))
            self.last_db.append(np.zeros((d1,)))
            
        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.states = []

    def forward(self, x):
        self.states.append(x)
        for i in range(self.nlayers):
            x = np.matmul(x, self.W[i]) + self.b[i]

            if i < self.num_bn_layers:
                x = self.bn_layers[i].forward(x, self.train_mode == False)

            x = self.activations[i](x)
            self.states.append(x)

        return x

    def zero_grads(self):
        for i in range(self.nlayers):
            self.dW[i].fill(0.0)
            self.db[i].fill(0.0)

    def step(self):
        for i in range(self.nlayers):
            self.last_dW[i] = self.momentum * self.last_dW[i] + self.lr * self.dW[i]
            self.last_db[i] = self.momentum * self.last_db[i] + self.lr * self.db[i]
            self.W[i] = self.W[i] - self.last_dW[i]
            self.b[i] = self.b[i] - self.last_db[i]

            if i < self.num_bn_layers:
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta

        self.zero_grads()
        self.states.clear()

    def backward(self, labels):
        self.criterion(self.states[-1], labels)
        div_dy = self.criterion.derivative()
        batch_size = div_dy.shape[0]

        assert(len(self.activations) == len(self.dW))
        assert(len(self.db) == len(self.states) - 1)

        for layer_id in range(self.nlayers-1, -1, -1):
            div_dz = div_dy * self.activations[layer_id].derivative()

            if layer_id < self.num_bn_layers:
                div_dz = self.bn_layers[layer_id].backward(div_dz)

            self.dW[layer_id] += np.matmul(np.transpose(self.states[layer_id]), 
                                           div_dz) / batch_size
            self.db[layer_id] += np.mean(div_dz, axis=0)
            div_dy = np.matmul(div_dz, np.transpose(self.W[layer_id]))
            assert(div_dy.shape[0] == batch_size)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train_data, train_label = dset[0]
    val_data, val_label = dset[1]
    test_data, test_label = dset[2]

    training_losses = np.zeros((nepochs,))
    training_errors = np.zeros((nepochs,))
    validation_losses = np.zeros((nepochs,))
    validation_errors = np.zeros((nepochs,))
    confusion_matrix = np.zeros((10,10))

    num_train_samples = train_data.shape[0]
    num_val_samples = val_data.shape[0]
    num_test_samples = test_data.shape[0]

    train_idx = np.arange(num_train_samples)
    for i in np.arange(nepochs):
        print (i)
        #train_idx = idx.copy()
        np.random.shuffle(train_idx)
        for batch in np.arange(0, num_train_samples, batch_size):
            train_d = train_data[train_idx[batch:batch + batch_size]]
            train_l = train_label[train_idx[batch:batch + batch_size]]

            mlp.forward(train_d)
            mlp.backward(train_l)
            mlp.step()

            training_losses[i] += np.sum(mlp.criterion.loss)
            predicted_l = np.argmax(mlp.criterion.sm, axis=1)
            true_l = np.argmax(train_l, axis=1)
            training_errors[i] += np.sum(predicted_l != true_l)

        training_losses[i] = training_losses[i] / num_train_samples
        training_errors[i] = training_errors[i] / num_train_samples

        # run on val data
        mlp.eval()
        for batch in np.arange(0, num_val_samples, batch_size):
            val_d = val_data[batch:batch + batch_size]
            val_l = val_label[batch:batch + batch_size]

            val_y = mlp.forward(val_d)
            mlp.states.clear()
            validation_losses[i] += np.sum(mlp.criterion.forward(val_y, val_l))
            assert(mlp.criterion.loss.shape[0] == batch_size)

            predicted_l = np.argmax(mlp.criterion.sm, axis=1)
            true_l = np.argmax(val_l, axis=1)
            validation_errors[i] += np.sum(predicted_l != true_l)

        validation_losses[i] = validation_losses[i] / num_val_samples
        validation_errors[i] = validation_errors[i] / num_val_samples

        mlp.train()

    # run on test data
    mlp.eval()
    for batch in np.arange(0, num_test_samples, batch_size):
        test_d = test_data[batch:batch + batch_size]
        test_l = test_label[batch:batch + batch_size]

        test_y = mlp.forward(test_d)
        mlp.states.clear()

        mlp.criterion.forward(test_y, test_l)
        predicted_l = np.argmax(mlp.criterion.sm, axis=1)
        true_l = np.argmax(test_l, axis=1)
        for predicted, true in zip(predicted_l, true_l):
            confusion_matrix[predicted][true] += 1

    return training_losses, training_errors, validation_losses, validation_errors, confusion_matrix


