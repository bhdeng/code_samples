import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), 'HW3_Q2_Data')

def load_data(name):
    return np.genfromtxt(os.path.join(DATA_PATH, name))

X_train = load_data('Xtrain.txt')
y_train = load_data('ytrain.txt')
X_test = load_data('Xtest.txt')
y_test = load_data('ytest.txt')

def validation(X, y, w, b):
    val_acc = 0.0
    for Xs, ys in zip(X,y):
        if ys * (np.matmul(w, Xs) + b) > 0:
            val_acc += 1
    return val_acc / len(y)

def perceptron():
    len_feat = X_train.shape[1]
    w = np.zeros(len_feat)
    b = 0

    num_iter = 100
    train_accuracies = []
    test_accuracies = []
    w_list = []
    b_list = []
    print ('Start training')
    for e in range(num_iter):
        train_error = 0
        for X, y in zip(X_train, y_train):
            if y * (np.matmul(w, X) + b) <= 0:
                w = w + y * X
                b = b + y
                train_error += 1

        if train_error == 0:
            print ('Finish training')
            break

        w_list.append(w)
        b_list.append(b)

        train_acc = 1.0 * (len(y_train) - train_error) / len(y_train)
        train_accuracies.append(train_acc)

        # validation
        val_acc = validation(X_test, y_test, w, b)
        test_accuracies.append(val_acc)

        print ('iteration %d: train acc: %.4f; test acc: %.4f'
            % (e+1, train_acc, val_acc))

    return w_list, b_list, train_accuracies, test_accuracies

if __name__ == '__main__':
    w_list, b_list, train_acc, test_acc = perceptron()
    train_idx = np.argmax(train_acc)
    test_idx = np.argmax(test_acc)
    print (train_idx, test_idx)
    print (w_list[train_idx], b_list[train_idx])
    print (w_list[test_idx], b_list[test_idx])
    print (w_list[-1], b_list[-1])
    #num_iter = range(1,101)
    #plt.plot(num_iter, train_acc, label="Train Accuracy")
    #plt.plot(num_iter, test_acc, label="Test Accuracy")
    #plt.legend(loc='upper right')
    #plt.show()

