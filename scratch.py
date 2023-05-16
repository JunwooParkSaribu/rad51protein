import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


def parser_tf(tfstring):
    val = tfstring.split('(')[1].split(',')[0].strip()
    return float(val)


if __name__ == '__main__':
    train_acc = []
    test_acc = []
    with open(f'./model/model1/log.txt') as f:
        lines = f.readlines()
        for line in lines[2:]:
            line = line.strip().split('\t')
            train_acc.append(parser_tf(line[2]))
            test_acc.append(parser_tf(line[3]))
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    plt.figure()
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['train_acc', 'validation_acc'])
    plt.show()
