import os
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/whyguu/satellite_caffe_train_basic.log'

accuracy = []
loss = []
class_0_accuracy = []
class_1_accuracy = []
lr = []

with open(path, 'r') as log:
    for i in range(4):
        line = log.readline(1000)

    print(line)
    while 1:
        line = log.readline(1000)
        if not line:
            break

        print(line)
        try:
            idx = line.index('loss = ')
        except ValueError:
            log.readline()
            line = log.readline(1000)
        idx = line.index('loss = ')
        loss.append(float(line[idx+6:-1]))

        line = log.readline(1000)
        idx = line.index('accuracy = ')
        accuracy.append(float(line[idx+10:-1]))

        log.readline(1000)

        line = log.readline(1000)
        idx = line.index('per_class_accuracy = ')
        class_0_accuracy.append(float(line[idx+20:-1]))

        line = log.readline(1000)
        idx = line.index('per_class_accuracy = ')
        class_1_accuracy.append(float(line[idx+20:-1]))

        line = log.readline(1000)
        idx = line.index('lr = ')
        lr.append(line[idx+4:-1])

accuracy = np.array(accuracy)
loss = np.array(loss)
lr = np.array(lr)
class_0_accuracy = np.array(class_0_accuracy)
class_1_accuracy = np.array(class_1_accuracy)

print(accuracy.shape)

plt.figure(1)
plt.plot(loss)
plt.legend(['loss'])

plt.figure(3)
plt.plot(lr)
plt.legend(['lr'])

plt.figure(2)
plt.plot(class_0_accuracy)
plt.plot(class_1_accuracy)
plt.plot(accuracy)
plt.legend(['class_0', 'class_1', 'total'])

plt.show()


