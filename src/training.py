import tensorflow as tf
import math


def no_scheduler(epoch, lr):
    return lr


def step_scheduler(epoch, lr):
    # 0.00005
    if epoch == 20:
        return lr / 2 # 0.00003
    elif epoch == 70:
        return lr / 2 # 0.00001
    else:
        return lr


def exp_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

