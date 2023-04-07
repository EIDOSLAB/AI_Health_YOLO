import tensorflow as tf
import math


def no_scheduler(epoch, lr):
    return lr


def step_scheduler(epoch, lr):
    if epoch <= 10: 
        return lr   # 0.00005
    elif epoch <= 35:
        return lr / 2 # 0.00003
    else:
        return lr / 2 # 0.00001


def exp_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

