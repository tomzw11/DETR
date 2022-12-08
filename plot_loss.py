#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a python code template
"""


from matplotlib import pyplot as plt

def test(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    loss_list = []
    for line in lines:
        if 'epoch' in line:
            los = line.split('loss:')[-1].split(',')[0]
            loss_list.append(float(los))
    return loss_list

if __name__ == '__main__':
    
    log1 = "/home/w30005666/DETR/outputs/train0.log"
    log2 = "/home/w30005666/1126/train0.log"

    loss_list_1 = test(log1)
    loss_list_2 = test(log2)
    
    x1 = range(len(loss_list_1))
    x2 = range(len(loss_list_2))

    plt.plot(x1, loss_list_1, linewidth=2, label="graph")
    plt.plot(x2, loss_list_2, linewidth=2, label="pynative")

    plt.legend(loc="upper right")

    plt.xlim([-10,6000])

    plt.title("DETR training loss")

    plt.savefig('loss_curve.png')


