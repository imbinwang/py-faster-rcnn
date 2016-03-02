#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
if sys.platform.startswith('linux'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math



def save_loss_curve(fname, num_subplot):

    rpn_cls_loss = []
    rpn_bbox_loss = []
    
    for line in open(fname):          
        if 'output' in line and '#0' in line and 'rpn_cls_loss' in line:
            txt = re.search(ur'rpn_cls_loss\s=\s([0-9\.]+)', line)
            rpn_cls_loss.append(float(txt.groups()[0]))
        if 'output' in line and '#1' in line and 'rpn_loss_bbox' in line:
            txt = re.search(ur'rpn_loss_bbox\s=\s([0-9\.]+)', line)
            rpn_bbox_loss.append(float(txt.groups()[0]))

    rpn_cls_loss = [math.log(l) for l in rpn_cls_loss]
    rpn_bbox_loss = [math.log(l) for l in rpn_bbox_loss]
    
    lim = -9

    plt.clf()
    plt.ylim(lim, 0)
    num_subplot = int(num_subplot)
    f, axarr = plt.subplots(num_subplot, figsize=(6,8), sharex=True)
    for sub_id in range(num_subplot):
        if sub_id==0:
            axarr[sub_id].plot(range(len(rpn_cls_loss)), rpn_cls_loss, 'r')
            axarr[sub_id].set_title('rpn_cls_loss')
        if sub_id==1:
            axarr[sub_id].plot(range(len(rpn_bbox_loss)), rpn_bbox_loss, 'g')
            axarr[sub_id].set_title('rpn_bbox_loss')
        
    plt.savefig('loss_curve_alt.png')
    plt.show()


if __name__ == '__main__':
    # argv[1]: log_file
    # argv[2]: num_subplot
    save_loss_curve(sys.argv[1], sys.argv[2])
