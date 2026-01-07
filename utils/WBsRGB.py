###########################################################################
# White-balance model class
#
# Copyright (c) 2018-present, Mahmoud Afifi
# York University, Canada
# mafifi@eecs.yorku.ca | m.3afifi@gmail.com
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown,
# "When color constancy goes wrong: Correcting improperly white-balanced
# images", CVPR 2019.
###########################################################################

import numpy as np
import cv2

def rgb_uv_hist(I, h=60):

    sz = np.shape(I)

    if sz[0] * sz[1] > 202500:
        factor = np.sqrt(202500 / (sz[0] * sz[1]))
        newH = int(np.floor(sz[0] * factor))
        newW = int(np.floor(sz[1] * factor))
        I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)

    I_reshaped = I[(I > 0).all(axis=2)]
    eps = 6.4 / h
    hist = np.zeros((h, h, 3))
    Iy = np.linalg.norm(I_reshaped, axis=1)

    for i in range(3):
        r = []
        for j in range(3):
            if j != i:
                r.append(j)

        Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
        Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])

        hist[:, :, i], _, _ = np.histogram2d(
            Iu, Iv, bins=h,
            range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2,
            weights=Iy
        )

        norm_ = hist[:, :, i].sum()
        hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)

    return hist