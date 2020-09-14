import numpy as np
import cv2
from glob import glob as gl
import os
import shutil
import json


for setname in ['game_Linh']:
    path_labeled = '../data/image0/'+setname+'/RGB'

    # X_gray = []
    # X_rgb = []
    X = []
    Y = []


    for imgname in os.listdir(path_labeled):
            imgpath = path_labeled+'/'+imgname
            print('imgpath', imgpath)

            image = cv2.imread(imgpath)

            image = cv2.resize(image, (64,64))
            # # rewrite image
            # cv2.imwrite(imgpath, image)

            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # X_gray.append(gray)
            # X_rgb.append(rgb)
            X.append(rgb)
            Y.append([0])

    print(len(X))
    X = np.array(X)
    Y = np.array(Y)
    print('X', X.shape)
    data_path_x = 'data_prepared/'+setname+'__X_rgb.npy'
    data_path_y = 'data_prepared/'+setname+'__Y.npy'
    np.save(data_path_x, X)
    np.save(data_path_y, Y)
    print('Save X to', data_path_x)
    print('Save Y to', data_path_y)


    # cv2.destroyAllWindows()
