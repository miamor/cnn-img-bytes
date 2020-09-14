import os
import shutil
import cv2

for folder in ['benign', 'malware', 'game_Linh']:
    for filename in os.listdir('../data/image0/{}/RGB'.format(folder)):
        img_path = '../data/image0/{}/RGB/{}'.format(folder, filename)
        # print('img_path', img_path, 'filename', filename, folder)

        im = cv2.imread(img_path)
        im = cv2.resize(im, (64,64))
        print(('Write to ../data/image64/{}/RGB/{}'.format(folder, filename)))
        cv2.imwrite('../data/image64/{}/RGB/{}'.format(folder, filename), im)