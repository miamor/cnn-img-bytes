#! /usr/bin/python3
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import json
from tqdm import tnrange
from glob import glob as gl
from tensorflow.python.keras.models import load_model, Model

import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Fixed error Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model_name = 'rgb_reported'
setname = 'test' # none | test | game_Linh

''' Load data '''
# X = np.load('data_prepared/'+setname+'__X_rgb.npy', allow_pickle=True)
# Y = np.load('data_prepared/'+setname+'__Y.npy', allow_pickle=True)
X = np.load('data_prepared/test__X_rgb__60game.npy', allow_pickle=True)
Y = np.load('data_prepared/test__Y__60game.npy', allow_pickle=True)


Y = Y.flatten()

# from keras.utils import to_categorical
# Y = to_categorical(Y, num_classes=2)
# print(Y)
# print(Y, categorical_Y)


''' Load model '''
model = load_model('models/'+model_name+'.h5')
model.summary()



''' Infer '''
predict = model.predict(X/255.0)
y_preds = predict.argmax(axis=1)
# print(predict)
# print(y_preds)
# print(Y)

acc = np.sum(Y == y_preds)
total = len(Y)
print('acc', acc, 'total', total, acc/total)

# tot_bgn = np.sum(Y == 0)
# tot_mal = np.sum(Y == 1)

# tpr = 0.0
# far = 0.0
# for k,v in enumerate(Y):
#     # print(k, Y[k], v)
#     if v == 1 and y_preds[k] == 1:
#         tpr += 1
#     if v == 0 and y_preds[k] == 1:
#         far += 1
# print('tpr', tpr, 'tot_mal', tot_mal, tpr/tot_mal)
# print('far', far, 'tot_bgn', tot_bgn, far/tot_bgn)


print('Confusion Matrix')
C = confusion_matrix(Y, y_preds)
Cm = C / C.astype(np.float).sum(axis=1)
print(Cm)
# print('Classification Report')
# print(classification_report(y, y_preds))

# # Plot non-normalized confusion matrix
# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(clf, X, y,
#                                  display_labels=[0, 1],
#                                  # cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)

#     print(title)
#     print(disp.confusion_matrix)

# plt.show()



# summarize feature map shapes
l = 0
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)
    l = i
print('l', l)

intermediate_model = Model(inputs=model.inputs, outputs=model.layers[l].output)
imgs = []
X = []
for set in ['benign', 'malware', 'game_Linh']:
    k = 0
    for imgname in os.listdir('../data/image64/'+set+'/RGB'):
        # if k == 5: break # test small sample
        # k += 1
        imgs.append('{}__{}'.format(set, imgname))
        p = '../data/image64/'+set+'/RGB/'+imgname
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        X.append(im)
X = np.array(X)
feature_maps = intermediate_model.predict(X)
print('feature_maps', feature_maps.shape)
for i in range(feature_maps.shape[0]):
    ft_map = feature_maps[i, :, :, feature_maps.shape[-1]-1]
    # print('./feature_maps/{}/{}.jpg'.format(imgs[i].split('__')[0], imgs[i].split('__')[1]))
    cv2.imwrite('./feature_maps/{}/{}.jpg'.format(imgs[i].split('__')[0], imgs[i].split('__')[1]), ft_map)








# ''' Visualize feature maps '''
# from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.vgg16 import preprocess_input
# import matplotlib.pyplot as plt


# # summarize feature map shapes
# for i in range(len(model.layers)):
#     layer = model.layers[i]
#     # check for convolutional layer
#     if 'conv' not in layer.name:
#         continue
#     # summarize output shape
#     print(i, layer.name, layer.output.shape)

# # redefine model to output right after the first hidden layer
# intermediate_model = Model(inputs=model.inputs, outputs=model.layers[1].output)


# # load the image with the required shape
# # img = load_img('../data/image64/game_Linh/RGB/Miniclip-Game-twiddlestix_RGB.png', target_size=(64,64))
# # # convert the image to an array
# # img = img_to_array(img)
# # # expand dimensions so that it represents a single 'sample'
# # img = np.expand_dims(img, axis=0)

# # prepare the image (e.g. scale pixel values for the vgg)
# # img = preprocess_input(img)

# imgs = ['../data/image64/game_Linh/RGB/Miniclip-Game-twiddlestix_RGB.png', 
# '../data/image64/game_Linh/RGB/avast_free_antivirus_setup_online_RGB.png', 
# '../data/image64/benign/RGB/075e550cf94840b806e88772a2c05dc12b68b0b55ed4a2a0b5bd69539d67a40e_RGB.png', 
# '../data/image64/benign/RGB/fce43087aef8626210709ff7e29c87f5e1f61f5a399ff7121de8b375de57c3d7_RGB.png', 
# '../data/image64/malware/RGB/2b2710d355091e4dc12d43e1e402ce45f4501a13ef9c96d89278b104d9f81c68_RGB.png', 
# '../data/image64/malware/RGB/2ee99f7590d2eac42d729053f08f3bc2989e8ab3ebe49dc63d58a1dc4ed576d0_RGB.png']
# X = []
# for img_path in imgs:
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     X.append(img)

# X = np.array(X)/255.

# # get feature map for first hidden layer
# feature_maps = intermediate_model.predict(X)
# print('feature_maps', feature_maps.shape)


# square = 6
# ax = {}
# for i in range(feature_maps.shape[0]):
#     fig = plt.figure(i)
#     fig.canvas.set_window_title('{} - {}'.format(imgs[i].split('/image64/')[1].split('/')[0], imgs[i].split('/')[-1]))

#     # plot all 16 maps in a 4x4 squares
#     ix = 1

#     img = cv2.imread(imgs[i])
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     for _ in range(square):
#         for _ in range(square):
#             # specify subplot and turn of axis
#             ax[i] = plt.subplot(square, square, ix)
#             # ax[i].set_title('map '+str(ix))
#             ax[i].set_xticks([])
#             ax[i].set_yticks([])

#             if ix == 1:
#                 plt.imshow(img, cmap='gray')
#             elif ix <= feature_maps.shape[-1]:
#                 # plot filter channel in grayscale
#                 plt.imshow(feature_maps[i, :, :, ix-1], cmap='gray')
#             ix += 1

# # show the figure
# plt.show()
