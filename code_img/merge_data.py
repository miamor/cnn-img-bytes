import os
import shutil
import numpy as np

x_train = np.load('data_prepared/train__X_rgb.npy')
x_game = np.load('data_prepared/game_Linh__X_rgb.npy')
y_train = np.load('data_prepared/train__Y.npy')
y_game = np.load('data_prepared/game_Linh__Y.npy')
print('y_game', y_game)

print('x_train', x_train.shape, x_train[30:,:].shape)
print('x_game', x_game.shape, x_game[:30,:].shape)

X = np.concatenate((x_train[30:,:], x_game[:30,:]), axis=0)
Y = np.concatenate((y_train[30:,:], y_game[:30,:]), axis=0)
print('X', X.shape)
print('Y', Y.shape)
# print('Y', Y)
# from keras.utils import to_categorical
# Y = to_categorical(Y, num_classes=2)
# print('Y', Y)

np.save('data_prepared/train__X_rgb__merge_30game.npy', X)
np.save('data_prepared/train__Y__merge_30game.npy', Y)
np.save('data_prepared/test__X_rgb__60game.npy', x_game[30:,:])
np.save('data_prepared/test__Y__60game.npy', y_game[30:,:])

