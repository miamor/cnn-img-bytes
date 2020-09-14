from model_efn import build_efn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# %% Load data
# X = np.load('data_prepared/train__X_rgb.npy')
# Y = np.load('data_prepared/train__Y.npy')
X_train = np.load('data_prepared/train__X_rgb__merge_30game.npy')
Y_train = np.load('data_prepared/train__Y__merge_30game.npy')
X_test = np.load('data_prepared/test__X_rgb_60game.npy')
Y_test = np.load('data_prepared/test__Y__60game.npy')

from keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)
# print(Y_train)
print(N, K, N - K)
print(X_train.shape, Y_train.shape)




# %% Build model 
model_final = build_efn(input_shape=(64, 64, 3), num_classes=2)

model_final.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

mcp_save = ModelCheckpoint('./models/EnetB0_CIFAR10_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

# %% Train
# print("Training....")
model_final.fit(X_train, Y_train,
              batch_size=32,
              epochs=40,
              validation_split=0.2,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)



# %% Save model
model_final.save("./model/model.h5")
# serialize model to JSON
model_json = model_final.to_json()
with open("./model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("./model/weights.h5")


# %% Infer

predict = model_final.predict(X_test/255.0)
y_preds = predict.argmax(axis=1)
# print(predict)
# print(y_preds)
# print(Y_test)

acc = np.sum(Y_test == y_preds)
total = len(Y_test)
print('acc', acc, 'total', total, acc/total)

print('Confusion Matrix')
C = confusion_matrix(Y_test, y_preds)
Cm = C / C.astype(np.float).sum(axis=1)
print(Cm)
