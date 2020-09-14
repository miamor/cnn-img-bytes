from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import efficientnet.keras as enet
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.models import Model


# %%
# Swish defination
from keras.backend import sigmoid


class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


def swish_act(x, beta=1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish_act': SwishActivation(swish_act)})


def build_efn(input_shape, num_classes, include_top=True):
    # loading B0 pre-trained on ImageNet without final aka feature extractor
    # model = enet.EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    model = enet.EfficientNetB5(
        include_top=False, input_shape=input_shape, pooling='max', weights='imagenet')
    # model.summary()

    # building 2 fully connected layer
    x = model.output

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)

    if not include_top:
        model_final = Model(input=model.input, output=x)
    else:
        # output layer
        predictions = Dense(num_classes, activation="softmax")(x)
        model_final = Model(inputs=model.input, outputs=predictions)

    # model_final.summary()

    return model_final
