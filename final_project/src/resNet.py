from keras.models import Model 
from keras import layers 
import keras.layers as layers
from keras.regularizers import l2
from keras import backend as K 

def _add_common_layer(l):
    # build BN->relu block
    l = layers.BatchNormalization()(l)
    l = layers.LeakyReLU()(l)

    return l 

def _add_common_layer_conv(l, filters, kernel_size, strides=(1,1)):
    # build Conv -> BN -> relu block
    l = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(l)
    l = _add_common_layer(l)

    return l

def _connect(input, residual):
    # add a shortcut between the input and residual 

    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input 
    print("input shape", input_shape)
    print("residual shape", residual_shape)
    # 1x1 conv if the shape is different; otherwise just identity 
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = layers.Conv2D(filters=residual_shape[ROW_AXIS],
                          kernel_size=(1,1),
                          strides=(stride_width, stride_height),
                          padding='valid',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(0.0001))(input)
        print("shortcut shape", K.int_shape(shortcut))
        print("1x1 conv for shortcut")

    return layers.merge.add([shortcut, residual])


def _residual_block(l, block_fn, filters, repetitions, is_first_layer):
    for i in range(repetitions):
        strides = (1,1)

        if i==0 and not is_first_layer:
            strides = (2,2)
        l = block_fn(filters=filters, init_strides=strides, is_first_block=(is_first_layer and i ==0))(l)

    return l 

def _dim_ordering():
    global ROW_AXIS 
    global COL_AXIS 
    global CHANNEL_AXIS 

    if K.image_dim_ordering == 'tf':
        ROW_AXIS = 1 
        COL_AXIS = 2 
        CHANNEL_AXIS = 3 
    else:
        CHANNEL_AXIS = 1 
        ROW_AXIS = 2 
        COL_AXIS = 3 

    print("channels first?", K.image_data_format == "channels_first")
    print("Row axis {}, column axis {}, channel axis {}".format(ROW_AXIS, COL_AXIS, CHANNEL_AXIS))


def buildResNet(num_outputs, block_fn, repetitions, input_shape=(64, 64, 3)):

    _dim_ordering()

    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same')(input)
    x = _add_common_layer_conv(x, filters=64, kernel_size=(7,7), strides=(2,2))
    x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    block = x 
    filters = 64 
    for i, r in enumerate(repetitions):
        block = _residual_block(block, block_fn, filters=filters, repetitions=r, is_first_layer=(i==0))
        filters = filters*2

    
    block = _add_common_layer(block)
    block_shape = K.int_shape(block)

    x = layers.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1,1))(block)
    x = layers.Flatten()(x)
    x = Dense(units=num_outputs, kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(inputs=input, outputs=dense)
    return model 
                                                

def basic_block_fn(filters, init_strides=(1,1), is_first_block=False):
    # basic 3x3 conv blocks 
    def f(input):
        if is_first_block:
            conv = layers.Conv2D(filters=filters, kernel_size=(3,3),
                                 strides=init_strides, 
                                 padding='same',
                                 kernel_initializer='he_normal', 
                                 kernel_regularizer=l2(1e-4))(input)
        else:
            conv = _add_common_layer_conv(filters=filters, kernel_size=(3,3),
                                          strides=init_strides)(input)

        residual = _add_common_layer_conv(conv, filters=filters, kernel_size=(3,3))
        return _connect(input, residual)

    return f 

def bottleneck_fn(filters, init_strides=(1,1), is_first_block=False):
    # bottleneck architecture as proposed in the follow-up paper 

    def f(input):
        if is_first_block:
            conv = conv2D(filters=filters, kernel_size=(1,1),
                          strides=init_strides, 
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(input)
        else:
            conv = _add_common_layer_conv(filters=filters, kernel_size=(1,1),
                                          strides=init_strides)(input)

        conv = _add_common_layer_conv(filters, kernel_size=(3,3))(conv)
        residual = _add_common_layer_conv(filters, kernel_size=(1,1))(conv)
        return _connect(conv, residual)

    return f 


def resnet_18(num_outputs):
    return buildResNet(num_outputs, basic_block_fn, [2, 2, 2, 2])

if __name__ == "__main__":
    model = resnet_18(200)
    model.summary()


