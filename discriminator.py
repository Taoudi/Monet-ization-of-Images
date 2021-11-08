from keras.layers import InstanceNormalization, Reshape, Dense, Input, LeakyReLU, Conv2D, Conv2DTranspose, Concatenate, ReLU, Dropout, ZeroPadding2D
from keras.initializers import RandomNormal

from keras.models import Model

class Discriminator:
    def __init__(self):
        initializer = RandomNormal(mean=0.,stddev=0.02)
        img_inp = Input(shape = (256, 256, 3))
        conv_1 = Conv2D(64,4,strides=2,use_bias=False,kernel_initializer=initializer,padding='same')(img_inp)
        act_1 = LeakyReLU(alpha=0.2)(conv_1)
    
        conv_2 = Conv2D(128,4,strides=2,use_bias=False,kernel_initializer=initializer,padding='same')(act_1)
        
        batch_norm_2 = InstanceNormalization(momentum=0.8)(conv_2)
        act_2 = LeakyReLU(alpha=0.2)(batch_norm_2)
    
        conv_3 = Conv2D(256,4,strides=2,use_bias=False,kernel_initializer=initializer,padding='same')(act_2)
        batch_norm_3 = InstanceNormalization(momentum=0.8)(conv_3)
        act_3 = LeakyReLU(alpha=0.2)(batch_norm_3)
    
        zero_pad = ZeroPadding2D()(act_3)
    
        # STRIDES = 2?
        conv_4 = Conv2D(512,4,strides=1,use_bias=False,kernel_initializer=initializer)(zero_pad)
        batch_norm_4 = InstanceNormalization(momentum=0.8)(conv_4)
        act_4 = LeakyReLU(alpha=0.2)(batch_norm_4)
    
        zero_pad_1 = ZeroPadding2D()(act_4)
        outputs = Conv2D(1,4,strides=1,use_bias=False,kernel_initializer=initializer)(zero_pad_1)
    
        self.model = Model(img_inp, outputs)
