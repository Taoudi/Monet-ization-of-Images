from keras.layers import Reshape, Dense, Input, LeakyReLU, Conv2D, Conv2DTranspose, Concatenate, ReLU, Dropout, ZeroPadding2D
from tensorflow_addons.layers import InstanceNormalization
from keras.initializers import RandomNormal

from keras.models import Model
from reflection import ReflectionPadding2D

#  C64->C128->C256->C512
class Discriminator:
    def __init__(self,padding ='valid',strides=(2,2),kernel=(4,4),initializer = RandomNormal(mean=0.,stddev=0.02),alpha=0.2):
        img_inp = Input(shape = (256, 256, 3))
        conv_1 = Conv2D(64,kernel,strides=2,use_bias=False,kernel_initializer=initializer,padding=padding)(img_inp)
        act_1 = LeakyReLU(alpha)(conv_1)
    
        conv_2 = Conv2D(128,kernel,strides=strides,use_bias=False,kernel_initializer=initializer,padding=padding)(act_1)
        
        batch_norm_2 = InstanceNormalization()(conv_2)
        act_2 = LeakyReLU(alpha)(batch_norm_2)
    
        conv_3 = Conv2D(256,kernel,strides=strides,use_bias=False,kernel_initializer=initializer,padding=padding)(act_2)
        batch_norm_3 = InstanceNormalization()(conv_3)
        act_3 = LeakyReLU(alpha)(batch_norm_3)
    
        #zero_pad = ZeroPadding2D()(act_3)
    
        # STRIDES = 2?
        conv_4 = Conv2D(512,kernel,strides=(1,1),use_bias=False,kernel_initializer=initializer)(act_3)
        batch_norm_4 = InstanceNormalization()(conv_4)
        act_4 = LeakyReLU(alpha)(batch_norm_4)
    
        #zero_pad_1 = ZeroPadding2D()(act_4)
        outputs = Conv2D(1,kernel,strides=1,use_bias=False,kernel_initializer=initializer)(act_4)
    
        self.model = Model(img_inp, outputs)


if __name__ == "__main__":
    image_shape = (256,256,3)
# create the model
    disc = Discriminator()
    # summarize the model
    disc.model.summary()