from keras.layers import Reshape, Dense, Input, ReLU, Conv2D, Conv2DTranspose, Concatenate, ReLU, Dropout, ZeroPadding2D
from tensorflow_addons.layers import InstanceNormalization

from keras.models import Model
from reflection import ReflectionPadding2D

class Generator:
    def __init__(self,k=64,n_res=8):
        img_inp = Input(shape = (256, 256, 3))
        c7s164 = ReflectionPadding2D(padding = (3,3))(img_inp)
        c7s164 = Conv2D(64,(7,7),(1,1))(c7s164)
        c7s164 = InstanceNormalization()(c7s164)
        c7s164 = ReLU()(c7s164)

        #d128 = ReflectionPadding2D()(c7s164)
        d128 = Conv2D(128,(3,3),(2,2),padding="same")(c7s164)
        d128 = InstanceNormalization()(d128)
        d128 = ReLU()(d128)

        #d256 = ReflectionPadding2D()(d128)
        d256 = Conv2D(256,(3,3),(2,2),padding="same")(d128)
        d256 = InstanceNormalization()(d256)
        d256 = ReLU()(d256)


        # RESIDUAL BLCOKS

        curr = d256
        res = d256
        k=256
        for _ in range(n_res):
            res = ReflectionPadding2D()(res)
            res = Conv2D(k,(3,3))(res)
            res = InstanceNormalization()(res)
            res = ReLU()(res)

            res = ReflectionPadding2D()(res)
            res = Conv2D(k,(3,3))(res)
            res = InstanceNormalization()(res)
            res = Concatenate()([res,curr])
            curr = res
            

        #u128 = ReflectionPadding2D()(res)
        u128 = Conv2DTranspose(128,(3,3),(2,2),padding="same")(res)
        u128 = InstanceNormalization()(u128)
        u128 = ReLU()(u128)

        #u64 = ReflectionPadding2D()(u128)
        u64 = Conv2DTranspose(64,(3,3),(2,2),padding="same")(u128)
        u64 = InstanceNormalization()(u64)
        u64 = ReLU()(u64)

        c7s13 = ReflectionPadding2D(padding=(3,3))(u64)
        c7s13 = Conv2D(3,(7,7))(c7s13)
        c7s13 = InstanceNormalization()(c7s13)
        c7s13 = ReLU()(c7s13)
        self.model = Model(img_inp, c7s13)



        
if __name__ == "__main__":
    image_shape = (256,256,3)
    # create the model
    disc = Generator()
    # summarize the model
    disc.model.summary()