from discriminator import Discriminator
from generator import Generator
from keras.layers import Input
from keras.models import Model

class CycleGAN:
    def __init__(self):
        self.genG = Generator().model
        self.genF = Generator().model
        self.discX = Discriminator().model
        self.discY = Discriminator().model

        inputX = Input(shape = (256, 256, 3)) 
        inputY = Input(shape = (256, 256, 3)) 

        # Reconstruction needed for Cycle Consistency loss
        gen_imageY = self.genG(inputX)
        gen_imageX = self.genF(inputY)
        recon_imageX = self.genF(gen_imageY)
        recon_imageY = self.genG(gen_imageX)

        # Identity
        gen_identityX = self.genF(inputX)
        gen_identityY = self.genG(inputY)

        # disc
        distinguishX = self.discX(gen_imageX)
        distinguishY = self.discY(gen_imageY)

        self.discX.trainable = False
        self.discY.trainable = False    

        self.model = Model([inputX, inputY], [distinguishX, distinguishY, recon_imageX, recon_imageY, gen_identityX, gen_identityY])


if __name__ == '__main__':
    CycleGAN().model.summary()