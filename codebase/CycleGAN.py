from data_loader import data
from discriminator import Discriminator
from generator import Generator
from keras.layers import Input
from keras.models import Model
from keras.losses import MeanAbsoluteError, MeanSquaredError
import tensorflow as tf
from keras import backend as K
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#from keras.optimizers import Adam

"""def generator_loss(gen):
    return MeanSquaredError(tf.ones_like(gen), gen)


def discriminator_loss(real,gen):
    return (MeanSquaredError(tf.ones_like(real),real) + MeanSquaredError(tf.zeros_like(gen),gen))/2"""

def gen_loss(predict):
    return MeanSquaredError(tf.ones_like(predict), predict)

def disc_loss(predict_real, predict_gen):
    return MeanSquaredError(predict_real,tf.ones_like(predict_real)) + predict_gen**2
    
class CycleGAN(Model):

    def __init__(self,shape=((256, 256, 3))):#,batch):
        super(CycleGAN,self).__init__()
        x = Input(shape=shape)
        y = Input(shape=shape)

        #x,y = batch
        self.genG = Generator().model
        self.genF = Generator().model
        self.discX = Discriminator().model
        self.discY = Discriminator().model


        self.cycle_weight = 10
        self.identity_weight = 0.5

        super(CycleGAN,self).compile()
        

        self.genG_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.genF_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.discX_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.discY_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

        self.cycle_loss = MeanAbsoluteError()
        self.identity_loss = MeanAbsoluteError()
        self.gen_loss = MeanSquaredError()
        self.disc_loss = MeanSquaredError()


        # Reconstruction needed for Cycle Consistency loss

    """def call(self,inputs):
        print(tf.shape(inputs))
        print(inputs)
        x,y = inputs
        gen_y = self.genG(x)
        gen_x = self.genF(y)
        recon_x = self.genF(gen_y)
        recon_y = self.genG(gen_x)

        # Identity
        identity_x = self.genF(x)
        identity_y = self.genG(y)

        # disc
        predict_x = self.discX(x)
        predict_gen_x = self.discX(gen_x)

        predict_y = self.discY(y)
        predict_gen_y = self.discY(gen_y)
        return gen_y,gen_x,recon_x,recon_y,identity_x,identity_y,predict_x,predict_gen_x,predict_y,predict_gen_y"""

    @tf.function
    def train_step(self,data_batch):
        x,y = data_batch
        with tf.GradientTape(persistent=True) as tape:
            gen_y = self.genG(x, training=True)
            gen_x = self.genF(y, training=True)
            recon_x = self.genF(gen_y, training=True)
            recon_y = self.genG(gen_x, training=True)

            # Identity
            identity_x = self.genF(x, training=True)
            identity_y = self.genG(y, training=True)

            # disc
            predict_x = self.discX(x, training=True)
            predict_gen_x = self.discX(gen_x, training=True)

            predict_y = self.discY(y, training=True)
            predict_gen_y = self.discY(gen_y, training=True)

            #def identity_loss(real, identity):
            #    return MeanAbsoluteError(real, identity) * self.identity_weight * self.cycle_weight

            #def cycle_loss(real, recon):
            #    return MeanAbsoluteError(real, recon) * self.cycle_weight

            #self.discX.trainable = False
            #self.discY.trainable = False
            
            

            G_identity_loss =  self.identity_loss(y,identity_y)* self.identity_weight * self.cycle_weight
            F_identity_loss = self.identity_loss(x, identity_x)* self.identity_weight * self.cycle_weight

            G_cycle_loss = self.cycle_loss(x, recon_x)* self.cycle_weight
            F_cycle_loss = self.cycle_loss(y, recon_y)* self.cycle_weight

            G_gen_loss = self.gen_loss(predict_gen_y,tf.ones_like(predict_gen_y))
            F_gen_loss = self.gen_loss(predict_gen_x,tf.ones_like(predict_gen_x))

            Y_disc_loss = self.disc_loss(predict_y,tf.ones_like(predict_y))/2 + self.disc_loss(predict_gen_y,tf.zeros_like(predict_gen_y))/2
            X_disc_loss = self.disc_loss(predict_x,tf.ones_like(predict_x))/2 + self.disc_loss(predict_gen_x,tf.zeros_like(predict_gen_x))/2

            G_total_loss = G_cycle_loss+G_identity_loss+G_gen_loss
            F_total_loss = F_cycle_loss+F_identity_loss+F_gen_loss
    
        gradsG = tape.gradient(G_total_loss, self.genG.trainable_variables)
        gradsF = tape.gradient(F_total_loss, self.genF.trainable_variables)

        discX_grads = tape.gradient(X_disc_loss, self.discX.trainable_variables)
        discY_grads = tape.gradient(Y_disc_loss, self.discY.trainable_variables)

        self.genG_optimizer.apply_gradients(
            zip(gradsG, self.genG.trainable_variables)
        )
        self.genF_optimizer.apply_gradients(
            zip(gradsF, self.genF.trainable_variables)
        )

        # Update the weights of the discriminators
        self.discX_optimizer.apply_gradients(
            zip(discX_grads, self.discX.trainable_variables)
        )
        self.discY_optimizer.apply_gradients(
            zip(discY_grads, self.discY.trainable_variables)
        )
        

        return {
            "G_loss": G_cycle_loss+G_identity_loss+G_gen_loss,
            "F_loss": F_cycle_loss+F_identity_loss+F_gen_loss,
            "D_X_loss": X_disc_loss,
            "D_Y_loss": Y_disc_loss,
        }

    #def train(self, data,epochs=10):
    #    for _ in range(epochs):
    #        for i in range(data.shape[0]):
    #            img_a = np.ex



if __name__ == '__main__':
    #real,monet = data()
    #print(monet)
    cgan = CycleGAN()
    cgan.fit(tf.data.Dataset.zip(data()),epochs=1)

    #print(type(real),type(monet))
    #cgan = CycleGAN()
    #cgan.fit([real,monet],epochs=1)
    #cgan.combine()
    #cgan.model.summary()
