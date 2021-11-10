from discriminator import Discriminator
from generator import Generator
from keras.layers import Input
from keras.models import Model
from keras.losses import MeanAbsoluteError, MeanSquaredError
import tensorflow as tf
#from keras.optimizers import Adam

"""def generator_loss(gen):
    return MeanSquaredError(tf.ones_like(gen), gen)


def discriminator_loss(real,gen):
    return (MeanSquaredError(tf.ones_like(real),real) + MeanSquaredError(tf.zeros_like(gen),gen))/2"""

class CycleGAN:

    def __init__(self,shape=((256, 256, 3))):#,batch):

        x = Input(shape=shape)
        y = Input(shape=shape)

        #x,y = batch
        self.genG = Generator().model
        self.genF = Generator().model
        self.discX = Discriminator().model
        self.discY = Discriminator().model



        self.genG_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.genF_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.discG_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.discF_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

        self.cycle_weight = 10
        self.identity_weight = 0.5

        # Reconstruction needed for Cycle Consistency loss

    def train_iteration(self,data_batch):
        with tf.GradientTape(persistent=True) as tape:
            x,y = data_batch
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

            def identity_loss(real, identity):
                return MeanAbsoluteError(real, identity) * self.identity_weight * self.cycle_weight

            def cycle_loss(real, recon):
                return MeanAbsoluteError(real, recon) * self.cycle_weight

            def gen_loss(predict):
                return MeanSquaredError(tf.ones_like(predict), predict)

            def disc_loss(predict_real, predict_gen):
                return MeanSquaredError(predict_real,tf.ones_like(predict_real)) + predict_gen**2

            self.discX.trainable = False
            self.discY.trainable = False

            G_cycle_loss = cycle_loss(x, recon_x)
            F_cycle_loss = cycle_loss(y, recon_y)

            G_identity_loss = identity_loss(y, identity_y)
            F_identity_loss = identity_loss(x, identity_x)

            G_gen_loss = gen_loss(predict_gen_y)
            F_gen_loss = gen_loss(predict_gen_x)

            G_disc_loss = disc_loss(predict_y, predict_gen_y)
            F_disc_loss = disc_loss(predict_x,predict_gen_x)

        gradsG = tape.gradient(G_cycle_loss+G_identity_loss+G_gen_loss, self.genG.trainable_variables)
        gradsF = tape.gradient(F_cycle_loss+F_identity_loss+F_gen_loss, self.genF.trainable_variables)

        discX_grads = tape.gradient(G_disc_loss, self.disc_X.trainable_variables)
        discY_grads = tape.gradient(F_disc_loss, self.disc_Y.trainable_variables)


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
            "D_X_loss": G_disc_loss,
            "D_Y_loss": F_disc_loss,
        }



            





       """ self.model = Model([x, y], [predict_gen_x, predict_gen_y, predict_x, predict_y, recon_x, recon_y, identity_x, identity_y])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

        self.discX.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.discY.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae','mae'],optimizer=self.optimizer)"""

    def train(data,epochs=10):
        for _ in range(epochs):
            for i in range(data.shape[0]):
                img_a = np.ex



if __name__ == '__main__':
    cgan = CycleGAN().model.summary()
    #cgan.combine()
    #cgan.model.summary()
