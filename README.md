# Monet-ization-of-Images
 


Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks - https://arxiv.org/pdf/1703.10593.pdf

Check out the Notebook/Writeup at https://www.kaggle.com/yousseftaoudi/cyclegan-monet-ization-of-photographs !


### Architecture

    Discriminator: C64-C128-C256-C512 as described in the paper.
    Generator: c7s1-64,d128,d256,
                R256,R256,R256,R256,R256,R256,R256,R256,R256,
                u128,u64,c7s1-3
