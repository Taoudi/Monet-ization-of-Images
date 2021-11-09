# Monet-ization-of-Images
 


Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks - https://arxiv.org/pdf/1703.10593.pdf

### Mode Collapse

To prevent Mode Collapse from domain X -> Y, we add a cyclic transformation. Both transformations F : X->Y and G: Y->X must be satisfied.
The transformation loss, known as cycle consistency loss is added onto the adverserial loss in training so that G(F(x))≈x and F(G(x))≈x

### Padding

Pixels on the border are convolved less frequently than pixels more to the center of an image and will therefore not be preserved very well by the network. To combat this, we introduce (reflection) padding where the images get an additional layer added on top of the borders.

### Architecture

    Discriminator: C64-C128-C256-C512 as described in the paper.
    Generator: c7s1-64,d128,d256,
                R256,R256,R256,R256,R256,R256,R256,R256,R256,
                u128,u64,c7s1-3
