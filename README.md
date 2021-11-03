# Monet-ization-of-Images
 



### Mode Collapse

To prevent Mode Collapse from domain X -> Y, we add a cyclic transformation. Both transformations F : X->Y and G: Y->X must be satisfied.
The transformation loss, known as cycle consistency loss is added onto the adverserial loss in training so that G(F(x))≈x anda F(G(x))≈x
