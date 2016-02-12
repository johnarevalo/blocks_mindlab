from blocks.bricks import Identity, Logistic, MLP
from blocks.initialization import Uniform, Constant
from math import sqrt
from theano import tensor


class Autoencoder(MLP):

    def __init__(self, ninput, nhidden):
        r = sqrt(6) / sqrt(nhidden + ninput + 1)
        super(Autoencoder, self).__init__(activations=[Logistic(), Identity()],
                                          dims=[ninput, nhidden, ninput],
                                          weights_init=Uniform(width=r),
                                          biases_init=Constant(0))

def sparsity_regularizer(h, rho, beta):
    p_hat = tensor.abs_(h).mean(axis=0)
    kl = rho * tensor.log(rho / p_hat) + (1 - rho) * \
        tensor.log((1 - rho) / (1 - p_hat))
    return beta * kl.sum()
