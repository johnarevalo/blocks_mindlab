from blocks.bricks import Identity, Logistic, MLP, cost, application
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


class SparsePenaltyCost(cost.SquaredError):

    def __init__(self, h, rho, beta):
        super(SparsePenaltyCost, self).__init__(h)
        self.rho = rho
        self.h = h
        self.beta = beta

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        error = super(SparsePenaltyCost, self).apply(y, y_hat)
        cost = tensor.sqr(y - y_hat).sum(axis=1).mean()
        p_hat = tensor.abs_(self.h).mean(axis=0)
        kl = self.rho * tensor.log(self.rho / p_hat) + (1 - self.rho) * \
            tensor.log((1 - self.rho) / (1 - p_hat))
        return cost + self.beta * kl.sum()
