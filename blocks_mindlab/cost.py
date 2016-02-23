import numpy
import theano
from blocks.bricks import application
from blocks.bricks.cost import SquaredError, Cost
from blocks.bricks.wrappers import WithExtraDims


class MAPError(Cost):

    @application(inputs=['y', 'y_hat'], outputs=["cost"])
    def apply(self, y, y_hat, **kwargs):
        denom = theano.tensor.clip(theano.tensor.abs_(y),
                                   1e-8,
                                   numpy.finfo(theano.config.floatX).max)
        return theano.tensor.abs_((y - y_hat) / denom).sum(axis=1)


class NDimensionalMAPError(MAPError):
    decorators = [WithExtraDims()]


class NDimensionalSquaredError(SquaredError):
    decorators = [WithExtraDims()]
