from blocks.bricks import (Softmax, Linear, Logistic, Initializable,
                           application, Brick, Sequence, BatchNormalization)
from blocks import initialization


class FCSoftmax(Initializable):

    def __init__(self, input_dim, output_dim, linear_range, **kwargs):
        super(FCSoftmax, self).__init__(**kwargs)
        self.linear = Linear(input_dim=input_dim, output_dim=output_dim,
                             name='linear_softmax')
        self.linear.weights_init = initialization.Uniform(width=linear_range)
        self.linear.biases_init = initialization.Constant(0)
        self.softmax = Softmax()
        self.children = [self.linear, self.softmax]
        self.categorical_cross_entropy = self.softmax.categorical_cross_entropy

    @application(inputs=['h'], outputs=['y_hat', 'linear_output'])
    def apply(self, h):
        linear_output = self.linear.apply(h)
        y_hat = self.softmax.apply(linear_output)
        return y_hat, linear_output


class FCLogistic(Initializable):

    def __init__(self, input_dim, output_dim, linear_range, **kwargs):
        super(FCLogistic, self).__init__(**kwargs)
        self.linear = Linear(input_dim=input_dim, output_dim=output_dim,
                             name='linear_logistic')
        self.linear.weights_init = initialization.Uniform(width=linear_range)
        self.linear.biases_init = initialization.Constant(0)
        self.logistic = Logistic()
        self.children = [self.linear, self.logistic]

    @application(inputs=['h'], outputs=['y_hat', 'linear_output'])
    def apply(self, h):
        linear_output = self.linear.apply(h)
        y_hat = self.logistic.apply(linear_output)
        return y_hat, linear_output


class TransposeBN(Brick):

    @application(inputs=['h'], outputs=['output'])
    def apply(self, h):
        return h.transpose(1, 0, 2)


class RecurrentBN(Sequence, Initializable):

    def __init__(self, input_dim, **kwargs):
        super(RecurrentBN, self).__init__([
            TransposeBN().apply,
            BatchNormalization(input_dim=input_dim).apply,
            TransposeBN().apply,
        ])
