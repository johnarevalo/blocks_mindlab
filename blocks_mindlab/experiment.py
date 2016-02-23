import numpy
import theano
from blocks.algorithms import GradientDescent, CompositeRule, Momentum, Restrict, VariableClipping, RMSProp, Adam, StepClipping
from blocks.extensions import FinishAfter, saveload, predicates, Printing, Timing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.training import TrackTheBest, SharedVariableModifier
from blocks_extras.extensions import plot
from blocks.monitoring import aggregation
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.initialization import Uniform, Constant
from blocks.model import Model
from blocks import main_loop
from blocks.roles import INPUT, WEIGHT
from blocks_mindlab import utils
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []


class Experiment(object):

    def __init__(self, model_name, train_stream, dev_stream):
        self._algorithm = None
        self._parameters = None
        self.model_name = model_name
        self.monitored_vars = []
        self.extensions = []
        self.step_rules = []
        self.train_stream = train_stream
        self.dev_stream = dev_stream

    def set_momentum(self, learning_rate, momentum):
        self.step_rules.append(
            Momentum(learning_rate=learning_rate, momentum=momentum))

    def set_rmsprop(self, learning_rate, decay_rate=0.95):
        self.step_rules.append(
            RMSProp(learning_rate=learning_rate, decay_rate=decay_rate))

    def set_adam(self, learning_rate):
        self.step_rules.append(Adam(learning_rate))

    def initialize_layers(self, w_inits, b_inits, bricks):
        for i, brick in enumerate(bricks):
            brick.weights_init = Uniform(width=w_inits[i])
            if b_inits[i] == 0:
                brick.biases_init = Constant(0)
            else:
                brick.biases_init = Uniform(width=b_inits[i])
            brick.initialize()

    def set_cost(self, cost):
        cost.name = 'cost'
        self.cost = cost
        self.cg = ComputationGraph(self.cost)

    def monitor_perf_measures(self, y, y_hat, threshold):
        tp, tn, fp, fn, pre, rec, f_score = utils.get_measures(
            y.flatten(), y_hat.flatten() > threshold)
        self.monitored_vars.extend([f_score, tp, tn, fp, fn])

    def monitor_w_norms(self, bricks=[], weights=[], owner=None):
        for i, brick in enumerate(bricks):
            var = brick.W.norm(2, axis=0).max()
            brick.add_auxiliary_variable(var, name='W_max_norm_' + brick.name)
            self.monitored_vars.append(var)
        for i, W in enumerate(weights):
            var = W.norm(2, axis=0).max()
            owner.add_auxiliary_variable(
                var, name='W_max_norm_weight_' + str(i))
            self.monitored_vars.append(var)

    def monitor_activations(self, mlp):
        var_filter = VariableFilter(theano_name_regex='linear.*output')
        outputs = var_filter(self.cg.variables)
        for i, output in enumerate(outputs):
            mlp.add_auxiliary_variable(output.mean(),
                                       name='mean_act_' + str(i))
            mlp.add_auxiliary_variable(output.mean(axis=0).max(),
                                       name='max_act_' + str(i))
            mlp.add_auxiliary_variable(output.mean(axis=0).min(),
                                       name='min_act_' + str(i))
        self.monitored_vars.extend(mlp.auxiliary_variables)

    def apply_dropout(self, dropout, variables=None):
        if dropout and dropout > 0:
            if variables == None:
                var_filter = VariableFilter(theano_name_regex='linear.*input_')
                variables = var_filter(self.cg.variables)
            self.cg = apply_dropout(self.cg, variables, dropout)
            self.cost = self.cg.outputs[0]

    def apply_noise(self, noise, weights=None):
        if noise and noise > 0:
            if weights == None:
                weights = VariableFilter(roles=[WEIGHT])(self.cg.variables)
            self.cg = apply_noise(self.cg, weights, noise)
            self.cost = self.cg.outputs[0]

    def regularize_max_norm(self, max_norms, weights=None):
        if weights == None:
            weights = VariableFilter(roles=[WEIGHT])(self.cg.variables)
        self.step_rules.extend([Restrict(VariableClipping(max_norm, axis=0), [w])
                                for max_norm, w in zip(max_norms, weights)])

    def clip_gradient(self, threshold):
        """Add StepClipping rule

        :threshold: max norm allowed for gradients
        """
        self.step_rules.append(StepClipping(threshold))

    def decay_learning_rate(self, learning_rate_decay):
        """Decay learning rate after each epoch

        :learning_rate_decay: decay coeff.
        """
        if learning_rate_decay not in (0, 1):
            learning_rate = self.step_rules[0].learning_rate
            self.extensions.append(SharedVariableModifier(learning_rate,
                                                          lambda n, lr: numpy.cast[theano.config.floatX](
                                                              learning_rate_decay * lr),
                                                          after_epoch=True, after_batch=False))

    def plot_channels(self, channels, url_bokeh, **kwargs):
        self.extensions.append(plot.Plot(self.model_name, server_url=url_bokeh,
                                         channels=channels, **kwargs))

    def track_best(self, channel, save_path=None, choose_best=min):
        tracker = TrackTheBest(channel, choose_best=choose_best)
        self.extensions.append(tracker)
        if save_path:
            checkpoint = saveload.Checkpoint(
                save_path, after_training=False, use_cpickle=True)
            checkpoint.add_condition(["after_epoch"],
                                     predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
            self.extensions.append(checkpoint)

    def load_model(self, load_path):
        load_pre = saveload.Load(load_path)
        self.extensions.append(load_pre)

    def finish_after(self, nepochs):
        self.extensions.append(FinishAfter(after_n_epochs=nepochs))

    def add_printing(self, **kwargs):
        self.extensions.append(Printing(**kwargs))

    def add_timing(self, **kwargs):
        self.extensions.append(Timing(**kwargs))

    def get_monitored_var(self, var_name):
        idx = [n.name for n in self.monitored_vars].index(var_name)
        return self.monitored_vars[idx]

    def add_monitored_var(self, variable):
        self.monitored_vars.append(variable)

    def add_aggregation_norm_vars():
        gradient_norm = aggregation.mean(self.algorithm.total_gradient_norm)
        step_norm = aggregation.mean(self.algorithm.total_step_norm)
        grad_over_step = gradient_norm / step_norm
        grad_over_step.name = 'grad_over_step'
        self.monitored_vars.insert(0, gradient_norm)
        self.monitored_vars.insert(0, step_norm)
        self.monitored_vars.insert(0, grad_over_step)
        self.monitored_vars.insert(0, self.cost)

    def add_stream_monitors(self, **kwargs):
        train_monitor = TrainingDataMonitoring(self.monitored_vars,
                                               prefix='tra', **kwargs)
        self.extensions.insert(0, train_monitor)
        if self.dev_stream:
            dev_monitor = DataStreamMonitoring(variables=self.monitored_vars,
                                               data_stream=self.dev_stream, prefix="dev", **kwargs)
            self.extensions.insert(0, dev_monitor)

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = self.cg.parameters
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    def get_num_params(self):
        return sum([c.size for c in self.parameters]).eval()

    @property
    def algorithm(self):
        if self._algorithm is None:
            self._algorithm = GradientDescent(cost=self.cost,
                                              parameters=self.parameters,
                                              step_rule=CompositeRule(self.step_rules))
        return self._algorithm

    def get_main_loop(self):
        return MainLoop(data_stream=self.train_stream, algorithm=self.algorithm,
                        model=Model(self.cost), extensions=self.extensions)
