from blocks.algorithms import GradientDescent, CompositeRule, Momentum, Restrict, VariableClipping, RMSProp, Adam
from blocks.extensions import FinishAfter, saveload, predicates
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.training import TrackTheBest
from blocks.monitoring import aggregation
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.initialization import Uniform, Constant
from blocks.model import Model
from blocks import main_loop
from blocks.roles import INPUT, WEIGHT
from blocks_mindlab import utils, plot
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []


class Experiment(object):

    def __init__(self, model_name, train_stream, dev_stream):
        self.model_name = model_name
        self.monitored_vars = []
        self.extensions = []
        self.step_rules = []
        self.train_stream=train_stream
        self.dev_stream=dev_stream

    def set_momentum(self, learning_rate, momentum):
        self.step_rules.append(Momentum(learning_rate=learning_rate, momentum=momentum))

    def set_rmsprop(self, learning_rate, decay_rate=0.95):
        self.step_rules.append(RMSProp(learning_rate=learning_rate, decay_rate=decay_rate))

    def set_adam(self):
        self.step_rules.append(Adam())

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
        self.cg = ComputationGraph(cost)

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
            owner.add_auxiliary_variable(var, name='W_max_norm_weight_' + str(i))
            self.monitored_vars.append(var)

    def monitor_activations(self, mlp):
        var_filter = VariableFilter(theano_name_regex='linear.*output')
        outputs = var_filter(self.cg.variables)
        for i, output in enumerate(outputs):
            mlp.add_auxiliary_variable(output.mean(), name='mean_act_' + str(i))
            mlp.add_auxiliary_variable(output.mean(axis=0).max(), name='max_act_' + str(i))
            mlp.add_auxiliary_variable(output.mean(axis=0).min(), name='min_act_' + str(i))
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

    def plot_channels(self, channels, url_bokeh, **kwargs):
        print '{0} : {1}'.format(self.model_name, utils.generate_url(self.model_name, url_bokeh))
        self.extensions.append(plot.Plot(self.model_name, server_url=url_bokeh,
                               channels=channels, before_first_epoch=True,
                               **kwargs))

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

    def get_main_loop(self):
        algorithm = GradientDescent(cost=self.cost, parameters=self.cg.parameters,
                                    step_rule=CompositeRule(self.step_rules))

        self.monitored_vars.insert(0, self.cost)
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        grad_over_step = gradient_norm / step_norm
        grad_over_step.name = 'grad_over_step'
        self.monitored_vars.insert(0, gradient_norm)
        self.monitored_vars.insert(0, step_norm)
        self.monitored_vars.insert(0, grad_over_step)

        if self.dev_stream:
            dev_monitor = DataStreamMonitoring(variables=self.monitored_vars,
                                           before_first_epoch=True, after_epoch=True,
                                           data_stream=self.dev_stream, prefix="dev")
            self.extensions.insert(0, dev_monitor)
        train_monitor = TrainingDataMonitoring(self.monitored_vars,
                                               before_first_epoch=True,
                                               after_batch=True, prefix='tra')
        self.extensions.insert(0, train_monitor)
        self.algorithm = algorithm

        return MainLoop(data_stream=self.train_stream, algorithm=algorithm,
                        model=Model(self.cost), extensions=self.extensions)

    def get_monitored_var(self, var_name):
        idx = [n.name for n in self.monitored_vars].index(var_name)
        return self.monitored_vars[idx]
