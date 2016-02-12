import numpy
import os.path
import yaml
from theano import tensor
from bokeh.plotting import output_server, cursession
from bokeh.document import Document
from fuel.transformers import AgnosticSourcewiseTransformer


def get_measures(y_true, y_pred):
    tp = (tensor.eq(y_pred + y_true, 2)).sum()
    tn = (tensor.eq(y_pred + y_true, 0)).sum()
    fp = (tensor.eq(y_pred - y_true, 1)).sum()
    fn = (tensor.eq(y_true - y_pred, 1)).sum()
    f_score = 2. * tp / (2. * tp + fp + fn)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    pre.name = 'pre'
    rec.name = 'rec'
    tp.name = 'tp'
    tn.name = 'tn'
    fp.name = 'fp'
    fn.name = 'fn'
    f_score.name = 'f_score'
    return tp, tn, fp, fn, pre, rec, f_score


class OneHotTransformer(AgnosticSourcewiseTransformer):

    def __init__(self, data_stream, nclasses, **kwargs):
        self.nclasses = nclasses
        self.I = numpy.eye(self.nclasses, dtype='int32')
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(OneHotTransformer, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        return self.I[source_data]
