import numpy
from fuel.transformers import AgnosticSourcewiseTransformer
from matplotlib import pyplot
from theano import tensor
from sklearn import metrics


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


def plot_confusion_matrix(y_true, y_pred, target_names, outfile,
                          title='Confusion matrix', cmap=pyplot.cm.Blues):
    acc = metrics.accuracy_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    pyplot.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    for i, cas in enumerate(cm):
        for j, c in enumerate(cas):
            if c > 0:
                pyplot.text(j - .2, i + .2, c, fontsize=14)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = numpy.arange(len(target_names))
    pyplot.xticks(tick_marks, target_names, rotation=90)
    pyplot.yticks(tick_marks, target_names)
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.savefig(outfile)
    pyplot.close()
