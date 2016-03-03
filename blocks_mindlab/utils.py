import numpy
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


def plot_learning_curve(train_scores, valid_scores, X_range, xlabel, ylabel, outfile):
    pyplot.figure()
    pyplot.plot(X_range, train_scores, 'b-', label='Train')
    pyplot.plot(X_range, valid_scores, 'g-', label='Valid')
    pyplot.gca().set_ylim(0.4, 1)
    pyplot.legend()
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.savefig(outfile)
    pyplot.close()
