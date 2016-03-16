import numpy
import operator
from matplotlib import pyplot
from sklearn import metrics
from theano import tensor


def sort_dict(dic, reverse=True, by_value=True):
    sort_by = 1 if by_value else 0
    return sorted(dic.items(), key=operator.itemgetter(sort_by), reverse=reverse)


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
    pyplot.title(title, fontsize='smaller')
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
