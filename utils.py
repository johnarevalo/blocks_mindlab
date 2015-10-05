import os.path
import yaml
from theano import tensor
from bokeh.plotting import output_server, cursession
from bokeh.document import Document

config = None


def get_config(filename=None, reload=False):
    global config
    if config is not None and not reload:
        return config
    if filename is None:
        raise Exception(
            'Configuration has not been loaded previously, filename parameter is required')
    with open(filename) as f:
        config = yaml.load(f)
    return config


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


def generate_url(docname, url_bokeh):
    output_server(docname, url=url_bokeh)
    s = cursession()
    s.use_doc(docname)
    d = Document()
    s.load_document(d)
    return s.object_link(d.context)
