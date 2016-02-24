import numpy
from fuel.transformers import AgnosticSourcewiseTransformer


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


class SequenceTransposer(AgnosticSourcewiseTransformer):

    def __init__(self, data_stream, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(SequenceTransposer, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        if source_data.ndim == 2:
            return source_data.T
        elif source_data.ndim == 3:
            return source_data.transpose(1, 0, 2)
        else:
            raise ValueError('Invalid dimensions of this source.')
