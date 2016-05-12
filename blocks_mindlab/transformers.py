import numpy
from fuel.transformers import AgnosticSourcewiseTransformer, Transformer


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


class PairwiseTransformer(Transformer):

    def __init__(self, data_stream, target_source, **kwargs):
        super(PairwiseTransformer, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.target_source = target_source

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            if source != self.target_source:
                sources.append(source + '_1')
                sources.append(source + '_2')
            else:
                sources.append(source)
        return tuple(sources)

    def transform_batch(self, batch):
        batches = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source_batch.shape[0] % 2 != 0:
                source_batch = source_batch[:-1]
            half_batch = source_batch.shape[0] / 2
            first_batch = source_batch[0:half_batch]
            second_batch = source_batch[half_batch:]
            if source == self.target_source:
                targets = numpy.equal(first_batch, second_batch)
                batches.append(targets)
            else:
                batches.extend([first_batch, second_batch])
        return tuple(batches)
