import nltk
import numpy
from fuel.transformers import AgnosticSourcewiseTransformer, Transformer
from .utils import sort_dict


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
            half_batch = int(source_batch.shape[0] / 2)
            first_batch = source_batch[0:half_batch]
            second_batch = source_batch[half_batch:]
            if source == self.target_source:
                targets = numpy.equal(first_batch, second_batch)
                batches.append(targets)
            else:
                batches.extend([first_batch, second_batch])
        return tuple(batches)


class SentTokenizer(Transformer):

    def __init__(self, word_to_ix, ix_to_word, data_stream, which_source, **kwargs):
        if not data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')

        self.which_source = which_source
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        super(SentTokenizer, self).__init__(data_stream, produces_examples=True,
                                            **kwargs)

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source == self.which_source:
                sources.append(source + '_mask')
        return tuple(sources)

    def transform_example(self, source_example):
        example_with_mask = []
        for source, example in zip(self.data_stream.sources, source_example):
            if source != self.which_source:
                example_with_mask.append(example)
                continue
            sentences = nltk.sent_tokenize(
                ' '.join([self.ix_to_word[ix] for ix in example]))
            sentences = [[self.word_to_ix[w] for w in sent.split()]
                         for sent in sentences]
            max_length = max([len(s) for s in sentences])
            batch = numpy.zeros((len(sentences), max_length), dtype='float32')
            mask = numpy.zeros((len(sentences), max_length), dtype='float32')
            for i, s in enumerate(sentences):
                batch[i, :len(s)] = s
                mask[i, :len(s)] = 1
            example_with_mask.append(batch)
            example_with_mask.append(mask)
        return tuple(example_with_mask)
