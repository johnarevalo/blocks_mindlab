import copy
import numpy
from collections import OrderedDict
from fuel import config
from fuel.schemes import BatchScheme
from picklable_itertools import imap
from picklable_itertools.extras import partition_all, roundrobin


class PairwiseScheme(BatchScheme):

    def __init__(self, targets, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        labels = numpy.unique(targets)
        self.targets = OrderedDict()
        for l in labels:
            indices = numpy.array(targets == l)
            self.targets[l] = numpy.nonzero(indices)[0].tolist()
        super(PairwiseScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        targets = [copy.deepcopy(v) for v in self.targets.values()]
        # Build positive pairs
        positives = []
        for target in targets:
            numpy.random.shuffle(target)
            nsamples = int(len(target) / 2 + ((len(target) / 2) % 2))
            positives.extend(target[:nsamples])
            target[:nsamples] = []

        # Build negative pairs
        numpy.random.shuffle(targets)
        negatives = list(roundrobin(*targets))
        indices = numpy.asarray(positives + negatives)
        if len(indices) % 2 != 0:
            indices = indices[:-1]

        odds = numpy.arange(0, len(indices), 2)
        self.rng.shuffle(odds)
        batched_indices = []
        half_batch_size = int(self.batch_size / 2)
        for x in range(0, len(odds), half_batch_size):
            batched_indices.extend(indices[odds[x:x + half_batch_size]])
            batched_indices.extend(
                indices[odds[x:x + half_batch_size] + 1])

        return imap(list, partition_all(self.batch_size, batched_indices))
