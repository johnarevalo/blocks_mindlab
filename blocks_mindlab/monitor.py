from blocks.monitoring.aggregation import MonitoredQuantity
from sklearn import metrics


class FScoreQuantity(MonitoredQuantity):

    def __init__(self, average='micro', threshold=0.5, **kwargs):
        self.average = average
        self.threshold = threshold
        super(FScoreQuantity, self).__init__(**kwargs)

    def initialize(self):
        self.total_f_score, self.examples_seen = 0.0, 0

    def aggregate(self, y, y_hat):
        self.total_f_score += metrics.f1_score(y, y_hat > self.threshold,
                                               average=self.average)
        self.examples_seen += 1

    def get_aggregated_value(self):
        res = self.total_f_score / self.examples_seen
        return res
