"""Calculate Word accuracy.
"""

def WordAccuracy(preds, targets):
    denominator = len(targets)
    numerator = 0

    for pred, target in zip(preds, targets):
        if pred == target:
            numerator += 1

    return numerator, denominator

class WordAccuracyStats:

    def __init__(self):
        self.correct = 0
        self.total = 0

    def append(self, preds, targets):
        """This function is for updating the stats according to the prediction
        and target in the current batch.

        Arguments
        ----------
        log_probabilities : tensor
            Predicted log probabilities (batch_size, time, feature).
        targets : tensor
            Target (batch_size, time).
        length: tensor
            Length of target (batch_size,).
        """
        numerator, denominator = WordAccuracy(preds, targets)
        self.correct += numerator
        self.total += denominator

    def summarize(self):
        """Computes the accuract metric."""
        return self.correct / self.total
