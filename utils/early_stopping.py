class EarlyStopper:
    """
    It early stops the training if the accuracy of a metric (to be defned by us)
    doesn't improve over some epochs. For the efficiency and speed, it's better to check
    image level metrics such as Image AUROC, or F1.

    ...

    Attributes
    ----------
    patience : int
        how long to watch for the improvement
    min_delta : float
        minimum expected improvement
    """
    def __init__(self, patience=3, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float("-inf")

    def check_improvement(self, current_metric):
        if current_metric > (self.best_metric + self.min_delta):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience