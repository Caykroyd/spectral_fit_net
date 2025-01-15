class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode="min"):
        """
        Monitors a metric (e.g., validation loss or accuracy) during training and stops training if no improvement is observed 
        after a specified number of epochs (patience). Early stopping helps prevent overfitting by halting training once the 
        model's performance plateaus.

        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): "min" if lower values are better (e.g., loss), "max" if higher values are better (e.g., accuracy).

        Usage:
            Instantiate EarlyStopping and call it each epoch with the monitored metric (score). Access `early_stop` attribute to determine
            whether to halt training.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0  # Reset when there is an improvement
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        elif self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            raise ValueError("Mode must be 'min' or 'max'")