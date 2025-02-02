import numpy as np


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling.

    This class implements a learning rate scheduler for use with optimizers.
    The learning rate is scheduled based on the current training step and the
    number of warmup steps, using a scaling method that adjusts the learning rate.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam, AdamW) to apply the learning rate schedule to.
        d_model (int): The model dimension, used to calculate the initial learning rate.
        n_warmup_steps (int): Number of warmup steps for the learning rate schedule.
    """

    def __init__(self, optimizer, d_model, n_warmup_steps):
        """Initializes the ScheduledOptim object.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to wrap for learning rate scheduling.
            d_model (int): Model dimension to compute the initial learning rate.
            n_warmup_steps (int): Number of warmup steps.
        """
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Performs a step with the optimizer and updates the learning rate.

        This method calls the inner optimizer's step function after updating the
        learning rate according to the scheduled rule.
        """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zeroes out the gradients of the optimizer.

        This method calls the inner optimizer's zero_grad function to reset the gradients
        of all model parameters.
        """
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        """Calculates the learning rate scaling factor.

        The scaling is based on the current step number and the number of warmup steps.
        The learning rate decreases as the number of steps increases.

        Returns:
            float: The calculated learning rate scaling factor.
        """
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """Updates the learning rate according to the scheduled rule.

        The learning rate is computed at each step based on the scaling factor,
        and it is applied to the optimizer's parameter groups.
        """
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
