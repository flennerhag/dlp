"""
Visualization function for diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_train_scores(trainer, burn_in=0, figsize=(12, 5), title=None):
    """Plot loss, train and test score and the gradient norm.

    Arg
        trainer (obj): fitted trainer object instance.
        burn_in (int, float): share of iterations (from 0) to skip
        figsize (tuple): figure size.
    """
    gradient_norm = trainer.norms[-1]["grad"]
    train_loss = trainer.loss
    train_score = trainer.train_score
    test_score = trainer.test_score

    stop = len(train_loss)
    start = int(np.floor(stop * burn_in))
    ival = trainer.eval_ival

    f, ax = plt.subplots(1, 3, figsize=figsize, sharex=True)

    if title is not None:
        f.suptitle(title)

    ax[0].set_title("Score")
    ax[0].plot(range(start, stop, ival),
               train_score[int(burn_in/ival):],
               linewidth=0.5, color='k', label="train")

    ax[0].plot(range(start, stop, ival),
               test_score[int(burn_in/ival):],
               linewidth=0.5, color='r', label="test")

    ax[0].legend(frameon=False)

    ax[1].set_title("Loss")
    ax[1].plot(range(start, stop), train_loss[burn_in:],
               linewidth=0.5, color='k')

    ax[2].set_title("Gradient norm")
    ax[2].plot(range(start, stop), gradient_norm[burn_in:],
               linewidth=0.5, color='k')

    return ax
