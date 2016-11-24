"""
This module contains utility code for drawing data
"""

import matplotlib.pyplot as plt

def draw_event(event, title=None, mask_ranges=None, tight=True, **kwargs):
    """
    Draw and format one 1D detector event with matplotlib.
    Params:
        event: data for one event in image format
        title: plot title
        mask_range: tuple of arrays, (lower, upper) defining a detector
            mask envelope that will be drawn on the display
        kwargs: additional keywords passed to pyplot.plot
    """
    plt.imshow(event.T, interpolation='none', aspect='auto',
               origin='lower', **kwargs)
    if title is not None:
        plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Pixel')
    plt.autoscale(False)
    if tight:
        plt.tight_layout()
    if mask_ranges is not None:
        plt.plot(mask_ranges[:,0], 'w:')
        plt.plot(mask_ranges[:,1], 'w:')

def draw_input_and_pred(event_input, event_pred, figsize=(9,4), mask_ranges=None):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    draw_event(event_input, title='Input', mask_ranges=mask_ranges)
    plt.subplot(122)
    draw_event(event_pred, title='Model prediction', mask_ranges=mask_ranges)

def draw_train_history(history, figsize=(9,4)):
    """Make plots of training and validation losses and accuracies"""
    plt.figure(figsize=figsize)
    # Plot loss
    plt.subplot(121)
    plt.plot(history.epoch, history.history['loss'], label='Training set')
    plt.plot(history.epoch, history.history['val_loss'], label='Validation set')
    plt.xlabel('Training epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot accuracy
    # No global accuracy available in multi-output data
    #plt.subplot(122)
    #plt.plot(history.epoch, history.history[