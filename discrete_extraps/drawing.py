"""
This module contains utility code for drawing data
"""
import math

import numpy as np
import matplotlib.pyplot as plt

def draw_layer(ax, data, title=None, **kwargs):
    """Draw one detector layer as an image"""
    ax.imshow(data.T, interpolation='none', aspect='auto',
              origin='lower', **kwargs)
    if title is not None:
        ax.set_title(title)

def draw_layers(event, ncols=5, truthx=None, truthy=None, figsize=(12,5)):
    """Draw each detector layer as a grid of images"""
    num_det_layers = event.shape[0]
    nrows = math.ceil(float(num_det_layers)/ncols)
    plt.figure(figsize=figsize)
    for ilay in range(num_det_layers):
        ax = plt.subplot(nrows, ncols, ilay+1)
        title = 'layer %i' % ilay
        draw_layer(ax, event[ilay], title=title)
        ax.autoscale(False)
        if truthx is not None and truthy is not None:
            ax.plot(truthx[ilay]-0.5, truthy[ilay]-0.5, 'w+')
    plt.tight_layout()

def draw_projections(event, truthx=None, truthy=None, figsize=(12,5)):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    kwargs = dict(interpolation='none',
                  aspect='auto',
                  origin='lower')
    plt.imshow(event.sum(axis=1).T, **kwargs)
    plt.xlabel('detector layer')
    plt.ylabel('pixel')
    plt.autoscale(False)
    if truthy is not None:
        plt.plot(np.arange(event.shape[0]-0.5), truthy-0.5, 'w-')
    plt.subplot(122)
    plt.imshow(event.sum(axis=2).T, **kwargs)
    plt.xlabel('detector layer')
    plt.ylabel('pixel')
    plt.tight_layout()
    plt.autoscale(False)
    if truthx is not None:
        plt.plot(np.arange(event.shape[0]-0.5), truthx-0.5, 'w-')

def draw_1d_event(event, title=None, mask_ranges=None, tight=True, **kwargs):
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

def draw_train_history(history, figsize=(12,5)):
    """Make plots of training and validation losses and accuracies"""
    plt.figure(figsize=figsize)
    # Plot loss
    plt.subplot(121)
    plt.plot(history.epoch, history.history['loss'], label='Training set')
    plt.plot(history.epoch, history.history['val_loss'], label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.epoch, history.history['acc'], label='Training set')
    plt.plot(history.epoch, history.history['val_acc'], label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.title('Training accuracy')
    plt.legend(loc=0)
    plt.tight_layout()