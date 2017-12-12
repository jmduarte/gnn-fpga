"""
This module contains the Estimator class implementation which provides
code for doing the training of a PyTorch model.
"""

from __future__ import print_function

import logging
from timeit import default_timer as timer

import numpy as np

import torch

class Estimator(object):
    """Estimator class"""
    
    def __init__(self, model, loss_func, opt='Adam', cuda=False):
        
        self.model = model
        if cuda:
            self.model.cuda()
        self.loss_func = loss_func
        if opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        
        logging.info('Model: \n%s' % model)
        logging.info('Parameters: %i' %
                     sum(param.numel() for param in model.parameters()))
    
    def training_step(self, inputs, targets):
        """Applies single optimization step on batch"""
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def fit(self, train_input, train_target, batch_size=32, n_epochs=1,
            valid_input=None, valid_target=None):
        """Runs batch training for a number of specified epochs."""
        n_samples = train_input.size(0)
        n_batches = (n_samples + batch_size - 1) // batch_size
        logging.info('Training samples: %i' % n_samples)
        logging.info('Batches per epoch: %i' % n_batches)
        if valid_input is not None:
            logging.info('Validation samples: %i' % valid_input.size(0))

        batch_idxs = np.arange(0, n_samples, batch_size)
        self.train_losses, self.valid_losses = [], []

        for i in range(n_epochs):
            logging.info('Epoch %i' % i)
            start_time = timer()
            sum_loss = 0

            for j in batch_idxs:
                # TODO: add support for more customized batching
                batch_input = train_input[j:j+batch_size]
                batch_target = train_target[j:j+batch_size]
                loss = self.training_step(batch_input, batch_target)
                sum_loss += loss.cpu().data[0]

            end_time = timer()
            avg_loss = sum_loss / n_batches
            self.train_losses.append(avg_loss)
            logging.info('  training loss %.3g time %gs' %
                         (avg_loss, (end_time - start_time)))
            
            # Evaluate the model on the validation set
            if (valid_input is not None) and (valid_target is not None):
                valid_loss = (self.loss_func(self.model(valid_input), valid_target)
                              .cpu().data[0])
                self.valid_losses.append(valid_loss)
                logging.info('  validate loss %.3g' % valid_loss)
