# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ******************************************************************************
# Copyright (c) 2018 Mejbah ul Alam, Justin Gottschlich, Abdullah Muzahid
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ******************************************************************************
"""plots.py

Plotting utilities for various aspects of AutoPerf.
"""

from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, \
    roc_curve, roc_auc_score

from autoperf.config import cfg

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16


def plot_loss_curves(train_loss: Sequence, val_loss: Sequence):
  """Plots the loss curves for an AutoPerf training run.

  Args:
      train_loss: Sequence of training losses.
      val_loss: Sequence of validation losses.
  """
  batch_size = len(train_loss) / len(val_loss)

  plt.figure(figsize=(6, 4))
  plt.plot(np.arange(len(train_loss)) / batch_size, train_loss, label='Training Loss', lw=3)
  plt.plot(np.arange(len(val_loss)), val_loss, label='Validation Loss', lw=3)
  plt.legend()
  plt.title(r'Training Loss Curves | $\eta$ = {}'.format(cfg.training.noise))
  plt.xlabel('Training Epochs')
  plt.ylabel(cfg.training.loss.replace('_', ' ').title())
  plt.show()


def plot_roc_curve(y_true: Sequence, y_pred: Sequence):
  """Plots the ROC curve for a given sequence of labels and predictions.

  Args:
      y_true: Ground truth labels.
      y_pred: Predicted labels from the classifier.
  """
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  auc = roc_auc_score(y_true, y_pred)
  with sns.axes_style('whitegrid'):
    plt.figure()
    plt.plot(fpr, tpr, lw=3, color='darkorange', label=f'AUC = {auc}')
    plt.plot([0, 1], [0, 1], color='navy', lw=3, alpha=0.5, linestyle='--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_pr_curve(y_true: Sequence, y_pred: Sequence):
  """Plots the PR curve for a given sequence of labels and predictions.

  Args:
      y_true: Ground truth labels.
      y_pred: Predicted labels from the classifier.
  """
  precision, recall, _ = precision_recall_curve(y_true, y_pred)
  ap = average_precision_score(y_true, y_pred)
  with sns.axes_style('whitegrid'):
    plt.figure()
    plt.plot(precision, recall, lw=3, color='darkorange', label=f'AP = {ap}')
    plt.plot([0, 1], [0.5, 0.5], color='navy', lw=3, alpha=0.5, linestyle='--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR Curve')
    plt.legend(loc='lower left')
    plt.show()


def text_to_str(texts: list) -> list:
  """Converts a list of 'Text' objects into a list of strings.

  Args:
      texts: A list of Text objects.

  Returns:
      list: A list of strings.
  """
  strings = [t.get_text() for t in texts]
  return strings


def plot_histograms(train: np.ndarray = None, test_nom: np.ndarray = None,
                    test_anom: np.ndarray = None, threshold: float = None,
                    flipped_axes: bool = False, fill: bool = False):
  """Plots a series of reconstruction MSE histograms

  Args:
      train (optional): Array of training sample reconstruction MSEs.
      test_nom (optional): Array of nominal test sample reconstruction MSEs.
      test_anom (optional): Array of anomolous test sample MSEs.
      threshold (optional): Anomolous threshold value, used to draw a dividing line.
      flipped_axes (optional): Whether or not to place anomalous MSEs below the x-axis.
      fill (optional): Alternative plot type that shows the proportion each bin contains of each label.
  """

  if train is test_nom is test_anom is None:
    raise ValueError('No data supplied for plotting.')

  if fill:
    flipped_axes = False

  combined_pd = pd.DataFrame({})
  cmap = []
  if train is not None:
    cmap.append('#6464fb')
    combined_pd = combined_pd.append(pd.DataFrame({'Result Type': 'Nominal (Train)', 'MSE': train}))

  if test_nom is not None:
    cmap.append('#00A896')
    combined_pd = combined_pd.append(pd.DataFrame({'Result Type': 'Nominal (Test)', 'MSE': test_nom}))

  if test_anom is not None and not flipped_axes:
    cmap.append('#f16868')
    combined_pd = combined_pd.append(pd.DataFrame({'Result Type': 'Anomalous (Test)', 'MSE': test_anom}))

  # Get the bounding box size
  dist_min = combined_pd['MSE'].min()
  dist_max = combined_pd['MSE'].max()
  if test_anom is not None:
    dist_min = np.min([dist_min, test_anom.max()])
    dist_max = np.max([dist_max, test_anom.max()])

  # Nicer plot style
  with sns.axes_style('whitegrid'):

    _, ax = plt.subplots(figsize=(9, 6))

    # Anomalous samples will be drawn below the x-axis, if specified
    old_texts, old_patches = None, None
    if (test_anom is not None) and (flipped_axes):

      anom_pd = pd.DataFrame({'Result Type': 'Anomalous (Test)', 'MSE': test_anom})
      hist = sns.histplot(ax=ax, data=anom_pd, x='MSE',                       # Provide data
                          legend=True, multiple='stack',                      # Arrangement
                          bins=50, binrange=[dist_min, dist_max],             # Bin setup
                          hue='Result Type', palette=['#f16868'], alpha=0.9,  # Color setup
                          linewidth=1.75, edgecolor='w')         # Thin bars

      for p in ax.patches:  # turn the histogram upside down
        p.set_height(-p.get_height())

      ax.axhline(0, c=(0.2, 0.2, 0.2), lw=1.5)
      old_texts, old_patches = hist.get_legend().get_texts(), hist.get_legend().get_patches()
      old_texts = text_to_str(old_texts)

    # Plot the combined pandas histogram
    multiple = 'fill' if fill else 'stack'
    combined_hist = sns.histplot(ax=ax, data=combined_pd, x='MSE',            # Provide data
                                 legend=True, multiple=multiple,              # Arrangement
                                 bins=50, binrange=[dist_min, dist_max],      # Bin setup
                                 hue='Result Type', palette=cmap, alpha=0.9,  # Color setup
                                 linewidth=1.75, edgecolor='w')  # Thin bars

    # Threshold dividing line
    thresh = None
    if threshold is not None:
      thresh = ax.axvline(threshold, ls='--', c='k', lw=3, label='Threshold')

    # Determine y-axis limits, with support for histogram stacking
    min_height, max_height = 0, 0
    pos, neg = {}, {}
    for p in ax.patches:
      if p.get_height() > 0:
        if p.get_x() in pos:
          pos[p.get_x()] += p.get_height()
        else:
          pos[p.get_x()] = p.get_height()
        max_height = pos[p.get_x()] if pos[p.get_x()] > max_height else max_height

      else:
        if p.get_x() in neg:
          neg[p.get_x()] -= p.get_height()
        else:
          neg[p.get_x()] = p.get_height()
        min_height = neg[p.get_x()] if neg[p.get_x()] < min_height else min_height

    # Set the y-axis limits appropriately, and remove the minus sign from any negative ticks
    if not fill:
      ax.set_ylim(min_height * 1.1, max_height * 1.1)
      ax.set_yticks(ax.get_yticks())       # suppresses a FixedLocator warning
      ax.set_yticklabels(ax.get_yticks())  # set the labels first
      ax.set_yticklabels([f'{np.abs(int(t.get_position()[1]))}' for t in ax.get_yticklabels()])

    # Remove the bounding box from the plt
    for sp in ax.spines.values():
      sp.set_visible(False)

    # Merge multiple histplot() legends, if present
    handles, labels = [], []

    if threshold is not None:
      handles.append(thresh)
      labels.append('Threshold')

    handles.extend(combined_hist.get_legend().get_patches())
    labels.extend(text_to_str(combined_hist.get_legend().get_texts()))

    if old_texts is not None and old_patches is not None:
      handles.extend(old_patches)
      labels.extend(old_texts)

    if fill:
      ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
      ax.legend(handles=handles, labels=labels)

    # Modify label paddings and set the title
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.set_title('Reconstruction Error Histogram', y=1.04)

    plt.tight_layout()
    plt.show()
