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
"""actions.py

Actions triggered by user input in __main__.py.
"""

import os
import sys
import shutil
import subprocess
import time
import logging
from tempfile import TemporaryFile

import numpy as np
from rich.progress import TimeElapsedColumn, BarColumn, Progress, track
from tqdm.rich import tqdm
from tensorflow.keras.utils import plot_model

from autoperf.autoperf import getPerfDataset, preprocessDataArray, getDatasetArray, \
    getReconstructionErrors, testModel, testAnomaly, writeTestLog
from autoperf import keras_autoencoder
from autoperf.config import cfg
from autoperf.counters import get_num_counters
from autoperf.utils import getAutoperfDir, set_working_directory
from autoperf.plots import plot_histograms

log = logging.getLogger('rich')


def runDetect(nominal_dir: str, anomalous_dir: str):
  """Detect performance anomalies in the current branch.

  Args:
      nominal_dir: Nominal data directory
      anomalous_dir: Anomalous data directory
  """
  autoencoder = keras_autoencoder.loadTrainedModel()

  nominalRuns = os.listdir(nominal_dir)
  nominalDataset = list()
  for run in track(nominalRuns, description='Loading Training Data', transient=True):
    datadir = os.path.join(nominal_dir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = preprocessDataArray(getDatasetArray(dataset))
    nominalDataset.extend(dataArray.tolist())
  nominalDataset = np.array(nominalDataset)

  anomalousRuns = os.listdir(anomalous_dir)
  anomalousDataset = list()
  for run in track(anomalousRuns, description='Loading Testing Data ', transient=True):
    datadir = os.path.join(anomalous_dir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = preprocessDataArray(getDatasetArray(dataset))
    anomalousDataset.extend(dataArray.tolist())
  anomalousDataset = np.array(anomalousDataset)

  nominalTestErrors = getReconstructionErrors(nominal_dir, autoencoder)
  anomalousTestErrors = getReconstructionErrors(anomalous_dir, autoencoder)
  thresholdError = np.load(getAutoperfDir('threshold.npy'))[0]

  log.info('%d NOM reconstruction errors above threshold', np.sum(nominalTestErrors > thresholdError))
  log.info('%d ANOM reconstruction errors above threshold', np.sum(anomalousTestErrors > thresholdError))

  plot_histograms(test_nom=np.array(nominalTestErrors),
                  test_anom=np.array(anomalousTestErrors),
                  threshold=thresholdError,
                  flipped_axes=True)

  keras_autoencoder.visualizeLatentSpace(autoencoder, nominalDataset, anomalousDataset)


def runEvaluate(train_dir: str, nominal_dir: str, anomalous_dir: str):
  """Evaluate the trained autoencoder with various datasets.

  Args:
      train_dir: Training data directory
      nominal_dir: Nominal data directory
      anomalous_dir: Anomalous data directory
  """
  autoencoder = keras_autoencoder.loadTrainedModel()

  trainRuns = os.listdir(train_dir)
  trainDataset = list()
  for run in trainRuns:
    datadir = os.path.join(train_dir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = preprocessDataArray(getDatasetArray(dataset))
    trainDataset.extend(dataArray.tolist())
  trainDataset = np.array(trainDataset)

  nominalRuns = os.listdir(nominal_dir)

  nominalDataset = list()
  for run in nominalRuns:
    datadir = os.path.join(nominal_dir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = preprocessDataArray(getDatasetArray(dataset))
    nominalDataset.extend(dataArray.tolist())
  nominalDataset = np.array(nominalDataset)

  anomalousRuns = os.listdir(anomalous_dir)
  anomalousDataset = list()
  for run in anomalousRuns:
    datadir = os.path.join(anomalous_dir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = preprocessDataArray(getDatasetArray(dataset))
    anomalousDataset.extend(dataArray.tolist())
  anomalousDataset = np.array(anomalousDataset)

  log.info('Collecting reconstruction errors for [training] data:')
  trainTestErrors = getReconstructionErrors(train_dir, autoencoder)
  log.info('Collecting reconstruction errors for [nominal test] data:')
  nominalTestErrors = getReconstructionErrors(nominal_dir, autoencoder)
  log.info('Collecting reconstruction errors for [anomalous test] data:')
  anomalousTestErrors = getReconstructionErrors(anomalous_dir, autoencoder)

  thresholdError = np.load(getAutoperfDir('threshold.npy'))[0]

  plot_histograms(train=np.array(trainTestErrors),
                  test_nom=np.array(nominalTestErrors),
                  test_anom=np.array(anomalousTestErrors),
                  threshold=thresholdError)

  with open(getAutoperfDir('report'), 'w') as report:
    test_result = testModel(autoencoder, thresholdError, nominal_dir, anomalous_dir, report)

    imageFile = getAutoperfDir('model_architecture.png')
    plot_model(autoencoder, to_file=imageFile, show_shapes=True, show_layer_names=True)

    print('\nConfusion Matrix: |  P  |  N  |', file=report)
    print('                P |{:^5}|{:^5}|'.format(test_result.true_positive, test_result.false_positive), file=report)
    print('                N |{:^5}|{:^5}|'.format(test_result.false_negative, test_result.true_negative), file=report)

    # calculate F score
    precision = 0
    if test_result.true_positive + test_result.false_positive != 0:
      precision = test_result.true_positive / (test_result.true_positive + test_result.false_positive)

    recall = test_result.true_positive / (test_result.true_positive + test_result.false_negative)
    fscore = 0
    if precision + recall != 0:
      fscore = 2 * (precision * recall) / (precision + recall)  # harmonic mean of precision and recall

    print("\nPrecision: ", precision, file=report)
    print("Recall: ", recall, file=report)
    print("Fscore: ", fscore, file=report)


def runClean():
  """Cleans the codebase using the user-specified instructions."""
  with set_working_directory(cfg.build.dir), TemporaryFile('w+') as file:
    with Progress('[bright_magenta][progress.description]{task.description}[/bright_magenta]',
                  BarColumn(), '[', TimeElapsedColumn(), ']', transient=True) as progress:
      task = progress.add_task('Cleaning', start=False)
      p = subprocess.Popen([cfg.clean.cmd], env=os.environ.copy(), shell=True,
                           stdout=file, stderr=subprocess.STDOUT)
      if p.wait() != 0:
        log.error('Clean command failed.')
        file.seek(0)
        print(file.read())
        sys.exit(1)
      progress.update(task, total=0, start=True, description='Cleaning (Finished)')
      progress.start_task(task)


def runBuild():
  """Builds the code using the user-specified instructions."""
  with set_working_directory(cfg.build.dir), TemporaryFile('w+') as file:
    if cfg.clean.cmd:
      runClean()
    with Progress('[bright_magenta][progress.description]{task.description}[/bright_magenta]',
                  BarColumn(), '[', TimeElapsedColumn(), ']') as progress:
      task = progress.add_task('Building', start=False)
      p = subprocess.Popen([cfg.build.cmd], env=os.environ.copy(), shell=True,
                           stdout=file, stderr=subprocess.STDOUT)
      if p.wait() != 0:
        log.error('Build command failed.')
        file.seek(0)
        print(file.read())
        sys.exit(1)
      progress.update(task, total=0, start=True, description='Building (Finished)')
      progress.start_task(task)


def runWorkload(out_dir: str, run_count: int):
  """Runs a specified workload and collects HPC measurements.

  Args:
      out_dir: Directory in which to save the measurements
      run_count: Run index (used for file names)
  """
  DEFAULT_OUTFILE_NAME = "perf_data.csv"

  currOutputDirName = f'{out_dir}/run_{run_count}'
  os.makedirs(currOutputDirName, exist_ok=True)

  with set_working_directory(cfg.workload.dir):

    shutil.copyfile(getAutoperfDir('COUNTERS'), 'COUNTERS')

    log.info('Saving hardware telemetry data to %s', currOutputDirName)
    for i in tqdm(range(0, get_num_counters())):
      os.environ["PERFPOINT_EVENT_INDEX"] = str(i)

      start_time = time.perf_counter()
      p = subprocess.Popen([cfg.workload.cmd], env=os.environ.copy(), shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
      p.wait()
      end_time = time.perf_counter()

      # copy data to a new file
      if os.path.exists(DEFAULT_OUTFILE_NAME):
        new_out_filename = f'{currOutputDirName}/event_{i}_{DEFAULT_OUTFILE_NAME}'
        # log.info('Copying data from %s to %s', DEFAULT_OUTFILE_NAME, new_out_filename)
        out_file = open(new_out_filename, 'w')
        in_file = open(DEFAULT_OUTFILE_NAME, 'r')

        out_file.write(f'INPUT: {cfg.workload.cmd}\n')
        out_file.write(f'TIME: {end_time - start_time}\n')

        for line in in_file.readlines():
          out_file.write(line)

        in_file.close()
        try:
          os.remove(DEFAULT_OUTFILE_NAME)
        except Exception:
          ...

        out_file.close()

      else:
        log.error('%s not generated... Something went wrong.', DEFAULT_OUTFILE_NAME)

    os.unlink('COUNTERS')


def runReport(directory):
  """Generates a performance regression report using a trained model.

  Args:
      directory: Path containing a series of AutoPerf runs
  """
  model = keras_autoencoder.loadTrainedModel()
  with open(getAutoperfDir('report'), 'w') as logFile:
    runs = os.listdir(directory)
    anomaly_summary, _ = testAnomaly(model, directory, runs, np.load(getAutoperfDir('threshold.npy')))
    writeTestLog(logFile, anomaly_summary)
