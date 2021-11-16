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
"""autoperf.py

Main script for training and testing AutoPerf models.
"""

import sys
import os
import logging
from collections import namedtuple
from typing import Tuple, TextIO

import numpy as np
from tensorflow.keras.models import Model
from rich.progress import track

from autoperf import keras_autoencoder, plots
from autoperf.config import cfg
from autoperf.counters import get_num_counters
from autoperf.descriptions import get_description
from autoperf.utils import compareDataset, getAutoperfDir

log = logging.getLogger('rich')

"""Definition of tuples used by this script."""
AnomalyTuple = namedtuple('AnomalyTuple', 'run, sample_count, anomalous_sample_count, ranking')
AccuracyTuple = namedtuple('AccuracyTuple', 'true_positive, false_negative, false_positive, true_negative')


def getPerfDataset(dirName: str, numberOfCounters: int) -> Tuple[list, list]:
  """Collects and parses all `event_*_perf_data.csv` files in `dirName`.

  Args:
      dirName: A path containing `event_*_perf_data.csv` files.
      numberOfCounters: How many files are present.

  Returns:
      A list of headers.
      A list of profile counter results.
  """
  datasetHeader = []
  dataset = []
  eventID = 0
  initialized = False
  notFound = []
  for i in range(0, numberOfCounters):

    filename = dirName + "/event_" + str(eventID) + "_perf_data.csv"
    eventID += 1
    try:
      with open(filename, 'r') as fp:

        containsHeader = False
        containsData = False
        for linenumber, line in enumerate(fp):

          if linenumber == 2:
            containsHeader = True
            # last one is the counter, 1 and 2  is thd id and instcouunt , 0 is mark id
            headers = line.strip().split(",")
            datasetHeader.append(headers[-1])

          elif linenumber > 2:
            containsData = True

            perfCounters = line.strip().split(",")
            # mark = int(perfCounters[0])  # unused
            threadCount = int(perfCounters[1])
            instructionCount = int(perfCounters[2])
            currCounter = int(perfCounters[3])

            normalizedCounter = currCounter / (instructionCount * threadCount)
            normalizedCounter *= cfg.training.scale_factor
            if not initialized:
              dataset.append([normalizedCounter])
            else:
              dataset[linenumber - 3].append(normalizedCounter)

        if not containsHeader:
          datasetHeader.append('NO_HEADER_FOUND')

        if not containsData:
          notFound.append(i)

        if containsHeader and containsData:
          initialized = True

    except FileNotFoundError:
      log.error('%s not found.', filename)
      datasetHeader.append('FILE_NOT_FOUND')
      notFound.append(i)

  # cleanly handle cases where .csv files are not found
  for nf in notFound:
    for d in dataset:
      d.insert(nf, 0.0)

  return datasetHeader, dataset


def getDatasetArray(dataset: list, datatype: str = 'float32') -> np.ndarray:
  """Converts a list to a numpy array.

  Args:
      dataset: The input list.
      datatype: The requested type of array values.

  Returns:
      The output numpy array.
  """

  dataArray = np.array(dataset, dtype=datatype)
  return dataArray


def getMSE(old: np.ndarray, new: np.ndarray) -> float:
  """Calculates the mean squared error (MSE) between two numpy arrays.

  Args:
      old: Original inputs.
      new: Reconstructed inputs.

  Returns:
      The mean squared error.
  """
  squaredError = ((old - new) ** 2)
  return np.mean(squaredError)


def rankAnomalousPoint(sampleErrors: list, rankingMap: dict) -> dict:
  """Adds a new list of sample errors to the majority voting map.

  Args:
      sampleErrors: A list of `(name, distance)` tuples.
      rankingMap: A mapping between counter names and vote count lists.

  Returns:
      A updated rankingMap.
  """
  sampleErrors.sort(key=lambda tup: tup[1], reverse=True)
  for index, errorTuple in enumerate(sampleErrors):
    rankingMap[errorTuple[0]][index] += 1

  return rankingMap


def reportRanks(rankingMap: dict) -> list:
  """Reports a counter ranking based on the majority voting result.

  Args:
      rankingMap: A mapping between counter names and vote count lists.

  Returns:
      A list of sorted counters, ranked by level of abnormality.
  """
  anomalousCounterRank = []
  for i in range(get_num_counters()):  # for each possible ranks
    maxVote = 0
    maxKey = "None"
    for key in rankingMap:
      if rankingMap[key][i] > maxVote:
        maxVote = rankingMap[key][i]
        maxKey = key

    anomalousCounterRank.append(maxKey)

  return anomalousCounterRank


def detectAnomalyPoints(realData: list, predictedData: list, datasetHeader: list,
                        thresholdLoss: float) -> Tuple[int, int, list]:
  """Detects anomalous points via reconstruction loss and majority voting.

  Args:
      realData: Original data.
      predictedData: Reconstructed data.
      datasetHeader: Headers taken from original data.
      thresholdLoss: Minimum reconstruction error to be considered anomalous.

  Returns:
      The length of the original dataset.
      The number of anomalies detected.
      A list of anomalies ranked by severity.
  """
  datasetLen = realData.shape[0]
  dataLen = realData.shape[1]
  anomalyCount = 0

  """
  map for ranking:
  key = counter name, val = list of len(NUMBER_OF_COUNTERS)
  each pos corresponds to rank, smaller is better
  """
  rankingMap = {}
  for counterName in datasetHeader:
    rankingMap[counterName] = [0] * get_num_counters()

  reconstructErrorList = []
  for x in range(datasetLen):
    reconstructionError = getMSE(realData[x], predictedData[x])
    reconstructErrorList.append(reconstructionError)
    if reconstructionError > thresholdLoss:
      anomalyCount += 1
      errorList = []  # for ranking
      for y in range(0, dataLen):
        dist = 1.0
        if realData[x][y] != 0:
          dist = abs(predictedData[x][y] - realData[x][y]) / realData[x][y]
        # collect errors and counter tuple
        errorList.append((datasetHeader[y], dist))
      # update ranking
      rankingMap = rankAnomalousPoint(errorList, rankingMap)

  votingResult = reportRanks(rankingMap)

  return datasetLen, anomalyCount, votingResult


def testAnomaly(model: Model, testDataDir: str, runs: list,
                thresholdError: float) -> list:
  """Collects AnomalyTuples from test run inferences in the given directory.

  Args:
      model: A Keras autoencoder.
      perfTestDataDir: A directory holding performance data.
      runs: A list of `run_*` directories.
      thresholdError: Minimum reconstruction error to be considered anomalous.

  Returns:
      A list of AnomalyTuples.
  """
  test_summary = []
  pcts = []

  runs = sorted(runs, key=lambda r: int(''.join(filter(str.isdigit, r))))

  for run in runs:
    datadir = os.path.join(testDataDir, run)
    datasetHeader, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)
    decoded_data = keras_autoencoder.predict(model, dataArray)
    datasetLen, anomalyCount, ranking = detectAnomalyPoints(dataArray, decoded_data, datasetHeader, thresholdError)
    log.info('Run [%d] anomalous samples: %d / %d - %.2f%%',
             int(run.split('_')[-1]), anomalyCount, datasetLen, 100 * anomalyCount / datasetLen)
    if anomalyCount > datasetLen * cfg.detection.threshold:
      test_summary.append(AnomalyTuple(run=run, anomalous_sample_count=anomalyCount,
                                       sample_count=datasetLen, ranking=ranking))
    pcts.append(anomalyCount / datasetLen)

  return test_summary, pcts


def preprocessDataArray(dataset: np.ndarray) -> np.ndarray:
  """Preprocesses the dataset; normalizes by max in each column.

  Args:
      dataset: Input data.

  Returns:
      Preprocessed data.
  """
  # zero centering
  # mean_vector = np.mean(dataset, axis=0)
  # processed_dataset = dataset - mean_vector

  # normalize
  # dataset /= np.std(dataset, axis=0)

  # normalize with max in each column : not working some 0 values?? why not avoid zeros
  datasetMax = np.max(dataset, axis=0)
  processed_dataset = np.nan_to_num(np.true_divide(dataset, datasetMax))

  return processed_dataset


def getTrainDataSequence(dataDir: str) -> list:
  """Aggregates data files from all profile runs.

  Args:
      dataDir: A directory holding performance data.

  Returns:
      A list of `event_*_perf_data.csv` files.

  TODO:
      Filter the `runs` list to only contain valid CSV files.
  """
  runs = os.listdir(dataDir)  # no sequence of directory assigned, it should not matter

  log.info('Total execution/directory found for training data: %d', len(runs))
  return runs


def analyzeVariationInData(dataDir: str, testDir: str = None,
                           validationDir: str = None) -> Tuple[list, list, list]:
  """Analyzes the variation of the dataset.

  Args:
      dataDir: A directory holding performance data.
      testDir (optional): A directory holding test data.
      validationDir (optional): A directroy holding validation data.

  Returns:
      A list of sorted runs, ranked by amount of deviation.
      A list of test dataset deviations.
      A list of validation dataset deviations.
  """
  runs = os.listdir(dataDir)
  results = []
  baseDataset = None
  for counter, run in enumerate(runs):
    datadir = os.path.join(dataDir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    if counter == 0:
      baseDataset = getDatasetArray(dataset)
      results.append((run, 0))
    else:
      if counter < 50:
        results.append((run, compareDataset(baseDataset, getDatasetArray(dataset))))

  testResults = []
  if testDir is not None:
    runs = os.listdir(testDir)
  for counter, run in enumerate(runs):
    datadir = os.path.join(dataDir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    testResults.append(compareDataset(baseDataset, getDatasetArray(dataset)))

  validationResults = []
  if validationDir is not None:
    runs = os.listdir(validationDir)
  for counter, run in enumerate(runs):
    datadir = os.path.join(dataDir, run)
    _, dataset = getPerfDataset(datadir, get_num_counters())
    validationResults.append(compareDataset(baseDataset, getDatasetArray(dataset)))

  sorted_results = sorted(results, key=lambda tup: tup[1], reverse=True)
  return sorted_results, testResults, validationResults


def writeTestLog(logFile: TextIO, anomalySummary: list):
  """Writes a series of AnomalyTuples to the given log.

  Args:
      logFile: An open file descriptor of a log file.
      anomalySummary: List of AnomalyTuples.
  """
  topRankings = []

  # sort the anomalies by their run index
  anomalySummary = sorted(anomalySummary, key=lambda s: int(''.join(filter(str.isdigit, s.run))))

  for summary in anomalySummary:
    pct = summary.anomalous_sample_count / summary.sample_count * 100.0
    print(f'{summary.run} - {summary.anomalous_sample_count} / {summary.sample_count} - \
{pct:.2f}% - {summary.ranking[:3]}', file=logFile)
    topRankings.append(summary.ranking[0])

  if len(topRankings) > 0:
    topRank = max(set(topRankings), key=topRankings.count)  # https://stackoverflow.com/a/40785493
    print(f'\nTop Performance Counter: [{topRank.strip()}]', file=logFile)

    desc = get_description(topRank.strip())
    if desc is None:
      desc = f'- Performance counter not found in {getAutoperfDir("descriptions.json")}.'
    else:
      print(f'- Explanation: [{desc}]', file=logFile)


def testModel(model: Model, thresholdError: float, nonAnomalousDataDir: str,
              anomalousDataDir: str, logFile: TextIO = None) -> AccuracyTuple:
  """Tests the model and reports classification performance metrics.

  Args:
      model: A Keras autoencoder.
      thresholdError: Minimum reconstruction error to be considered anomalous.
      nonAnomalousDataDir: A path to a directory containing nominal data.
      anomalousDataDir: A path to a directory containing anomalous data.
      logFile (optional): An open file descriptor of a log file.

  Returns:
      An `AccuracyTuple` containing (true / false) positives + negatives.
  """
  log.info("Testing Non-Anomalous")
  negative_runs = os.listdir(nonAnomalousDataDir)
  anomaly_summary, nominal_pcts = testAnomaly(model, nonAnomalousDataDir, negative_runs, thresholdError)
  if logFile is not None:
    print("\n..Testing nonAnomalousData..\n", file=logFile)
    writeTestLog(logFile, anomaly_summary)
  false_positive = len(anomaly_summary)
  true_negative = len(negative_runs) - false_positive

  log.info("Testing Anomalous")
  positive_runs = os.listdir(anomalousDataDir)
  anomaly_summary, anomalous_pcts = testAnomaly(model, anomalousDataDir, positive_runs, thresholdError)
  if logFile is not None:
    print("\n..Testing AnomalousData..\n", file=logFile)
    writeTestLog(logFile, anomaly_summary)
  true_positive = len(anomaly_summary)
  false_negative = len(positive_runs) - true_positive

  error_labels = [*np.zeros(len(nominal_pcts)), *np.ones(len(anomalous_pcts))]
  combined_errors = [*nominal_pcts, *anomalous_pcts]

  plots.plot_roc_curve(error_labels, combined_errors)
  plots.plot_pr_curve(error_labels, combined_errors)

  return AccuracyTuple(true_positive=true_positive, false_positive=false_positive,
                       true_negative=true_negative, false_negative=false_negative)


def getReconstructionErrors(perfTestDataDir: str, model: Model) -> list:
  """Runs inference on input data and collections the reconstruction errors.

  Args:
      perfTestDataDir: A directory holding performance testing data.
      model: A Keras autoencoder.

  Returns:
      A list of reconstruction errors.
  """
  runs = os.listdir(perfTestDataDir)
  reconstructErrorList = []
  for run in track(runs, description=perfTestDataDir):
    # print ("Reconstruction of execution ", run)
    datadir = os.path.join(perfTestDataDir, run)

    _, dataset = getPerfDataset(datadir, get_num_counters())
    dataArray = getDatasetArray(dataset)
    dataArray = preprocessDataArray(dataArray)

    datasetLen = dataArray.shape[0]
    decodedData = keras_autoencoder.predict(model, dataArray)
    for x in range(datasetLen):
      reconstructionError = getMSE(dataArray[x], decodedData[x])
      # for debugging
      reconstructErrorList.append(reconstructionError)
  return reconstructErrorList


def aggregateAndTrain(perfTrainDataDir: str, autoencoder: Model, saveTrainedNetwork: bool = False,
                      outputDir: str = None) -> Model:
  """Aggregates and trains on the provided data.

  TODO: How is this different from `perfAnalyzerMainTrain()`?

  Args:
      perfTrainDataDir: A directory holding performance training data.
      autoencoder: A Keras autoencoder.
      saveTrainedNetwork (optional): Whether or not to save the network to disk.
      outputDir (optional): The directory where the network is saved, if requested.

  Returns:
      A trained Keras autoencoder.
  """
  training_sequence = getTrainDataSequence(perfTrainDataDir)
  train_loss_list = []
  # reconstruction_error_list = []  # unused
  validation_loss_list = []
  dataset = []
  for train_run in training_sequence:
    datadir = os.path.join(perfTrainDataDir, train_run)
    _, additionalDatatset = getPerfDataset(datadir, get_num_counters())
    processed_dataset_array = preprocessDataArray(getDatasetArray(additionalDatatset))
    dataset.extend(processed_dataset_array.tolist())

  if len(dataset) < get_num_counters() * 2:
    log.error("Not enough data for training this iteration")
    sys.exit(1)

  trainingDataset = getDatasetArray(dataset)
  monitor = {'batch_end': ['loss'], 'epoch_end': ['val_loss']}

  model, history = keras_autoencoder.trainAutoencoder(autoencoder, trainingDataset, monitor)
  train_loss_list.extend(history.record['batch_end']['loss'])
  validation_loss_list.extend(history.record['epoch_end']['val_loss'])

  # plots.plot_loss_curves(train_loss_list, validation_loss_list)

  if saveTrainedNetwork:
    model.save(os.path.join(outputDir, cfg.model.filename))

  return model


def calcThresoldError(reconstructionErrors: list) -> float:
  """Calculates threshold of reconstruction errors (3 STDs away from the mean).

  Args:
      reconstructionErrors: A list of measured reconstruction errors.

  Returns:
      The calculated reconstruction error threshold.
  """
  meanVal = np.mean(reconstructionErrors)
  log.info('meanVal = %f', meanVal)
  meanVal += (3 * np.std(reconstructionErrors))
  log.info('std = %f', np.std(reconstructionErrors))
  return meanVal


def trainAndEvaluate(model: Model, trainDataDir: str):
  """Trains and performs some basic evaluation on an autoencoder.

  Args:
      model: A Keras autoencoder
      trainDataDir: Training data directory
  """
  model = aggregateAndTrain(trainDataDir, model,
                            saveTrainedNetwork=True, outputDir=getAutoperfDir())

  log.info("Training Complete")
  datasetTrainErrors = getReconstructionErrors(trainDataDir, model)
  thresholdError = calcThresoldError(datasetTrainErrors)
  np.save(getAutoperfDir('threshold.npy'), np.array([thresholdError]))

  plots.plot_histograms(train=np.array(datasetTrainErrors),
                        threshold=thresholdError)


def checkExistence(directories: list) -> bool:
  """Checks if every directory in the input list exists.

  Args:
      directories: A list of input directories.

  Returns:
    True if all directories exist; else False.
  """
  for d in directories:
    if not os.path.exists(d):
      log.error('Directory [%s] does not exist. Please try again.', d)
      return False
  return True
