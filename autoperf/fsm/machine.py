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
"""machine.py

This file contains all code regarding the finite state machine that drives the
order of execution.
"""

import os
from enum import IntFlag, Enum, auto
import pickle
import logging

from transitions import Machine
# from transitions.extensions import GraphMachine as Machine

from autoperf.config import cfg
from autoperf.utils import getAutoperfDir

logging.getLogger('transitions').setLevel(logging.CRITICAL)
log = logging.getLogger("rich")


class AutoName(Enum):
  """Automatically name Enums using the `auto()` function."""
  def _generate_next_value_(name, start, count, last_values):
    return name


class Modes(IntFlag):
  """Mode of operation flags. More than one can be active at a time."""
  TRAIN = auto()
  DETECT = auto()


class States(str, AutoName):
  """Underlying state names of the FSM."""
  DIFF = auto()
  STASH = auto()
  ANNOTATE = auto()
  BUILD = auto()
  MEASURE = auto()
  CLUSTER = auto()
  TRAIN = auto()
  POP = auto()
  DETECT = auto()
  REPORT = auto()
  FINISHED = auto()


class AutoPerfMachine(Machine):
  """Full sequential FSM with checkpoints and recovery."""

  transitions = [
      {'trigger': 'next', 'source': States.DIFF, 'dest': States.STASH},

      {'trigger': 'next', 'source': States.STASH, 'dest': States.ANNOTATE},

      {'trigger': 'next', 'source': States.POP, 'dest': States.ANNOTATE,
       'conditions': 'do_detect'},

      {'trigger': 'next', 'source': States.POP, 'dest': States.FINISHED,
       'unless': 'do_detect'},

      {'trigger': 'next', 'source': States.ANNOTATE, 'dest': States.BUILD},

      {'trigger': 'next', 'source': States.BUILD, 'dest': States.MEASURE,
       'after': 'reset_workload_runs'},

      {'trigger': 'next', 'source': States.MEASURE, 'dest': States.CLUSTER,
       'conditions': 'do_train', 'unless': 'keep_collecting', 'after': 'increment_workload'},

      {'trigger': 'next', 'source': States.MEASURE, 'dest': '=',
       'conditions': 'keep_collecting', 'after': 'increment_workload'},

      {'trigger': 'next', 'source': States.CLUSTER, 'dest': States.TRAIN},

      {'trigger': 'next', 'source': States.TRAIN, 'dest': States.POP,
       'after': 'stop_train'},

      {'trigger': 'next', 'source': States.MEASURE, 'dest': States.DETECT,
       'conditions': 'do_detect', 'unless': ['keep_collecting', 'do_train'],
       'after': 'increment_workload'},

      {'trigger': 'next', 'source': States.DETECT, 'dest': States.REPORT},

      {'trigger': 'next', 'source': States.REPORT, 'dest': States.FINISHED}
  ]

  mode = None
  max_workload_runs = None
  workload_run = None

  def _save_checkpoint(self):
    with open(getAutoperfDir('checkpoint.p'), 'wb') as checkpoint:
      pickle.dump(self, checkpoint, 4)

  def _load_checkpoint(self):
    with open(getAutoperfDir('checkpoint.p'), 'rb') as checkpoint:
      tmp_dict = pickle.load(checkpoint)
      self.__dict__.clear()
      self.__dict__.update(tmp_dict.__dict__)

  def do_train(self):
    """Check if we should train the model."""
    return Modes.TRAIN in self.mode

  def stop_train(self):
    """Disable training; useful in TRAIN|DETECT mode."""
    self.mode &= ~Modes.TRAIN

  def do_detect(self):
    """Check if we should try to detect anomalies."""
    return Modes.DETECT in self.mode

  def stop_detect(self):
    """Disable anomaly detection."""
    self.mode &= ~Modes.DETECT

  def increment_workload(self):
    """Increment the current (active) workload ID."""
    self.workload_run += 1

  def keep_collecting(self):
    """Check if more data should be collected."""
    return self.workload_run < self.max_workload_runs

  def reset_workload_runs(self):
    """Change back to the default workload ID."""
    self.workload_run = 1

  def __init__(self, mode: Modes, runs: int = 0, checkpoint: bool = True):
    """Initialize the FSM!

    Args:
        mode: Mode of operation.
        runs: Number of workload runs to conduct.
        checkpoint: Whether checkpoints should be saved / recovered from.
    """
    if checkpoint:
      try:
        self._load_checkpoint()
        log.info('[yellow]Recovering from checkpoint.')

      except FileNotFoundError:
        log.warning('Checkpoint not found, starting from scratch.')

    # Either the checkpoint load failed, or checkpoints are not active.
    if self.mode is None:

      self.mode = mode
      self.max_workload_runs = runs
      self.workload_run = 1
      initial_state = States.DIFF

      if self.mode == Modes.DETECT:
        if not os.path.exists(getAutoperfDir(cfg.model.filename)):
          self.mode |= Modes.TRAIN

        elif runs > 0:
          initial_state = States.ANNOTATE

        else:
          initial_state = States.DETECT

      Machine.__init__(self, states=States, transitions=self.transitions,
                       initial=initial_state, after_state_change='_save_checkpoint')

  def __iter__(self):
    """Turn the FSM into an iterable."""
    return AutoPerfIterator(self)


class AutoPerfIterator():
  """Iterator of the AutoPerf FSM, allowing the machine to be executed through
  a `for state in fsm` loop."""

  def __init__(self, obj):
    """Start at the initial state."""
    self._autoperf = obj
    self._index = 0

  def __next__(self):
    """Advance to the next state."""
    if self._index == 0:
      self._index += 1
      return self._autoperf.model.state.name

    if self._autoperf.is_FINISHED():
      os.unlink(getAutoperfDir('checkpoint.p'))
      raise StopIteration

    self._index += 1
    self._autoperf.next()
    return self._autoperf.model.state.name
