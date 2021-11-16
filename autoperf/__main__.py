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
"""Main entrypoint to Autoperf. Routes user commands to the correct functions."""

import os
import sys
import argparse
import logging
import warnings

from typing import List

from rich.logging import RichHandler
from rich import traceback

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from .annotation import annotation

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
traceback.install()

logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True
logging.getLogger('subprocess.Popen').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
log = logging.getLogger("rich")

# Filter out this warning about Rich TQDM progress bars
warnings.filterwarnings("ignore", message="rich is experimental/alpha")


def init(args):
  """Initializes the .autoperf directory at the root of the git repo.

  Args:
      args: Unused, but required by argparse.
  """
  from autoperf.config import CLI
  CLI()


def annotate(args):
  """Apply Autoperf HPC annotations to C/C++ code.

  Args:
      args: Provided by argparse.
  """
  from .annotation.annotation import annotate as anno
  anno(args)


def measure(args):
  """Run the specified workload and collect HPC measurements.

  Args:
      args: Provided by argparse.
  """
  from .actions import runBuild, runWorkload
  runBuild()
  for r in args.run_count:
    runWorkload(args.out_dir, r)


def train(args):
  """Train a Keras autoencoder using collected HPC measurements.

  Args:
      args: Provided by argparse.
  """
  from autoperf import autoperf, keras_autoencoder
  from autoperf.counters import get_num_counters
  from autoperf.config import cfg

  autoencoder = keras_autoencoder.getAutoencoder(get_num_counters(), cfg.model.encoding, cfg.model.hidden)
  if args.hidden is not None and args.encoding is not None:
    autoencoder = keras_autoencoder.getAutoencoder(get_num_counters(), args.encoding, args.hidden)

  log.info('Autoencoder Summary:')
  for line in autoencoder.summary():
    log.info(line)
  autoperf.trainAndEvaluate(autoencoder, args.train_dir)


def evaluate(args):
  """Evaluate an autoencoder with train (nominal) + test (nominal/anomalous)
  HPC measurements.

  Args:
      args: Provided by argparse.
  """
  from .actions import runEvaluate
  runEvaluate(args.train, args.nominal, args.anomalous)


def detect(args):
  """Detect performance anomalies using a previously trained autoencoder.

  Args:
      args: Provided by argparse.
  """
  from autoperf.fsm import Modes
  runMachine(Modes.DETECT, args)


def clean(args):
  """Clean up the .autoperf directory, besides configuration files.

  Args:
      args: Provided by argparse.
  """
  import shutil
  from glob import glob
  from autoperf.utils import getAutoperfDir

  # This would force the user to give explicit permission before clearing the
  # directory. Temporarily disabled, to match other common CLI apps.
#   from rich.prompt import Confirm
#   if Confirm.ask("[red]Would you like to remove all non-configuration files in \
# the [code].autoperf[/code] directory?"):

  for file in glob(getAutoperfDir('*')):
    if file.split('/')[-1] not in ['config.ini', 'COUNTERS']:
      log.info('Removing [code]%s', file)
      try:
        os.unlink(file)
      except IsADirectoryError:
        try:
          shutil.rmtree(file)
        except Exception:
          ...


def runMachine(mode, args):
  """Run the finite state machine until all states are exhausted.

  Args:
      mode: What the machine should do: (TRAIN | DETECT).
      args: Provided by argparse.
  """
  from git import Repo
  from autoperf.fsm import AutoPerfMachine, States, Modes
  from autoperf.utils import getAutoperfDir
  from autoperf.config import cfg

  ap_fsm = AutoPerfMachine(mode=mode, runs=args.run_count)
  for state in ap_fsm:

    log.info('[bold green]State - [%s]', state)

    # -------------------------------------------------------------------------
    if state == States.DIFF:
      from .annotation.annotation import annotate as runAnnotate
      from .annotation.annotation import configure_args

      anno_arg_str = f'{cfg.build.dir} --diff {cfg.git.main} --recursive'
      anno_arg_parser = configure_args()
      anno_args = anno_arg_parser.parse_args(anno_arg_str.split(' '))
      runAnnotate(anno_args)

    # -------------------------------------------------------------------------
    elif state == States.STASH:
      repo = Repo(os.getcwd(), search_parent_directories=True)
      repo.git.stash()
      repo.git.checkout(cfg.git.main)

    # -------------------------------------------------------------------------
    elif state == States.ANNOTATE:
      from .annotation.annotation import annotate as runAnnotate
      from .annotation.annotation import configure_args

      repo = Repo(os.getcwd(), search_parent_directories=True)
      onlyfile = getAutoperfDir(str(repo.head.ref) + '.json')

      if os.path.exists(onlyfile):
        anno_arg_str = f'{cfg.build.dir} --only {onlyfile} --recursive --apply --inject'
        anno_arg_parser = configure_args()
        anno_args = anno_arg_parser.parse_args(anno_arg_str.split(' '))
        runAnnotate(anno_args)

    # -------------------------------------------------------------------------
    elif state == States.BUILD:
      from .actions import runBuild
      runBuild()

    # -------------------------------------------------------------------------
    elif state == States.MEASURE:
      from .actions import runWorkload
      from glob import glob

      out_dir = getAutoperfDir()
      if Modes.TRAIN in ap_fsm.mode:
        out_dir = os.path.join(out_dir, 'train')
      elif Modes.DETECT in ap_fsm.mode:
        out_dir = os.path.join(out_dir, 'detect')

      workload_run = 0
      for folder in glob(out_dir + '/run_*'):
        run = int(folder.split('_')[-1])
        if run > workload_run:
          workload_run = run

      runWorkload(out_dir, workload_run + 1)

    # -------------------------------------------------------------------------
    elif state == States.CLUSTER:
      log.warning('  Not yet implemented.')

    # -------------------------------------------------------------------------
    elif state == States.TRAIN:
      from autoperf import autoperf, keras_autoencoder
      from autoperf.counters import get_num_counters

      autoencoder = keras_autoencoder.getAutoencoder(get_num_counters(), cfg.model.encoding, cfg.model.hidden)
      autoperf.trainAndEvaluate(autoencoder, getAutoperfDir('train'))

    # -------------------------------------------------------------------------
    elif state == States.POP:
      repo = Repo(os.getcwd(), search_parent_directories=True)
      repo.git.reset('--hard')
      repo.git.checkout('-')
      try:
        repo.git.stash('pop')
      except Exception:
        ...  # Just means the previous stash didn't save anything, that's okay.

    # -------------------------------------------------------------------------
    elif state == States.DETECT:
      from .actions import runDetect
      train_dir = getAutoperfDir('train')
      detect_dir = getAutoperfDir('detect')
      runDetect(train_dir, detect_dir)

    # -------------------------------------------------------------------------
    elif state == States.REPORT:
      from .actions import runReport
      runReport(getAutoperfDir('detect'))

    # -------------------------------------------------------------------------
    elif state == States.FINISHED:
      from .annotation.annotation import annotate as runAnnotate
      from .annotation.annotation import configure_args

      log.info('Cleaning...')
      repo = Repo(os.getcwd(), search_parent_directories=True)
      onlyfile = getAutoperfDir(str(repo.head.ref) + '.json')

      if os.path.exists(onlyfile):
        anno_arg_str = f'{cfg.build.dir} --only {onlyfile} --recursive --erase'
        anno_arg_parser = configure_args()
        anno_args = anno_arg_parser.parse_args(anno_arg_str.split(' '))
        runAnnotate(anno_args)


def main(argv: List[str] = None):
  """Main Autoperf entrypoint, routes user commands to function calls.

  Args:
      argv: Argument vector, typically derived from user input.
  """
  parser = argparse.ArgumentParser(prog='autoperf',
                                   description='AutoPerf is a performance regression monitoring system.')
  parser.set_defaults(func=lambda x: parser.print_help())
  subparsers = parser.add_subparsers(title='commands')

  # --------------------------------------------------------------------------

  parser_detect = subparsers.add_parser('detect',
                                        help='run AutoPerf end-to-end and report any discovered anomalies',
                                        description='Run AutoPerf end-to-end and report any discovered anomalies.')
  parser_detect.add_argument('run_count', metavar='R', type=int, help='Number of workload runs to execute')
  parser_detect.set_defaults(func=detect)

  # --------------------------------------------------------------------------

  parser_init = subparsers.add_parser('init',
                                      help='initialize the .autoperf folder + configs',
                                      description='Initialize the .autoperf folder + configs.')
  parser_init.set_defaults(func=init)

  # --------------------------------------------------------------------------

  parser_clean = subparsers.add_parser('clean',
                                       help='clean the .autoperf folder except for configs',
                                       description='Clean the .autoperf folder except for configs.')
  parser_clean.set_defaults(func=clean)

  # --------------------------------------------------------------------------

  parser_measure = subparsers.add_parser('measure',
                                         help='run the program under test and collect measurements',
                                         description='Run the program under test and collect measurements.')
  parser_measure.add_argument('out_dir', type=str, help='Output directory for HPC results')
  parser_measure.add_argument('run_count', metavar='R', nargs='+', type=int, help='Run index')
  parser_measure.set_defaults(func=measure)

  # --------------------------------------------------------------------------

  parser_train = subparsers.add_parser('train',
                                       help='train an autoencoder with collected measurements',
                                       description='Train an autoencoder with collected measurements.')
  parser_train.add_argument('--hidden', metavar='H', type=int, default=None,
                            nargs='+', help='List of hidden layer dimensions')
  parser_train.add_argument('--encoding', metavar='E', type=int, default=None,
                            help='Encoding layer dimension')
  parser_train.add_argument('train_dir', type=str, help='Nominal training data directory')
  parser_train.set_defaults(func=train)

  # --------------------------------------------------------------------------

  parser_evaluate = subparsers.add_parser('evaluate',
                                          help='evaluate a trained autoencoder with test data',
                                          description='Evaluate a trained autoencoder with test data.')
  parser_evaluate.add_argument('train', type=str, help='Training data directory')
  parser_evaluate.add_argument('nominal', type=str, help='Nominal test data directory')
  parser_evaluate.add_argument('anomalous', type=str, help='Anomalous test data directory')
  parser_evaluate.set_defaults(func=evaluate)

  # --------------------------------------------------------------------------

  annotation.configure_args(subparsers)

  # --------------------------------------------------------------------------

  # Execute the user command
  try:
    args = parser.parse_args(argv)
    args.func(args)

  # Handle Ctrl-C gracefully
  except KeyboardInterrupt:
    print('\n')
    sys.exit(0)


if __name__ == "__main__":
  sys.exit(main())
