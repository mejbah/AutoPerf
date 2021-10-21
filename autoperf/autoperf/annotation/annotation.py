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
"""annotation.py

This file contains core functions for automatic perfpoint C/C++ annotations.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path

from typing import Callable, List, Union
from functools import partial
import multiprocessing as mp
import argparse

import psutil
from git import Repo
from rich.live import Live
# from rich import print as rprint

from autoperf.utils import getAutoperfDir, set_working_directory
from autoperf.annotation.interface import CLI
from autoperf.annotation.parsing import ClangParser
from autoperf.annotation.utils import Injector, Eraser

log = logging.getLogger('rich')

# valid c/c++ extensions
extensions = ['.cpp', '.hpp', '.c', '.h', '.cc', '.hh']

# ------------------------------------------------------------------------------


def ranges_in_chunk(chunk: list) -> list:
  """Convert a git diff chunk into a list of line numbers:
    [[X,N], [Y,N]]
  Where X represents the starting line in file 'a' (and Y in file 'b'), with N
  being the number of lines that were modified in each file.

  Args:
      chunk: A chunk received from the git diff.

  Returns:
      A 2D list of line ranges, explained in detail above.
  """
  strings = re.findall(r'[-\d]+', chunk[0].split('@@')[1])
  ranges = [int(s) for s in strings]

  if len(ranges) == 4:
    ranges = [ranges[:2], ranges[2:]]

  elif len(ranges) == 3:
    if re.match(r',\d.*\s', chunk[0]) is not None:  # look for '-X +Y,N'
      ranges = [ranges[:2], [ranges[2], 1]]
    else:                                      # look for '-X,N +Y'
      ranges = [[ranges[0], 1], ranges[1:]]

  elif len(ranges) == 2:
    ranges = [[ranges[0], 1], [ranges[1], 1]]

  ranges[0][0] *= -1
  return ranges


def gitDiff(branch1: str, branch2: str) -> List[Union[List[str], List[str], List[List[str]]]]:
  """Perform a git diff on the current repository between two branches, and save
  some metadata to {branch1}.json / {branch2}.json in the .autoperf directory.

  Args:
      branch1: Name of base branch.
      branch2: Name of new branch.

  Returns:
      List of filenames changed in branch1.
      List of filenames changed in branch2.
      List of modified chunks associated with the previous lists.
  """
  r = Repo(os.getcwd(), search_parent_directories=True)
  fmt = ['-U 0', '--diff-filter=M']
  differ = r.git.diff(f'{branch1}..{branch2}', fmt).split("\n")
  differ = filter(lambda x: len(x), differ)

  a_files, b_files = [], []
  diffs, cur_diff, cur_chunk = [], [], []
  for line in differ:
    if line.startswith('diff'):
      a_files += [re.findall(r'a\/(.*)\s', line)[0]]
      b_files += [re.findall(r'b\/(.*)', line)[0]]
      if len(cur_chunk) > 0 or len(cur_diff) > 0:
        if len(cur_chunk) > 1:
          cur_diff += [cur_chunk]
        diffs += [cur_diff]
        cur_diff = []
        cur_chunk = []

    elif line.startswith('@@') and len(cur_chunk) > 0:
      if len(cur_chunk) > 1:
        cur_diff += [cur_chunk]
      cur_chunk = [line]

    elif not line.startswith(('---', '+++', 'index')):
      if len(line[1:].strip()) > 0:
        cur_chunk += [line]

  if len(cur_chunk) > 0 or len(cur_diff) > 0:
    if len(cur_chunk) > 1:
      cur_diff += [cur_chunk]
    diffs += [cur_diff]
    cur_diff = []
    cur_chunk = []

  base_json, modified_json = {}, {}
  for a, b, diff in zip(a_files, b_files, diffs):
    for chunk in diff:
      ranges = ranges_in_chunk(chunk)

      if a not in base_json:
        base_json[a] = [ranges[0][0]]
      else:
        base_json[a] += [ranges[0][0]]

      if b not in modified_json:
        modified_json[b] = [ranges[1][0]]
      else:
        modified_json[b] += [ranges[1][0]]

  if base_json:
    with open(getAutoperfDir(f'{branch1}.json'), 'w') as write_file:
      json.dump(base_json, write_file)

  if modified_json:
    with open(getAutoperfDir(f'{branch2}.json'), 'w') as write_file:
      json.dump(modified_json, write_file)

  return a_files, b_files, diffs

# ------------------------------------------------------------------------------


def get_working_directory(input_work_dir: str, input_path: str) -> str:
  """Try to identify the working directory based on user input or location of
  a Makefile.

  Args:
      input_work_dir: User-specified working directory (--work_dir)
      input_path: Files the user wants to parse.

  Returns:
      Path to the working directory.
  """

  # First, check if the user specified a working directory
  work_dir = None
  if input_work_dir is not None:
    work_dir = Path(input_work_dir).resolve()
    if not work_dir.exists():
      work_dir = None
      log.warning('[red]The provided [code]--work_dir[/code] does not exist. Attempting to resolve:[/red]')

  # If a top directory isn't specified, try to intelligently determine it
  if work_dir is None:
    common_path = os.path.commonprefix(input_path)
    work_dir = Path(common_path).resolve()
    work_dir = work_dir if work_dir.is_dir() else work_dir.parent
    if input_work_dir is not None:
      log.info('[code][dark_red]%s[/dark_red][/code] [red]has been selected.[/red]', work_dir)

  return work_dir

# ------------------------------------------------------------------------------


def resolve_paths_to_new_working_dir(paths: list, new_work_dir: str) -> str:
  """Resolve a list of file paths relative to the new working directory.

  Args:
      paths: User-provided paths.
      new_work_dir: The new working directory.

  Returns:
      List of new paths.
  """
  return [Path(p).resolve().relative_to(new_work_dir) for p in paths]


def convert_paths_to_files(paths: list, recursive: bool = False) -> list:
  """Convert a list of paths to unique files, sorted by depth - so higher items
  are tackled first.

  Args:
      paths: User-provided paths.
      recursive: Whether or not to search directories recursively. Defaults to False.

  Returns:
      List of unique files.
  """
  files = []
  for p in paths:

    # If it's a directory, search it for matching files
    if p.is_dir():
      prefix = '**/' if recursive else ''
      for f in Path(p).glob(prefix + '*'):
        if f.suffix in extensions:
          files.append(f)

    # Otherwise, append it straight to the list
    else:
      files.append(p)

  # Delete duplicates
  unique_files = list(set(files))

  # Sort list of files by the depth of their folder nesting
  # This ensures higher items in the tree are tackled first
  unique_files.sort(key=lambda x: len(x.parents), reverse=True)

  return unique_files


def configure_args(base: argparse.ArgumentParser = None):
  """Configure the argparser with help information + arguments."""

  usage_examples = \
      """Examples:
  python function_parser.py /path/to/file.c /path/to/dir
  python function_parser.py /path/to/dir/*.c
  python function_parser.py --recursive /path/to/dir
  python function_parser.py --parallel 4 /path/to/dir
  """

  if base:
    parser = base.add_parser('annotate',
                             help='annotate functions with AutoPerf profiling code',
                             description='Annotate functions with AutoPerf profiling code.',
                             formatter_class=argparse.RawDescriptionHelpFormatter,
                             epilog=usage_examples)
  else:
    parser = argparse.ArgumentParser()

  parser.add_argument('paths', type=str, nargs='+', help='A file or directory \
                      to parse - can provide an arbitrary number of either')
  parser.add_argument('-p', '--parallel', type=int, default=-1, metavar='P',
                      help='Number of parallel processes to use')

  parser.add_argument('-a', '--args', type=str, default='', help='Optional \
                      clang arguments to use while parsing')
  parser.add_argument('--work_dir', type=str, default=None, help='Top source \
                      directory of the repository containing a valid Makefile')

  parser.add_argument('-i', '--inject', action='store_true', default=False,
                      help='Inject AutoPerf annotations at the start and end \
                      of function bodies')
  parser.add_argument('-e', '--erase', action='store_true', default=False,
                      help='Erase AutoPerf annotations at the start and end \
                      of function bodies')

  parser.add_argument('--diff', type=str, default=None, nargs='?', const='master',
                      metavar='BRANCH', help='Perform a diff vs. a specified \
                      branch of the current git repository')
  parser.add_argument('--apply', action='store_true', default=False,
                      help='Apply a stored annotation of the current branch')

  parser.add_argument('--only', type=str, default=None, metavar='JSON',
                      help='A `.json` file that specifies files + functions \
                      to annotate or erase')

  parser.add_argument('-r', '--recursive', action='store_true', default=False,
                      help='Recursively search the directory tree')
  parser.add_argument('--no_collapse', action='store_true', default=False,
                      help='Collapse subfolders in large directories (helpful \
                      with --recursive)')

  parser.add_argument('--detailed', action='store_true', default=False,
                      help='Print detailed information about each file')
  parser.add_argument('--include_preproc', action='store_true', default=False,
                      help='Include preprocessor macros in the detailed report')
  parser.add_argument('-f', '--fast', action='store_true', default=False,
                      help='Parse files faster by disabling recursive inclusion, \
                      but potentially miss some functions')

  parser.add_argument('--libclang', type=str, default=None, help='Location of \
                      the installed libclang.so/.dll/.dylib file')

  parser.set_defaults(func=annotate)
  return parser


def validate_paths(paths: list):
  """Check if any provided paths are invalid.

  Args:
      paths: List of file paths.
  """
  if any(not Path(p).exists() for p in paths):
    log.error('[red]One or more input paths are invalid.[/red]')
    sys.exit()


def validate_annotations(args: argparse.Namespace):
  """Don't allow users to inject + erase annotations at the same time.

  Args:
      args: User-provided args.
  """
  if args.inject and args.erase:
    log.error('[red]Cannot inject and erase annotations simultaneously.[/red]')
    sys.exit()


def validate_only(path: str):
  """Check that the user-provided `only.json` file exists.

  Args:
      path: Path to only.json file.
  """
  if path is not None and not Path(path).exists():
    log.error('[red]Specified `only` file does not exist.[/red]')
    sys.exit()


def print_num_parallel_procs(num: int):
  """Intelligently print the number of parallel processes that will be used.

  Args:
      num: Number of processes.
  """
  plural = 'es' if num != 1 else ''
  log.info('[bright_blue]Splitting tasks across [code]%d process%s[/code].[/bright_blue]', num, plural)


def get_action(inject: bool, erase: bool, only: dict) -> Callable:
  """Retrieve the action specified by the user.

  Args:
      inject: Flag to inject annotations.
      erase: Flag to erase annotations.
      only: List of files / functions that should be modified.

  Returns:
      A function that can be called - either `inject()` or `erase()`.
  """

  # Default case, no-op
  def do_nothing(*args):
    pass
  action = do_nothing

  if inject:
    injector = Injector(only)
    action = injector.inject
    log.info("Injecting AutoPerf annotations.")

  elif erase:
    eraser = Eraser(only)
    action = eraser.erase
    log.info("Removing all AutoPerf annotations.")

  return action


def annotate(input_args: argparse.Namespace):
  """Perform end-to-end annotation based on the input arguments.

  Args:
      input_args: Provided by argparse.
  """
  validate_paths(input_args.paths)
  validate_annotations(input_args)
  validate_only(input_args.only)

  # Perform a diff on the current repository and save off the results
  repo = Repo(os.getcwd(), search_parent_directories=True)
  if input_args.diff:
    if input_args.diff == str(repo.head.ref):
      log.error('Cannot diff a branch with itself [code](%s)[/code]', repo.head.ref)
      sys.exit(-1)
    gitDiff(input_args.diff, repo.head.ref)
    log.info('Generating diff between [code]%s[/code] and [code]%s[/code] \
branches.', input_args.diff, repo.head.ref)
    if not input_args.apply:
      return

  # Load in the list of specified files / functions
  only = None
  if input_args.apply:
    only = json.load(open(getAutoperfDir(f'{repo.head.ref}.json')))
  elif input_args.only is not None:
    only = json.load(open(input_args.only))

  # Use all available cores, ignoring hyperthreading
  if input_args.parallel == -1:
    input_args.parallel = psutil.cpu_count(logical=False)
  print_num_parallel_procs(input_args.parallel)

  work_dir = get_working_directory(input_args.work_dir, input_args.paths)
  paths = resolve_paths_to_new_working_dir(input_args.paths, work_dir)

  # Change the working directory to the new path temporarily
  with set_working_directory(work_dir):

    cparser = ClangParser(fast=input_args.fast,
                          preproc=input_args.include_preproc,
                          lib=input_args.libclang)
    workload = partial(cparser.parse, args=input_args.args)

    log.info('ClangParser initialized.')

    action = get_action(input_args.inject, input_args.erase, only)

    files = convert_paths_to_files(paths, input_args.recursive)

    if only:
      temp_files = []
      for f in files:
        if any(w in str(f) for w in only):
          temp_files.append(f)
      files = temp_files

    cli = CLI(files, work_dir.name, collapse=(not input_args.no_collapse))

    with Live(cli.live_grid, refresh_per_second=4) as live:

      cli.start(live=live)
      with mp.Pool(processes=input_args.parallel) as pool:
        for result in pool.imap_unordered(workload, files):
          cli.update(result)
          action(result)
      cli.finish()

    if input_args.detailed:
      cli.print_results(detailed=input_args.detailed)
