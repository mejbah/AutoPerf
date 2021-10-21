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
"""interface.py

This file contains classes and functions that implement the annotation CLI.
"""

from timeit import default_timer as timer

from copy import deepcopy
from pathlib import Path

import numpy as np
from rich import print as rprint
from rich.text import Text
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeRemainingColumn

# ------------------------------------------------------------------------------


class DirectoryTree():
  """Class to represent a directory tree, using Rich for display purposes."""

  root = None
  dirs = None     # Directories are stored as branches
  files = None    # Individual files are stored as leaves

  def __init__(self, paths, top_dir):
    self.root = Tree(f":open_file_folder: [link file://{Path('.').resolve()}]{top_dir}/",
                     style='bright_black', guide_style="bright_blue")
    self.dirs = {'.': self.root}
    self.files = {}
    for p in paths:
      self.add_file(p)

  def add_file(self, path: str):
    """Add a file path to the directory tree.

    Args:
        path: Full path to the file.
    """
    # First, add the required subdirectories to the tree
    d = path.parts
    list_of_subdirs = ['/'.join(d[:i]) for i in range(1, len(d))]
    for i, subd in enumerate(list_of_subdirs):
      prev_subd = '.'
      if i > 0:
        prev_subd = list_of_subdirs[i - 1]

      if subd not in self.dirs:
        final_dir = subd.split('/')[-1]
        self.dirs[subd] = self.dirs[prev_subd].add(f":open_file_folder: [link file://{Path(subd).resolve()}]{final_dir}/",
                                                   guide_style="bright_blue")

    # Then, add the file as a leaf node
    text_filename = Text(path.name, "bold bright_black")
    text_filename.stylize(f"link file://{path.resolve()}")
    icon = "📄 "
    self.files[path] = self.dirs[str(path.parent)].add(Text(icon) + text_filename)

  def finish_file(self, name: str):
    """Finish a file, change text to green + give a checkmark.

    Args:
        name: Full path to the file.
    """
    self.files[name].label = Text('✔️ ') + self.files[name].label[1:]
    self.files[name].label.stylize('green')

  def finish_dir(self, name: str, collapse: bool = False):
    """Finish a directory, change text to green + give a check (and collapse if
    requested, to minimize terminal bloat.)

    Args:
        name: Full path to the file
        collapse: Whether or not to collapse the folder. Defaults to False.
    """
    self.dirs[name].style = 'bold green'
    self.dirs[name].guide_style = 'dark_green'
    self.dirs[name].label = '✔️ ' + self.dirs[name].label.split(':open_file_folder:')[-1]
    self.dirs[name].expanded = not collapse  # Collapse folders as they complete

  def finish_root(self):
    """Mark the root directory as finished."""
    self.root.style = 'bold green'
    self.root.guide_style = 'dark_green'
    self.root.label = '✔️ ' + self.root.label.split(':open_file_folder:')[-1]

# ------------------------------------------------------------------------------


class CLI():
  """Function parsing command-line interface, using the DirectoryTree."""

  num_files = None
  files = None
  results = None

  dt = None
  progress = None
  task = None
  live_grid = None
  live = None

  start_time = None
  end_time = None
  total_time = None

  def __init__(self, files, work_dir, collapse):
    self.num_files = len(files)
    self.files = deepcopy(files)
    self.collapse = collapse

    # Build a file tree for visualization
    self.dt = DirectoryTree(files, work_dir)

    self.progress = Progress("{task.completed}/{task.total}",
                             BarColumn(),
                             "[progress.percentage]{task.percentage:>3.0f}%",
                             TimeRemainingColumn())
    self.task = self.progress.add_task('', total=self.num_files)

    self.live_grid = Table.grid()
    self.live_grid.add_row()
    self.live_grid.add_row(Panel(self.progress, padding=(0, 0), title='Progress'), style='red')
    self.live_grid.add_row(Panel(self.dt.root, title='Directory Structure', border_style='bright_blue', padding=(0, 0)))

  def start(self, live):
    """Start the CLI + timer.

    Args:
        live: Rich `Live` object.
    """
    self.live = live
    self.results = []
    self.start_time = timer()

  def update(self, result):
    """Update the CLI with a new result (e.g. a finished file).

    Args:
        result: Output of ClangParser.
    """
    self.progress.advance(self.task)
    self.results.append(result)

    # Mark the file as finished and remove it from the structure
    self.files.remove(result.filename)
    self.dt.finish_file(result.filename)

    # Check if all files within the subfolder(s) are finished parsing
    for subd in result.filename.parents:
      last_file_in_dir = not any([str(f).startswith(str(subd)) for f in self.files])
      if (str(subd) != '.' and last_file_in_dir):
        self.dt.finish_dir(str(subd), self.collapse)

  def finish(self):
    """Finish the CLI, record timings, etc."""
    self.dt.finish_root()
    self.live_grid.rows[1].style = 'green'
    self.live.update(self.live_grid)
    self.end_time = timer()
    self.total_time = self.end_time - self.start_time

  def print_results(self, detailed):
    """Print tables of result data, at varying levels of detail."

    Args:
        detailed: Whether or not to include function-level information.
    """
    if detailed:
      all_tables = Table.grid()
      for r in self.results:
        all_tables.add_row(r.get_table())
      table_panel = Panel(all_tables,
                          title='Detailed Information',
                          expand=False,
                          padding=(1, 1, 0, 1))
      rprint('', table_panel)

    stats = [r.get_stats() for r in self.results]
    total_declared = np.sum([s['declared'] for s in stats])
    total_defined = np.sum([s['defined'] for s in stats])

    table = Table(title=Text("Results", "italic bold"))
    table.add_column('Files Parsed', justify="center", style='bold white')
    table.add_column('Time', justify="center", style='cyan')
    table.add_column('Speed', justify="center", style='cyan')
    table.add_column('Declarations', justify="center", style='bold red')
    table.add_column('Definitions', justify="center", style='bold green')
    table.add_row(f'{self.num_files}',
                  f'{self.total_time:.3f} s',
                  f'{self.num_files / self.total_time:3f} files/s',
                  f'{total_declared}',
                  f'{total_defined}')
    rprint('', table, '')
