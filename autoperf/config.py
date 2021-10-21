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
"""config.py

This file contains configuration variables and utility functions for
accessing network parameters and topologies.
"""

import os
import ast
import sys
import errno
import configparser
from dataclasses import dataclass
from string import Template

from git import Repo
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

from autoperf.counters import save_counters
from autoperf.utils import getAutoperfDir


class ConfigTemplate(Template):
  """Template of the configuration file, serving as a bridge between the
  dataclass implementation and the on-disk `config.ini`."""

  defaults = {
      'build': 'make',
      'build_dir': '.',
      'clean': 'make clean',
      'workload': 'make eval-perfpoint',
      'workload_dir': '.',
      'branch': 'master',
      'noise': 0.25,
      'hidden': [16, 8],
      'encoding': 4,
      'activation': 'tanh',
      'filename': 'trained_network',
      'epochs': 12,
      'batch_size': 64,
      'optimizer': 'Adam',
      'learning_rate': 0.00001,
      'loss': 'mean_squared_error',
      'scale_factor': 1.0,
      'threshold': 0.05
  }

  config_str = Template("""
[build]
  cmd = ${build}
  dir = ${build_dir}

[clean]
  cmd = ${clean}

[workload]
  cmd = ${workload}
  dir = ${workload_dir}

[git]
  main = ${branch}

[model]
  hidden = ${hidden}
  encoding = ${encoding}
  activation = ${activation}
  filename = ${filename}

[training]
  epochs = ${epochs}
  batch_size = ${batch_size}
  optimizer = ${optimizer}
  learning_rate = ${learning_rate}
  loss = ${loss}
  noise = ${noise}
  scale_factor = ${scale_factor}

[detection]
  threshold = ${threshold}
""")

  def __init__(self):
    ...

  def update(self, settings: dict) -> str:
    """Update the template with new settings and return the substituted string.

    Args:
        settings: Dictionary containing key-value pairs of config settings.

    Returns:
        Substituted string with combination of default + configured options.
    """
    return self.config_str.substitute({**self.defaults, **settings})

# ------------------------------------------------------------------------------


def mkdir_p(path: str):
  """Recursively makes a directory by also creating all intermediate dirs.

  Args:
      path: Directory path to create.
  """
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

# ------------------------------------------------------------------------------


@dataclass
class Sections:
  """INI `[section]` headings."""
  raw_sections: dict

  def __post_init__(self):
    for section_key, section_value in self.raw_sections.items():
      setattr(self, section_key, SectionContent(section_value.items()))


@dataclass
class SectionContent:
  """INI key-value configuration mappings."""
  raw_section_content: dict

  def __post_init__(self):
    for section_content_k, section_content_v in self.raw_section_content:
      v = section_content_v
      try:
        v = ast.literal_eval(section_content_v)
      except Exception:
        ...
      setattr(self, section_content_k, v)


class Config(Sections):
  """Configuration dataclass wrapper, allowing the following:
    [section]
      key = value

  To be accessed programatically like:
    cfg.section.key
  """
  def __init__(self, raw_config_parser):
    Sections.__init__(self, raw_config_parser)

# ------------------------------------------------------------------------------


# Read the config file + load into a dataclass that can be accessed in any module
cfg = configparser.ConfigParser()
cfg.read(getAutoperfDir('config.ini'))
cfg = Config(cfg._sections)

# ------------------------------------------------------------------------------


def CLI():
  """A CLI for initialization / configuration."""

  banner = r"""[bright_green]    ___         __        ____            ____
   /   | __  __/ /_____  / __ \___  _____/ __/
  / /| |/ / / / __/ __ \/ /_/ / _ \/ ___/ /_
 / ___ / /_/ / /_/ /_/ / ____/  __/ /  / __/
/_/  |_\__,_/\__/\____/_/    \___/_/  /_/
"""
  print()
  rprint(Panel(banner, padding=(0, 2), title='Welcome To', expand=False, border_style='green'))

  configfile = getAutoperfDir('config.ini')
  if os.path.exists(configfile):
    if not Confirm.ask("[red]AutoPerf has already been configured in this repository.\n\
Would you like to start from scratch?"):
      print()
      sys.exit()

  else:
    rprint('[bright_blue]Before we can get started, we need some information.')

  print()

  mkdir_p(getAutoperfDir())

  # settings
  config_template = ConfigTemplate()
  s = {}

  s['build'] = Prompt.ask('[italic]Enter the command you use to build your codebase',
                          default=config_template.defaults['build'])
  s['build_dir'] = Prompt.ask('[italic]Enter the path where this command should be run',
                              default=config_template.defaults['build_dir'])
  if Confirm.ask('[italic][yellow]Would you like to clean the repository before building?'):
    s['clean'] = Prompt.ask('↪ [italic]Enter the command you use to clean your codebase',
                            default=config_template.defaults['clean'])
  else:
    s['clean'] = ''

  s['workload'] = Prompt.ask('\n[italic]Enter the workload under test',
                             default=config_template.defaults['workload'])
  s['workload_dir'] = Prompt.ask('[italic]Enter the path where this command should be run',
                                 default=config_template.defaults['workload_dir'])

  s['branch'] = Prompt.ask('\n[italic]Enter the name of your repository\'s main branch',
                           default=config_template.defaults['branch'])

  if Confirm.ask("[green]\nWould you like to configure some additional advanced options?"):

    if Confirm.ask("[yellow]↪ Would you like to configure the architecture of the autoencoders?"):
      s['hidden'] = Prompt.ask('  ↪ [italic]Configure the hidden layers arranged prior to the '
                               'latent space', default=str(config_template.defaults['hidden']))
      s['encoding'] = IntPrompt.ask('  ↪ [italic]Configure the size of the latence space',
                                    default=config_template.defaults['encoding'])
      s['activation'] = Prompt.ask('  ↪ [italic]Select the activation function to use throughout '
                                   'the network', default=config_template.defaults['activation'],
                                   choices=['tanh', 'sigmoid', 'relu', 'swish'])
      s['filename'] = Prompt.ask('  ↪ [italic]Set the name of the trained autoencoder',
                                 default=config_template.defaults['filename'])

    if Confirm.ask("[yellow]↪ Would you like to configure how the autoencoders are trained?"):
      s['epochs'] = IntPrompt.ask('  ↪ [italic]Configure the number of training epochs',
                                  default=config_template.defaults['epochs'])
      s['batch_size'] = IntPrompt.ask('  ↪ [italic]Configure the minibatch size',
                                      default=config_template.defaults['batch_size'])
      s['optimizer'] = Prompt.ask('  ↪ [italic]Configure the optimizer',
                                  default=config_template.defaults['optimizer'])
      s['learning_rate'] = FloatPrompt.ask('  ↪ [italic]Configure the optimizer\'s learning rate',
                                           default=config_template.defaults['learning_rate'])
      s['loss'] = Prompt.ask('  ↪ [italic]Configure the loss function',
                             default=config_template.defaults['loss'])
      s['noise'] = FloatPrompt.ask('  ↪ [italic]Configure the intensity of noise added during training',
                                   default=config_template.defaults['noise'])
      s['scale_factor'] = FloatPrompt.ask('  ↪ [italic]Configure the scale factor of the HPC samples',
                                          default=config_template.defaults['scale_factor'])

    if Confirm.ask("[yellow]↪ Would you like to configure the detection process?"):
      s['threshold'] = FloatPrompt.ask('  ↪ [italic]Set the % anomalous detection threshold',
                                       default=config_template.defaults['threshold'])

  parser = configparser.ConfigParser()
  parser.read_string(config_template.update(s))

  with open(configfile, 'w') as f:
    parser.write(f)

  save_counters()

  repo = Repo(os.getcwd(), search_parent_directories=True)
  with open(os.path.join(repo.working_tree_dir, '.gitignore'), 'a+') as f:
    f.seek(0, os.SEEK_SET)
    if len(f.read()) > 0:
      f.write('\n')
    f.write('# ------------ Added by AutoPerf ------------ #\n')
    f.write('.autoperf\n')
    f.write('# To track configs, uncomment the following:\n')
    f.write('# .autoperf/*\n')
    f.write('# !.autoperf/config.ini\n')
    f.write('# !.autoperf/COUNTERS\n')
    f.write('# ------------------------------------------- #')

  rprint('\n[bright_blue]Your responses (along with some default parameters) have been saved to:')
  rprint(f'  {configfile}\n')
