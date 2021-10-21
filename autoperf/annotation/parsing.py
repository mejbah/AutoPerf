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
"""parsing.py

This file contains all code required to create, traverse, and parse the abstract
syntax trees (ASTs) of C/C++ source files via Clang.
"""

import os
import logging
from pathlib import Path
from glob import glob

import clang.cindex

# from rich import print as rprint
from rich import box
from rich.text import Text
from rich.table import Table
from rich.style import Style

log = logging.getLogger('rich')

# valid c/c++ extensions
extensions = ['.cpp', '.hpp', '.c', '.h', '.cc', '.hh']

# ------------------------------------------------------------------------------


def get_parser_options(fast: bool = False, preproc: bool = False) -> int:
  """Set and retrieve parser options based on user input.

  Args:
      fast (bool, optional): Flag for ultra-fast parsing at the expense of accuracy. Defaults to False.
      preproc (bool, optional): Include preprocessor macros in parsing. Defaults to False.

  Returns:
      An integer bitfield representing the selected options.
  """

  # clang.cindex.TranslationUnit does not have all latest flags
  CXTranslationUnit_None = 0x0
  CXTranslationUnit_DetailedPreprocessingRecord = 0x01
  CXTranslationUnit_Incomplete = 0x02
  CXTranslationUnit_PrecompiledPreamble = 0x04
  CXTranslationUnit_CacheCompletionResults = 0x08
  CXTranslationUnit_ForSerialization = 0x10
  CXTranslationUnit_CXXChainedPCH = 0x20
  CXTranslationUnit_SkipFunctionBodies = 0x40
  CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = 0x80
  CXTranslationUnit_CreatePreambleOnFirstParse = 0x100
  CXTranslationUnit_KeepGoing = 0x200
  CXTranslationUnit_SingleFileParse = 0x400
  CXTranslationUnit_LimitSkipFunctionBodiesToPreamble = 0x800
  CXTranslationUnit_IncludeAttributedTypes = 0x1000
  CXTranslationUnit_VisitImplicitAttributes = 0x2000
  CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles = 0x4000
  CXTranslationUnit_RetainExcludedConditionalBlocks = 0x8000

  default_parser_options = (CXTranslationUnit_VisitImplicitAttributes  # only way to get class methods parsed correctly
                            | CXTranslationUnit_CacheCompletionResults  # potentially speeds up parsing of large baselines
                            | CXTranslationUnit_SkipFunctionBodies  # skips parsing inside of functions, much faster
                            | CXTranslationUnit_LimitSkipFunctionBodiesToPreamble  # only skips function preambles
                            | CXTranslationUnit_RetainExcludedConditionalBlocks  # keep includes inside ifdef blocks
                            | CXTranslationUnit_KeepGoing)  # don't stop on errors

  # Don't parse include files recursively - major speedup
  if fast:
    default_parser_options |= CXTranslationUnit_SingleFileParse

  # Needed to parse preprocessor macros
  if preproc:
    default_parser_options |= CXTranslationUnit_DetailedPreprocessingRecord

  return default_parser_options


def get_kind(kind: clang.cindex.CursorKind) -> str:
  """Retrieve the kind of clang Cursor - method, function, constructor, etc.

  Args:
      kind: What Clang found.

  Returns:
      Plaintext name of the type of Cursor.
  """
  found_kind = None

  if kind == clang.cindex.CursorKind.CXX_METHOD:
    found_kind = 'methods'
  elif kind == clang.cindex.CursorKind.CONSTRUCTOR:
    found_kind = 'constructors'
  elif kind == clang.cindex.CursorKind.FUNCTION_DECL:
    found_kind = 'functions'
  elif kind == clang.cindex.CursorKind.FUNCTION_TEMPLATE:
    found_kind = 'functions'
  elif kind == clang.cindex.CursorKind.INCLUSION_DIRECTIVE:
    found_kind = 'includes'
  elif kind == clang.cindex.CursorKind.MACRO_DEFINITION:
    found_kind = 'macros'

  return found_kind

# ------------------------------------------------------------------------------


class ParseResult():
  """Wrapper of the result of Clang's file parsing."""

  d = None
  filename = None

  def __init__(self, filename: str):
    self.d = {}
    self.filename = filename

  def has_kind(self, kind: str) -> bool:
    """Determines whether or not the result has a type of Cursor yet.

    Args:
        kind: Type of cursor.

    Returns:
        Whether or not it's in the current dictionary.
    """
    return kind in self.d

  def handle_overloads(self, kind: str, cursor: clang.cindex.Cursor, inner_dict: dict) -> dict:
    """Handle function overloading by including arguments in name

    Args:
        kind: Type of cursor.
        cursor: Clang's Cursor object.
        inner_dict: Dictionary that is inserted for each parsed object.

    Returns:
        Modified inner_dict.
    """
    if self.has_kind(kind):
      for other_cursor in self.d[kind].values():
        if other_cursor['spelling'] == cursor.spelling \
           and other_cursor['definition'] == cursor.is_definition():

          inner_dict['use_displayname'] = True
          other_cursor['use_displayname'] = True
    return inner_dict

  def insert(self, kind: str, cursor: clang.cindex.Cursor):
    """Tracks the contents of a Cursor in the internal dictionary.

    Args:
        kind: Type of cursor.
        cursor: Clang's Cursor object.
    """
    inner_dict = {'start': cursor.location.line,
                  'end': cursor.extent.end.line,
                  'indent': cursor.extent.start.column - 1,
                  'definition': cursor.is_definition(),
                  'spelling': cursor.spelling,
                  'displayname': cursor.displayname,
                  'use_displayname': False}

    if self.has_kind(kind):
      inner_dict = self.handle_overloads(kind, cursor, inner_dict)
      self.d[kind][cursor.hash] = inner_dict
    else:
      self.d[kind] = {cursor.hash: inner_dict}

  def get_table(self) -> Table:
    """Generate and return a Rich table containing all of the file's parsed information.

    Returns:
        The generated table.
    """
    title = Text(f'{self.filename}', style=Style(italic=True, bold=True, link=f'file://{self.filename.resolve()}'))
    table = Table(title=title, box=box.HEAVY_HEAD)

    if len(self.d.items()) == 0:
      table.add_column('No Functions Found', justify="center", style="red")
      if self.filename.suffix in extensions:
        table.add_row('Preprocessor Macros / Variable Decl. Only')
      else:
        table.add_row('Not a C/C++ Function')
      return table

    table.add_column("Type", justify="center", style="cyan", no_wrap=True)
    table.add_column("Name", justify="center", style="white")
    table.add_column("Line #", justify="center", style="cyan")
    table.add_column("Indent", justify="center", style="cyan")
    table.add_column("Definition?", justify="center", style="cyan", no_wrap=True)

    for kind, cursor in self.d.items():
      for i, (name, info) in enumerate(cursor.items()):
        func_name = info['displayname'] if info['use_displayname'] else info['spelling']

        line_num = f"{info['start']}:{info['end']}" if info['start'] != info['end'] else f"{info['start']}"
        end = i == (len(cursor) - 1)

        func_type = Text('n/a', 'bright_black')
        if kind in ['functions', 'methods', 'constructors']:
          func_type = Text('yes', 'green') if info['definition'] else Text('no', 'red')

        table.add_row(f"{kind.capitalize()[:-1]}",
                      func_name,
                      line_num,
                      f"{info['indent']}",
                      func_type,
                      end_section=end)
    return table

  def get_stats(self) -> dict:
    """Retrieve high-level stats of a particular file - how many functions are
    declared, etc.

    Returns:
        Dictionary containing number of declared & defined functions, methods,
        and constructors.
    """
    stats = {'declared': 0, 'defined': 0}
    keys = ['functions', 'methods', 'constructors']
    for k in keys:
      if k in self.d:
        for func in self.d[k].values():
          if func['definition']:
            stats['defined'] += 1
          else:
            stats['declared'] += 1
    return stats

# ------------------------------------------------------------------------------


index = None
find_clang = glob(os.environ['CONDA_PREFIX'] + '/lib/libclang.*')
default_lib = None if len(find_clang) == 0 else find_clang[0]

makefile_found = \
    """[bright_magenta]Makefile found at:[/bright_magenta]
    [link file://{mkfile}][code]{mkfile}[/code]"""

makefile_args = \
    """[bright_magenta]Using the following compile/link arguments:[/bright_magenta]
    [code]{args}[/code]"""

no_makefile_args = "[bright_magenta]No [code]CFLAGS[/code] specified.[/bright_magenta]"

# Maximum number of directories to search upward for a Makefile
MAX_ATTEMPTS = 3


class ClangParser():
  """High-level Clang parsing object that traverses a given directory."""

  fast = None
  preproc = None
  parse_opts = None
  args = ''

  def __init__(self, fast=False, preproc=False, lib=None):
    lib = default_lib if lib is None else lib  # Handle when no lib is passed
    if not clang.cindex.Config.loaded:
      clang.cindex.Config.set_library_file(lib)

    global index
    index = clang.cindex.Index.create()

    self.fast = fast
    self.preproc = preproc
    self.parse_opts = get_parser_options(fast=fast, preproc=preproc)
    self.get_args_from_makefile()

  def get_args_from_makefile(self, root: Path = Path('.'), attempt: int = 0):
    """Use the given Makefile to identify any CFLAGS necessary to compile the
    program under test; Clang needs these to traverse the AST.

    Args:
        root: Location to search for a Makefile. Defaults to Path('.').
        attempt: Number of directories that have been searched. Defaults to 0.
    """
    # TODO: Makefiles can include other Makefiles, each with their own
    #       additional arguments. Should probably handle this.

    root = root.resolve()
    mkfile = root / 'Makefile'
    try:
      with open(mkfile) as f:
        log.info(makefile_found.format(mkfile=mkfile))
        for item in f.read().split('\n'):
          # Remove 'CFLAGS =', any comments, and strip whitespace
          if item.startswith('CFLAGS'):
            args = item.split('=')[1].split('#')[0].strip()
            log.info(makefile_args.format(args=args))
            self.args = args
            return

    except FileNotFoundError:
      if attempt < MAX_ATTEMPTS:
        self.get_args_from_makefile(root.parent, attempt + 1)
      else:
        log.warning('\n[bright_magenta]No Makefile found.[/bright_magenta]')

    log.info(no_makefile_args)

  def parse(self, filename: str, args: str = '') -> ParseResult:
    """Parses a given file with the specified CFLAGS in `args`.

    Args:
        filename: Path to the file that needs to be parsed.
        args: CFLAGS needed for AST traversal. Defaults to ''.

    Returns:
        A ParseResult object.
    """
    # Needed for correct AST
    args = '-xc++ -std=c++98 ' + self.args + args
    tu = index.parse(str(filename), args=args.split(), options=self.parse_opts)

    # Iterate through the abstract syntax tree (AST)
    result = ParseResult(filename=filename)
    for cursor in tu.cursor.walk_preorder():

      # Only include cursors generated by the file in question
      if str(cursor.location.file).startswith(str(filename)):

        kind = get_kind(cursor.kind)
        if kind is not None:
          result.insert(kind, cursor)
        else:
          continue

    return result
