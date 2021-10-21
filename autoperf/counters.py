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
"""counters.py

This file contains helper functions for accessing the selected PAPI counters,
along with some helpful defaults.
"""

import os

from autoperf.utils import getAutoperfDir

counters = \
    """PAPI_L1_ICM
PAPI_L1_DCM
PAPI_L2_ICM
PAPI_L2_TCM
PAPI_L3_TCM
PAPI_TLB_IM
PAPI_L1_LDM
PAPI_L1_STM
PAPI_L2_STM
PAPI_STL_ICY
PAPI_BR_CN
PAPI_BR_NTK
PAPI_BR_MSP
PAPI_LD_INS
PAPI_SR_INS
PAPI_BR_INS
PAPI_TOT_CYC
PAPI_L2_DCA
PAPI_L2_DCR
PAPI_L3_DCR
PAPI_L2_DCW
PAPI_L3_DCW
PAPI_L2_ICH
PAPI_L2_ICA
PAPI_L3_ICA
PAPI_L2_ICR
PAPI_L3_ICR
PAPI_L2_TCA
PAPI_L3_TCA
PAPI_L2_TCR
PAPI_L3_TCR
PAPI_L2_TCW
PAPI_L3_TCW"""


def save_counters():
  """Save the performance counters in this file to disk."""
  counter_file = getAutoperfDir('COUNTERS')
  with open(counter_file, 'w') as f:
    f.write(counters)


def get_counters() -> list:
  """Retrieve a list of performance counters housed in the COUNTERS file.

  Returns:
      A list of performance counters.
  """
  counter_file = getAutoperfDir('COUNTERS')
  if os.path.exists(counter_file):
    with open(counter_file, 'r') as f:
      return f.readlines()
  else:
    return counters


def get_num_counters() -> int:
  """Retrieve the number of performance counters housed in the COUNTERS file.

  Returns:
      The number of performance counters.
  """
  return len(get_counters())
