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
"""descriptions.py

This file contains helper functions for saving and accessing the descriptions
of PAPI counters.
"""

import os
import json

from autoperf.utils import getAutoperfDir

descriptions = {
    "PAPI_L1_DCM": "Level 1 data cache misses",
    "PAPI_L1_ICM": "Level 1 instruction cache misses",
    "PAPI_L2_DCM": "Level 2 data cache misses",
    "PAPI_L2_ICM": "Level 2 instruction cache misses",
    "PAPI_L3_DCM": "Level 3 data cache misses",
    "PAPI_L3_ICM": "Level 3 instruction cache misses",
    "PAPI_L1_TCM": "Level 1 total cache misses",
    "PAPI_L2_TCM": "Level 2 total cache misses",
    "PAPI_L3_TCM": "Level 3 total cache misses",
    "PAPI_CA_SNP": "Requests for a Snoop",
    "PAPI_CA_SHR": "Requests for access to shared cache line (SMP)",
    "PAPI_CA_CLN": "Requests for access to clean cache line (SMP)",
    "PAPI_CA_INV": "Cache Line Invalidation (SMP)",
    "PAPI_CA_ITV": "Cache Line Intervention (SMP)",
    "PAPI_L3_LDM": "Level 3 load misses",
    "PAPI_L3_STM": "Level 3 store misses",
    "PAPI_BRU_IDL": "Cycles branch units are idle",
    "PAPI_FXU_IDL": "Cycles integer units are idle",
    "PAPI_FPU_IDL": "Cycles floating point units are idle",
    "PAPI_LSU_IDL": "Cycles load/store units are idle",
    "PAPI_TLB_DM": "Data translation lookaside buffer misses",
    "PAPI_TLB_IM": "Instruction translation lookaside buffer misses",
    "PAPI_TLB_TL": "Total translation lookaside buffer misses",
    "PAPI_L1_LDM": "Level 1 load misses",
    "PAPI_L1_STM": "Level 1 store misses",
    "PAPI_L2_LDM": "Level 2 load misses",
    "PAPI_L2_STM": "Level 2 store misses",
    "PAPI_BTAC_M": "Branch target address cache (BTAC) misses",
    "PAPI_PRF_DM": "Pre-fetch data instruction caused a miss",
    "PAPI_L3_DCH": "Level 3 Data Cache Hit",
    "PAPI_TLB_SD": "Translation lookaside buffer shootdowns (SMP)",
    "PAPI_CSR_FAL": "Failed store conditional instructions",
    "PAPI_CSR_SUC": "Successful store conditional instructions",
    "PAPI_CSR_TOT": "Total store conditional instructions",
    "PAPI_MEM_SCY": "Cycles Stalled Waiting for Memory Access",
    "PAPI_MEM_RCY": "Cycles Stalled Waiting for Memory Read",
    "PAPI_MEM_WCY": "Cycles Stalled Waiting for Memory Write",
    "PAPI_STL_ICY": "Cycles with No Instruction Issue",
    "PAPI_FUL_ICY": "Cycles with Maximum Instruction Issue",
    "PAPI_STL_CCY": "Cycles with No Instruction Completion",
    "PAPI_FUL_CCY": "Cycles with Maximum Instruction Completion",
    "PAPI_HW_INT": "Hardware interrupts",
    "PAPI_BR_UCN": "Unconditional branch instructions executed",
    "PAPI_BR_CN": "Conditional branch instructions executed",
    "PAPI_BR_TKN": "Conditional branch instructions taken",
    "PAPI_BR_NTK": "Conditional branch instructions not taken",
    "PAPI_BR_MSP": "Conditional branch instructions mispredicted",
    "PAPI_BR_PRC": "Conditional branch instructions correctly predicted",
    "PAPI_FMA_INS": "FMA instructions completed",
    "PAPI_TOT_IIS": "Total instructions issued",
    "PAPI_TOT_INS": "Total instructions executed",
    "PAPI_INT_INS": "Integer instructions executed",
    "PAPI_FP_INS": "Floating point instructions executed",
    "PAPI_LD_INS": "Load instructions executed",
    "PAPI_SR_INS": "Store instructions executed",
    "PAPI_BR_INS": "Total branch instructions executed",
    "PAPI_VEC_INS": "Vector/SIMD instructions executed",
    "PAPI_FLOPS": "Floating Point Instructions executed per second",
    "PAPI_RES_STL": "Cycles processor is stalled on resource",
    "PAPI_FP_STAL": "Cycles any FP units are stalled",
    "PAPI_TOT_CYC": "Total cycles",
    "PAPI_IPS": "Instructions executed per second",
    "PAPI_LST_INS": "Total load/store instructions executed",
    "PAPI_SYC_INS": "Synchronization instructions executed",
    "PAPI_L1_DCH": "L1 data cache hits",
    "PAPI_L2_DCH": "L2 data cache hits",
    "PAPI_L1_DCA": "L1 data cache accesses",
    "PAPI_L2_DCA": "L2 data cache accesses",
    "PAPI_L3_DCA": "L3 data cache accesses",
    "PAPI_L1_DCR": "L1 data cache reads",
    "PAPI_L2_DCR": "L2 data cache reads",
    "PAPI_L3_DCR": "L3 data cache reads",
    "PAPI_L1_DCW": "L1 data cache writes",
    "PAPI_L2_DCW": "L2 data cache writes",
    "PAPI_L3_DCW": "L3 data cache writes",
    "PAPI_L1_ICH": "L1 instruction cache hits",
    "PAPI_L2_ICH": "L2 instruction cache hits",
    "PAPI_L3_ICH": "L3 instruction cache hits",
    "PAPI_L1_ICA": "L1 instruction cache accesses",
    "PAPI_L2_ICA": "L2 instruction cache accesses",
    "PAPI_L3_ICA": "L3 instruction cache accesses",
    "PAPI_L1_ICR": "L1 instruction cache reads",
    "PAPI_L2_ICR": "L2 instruction cache reads",
    "PAPI_L3_ICR": "L3 instruction cache reads",
    "PAPI_L1_ICW": "L1 instruction cache writes",
    "PAPI_L2_ICW": "L2 instruction cache writes",
    "PAPI_L3_ICW": "L3 instruction cache writes",
    "PAPI_L1_TCH": "L1 total cache hits",
    "PAPI_L2_TCH": "L2 total cache hits",
    "PAPI_L3_TCH": "L3 total cache hits",
    "PAPI_L1_TCA": "L1 total cache accesses",
    "PAPI_L2_TCA": "L2 total cache accesses",
    "PAPI_L3_TCA": "L3 total cache accesses",
    "PAPI_L1_TCR": "L1 total cache reads",
    "PAPI_L2_TCR": "L2 total cache reads",
    "PAPI_L3_TCR": "L3 total cache reads",
    "PAPI_L1_TCW": "L1 total cache writes",
    "PAPI_L2_TCW": "L2 total cache writes",
    "PAPI_L3_TCW": "L3 total cache writes",
    "PAPI_FML_INS": "Floating Multiply instructions",
    "PAPI_FAD_INS": "Floating Add instructions",
    "PAPI_FDV_INS": "Floating Divide instructions",
    "PAPI_FSQ_INS": "Floating Square Root instructions",
    "PAPI_FNV_INS": "Floating Inverse instructions"
}


def save_descriptions():
  """Save the performance counter descriptions in this file to disk."""
  description_file = getAutoperfDir('descriptions.json')
  json.dump(descriptions, open(description_file, 'w'))


def get_descriptions() -> list:
  """Retrieve a list of performance counter descriptions housed in the
  descriptions.json file.

  Returns:
      A list of performance counter descriptions.
  """
  description_file = getAutoperfDir('descriptions.json')
  if os.path.exists(description_file):
    with open(description_file, 'r') as f:
      return f.readlines()
  else:
    return descriptions


def get_description(key: str) -> str:
  """Retrieve the description of a particular performance counter.

  Args:
      key: The performance counter to describe.

  Returns:
      The description. If the performance counter can't be found, returns None.
  """
  try:
    return get_descriptions()[key]
  except KeyError:
    return None
