LEGACY
==================================
Profiling
*************
- Autoperf uses [PAPI](http://icl.cs.utk.edu/papi/index.html) interface for performance counters. Extract and install from source (papi-5.5.1.tar.gz)
    - Build profiler library:
    - AutoPerf/proflier
    - make
- Prepare candidate program:
    - Annotate functions:
    - add header : `#include "perfpoint.h"`
    - mark start : `perfpoint_START(marker_id)`
    - mark end : `perfpoint_END()`
    - NOTE: use mark_id as parameter to uniquely identify code region
    - Link profiler library `libperfpoint.so` with candidate "program" or use LD_PRELOAD=/path/to/libperfpoint.so (example: Default.mk in tests dir)
- Run program:
    - create list of performance counter names in file named `COUNTERS` in binary path [ or copy the file Autoperf/profiler/scripts/COUNTERS]
    - copy Autoperf/profiler/scripts/run_profiler.py in binary path
    - set `PERFPOINT_LIB_PATH="path/to/libperfpoint"` in `run_profiler.py`
    - `python run_profiler.py PATH/TO/OUTPUT/PROFILE_DATA PROGRAM_BINARY PROGRAM_ARGS [runID]`
    - NOTE: use optional `runID` to store multiple executions data in separate directories

Anomaly Detection
**************************
- cd AutoPerf/autoperf
- set `NUMBER_OF_COUNTERS` and `NO_OF_HIDDEN_LAYER_TO_SEARCH` in `configs.py`
- `python autoperf.py PATH/TO/PROFILE_DATA_FOR_TRAINING PATH/TO/PROFILE_DATA_FOR_TEST PATH/TO/OUTPUT_DETECTION_RESULTS`
- Output files:
    - accuracy.out : Detection results
    - network_training.log : Network configs + training error + validation error