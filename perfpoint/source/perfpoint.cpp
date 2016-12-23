#include "perfpoint.h"
#include "xthread.h"
#include "threadmods.h"
#include "perfevent.h"

void perfpoint_START(){
  int tindex = getThreadIndex();
  thread_t* thread = xthread::getInstance().getThreadInfoByIndex(tindex);
  xPerf::getInstance().start_perf_counters(thread);
}

void perfpoint_END(){
  int tindex = getThreadIndex();
  thread_t* thread = xthread::getInstance().getThreadInfoByIndex(tindex);
  xPerf::getInstance().stop_and_record_perf_counters(thread);

}


