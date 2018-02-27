/***
Copyright (c) 2018 Mejbah ul Alam, Justin Gottschlich, Abdullah Muzahid

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

***/

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdarg.h>
#include "xthread.h"
#include "xrun.h"
#include "recordentries.hh"
#include "perfevent.h"


void initializer (void) __attribute__((constructor));
void finalizer (void)   __attribute__((destructor));
bool initialized = false;
  
__thread thread_t * current = NULL;	
bool _isMultithreading = false;

#if 0
char* g_event_list[NUM_EVENTS] =
{
  "PAPI_TOT_INS", //0
  "OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_REMOTE_DRAM", 
  "OFFCORE_RESPONSE_0:ANY_REQUEST:HITM",
  "L3_LAT_CACHE:MISS",
  "L3_LAT_CACHE:REFERENCE",
  "PAPI_L1_ICM",  //1
  "PAPI_L1_DCM",  //2
  "PAPI_L2_ICM",  //3
  "L2_LINES_IN:S",//4
  "PAPI_L2_TCM", //5
  "PAPI_L3_TCM", //6
  "PAPI_TLB_IM", //7
  "PAPI_L1_LDM", //8
  "PAPI_L1_STM", //9
  "PAPI_L2_STM", //10
  "PAPI_STL_ICY", //11
  "PAPI_BR_CN", //12
  "PAPI_BR_NTK", //13
  "PAPI_BR_MSP", //14
  "PAPI_LD_INS", //15
  "PAPI_SR_INS", //16
  "PAPI_BR_INS", //17
  "PAPI_TOT_CYC", //18
  "PAPI_L2_DCA", //19
  "PAPI_L2_DCR", //20
  "PAPI_L3_DCR", //21
  "PAPI_L2_DCW", //22
  "PAPI_L3_DCW", //23
  "PAPI_L2_ICH", //24   0x8000004a  Yes   No   Level 2 instruction cache hits
  "PAPI_L2_ICA", //25  0x8000004d  Yes   No   Level 2 instruction cache accesses
  "PAPI_L3_ICA", //26  0x8000004e  Yes   No   Level 3 instruction cache accesses
  "PAPI_L2_ICR", //27  0x80000050  Yes   No   Level 2 instruction cache reads
  "PAPI_L3_ICR", //28  0x80000051  Yes   No   Level 3 instruction cache reads
  "PAPI_L2_TCA", //29  0x80000059  Yes   Yes  Level 2 total cache accesses
  "PAPI_L3_TCA", //30  0x8000005a  Yes   No   Level 3 total cache accesses
  "PAPI_L2_TCR",  //31 0x8000005c  Yes   Yes  Level 2 total cache reads
  "PAPI_L3_TCR",  //32 0x8000005d  Yes   Yes  Level 3 total cache reads
  "PAPI_L2_TCW", //33  0x8000005f  Yes   No   Level 2 total cache writes
  "PAPI_L3_TCW", //34  0x80000060  Yes   No   Level 3 total cache writes
//  "PAPI_FDV_INS" //32 0x80000063  Yes   No   Floating point divide instructions
  
};

#endif

void initializer (void) {
  init_real_functions();

#ifdef PERF_EVENT
  xPerf::getInstance().init();
#endif
  xrun::getInstance().initialize();

  initialized = true;
  
  fprintf(stderr, "Perfpoint initialization complete\n"); 
  
}

void finalizer (void) {
  initialized = false;
  xrun::getInstance().finalize();
}



  // Intercept the pthread_create function.
int pthread_create (pthread_t * tid, const pthread_attr_t * attr, void *(*start_routine) (void *), void * arg)
{
  //printf("In my thread_create\n");
  return xthread::getInstance().thread_create(tid, attr, start_routine, arg);
}

  // Intercept the pthread_join function. Thus, 
  // we are able to know that how many threads have exited.
int pthread_join(pthread_t thread, void **retval) {
  return xthread::getInstance().thread_join(thread, retval);
}


