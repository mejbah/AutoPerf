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

#ifdef PERF_EVENT
#ifdef PERFPOINT_EVENT_SET_1
char* g_event_list[NUM_EVENTS] =
{
  "PAPI_TOT_INS", //1
  "PAPI_L1_ICM",  //2
  "PAPI_L1_DCM",  //3
  "PAPI_L2_DCM",  //4
  "PAPI_L2_ICM"  //5
};
#elif PERFPOINT_EVENT_SET_2
char* g_event_list[NUM_EVENTS] =
{
  "PAPI_TOT_INS",
  "PAPI_BR_UCN", //6
  "PAPI_BR_MSP", //7
  "L2_LINES_IN:S",//8
  "PAPI_BR_CN"
  //"PAPI__"
  //"perf::PAGE-FAULTS",
};
#else
char* g_event_list[NUM_EVENTS] =
{
  "PAPI_TOT_INS",
  "RESOURCE_STALLS:ANY",//10
  "RESOURCE_STALLS:SB",//11
  "OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_HITM:HITM",//12
  "PAPI_TOT_CYC"
//  "perf::PERF_COUNT_SW_CONTEXT_SWITCHES",//13
//  "MACHINE_CLEARS:MEMORY_ORDERING",
//  "MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_HITM",
//  "MEM_LOAD_LLC_HIT_RETIRED:XSNP_HITM"
};

#endif
#endif

void initializer (void) {
  init_real_functions();

#ifdef PERF_EVENT
  xPerf::getInstance().init(NUM_EVENTS);
#endif

  xrun::getInstance().initialize();

  initialized = true;
  
  fprintf(stderr, "Now we have initialized successfuuly\n"); 
  
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


