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
#ifndef NEXT_EVENT_SET
char* g_event_list[NUM_EVENTS] =
{
  "PAPI_TOT_INS",
  "PAPI_L1_ICM",
  "PAPI_L1_DCM",
  "PAPI_L2_TCM",
//  "PAPI_L3_TCM",
//  "MEM_LOAD_UOPS_LLC_MISS_RETIRED:REMOTE_HITM"
  "OFFCORE_RESPONSE_0:LLC_HITM",
//  "OFFCORE_RESPONSE_1:LLC_HITM"
};
//int papi_events[NUM_EVENTS] = 
//{
//	PAPI_TOT_INS,
//	PAPI_L1_DCM,
//	PAPI_L1_ICM,
//	PAPI_L2_TCM,
//	PAPI_L3_TCM
//};
#else
int papi_events[NUM_EVENTS] = 
{
//	PAPI_FP_INS
	PAPI_BR_INS,
//	PAPI_BR_MSP,
	PAPI_BR_NTK,
//	PAPI_MEM_LOAD_UOPS_LLC_HIT_RETIRED|XSNP_HITM
//	PAPI_TLB_DM,
	
//	PAPI_TLB_IM
//	PAPI_STL_ICY
//	OFFCORE_RESPONSE_1|LLC_HITM | SNP_MISS,
//	PAPI_BR_MSP, 
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


