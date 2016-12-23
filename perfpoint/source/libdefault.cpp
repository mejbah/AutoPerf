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
int papi_events[NUM_EVENTS] = 
{
	PAPI_TOT_INS,
	PAPI_TOT_CYC,
	PAPI_L1_DCM 
//	PAPI_BR_MSP, 
};
#endif

void initializer (void) {
  init_real_functions();
#ifdef PAPI_PERF
  //initialize papi library
  initPerfEvents();
#endif
#ifdef PERF_EVENT
  xPerf::getInstance().init(papi_events, NUM_EVENTS);
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


