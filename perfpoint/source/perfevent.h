#ifndef PERF_SAMPLER
#define PERF_SAMPLER

#include "papi.h"
#include "pthread.h"
#include "threadmods.h"
#include "xdefines.h"




class xPerf {

private:
  int *perf_event;
  int num_events;
  int eventSet[xdefines::MAX_THREADS]; //eventSet for each thread

  xPerf(){}

public:
  static xPerf& getInstance(){
	static char buf[sizeof(xPerf)];
	static xPerf *singleton = new (buf) xPerf();
	return *singleton;
  }
  
  void init(int *events, int numberOfEvents){
	//memset(&perf_event, 0, sizeof(int));
	//memset(&eventSet, PAPI_NULL, sizeof(int));
	num_events = numberOfEvents;
	perf_event = new int[numberOfEvents];
	for(int i=0; i < numberOfEvents; i++){
	  perf_event[i]= events[i];
	}

	/* Init PAPI library */
	int retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
     fprintf(stderr,  "PAPI_library_init : retval %s, file %s, line %d\n",
			   PAPI_strerror(retval), __FILE__, __LINE__);
	}

	printf("Initialized PAPI library\n");

  }
#if 0  //TODO: fix the parameter
  void initPerfEventsforThread(pthread_t thread_self){
  
	int retval = PAPI_thread_init( ( unsigned long ( * )( void ) )
                   ( thread_self ) );
	if ( retval != PAPI_OK ) {
	  fprintf(stderr, "PAPI_thread_init : retval %s, file %s, line %d\n",
		PAPI_strerror(retval), __FILE__, __LINE__);
	}
  // retval = pthread_attr_setscope( &attr, PTHREAD_SCOPE_SYSTEM );
  //  if ( retval != 0 )

  }
#endif
  int registerThreadForPAPI(){
	int retval = PAPI_register_thread(  );
	return retval;
  }

  int unregisterThreadForPAPI(){
	int retval = PAPI_unregister_thread(  );
	return retval;
  }  


  void set_perf_events( thread_t *thd ){
  
    char event_name[PAPI_MAX_STR_LEN];
    int counters = 0;
    int tid =  thd->index;
    eventSet[tid]  = PAPI_NULL;
  
    const PAPI_component_info_t* cmpinfo;
  
    counters = PAPI_num_hwctrs(  );
  
    if (counters<=0) {
    	cmpinfo = PAPI_get_component_info( 0 );
    	fprintf(stderr,"\nComponent %s disabled due to %s\n",
    		cmpinfo->name, cmpinfo->disabled_reason);
    }
  
    if(counters < num_events){
	  fprintf(stderr, "\n More events(%d) than avaliable counters(%d)\n",
				 num_events, counters);
    }
  
    
    /* create the eventset */
    int retval = PAPI_create_eventset( &eventSet[tid] );
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_create_eventset File %s Line %d retval %d\n",
			 __FILE__, __LINE__, retval );
    }
  
    //retval = PAPI_add_events( thd->eventSet, papi_events, NUM_EVENTS );
    for(int i=0; i<num_events; i++){ 
	  retval = PAPI_add_event( eventSet[tid], perf_event[i] );
  	  PAPI_event_code_to_name(perf_event[i], event_name);
  	  if( retval != PAPI_OK ){
  	    	  fprintf( stderr, "Error : PAPI_add_event %s File %s Line %d retval %s\n", 
				  event_name,  __FILE__, __LINE__, PAPI_strerror(retval) );
  	  }
  	  //else{
  	  //  printf("\n %s event added\n", event_name);
  	  //}
  
    }
  
  }

  int start_perf_counters( thread_t *thd, int mark )
  {
    //int eventSet = set_perf_events();
	int tid = thd->index;
	thd->current_mark = mark;
    int retval = PAPI_start(eventSet[tid]);
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_start retval %s\n", PAPI_strerror(retval) );
    }
	//  int retval = PAPI_start_counters(papi_events, NUM_EVENTS);
	//  if ( retval != PAPI_OK) {
	//      fprintf(stderr, "PAPI_start_counters - FAILED , errorcode: %d\n",retval);
	//      exit(1);
	//  }
    return 1;
    
  }
  
  int stop_and_record_perf_counters( thread_t *thd )
  {
  
    long long *values;
    values =  ( long long * ) malloc( ( size_t ) num_events *
  									sizeof ( long long ) );
	int tid = thd->index;
    int retval = PAPI_stop( eventSet[tid], values );
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_stop File %s Line %d retval %d\n",
		   __FILE__, __LINE__, retval );
    }
	//   if (PAPI_stop_counters(values, NUM_EVENTS) != PAPI_OK) {
	//      fprintf(stderr, "PAPI_stop_counters - FAILED\n");
	//      exit(1);
	//  }
  
  
    //for(int i=0; i<NUM_EVENTS; i++){
    //  printf("%lld\n", values[i]);
    //}
    //record the values in sample
    int next_index = thd->perfRecords.get_next_index();
	(thd->perfRecords.getEntry(next_index))->mark = thd->current_mark;
    for(int i=0; i<num_events; i++){
	  (thd->perfRecords.getEntry(next_index))->count[i] = values[i];
    }
    //sample->cycleCount = values[1];
    //sample->cycleCount = values[2];
    //sample->cycleCount = values[1];
     
  
  }

};







#endif
