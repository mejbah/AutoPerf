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
  int num_of_hw_counters;
  bool event_multiplex;
  int eventSet[xdefines::MAX_THREADS]; //eventSet for each thread

  xPerf(){}

public:
  static xPerf& getInstance(){
	static char buf[sizeof(xPerf)];
	static xPerf *singleton = new (buf) xPerf();
	return *singleton;
  }
  
  void init(int numberOfEvents){
	//memset(&perf_event, 0, sizeof(int));
	//memset(&eventSet, PAPI_NULL, sizeof(int));
	num_events = numberOfEvents;
	event_multiplex = false;
	//perf_event = new int[numberOfEvents];
	//for(int i=0; i < numberOfEvents; i++){
	//  perf_event[i]= events[i];
	//}

	/* Init PAPI library */
	int retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
     fprintf(stderr,  "papi_library_init : retval %s, file %s, line %d\n",
			   PAPI_strerror(retval), __FILE__, __LINE__);
	}

	printf("Initialized PAPI library\n");

	
    const PAPI_component_info_t* cmpinfo;
  
    num_of_hw_counters = PAPI_num_hwctrs(  );
  
    if (num_of_hw_counters<=0) {
    	cmpinfo = PAPI_get_component_info( 0 );
    	fprintf(stderr,"\nComponent %s disabled due to %s\n",
    		cmpinfo->name, cmpinfo->disabled_reason);
    }
  
    if(num_of_hw_counters < num_events){
	  fprintf(stderr, "\n EVENT MULTIPLEX enabled as we found more events(%d) than avaliable counters(%d)\n",
				 num_events, num_of_hw_counters);
	  event_multiplex = true;
	  if(PAPI_multiplex_init() != PAPI_OK){
		fprintf(stderr,  "papi_multiplex_init : retval %s, file %s, line %d\n",
			   PAPI_strerror(retval), __FILE__, __LINE__);	  
	  }
	}


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
    int tid =  thd->index;
    eventSet[tid]  = PAPI_NULL;
    
    /* create the eventset */
    int retval = PAPI_create_eventset( &eventSet[tid] );
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_create_eventset File %s Line %d retval %d\n",
			 __FILE__, __LINE__, retval );
    }

	if(event_multiplex){
	  if(PAPI_set_multiplex(eventSet[tid]) != PAPI_OK){
		fprintf( stderr, "Error : PAPI_set_multiplex File %s Line %d retval %d\n",
			 __FILE__, __LINE__, retval );	
	  }
	}

    for(int i=0; i<num_events; i++){ 
	  int native = 0x0;
	  retval = PAPI_event_name_to_code(g_event_list[i],&native);
	  if( retval != PAPI_OK ){
		  fprintf( stderr, "ERROR: PAPI_event_code_to_name %s File %s Line %d retval %s\n", 
				   g_event_list[i], __FILE__, __LINE__,PAPI_strerror(retval) );
	  }
#ifdef PERFPOINT_DEBUG
	  else{
		
		PAPI_event_code_to_name(native, event_name);
		fprintf(stderr, "Native event code %x for event %s\n", native, event_name);
	  }
#endif
	  retval = PAPI_add_event( eventSet[tid], native);
  	  if( retval != PAPI_OK ){
  	    	  fprintf( stderr, "ERROR: PAPI_add_event %s File %s Line %d retval %s\n", 
				   g_event_list[i], __FILE__, __LINE__,	  PAPI_strerror(retval) );
  	  }
    }

  
    //for(int i=0; i<num_events; i++){ 
	//  retval = PAPI_add_event( eventSet[tid], perf_event[i] );
  	//  PAPI_event_code_to_name(perf_event[i], event_name);
  	//  if( retval != PAPI_OK ){
  	//    	  fprintf( stderr, "ERROR: File %s Line %d retval %s\n", 
	//			  event_name,  __FILE__, __LINE__, PAPI_strerror(retval) );
  	//  }
    //}
		//retval = PAPI_event_name_to_code("PAPI_TOT_INS",&native);
	
	//retval = PAPI_add_event( eventSet[tid], native );
	//if( retval != PAPI_OK ){
  	//    	  fprintf( stderr, "Native event name to code %s\n", 
	//			   strerror(retval) );
  	//}
	//else{
	//  fprintf(stderr, "Native event code %x\n", native);
	//}
  
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
