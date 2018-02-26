/***
Copyright 2018 Mejbah ul Alam mejbah.alam@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

***/



#ifndef PERF_SAMPLER
#define PERF_SAMPLER
#include "papi.h"
#include "pthread.h"
#include "threadmods.h"
#include "xdefines.h"
#include "cstdlib"




class xPerf {

private:
  int *perf_event;
  int num_events;
  int event_number_in_global_list; 
  //unsigned int monitoring_event_code; //used input/defined event code to monitor other than TOT_INST. Perfpoint sample TOT_INS and this event
  int monitoring_event_code; //used input/defined event code to monitor other than TOT_INST. Perfpoint sample TOT_INS and this event
  //char monitoring_event_name[100];
  char *monitoring_event_name;
  int num_of_hw_counters;
  bool eventset_initialized;
  bool event_multiplex;
  int eventSet[xdefines::MAX_THREADS]; //eventSet for each thread

  xPerf(){}

public:
  static xPerf& getInstance(){
    static char buf[sizeof(xPerf)];
    static xPerf *singleton = new (buf) xPerf();
    return *singleton;
  }
  
  void init(){
    num_events = xdefines::NUM_EVENTS_TO_MONITOR;
    event_multiplex = false;
    eventset_initialized = false;

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

    
    setMonitoringEventCode();

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

  void setMonitoringEventCode(){
    char* env_p = std::getenv("PERFPOINT_EVENT_INDEX");
    int event_index_in_list = atoi(env_p);
    //event name can be found at the provided line number of the file "COUNTERS"
    FILE * fp;
    size_t len = 0;
    size_t nread;
    ssize_t read;

    //fp = fopen(xdefines::COUNTER_LIST_FILE, "r");
    fp = fopen("COUNTERS", "r");
    if (fp == NULL){
      fprintf(stderr, "PERFPOINT:: cannot open COUNTERS file\n");
      exit(1); 

    }
    int i=0;

    //while (fgets(monitoring_event_name, sizeof(monitoring_event_name), fp)) {
    monitoring_event_name = NULL;
    while ((nread = getline(&monitoring_event_name, &len, fp)) != -1) {
      if(i==event_index_in_list){
        printf("Monitoring event name: %s", monitoring_event_name);
        break;
      }
      i++;
    }
    fclose(fp);
   
    if(i!=event_index_in_list){
      fprintf(stderr,"PERFPOINT:: wrong input event index %d\n", event_index_in_list);
      exit(1);
    } 
    //trim new line char
    if(nread>0 && monitoring_event_name[nread-1] == '\n'){
      monitoring_event_name[nread-1] = '\0';
    }
    

    monitoring_event_code = 0x0;
	  int retval = PAPI_event_name_to_code(monitoring_event_name,&monitoring_event_code); 
	  //int retval = PAPI_event_name_to_code(event_name,&monitoring_event_code); 
	  //int retval = PAPI_event_name_to_code("OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_REMOTE_DRAM",&monitoring_event_code); 

	  if( retval != PAPI_OK ){
	    fprintf( stderr, "ERROR: PAPI_event_name_to_code %s File %s Line %d retval %s\n", 
				   monitoring_event_name, __FILE__, __LINE__,PAPI_strerror(retval) );
      exit(1);
	  }
    
  }

  void setMonitoringEventName(){ //from  env variable
    char* env_p = std::getenv("PERFPOINT_EVENT_INDEX");
    event_number_in_global_list = atoi(env_p);
    assert(event_number_in_global_list >=0 && event_number_in_global_list < NUM_EVENTS);
  }
  char* getMonitringEventName(){
    return monitoring_event_name;
  }

  //set TOT_INS and one other event 
  void set_perf_events( thread_t *thd ){
    //setMonitoringEventName(); //from  env variable
    char event_name[PAPI_MAX_STR_LEN];
    int tid =  thd->index;
    eventSet[tid]  = PAPI_NULL;
	
    /* create the eventset */
    int retval = PAPI_create_eventset( &eventSet[tid] );
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_create_eventset File %s Line %d retval %d\n",
         __FILE__, __LINE__, retval );
       exit(1);
    }

    if(event_multiplex){
      if(PAPI_set_multiplex(eventSet[tid]) != PAPI_OK){
      fprintf( stderr, "Error : PAPI_set_multiplex File %s Line %d retval %d\n",
         __FILE__, __LINE__, retval );	
      }
    }

    int native = 0x0;
    char inst_counter_name[] = "PAPI_TOT_INS";
    //unsigned int native = 0x0;
    //retval = PAPI_event_name_to_code("PAPI_TOT_INS",&native); //TOT_INS
    retval = PAPI_event_name_to_code(inst_counter_name,&native); //TOT_INS
    if( retval != PAPI_OK ){
      fprintf( stderr, "ERROR: PAPI_event_code_to_name PAPI_TOT_INS in File %s Line %d retval %s\n", 
          __FILE__, __LINE__,PAPI_strerror(retval) );
      exit(1);
    }
#ifdef PERFPOINT_DEBUG
    else{
      PAPI_event_code_to_name(native, event_name);
      fprintf(stderr, "Native event code %x for event %s\n", native, event_name);
    }
#endif
    retval = PAPI_add_event( eventSet[tid], native);
    if( retval != PAPI_OK ){
      fprintf( stderr, "ERROR: PAPI_add_event File %s Line %d retval %s\n", 
       __FILE__, __LINE__,	  PAPI_strerror(retval) );
    }

    //fprintf( stderr, "Event PAPI_TOT_INS added for thread %d\n", tid );


    //add monitoring event
    retval = PAPI_add_event( eventSet[tid], monitoring_event_code);
    if( retval != PAPI_OK ){
        fprintf( stderr, "ERROR: PAPI_add_event code %u File %s Line %d retval %s\n", 
          monitoring_event_code, __FILE__, __LINE__,	  PAPI_strerror(retval) );
        exit(1);
    }
    eventset_initialized = true;
    //fprintf( stderr, "Event code 0x%x added for thread %d\n" ,monitoring_event_code, tid );
  
  }

  int start_perf_counters( thread_t *thd, int mark )
  {
    //int eventSet = set_perf_events();
    assert(eventset_initialized);
    int tid = thd->index;
    thd->current_mark = mark;

    int retval = PAPI_start(eventSet[tid]);
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_start retval %s\n", PAPI_strerror(retval) );
    }

    return 1;
  }
  
  int stop_and_record_perf_counters( thread_t *thd )
  {
  
    long long *values;
    values =  ( long long * ) malloc( ( size_t ) num_events * sizeof ( long long ) );
    int tid = thd->index;
    int retval = PAPI_stop( eventSet[tid], values );
    if ( retval != PAPI_OK ) {
       fprintf( stderr, "Error : PAPI_stop File %s Line %d retval %d\n",
      __FILE__, __LINE__, retval );
    }
  
    //record the values in sample
    int next_index = thd->perfRecords.get_next_index();
    (thd->perfRecords.getEntry(next_index))->mark = thd->current_mark;
    for(int i=0; i<num_events; i++){
      (thd->perfRecords.getEntry(next_index))->count[i] = values[i];
    }
  }
};


#endif
