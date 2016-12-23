#ifndef _THREADMODS_
#define _THREADMODS_
#include "finetime.h"
#include "libfuncs.h"
//#include "xdefines.h"
#include "recordentries.hh"


typedef void * threadFunction (void *);
#define gettid() syscall(SYS_gettid)

#define LOG_SIZE 4096


typedef struct thread {
  // The heap index and thread index to the threads pool.
  // index means that the current thread is the index-th thread in the system.
  // heapid is got by checking the availablity of threads in the system.
  // We don't want to make two alive threads to use the same subheap to avoid 
  // false sharing problem. 
  int       index;
  int       heapid;

  // What is the actual thread id. tid can be greater than 1024.
  int       tid;
  
  // Who is my parent;
  int       pindex;
  pid_t     ptid;

  // Who is my children in this phase;
  int childBeginIndex;
  int childEndIndex;

  pthread_t self; // Results of pthread_self
  char outputBuf[LOG_SIZE];	

  // The following is the parameter about starting function. 
  threadFunction * startRoutine;
  void * startArg; 

  // How much latency for all accesses on this thread?
  struct timeinfo startTime;
  unsigned long actualRuntime;
  unsigned long parentRuntime;
  unsigned long levelIndex; // In which phase
  
  RecordEntries<perf_record_t> perfRecords; //TODO: create structure of perfEvents
  int eventSet; //PAPI eventset

  // We used this to record the stack range
  void * stackBottom;
  void * stackTop;	

} thread_t;

// Whether current thread is inside backtrace phase
// If yes, then we do not need to get backtrace for current malloc.
extern __thread thread_t * current;
extern __thread bool isBacktrace; 
extern bool initialized;
extern bool _isMultithreading;
   

// inline char getThreadBuffer()
inline char * getThreadBuffer() {
	return current->outputBuf;
}

// Get thread index
inline int getTid(void) {
  return current->tid;
}

// Get thread index
inline int getThreadIndex(void) {
  return current->index;
}
   // Get thread stackTop
//inline unsigned int getThreadStackTop(void) {
//  return (unsigned int)current->stackTop;
//}
#endif
