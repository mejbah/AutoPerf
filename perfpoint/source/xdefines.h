
#ifndef _XDEFINES_H_
#define _XDEFINES_H_

#include <sys/mman.h>
#include <sys/types.h>
#include <syscall.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <ucontext.h>
#include <pthread.h>
//#include <new>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <syscall.h>

#include "finetime.h"
#include "libfuncs.h"


#define NUM_EVENTS 5 // 5 is the highest number without multiplexing

typedef struct {

  int mark; //uniqe identified of code location?
  long long count[NUM_EVENTS];

}perf_record_t;


extern char *g_event_list[];
 // Whether current thread is inside backtrace phase
  // If yes, then we do not need to get backtrace for current malloc.
//  extern __thread thread_t * current;
  //extern __thread bool isBacktrace; 
//  //extern bool initialized;
//	extern bool _isMultithreading;
//	
//
//	// inline char getThreadBuffer()
//	inline char * getThreadBuffer() {
//		return current->outputBuf;
//	}
//
//  // Get thread index
//  inline int getTid(void) {
//    return current->tid;
//  }
//
//  // Get thread index
//  inline int getThreadIndex(void) {
//    return current->index;
//  }
//	// Get thread stackTop
//	inline unsigned int getThreadStackTop(void) {
//    return (unsigned int)current->stackTop;
//  }
//
//  enum { USER_HEAP_BASE     = 0x40000000 }; // 1G
//  enum { USER_HEAP_SIZE = 1048576UL * 8192  * 8}; // 8G
//  enum { MAX_USER_SPACE     = USER_HEAP_BASE + USER_HEAP_SIZE };
//  enum { INTERNAL_HEAP_BASE = 0x100000000000 };
//  enum { MEMALIGN_MAGIC_WORD = 0xCFBECFBECFBECFBE };
//  enum { CALL_SITE_DEPTH = 2 };

  inline size_t alignup(size_t size, size_t alignto) {
    return ((size + (alignto - 1)) & ~(alignto -1));
  }
  
  inline size_t aligndown(size_t addr, size_t alignto) {
    return (addr & ~(alignto -1));
  }
  
//  // It is easy to get the 
//  inline size_t getCacheStart(void * addr) {
//    return aligndown((size_t)addr, CACHE_LINE_SIZE);
//  }

  inline unsigned long getMin(unsigned long a, unsigned long b) {
    return (a < b ? a : b);
  }
  
  inline unsigned long getMax(unsigned long a, unsigned long b) {
    return (a > b ? a : b);
  }

class xdefines {
public:
  enum { STACK_SIZE = 1024 * 1024 };
  enum { PHEAP_CHUNK = 1048576 };

  enum { INTERNALHEAP_SIZE = 1048576UL * 1024 * 8};
  enum { PAGE_SIZE = 4096UL };
  enum { PageSize = 4096UL };
  enum { PAGE_SIZE_MASK = (PAGE_SIZE-1) };

  enum { MAX_THREADS = 2048 };//4096 };
  enum { MAX_PERF_RECORDS_PER_THREAD = 1000 };
  
	//enum { MAX_SYNC_ENTRIES = 0x10000 };

	//                 e5ccc = 941260
	enum { MAX_SYNC_ENTRIES = 1000000 };
		
	// We only support 64 heaps in total.
  enum { NUM_HEAPS = 128 };
  enum { MAX_ALIVE_THREADS = NUM_HEAPS };
  enum { MAX_THREAD_LEVELS = 256 };
  
  // 2^6 = 64
  // Since the "int" is most common word, we track reads/writes based on "int"
//  enum { WORD_SIZE = sizeof(int) };
//  enum { WORDS_PER_CACHE_LINE = CACHE_LINE_SIZE/WORD_SIZE };
//
  enum { ADDRESS_ALIGNMENT = sizeof(void *) };

	// FIXME: should be adjusted according to real situation.
	enum { CYCLES_PER_NONFS_ACCESS = 3 }; 

  // sizeof(unsigned long) = 8;
  enum { ADDRESS_ALIGNED_BITS = 0xFFFFFFFFFFFFFFF8 };

  // We start to track all accceses only when writes is larger than this threshold.
  // If not, then we only need to track writes. 
  enum { THRESHOLD_TRACK_DETAILS = 2 };
 
  // We should guarantee that sampling period should cover the prediction phase.
  enum { SAMPLE_ACCESSES_EACH_INTERVAL = THRESHOLD_TRACK_DETAILS};
  //enum { SAMPLE_INTERVAL = SAMPLE_ACCESSES_EACH_INTERVAL * 100 };
  enum { SAMPLE_INTERVAL = SAMPLE_ACCESSES_EACH_INTERVAL * 1 };
  //enum { SAMPLE_INTERVAL = SAMPLE_ACCESSES_EACH_INTERVAL * 100 };


};
#endif
