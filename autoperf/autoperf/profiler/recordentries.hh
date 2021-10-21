#ifndef RECORD_ENTRIES_HH_
#define RECORD_ENTRIES_HH_

/*
 * @file   recordentires.h
 * @brief  Managing record entry for each thread. Since each thread will have different entries,
 There is no need to use lock here at all.
 The basic idea of having pool is
 to reduce unnecessary memory allocation and deallocation operations, similar
 to slab manager of Linux system. However, it is different here.
 There is no memory deallocation for each pool.
 In the same epoch, we keep allocating from
 this pool and udpate correponding counter, updating to next one.
 When epoch ends, we reset the counter to 0 so that we can reuse all
 memory and there is no need to release the memory of recording entries.

 * @author Tongping Liu <http://www.cs.umass.edu/~tonyliu>
 */

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
//#include <string>
#include <unistd.h>

//#include "log.hh"
#include "mm.hh"
#include "xdefines.h"
//Each thread will have a class liked this. Thus, we can avoid 
//memory allocation when we are trying to record a synchronization event.
//The total number of entries for each thread is xdefines::MAX_SYNCEVENT_ENTRIES.
//Thus, there is no need to reclaim it. 
//When an epoch is finished, we will call   
template <class Entry> class RecordEntries {
public:
  RecordEntries() {}

  void initialize(int entries) {
    void* ptr;
    size_t _size;

    _size = alignup(entries * sizeof(Entry), xdefines::PageSize);
    ptr = MM::mmapAllocatePrivate(_size);
    if(ptr == NULL) {
     // PRWRN("%d fail to allocate sync event pool entries : %s\n", getpid(), strerror(errno));
			printf("%d fail to allocate sync event pool entries \n", getpid());
      ::abort();
    }

    //  PRINF("recordentries.h::initialize at _cur at %p. memory from %p to 0x%lx\n", &_cur, ptr,
    // (((unsigned long)ptr) + _size));
    // start to initialize it.
    _start = (Entry*)ptr;
    _cur = 0;
    _total = entries;
    return;
  }
#if 0
  Entry* alloc() {
    Entry* entry = NULL;
		
	//	PRINF("allocEntry, _cur %ld\n", _cur);
    if(_cur < _total) {
			int val = __atomic_fetch_add(&_cur,1, __ATOMIC_RELAXED);
			entry = (Entry*)&_start[val];
    } else {
      // There are no enough entries now; re-allocate new entries now.
      printf("Not enough entries, now _cur %lu, _total %lu at %p!!!\n", _cur, _total, &_cur);
      ::abort();
    }
    return entry;
  }
#endif
	size_t get_next_index() {
		//int val = __atomic_fetch_add(&_cur,1, __ATOMIC_RELAXED);
		int val = __sync_fetch_and_add(&_cur,1);

		if(val < _total){
			return val;		
		} else {
      // There are no enough entries now; re-allocate new entries now.
      printf("Not enough entries, now _cur %lu, _total %lu at %p!!!\n", _cur, _total, &_cur);
      ::abort();
    }
  }

  void cleanup() {
    //_iter = 0;
    _cur = 0;
  }

  
  inline Entry* getEntry(size_t index) { return &_start[index]; }

  size_t getEntriesNumb() { return _cur; }

private:
  Entry* _start;
  size_t _total;
  volatile size_t _cur;

};

#endif   // RECORD_ENTRIES_HH_
