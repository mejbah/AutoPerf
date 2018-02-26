#if !defined(DOUBLETAKE_MM_H)
#define DOUBLETAKE_MM_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
//#include <string>
#include <sys/mman.h>

//#include "log.hh"
//#include "real.hh"

class MM {
public:
#define ALIGN_TO_CACHELINE(size) (size % 64 == 0 ? size : (size + 64) / 64 * 64)

  //static void mmapDeallocate(void* ptr, size_t sz) { Real::munmap(ptr, sz); }

  //static void* mmapAllocateShared(size_t sz, int fd = -1, void* startaddr = NULL) {
  //  return allocate(true, sz, fd, startaddr);
  //}

	static void* mmapAllocatePrivate(size_t sz, void* startaddr = NULL, int fd = -1) {
    return allocate(false, sz, fd, startaddr);
	}

private:
  static void* allocate(bool isShared, size_t sz, int fd, void* startaddr) {
    int protInfo = PROT_READ | PROT_WRITE;
    int sharedInfo = isShared ? MAP_SHARED : MAP_PRIVATE;
    sharedInfo |= ((fd == -1) ? MAP_ANONYMOUS : 0);
    sharedInfo |= ((startaddr != (void*)0) ? MAP_FIXED : 0);
    sharedInfo |= MAP_NORESERVE;

    //void* ptr = Real::mmap(startaddr, sz, protInfo, sharedInfo, fd, 0);
	void* ptr = mmap(startaddr, sz, protInfo, sharedInfo, fd, 0);
    if(ptr == MAP_FAILED) {
     printf("Couldn't do mmap (%s) : startaddr %p, sz %lu, protInfo=%d, sharedInfo=%d\n",
           strerror(errno), startaddr, sz, protInfo, sharedInfo);
      abort();
    } else {
      //      fprintf (stderr, "Successful map %lu.\n", sz);
    }

    return ptr;
  }
};


#endif
