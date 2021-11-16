include ../defines.mk

TEST_NAME = dedup
TEST_FILES = binheap decoder dedup encoder hashtable queue rabin tree util

ifeq ($(NCORES), 32)
  THREADS := 8
else ifeq ($(NCORES), 16)
  THREADS := 4
else ifeq ($(NCORES), 12)
  THREADS := 3
else ifeq ($(NCORES), 8)
  THREADS := 2
else
  THREADS := 1
endif

ifeq ($(DATASET), native)
	TEST_ARGS = -c -p -f -t $(THREADS) -i $(DATASET_HOME)/dedup/FC-6-x86_64-disc1.iso -o output.dat.ddp
else ifeq ($(DATASET), large)
	TEST_ARGS = -c -p -f -t $(THREADS) -i $(DATASET_HOME)/dedup/media.dat -o output.dat.ddp
endif


#CFLAGS = -D_XOPEN_SOURCE=600 -Iinclude -DENABLE_THREADS -DPARALLEL -std=gnu99 -D_XOPEN_SOURCE=600 -DMD32_REG_T=int -funroll-loops -fprefetch-loop-arrays -fstrict-aliasing -Wstrict-aliasing -O5 -fkeep-inline-functions -DOPENSSL_THREADS -D_REENTRANT -DL_ENDIAN -DTERMIO -DUSE_MMAP
CFLAGS = -Iinclude -DENABLE_THREADS -DPARALLEL -std=gnu99 -DMD32_REG_T=int -D_REENTRANT -DL_ENDIAN -DUSE_MMAP -D_XOPEN_SOURCE=600
LIBS = -Llib -lssl -lz -lcrypto


include $(HOME)/Default.mk

