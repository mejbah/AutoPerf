include ../defines.mk

TEST_NAME = test
TEST_FILES = test
THREADS = $(shell echo $(NCORES)-1 | bc)


ifeq ($(DATASET), native)
	TEST_ARGS = $(THREADS) 
else ifeq ($(DATASET), large)
	TEST_ARGS = $(THREADS) 
endif

CFLAGS = -DENABLE_THREADS  #-DEFF_SYNC 

LIBS = -lm -lstdc++

include $(HOME)/Default.mk


