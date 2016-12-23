include ../defines.mk

TEST_NAME = canneal
TEST_FILES = annealer_thread rng netlist main netlist_elem
THREADS = $(shell echo $(NCORES)-1 | bc)


ifeq ($(DATASET), native)
	TEST_ARGS = $(THREADS) 15000 2000 $(DATASET_HOME)/canneal/2500000.nets 6000
else ifeq ($(DATASET), large)
	TEST_ARGS = $(THREADS) 15000 2000 $(DATASET_HOME)/canneal/400000.nets 128
endif

CFLAGS = -DENABLE_THREADS #-DEFF_SYNC

LIBS = -lm -lstdc++

include $(HOME)/Default.mk


