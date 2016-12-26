include ../defines.mk

TEST_NAME = linear_regression
TEST_FILES = linear_regression

THREADS = $(shell echo $(NCORES)-1 | bc)
TEST_ARGS = $(DATASET_HOME)/linear_regression/key_file_500MB.txt $(THREADS)
CFLAGS = -DPERFPOINT -DNO_FALSE_SHARING
include $(HOME)/Default.mk

