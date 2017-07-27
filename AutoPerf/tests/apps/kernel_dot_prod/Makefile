include ../defines.mk

TEST_NAME = kernel_dot_prod
TEST_FILES = kernel_dot_prod
THREADS = $(shell echo $(NCORES)-1 | bc)
TEST_ARGS = 100000000 $(THREADS)
#TEST_ARGS=1000000 $(THREADS)
#USER_DEFINES=-DFALSE_SHARING
CFLAGS = -DFALSE_SHARING
include $(HOME)/Default.mk

