include ../defines.mk

TEST_NAME = bodytrack
TEST_FILES = AnnealingFactor BodyGeometry CameraModel ImageMeasurements main TrackingModel AsyncIO BodyPose CovarianceMatrix ImageProjection RandomGenerator TrackingModelPthread

THREADS = $(NCORES)

ifeq ($(DATASET), native)
	TEST_ARGS = $(DATASET_HOME)/bodytrack/sequenceB_261 4 261 4000 5 0 $(THREADS)
else ifeq ($(DATASET), large)
	TEST_ARGS = $(DATASET_HOME)/bodytrack/sequenceB_4 4 4 4000 5 0 $(THREADS)
endif

#CFLAGS = -DINSERT_PADDING -fexceptions -DENABLE_THREADS -DUSE_THREADS -DPARALLEL -DHAVE_LIBPTHREAD -I./ -I./FlexImageLib/ 
CFLAGS = -fexceptions -DENABLE_THREADS -DUSE_THREADS -DPARALLEL -DHAVE_LIBPTHREAD -I./ -I./FlexImageLib/ 
CFLAGS += -DINSERT_PADDING
LIBS = -lm -lstdc++ -Llib -lflex -lthreads


include $(HOME)/Default.mk

