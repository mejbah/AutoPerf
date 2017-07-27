HOME = ../..

BOOST_INCLUDE = /home/mejbah/git_clones/Perf-Anomaly/perfpoint/tests/real_apps/boost/boost_build/include
BOOST_LIB = /home/mejbah/git_clones/Perf-Anomaly/perfpoint/tests/real_apps/boost/boost_build/lib


TEST_NAME = boost_false_sharing
TEST_FILES = false_sharing

CFLAGS += -I$(BOOST_INCLUDE)
APP_LIBS = -lboost_thread -lboost_system
APP_LIB_DIR = $(BOOST_LIB)

include ../../Default.mk


