CC  = gcc
CXX = g++
AR = /usr/bin/ar cru
CFLAGS= -DHAVE_CONFIG_H -I. -I../.. -lpthread -O3 -g -funroll-loops -fprefetch-loop-arrays -fpermissive -fno-exceptions -static-libgcc -Wl,--hash-style=both,--as-needed -DPARSEC_VERSION=3.0-beta-20150206 -fexceptions -Wall -Wno-unknown-pragmas  -fPIC -DPIC

LIB_FILES = Thread ThreadGroup WorkerGroup Mutex Condition Barrier RWLock
LIB_OBJS = $(addprefix obj/, $(addsuffix .o, $(LIB_FILES)))

default: all
all: $(LIB_OBJS)
	$(AR) libthread.a $(LIB_OBJS)
	/usr/bin/ranlib ./libthread.a
	cp -f libthread.a ../lib

clean:
	find ./obj -name "*.o" -type f -delete
	rm -f libthread.a libthreads.la 
	rm -f $(LIBRARY) #obj/*


obj/%.o: %.c
	mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@ -I$(HOME)/include

obj/%.o: %.cpp
	mkdir -p obj
	$(CXX) $(CFLAGS) -c $< -o $@ -I$(HOME)/include

#libthread.a:$(LIB_OBJS)
#	$(CXX) $(CFLAGS) -o $@ $(LIB_OBJS) 
