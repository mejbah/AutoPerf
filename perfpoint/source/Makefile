CC = gcc
#CFLAGS= -I. -D_GNU_SOURCE -fPIC -DORIGINAL -DMY_DEBUG
CFLAGS= -g -O2 -fno-omit-frame-pointer -I. -D_GNU_SOURCE -fPIC -DWITH_TRYLOCK -DREPORT_LINE_INFO #-DCONTEXT_SORT #-DGET_STATISTICS  
LD = $(CC)
LDFLAGS = -L /usr/local/lib -Wl,-rpath /usr/local/lib -lpapi -lpthread 
#LDFLAGS += -L /home/mejbah/papi/papi-5.5.1/src -Wl,-rpath=/home/mejbah/papi/papi-5.5.1/src -lpapi -lpthread
#LDFLAGS += -lpthread -ldl  -shared -lpapi
#CFLAGS += -I/home/mejbah/papi/papi-5.5.1/src -DPERF_EVENT #-DPAPI_PERF
CFLAGS += -DPERF_EVENT  #-DPAPI_PERF
TARGET = libperfpoint.so 

SRCS = $(wildcard *.c)
CPP_SRCS = $(wildcard *.cpp)
OBJS = $(patsubst %.c,%.o,$(SRCS))
CPP_OBJS = $(patsubst %.cpp,%.o,$(CPP_SRCS))

#all: $(TARGET) CSCOPE
all: $(TARGET) 

$(TARGET) : $(OBJS) $(CPP_OBJS)
	$(LD) -shared $^ $(LDFLAGS) -o $@  
%.o : %.c
	$(CC) $(CFLAGS) -c $<
%.o : %.cpp
	$(CC) $(CFLAGS) -fpermissive -c $<
%.o : %.S
	$(CC) $(CFLAGS) -c $<
#CSCOPE:
#	`find -name '*.c' -o -name '*.cpp' -o -name '*.h' -name '*.hh'> cscope.files`
#	`cscope -b -q -k`

clean:
	rm -f liblockperf.so *.o
