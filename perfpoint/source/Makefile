CC = gcc
CFLAGS= -g -O2 -fno-omit-frame-pointer -I.  -fPIC 
LD = $(CC)
LDFLAGS = -L /usr/local/lib -Wl,-rpath /usr/local/lib -lpapi -lpthread 


#LDFLAGS += -L /home/mejbah/papi/papi-5.5.1/src -Wl,-rpath=/home/mejbah/papi/papi-5.5.1/src -lpapi -lpthread
#LDFLAGS += -lpthread -ldl  -shared -lpapi
#CFLAGS += -I/home/mejbah/papi/papi-5.5.1/src -DPERF_EVENT 


CFLAGS += -DPERF_EVENT #-DPERFPOINT_DEBUG
#CFLAGS += -DPERFPOINT_EVENT_SET_2 #-DPERFPOINT_EVENT_SET_1 
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
