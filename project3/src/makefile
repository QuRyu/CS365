CC = 

CFLAGS = 
LDFLAGS = 

OSNAME = $(shell uname)

ifeq ($(OSNAME), Darwin) 
    CC = clang++
    CFLAGS = -I/usr/local/include/opencv4 -I../include -std=c++17 -DmacOS_Qingbo
    LDFLAGS = -L/usr/local/lib 
endif 

ifeq ($(OSNAME), Linux) 
    CC = g++
    CFLAGS = -I../include -Dlinux_Iris
    LDFLAGS = -L/usr/lib/x86_64-linux-gnu 

endif 
	

CXXFLAGS = $(CFLAGS)
CXX = $(CC)

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect -lopencv_ml
		 

BINDIR = ../bin

obd: utilities.o imgproc.o features.o classifier.o obj_detect.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS) 

debug: CFLAGS += -DDEBUG -g 
debug: obd


clean:
	rm -f *.o *~ 
