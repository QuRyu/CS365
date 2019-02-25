CC = 

CFLAGS = 
LDFLAGS = 

OSNAME = $(shell uname)

ifeq ($(OSNAME), Darwin) 
    CC = clang++
    CFLAGS = -I/usr/local/include/opencv4 -I../include -std=c++17 
    LDFLAGS = -L/usr/local/lib
endif 

ifeq ($(OSNAME), Linux) 
    CC = g++
    CFLAGS = --I../include 
    LDFLAGS = -L/usr/lib/x86_64-linux-gnu
endif 
	

CXXFLAGS = $(CFLAGS)
CXX = $(CC)

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect
		 



BINDIR = ../bin

cbir: cbir.o 
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS) 

clean:
	rm -f *.o *~ 
