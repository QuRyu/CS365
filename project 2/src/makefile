CC = clang++

CXX = $(CC)

CFLAGS = -I/usr/local/include/opencv4 -I../include -std=c++17 
CXXFLAGS = $(CFLAGS)
LDFLAGS = -L/usr/local/lib

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect
		 



BINDIR = ../bin

cbir: cbir.o 
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS) 

clean:
	rm -f *.o *~ 
