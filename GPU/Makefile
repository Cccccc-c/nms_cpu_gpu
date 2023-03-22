
OPENCV_LIBPATH=-I/usr/local/lib -I/usr/local/lib/x86_64-linux-gnu/ -I/usr/lib/x86_64-linux-gnu/

OPENCV_INCLUDEPATH=-I/usr/include/opencv4

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/lib/cuda/include

NVCC_OPTS=-O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64 

GCC_OPTS=-m64 

nms: nms.o 
	nvcc -o nms_2d nms_2d.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS) 

#main.o: main.cpp 
#g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

nms.o: nms_2d.cu
	nvcc -c nms_2d.cu $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)


clean:
	rm -f *.o *.png hw
