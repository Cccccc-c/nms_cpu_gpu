#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdbool.h>
#include <math.h>

using namespace cv;
using namespace std;

#define BLOCKSIZE 32 //The number of threads per block should be not greater than 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct
{
    float x,y,w,h,s;
}box;

struct Detect2dObject {
  float x;
  float y;
  float w;
  float h;
  float class_prob;
  int class_idx;
};

__device__
float IOUcalc(box b1, box b2)
{
    float ai = (float)(b1.w)*(b1.h);
    float aj = (float)(b2.w)*(b2.h);
    float x_inter, x2_inter, y_inter, y2_inter;

    x_inter = max(b1.x,b2.x);
    y_inter = max(b1.y,b2.y);

    x2_inter = min((b1.x + b1.w),(b2.x + b2.w));
    y2_inter = min((b1.y + b1.h),(b2.y + b2.h));

    float w = (float)max((float)0, x2_inter - x_inter);  
    float h = (float)max((float)0, y2_inter - y_inter);  

    float inter = ((w*h)/(ai + aj - w*h));
    return inter;
}

__global__
void NMS_GPU_1024(box *d_b, bool *d_res)
{
    int abs_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int abs_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    float theta = 0.6;
    if(d_b[abs_x].s < d_b[abs_y].s)  //得分大的跟得分小的比
    {
        if(IOUcalc(d_b[abs_y],d_b[abs_x])>theta) //iou满足>theta，得分小的去掉
        {
            d_res[abs_x] = false; 
        }
    }
}

__global__
void NMS_GPU_64(box *d_b, bool *d_res)
{
    // blockIdx.x = 94 , blockIdx.y = 94   threadIdx.x =64
    int rows = blockIdx.x * 64 + threadIdx.x;
    int cols = blockIdx.y * 64;
    float theta = 0.6;
    for(int i=cols;i<cols+64;i++)
    {
        if(d_b[i].s < d_b[rows].s)  //得分大的跟得分小的比
        {
            if(IOUcalc(d_b[rows],d_b[i])>theta) //iou满足>theta，得分小的去掉
            {
                d_res[i] = false; 
            }
        }
    }
}

__global__
void NMS_GPU_64_shared(box *d_b, bool *d_res)
{
    // blockIdx.x = 94 , blockIdx.y = 94   threadIdx.x =64
    int rows = blockIdx.x * 64 + threadIdx.x;
    int cols = blockIdx.y * 64;
    float theta = 0.6;
    int tid = threadIdx.x;

    __shared__ box block_boxes[64]; // 64*4
    // 共享内存，把同一线程块中频繁访问的64个bbox的信息放到共享内存
    // 共享内存对同一线程块中的所有内存共享
    // 这里每个线程，负责把一个bbox放到共享内存中 
    if (tid < 64) {
        block_boxes[tid] = d_b[cols + tid];
    }
    __syncthreads(); // 同步！使用共享内存一定要同步，等64个线程把bbox放到共享内存后，再计算后面的iou

    for(int i=0;i<64;i++)
    {
        if(block_boxes[i].s < d_b[rows].s)  //得分大的跟得分小的比
        {
            if(IOUcalc(d_b[rows],block_boxes[i]) > theta) //iou满足>theta，得分小的去掉
            {
                d_res[cols +i] = false;
            }
        }
    }
}

__global__
void NMS_GPU_64_shared_upper_triangle(box *d_b, bool *d_res)
{
    if(blockIdx.y>blockIdx.x)
        return;
    // blockIdx.x = 94 , blockIdx.y = 94   threadIdx.x =64
    int rows = blockIdx.x * 64 + threadIdx.x;
    int cols = blockIdx.y * 64;
    float theta = 0.6;
    int tid = threadIdx.x;

    __shared__ box block_boxes[64]; // 64*4
    // 共享内存，把同一线程块中频繁访问的64个bbox的信息放到共享内存
    // 共享内存对同一线程块中的所有内存共享
    // 这里每个线程，负责把一个bbox放到共享内存中 
    if (tid < 64) {
        block_boxes[tid] = d_b[cols + tid];
    }
    __syncthreads(); // 同步！使用共享内存一定要同步，等64个线程把bbox放到共享内存后，再计算后面的iou

    for(int i=0;i<64;i++)
    {
        if(block_boxes[i].s < d_b[rows].s)  //得分大的跟得分小的比
        {
            if(IOUcalc(d_b[rows],block_boxes[i]) > theta) //iou满足>theta，得分小的去掉
            {
                d_res[cols +i] = false;
            }
        }
    }
}
int main()
{
    int count = 6000;

    bool *h_res =(bool *)malloc(sizeof(bool)*count);

    for(int i=0; i<count; i++)
    {
        h_res[i] = true;
    }

    box b[count];
    
    std::ifstream in;
    std::string line;
    
    in.open("../boxes.txt"); //y1, x1, y2, x2
    if (in.is_open()) 
    {
        int i = 0;
        while(getline(in, line))
        {
            istringstream iss(line);
            iss >> b[i].y;
            iss >> b[i].x;
            iss >> b[i].h; //y2
            iss >> b[i].w; //x2
            b[i].h-=b[i].y; //y2 -> h
            b[i].w-=b[i].x; //x2 -> w
            i+=1;
            if(i==count) break;
        }
    }
    in.close();
    
    in.open("../scores.txt");
    if (in.is_open()) 
    {
        int i = 0;
        while(in >> b[i].s)
        {
            i+=1;
            if(i==count) break;
        }
    }
    in.close();
    
    box *d_b;
    bool *d_res;

    gpuErrchk(cudaMalloc((void**)&d_res, count*sizeof(bool)));
    gpuErrchk(cudaMemcpy(d_res, h_res,sizeof(bool)*count, cudaMemcpyHostToDevice));

    for(int i=0;i<10;i++)
    {
        printf("%f %f\n",b[i].x,b[i].s);
    }
    gpuErrchk(cudaMalloc((void**)&d_b,sizeof(box)*count));
    gpuErrchk(cudaMemcpy(d_b, b,sizeof(box)*count, cudaMemcpyHostToDevice));
    
    //Setting 1: can only work when count <= 1024
    //NMS_GPU<<<dim3(1,count,1),count>>>(d_b,d_res);
    
    //Setting 2: work when count > 1024
    //NMS_GPU<<<dim3(count,count,1), 1>>>(d_b,d_res);
    
    //Setting 3: work when count > 1024, faster than Setting 2
    // 1 开辟num_bboxes * num_bboxes个线程，每个线程计算两个bbox之间的iou。显然这个方式，线程数太多，太消耗资源
    // dim3 gridSize(int(ceil(float(count)/BLOCKSIZE)), int(ceil(float(count)/BLOCKSIZE)),1); //gridSize(6000/32,6000/32,1) = (188,188,1)
    // dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);  // 32,32,1
    // NMS_GPU_1024<<<gridSize, blockSize>>>(d_b,d_res);
    // 2
    int threadsPerBlock = 64;
    int col_blocks = int((count+threadsPerBlock-1)/threadsPerBlock);
    dim3 gridSize(col_blocks,col_blocks,1); // (94,94,1)
    dim3 blockSize(threadsPerBlock, 1, 1);
    NMS_GPU_64_shared<<<gridSize, blockSize>>>(d_b,d_res);
    
    cudaThreadSynchronize();

    gpuErrchk(cudaMemcpy(h_res, d_res, sizeof(bool)*count, cudaMemcpyDeviceToHost));

    int result_num = 0;
    for(int i =0; i<count ; i++)
    {
        if(h_res[i] != true)
        {
            result_num++;
        } 
    }
    printf("Suppressed  number:%d\n",result_num);

    return 0;
}
