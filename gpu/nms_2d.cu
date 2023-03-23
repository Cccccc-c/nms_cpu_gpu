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
#include <map>

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

struct Detect2dObject {
  float x;
  float y;
  float w;
  float h;
  float s;
  int class_idx;
};

bool compObject(const Detect2dObject &v1, const Detect2dObject &v2)
{  
    return v1.s > v2.s;
} 

__device__
float IOUcalc(Detect2dObject b1, Detect2dObject b2)
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
void NMS_GPU_1024(Detect2dObject *d_b, bool *d_res)
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
void NMS_GPU_64(Detect2dObject *d_b, bool *d_res)
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
void NMS_GPU_64_shared(Detect2dObject *d_b, bool *d_res)
{
    // blockIdx.x = 94 , blockIdx.y = 94   threadIdx.x =64
    int rows = blockIdx.x * 64 + threadIdx.x;
    int cols = blockIdx.y * 64;
    float theta = 0.6;
    int tid = threadIdx.x;

    __shared__ Detect2dObject block_boxes[64]; // 64*4
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
void NMS_GPU_64_shared_upper_triangle(Detect2dObject *d_b, bool *d_res)
{
    if(blockIdx.x>blockIdx.y)
        return;
    // blockIdx.x = 94 , blockIdx.y = 94   threadIdx.x =64
    int rows = blockIdx.x * 64 + threadIdx.x;
    int cols = blockIdx.y * 64;
    float theta = 0.6;
    int tid = threadIdx.x;

    __shared__ Detect2dObject block_boxes[64]; // 64*4
    // 共享内存，把同一线程块中频繁访问的64个bbox的信息放到共享内存
    // 共享内存对同一线程块中的所有内存共享
    // 这里每个线程，负责把一个bbox放到共享内存中 
    if (tid < 64) {
        block_boxes[tid] = d_b[cols + tid];
    }
    __syncthreads(); // 同步！使用共享内存一定要同步，等64个线程把bbox放到共享内存后，再计算后面的iou

    int start = 0;
    if (blockIdx.x == blockIdx.y) {
      start = tid + 1; 
    }
    for(int i=start;i<64;i++)
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

int main() {
    std::ifstream in,in1;
    std::string line; 
  	
	int count=6000;
	std::map<int,std::vector<Detect2dObject>> m;
	in.open("../data/boxes.txt"); //y1, x1, y2, x2
	in1.open("../data/scores.txt");
    if (in.is_open() && in1.is_open()) {
        int class_id = 0;
        while(getline(in, line)){
			istringstream iss(line);
            Detect2dObject tmp;
			iss >> tmp.y;
			iss >> tmp.x;
			iss >> tmp.h;
			iss >> tmp.w;
			tmp.h -= tmp.y; //y2 -> h
            tmp.w -= tmp.x; //x2 -> w
			in1 >> tmp.s;
			tmp.class_idx = 0;
			// tmp.y *= 640;
			// tmp.x *= 640;
			// tmp.h *= 640;
			// tmp.w *= 640;

			if (m.find(0) == m.end()) {
				m[class_id] = std::vector<Detect2dObject>{std::move(tmp)};
			} else {
				m[class_id].emplace_back(std::move(tmp));
			}
        }
    }
    in.close();
	in1.close();

    bool h_res[count] = {true};
    memset(h_res,'1',sizeof(bool)*count);
    
    bool *d_res;
    gpuErrchk(cudaMalloc((void**)&d_res, count*sizeof(bool)));
    gpuErrchk(cudaMemcpy(d_res, h_res,sizeof(bool)*count, cudaMemcpyHostToDevice));

    Detect2dObject *d_b;
    std::map<int,std::vector<Detect2dObject>>::iterator iter;
    iter = m.lower_bound(0);
    std::vector<Detect2dObject> &bb = iter->second;
    // std::sort(bb.begin(),bb.end(),compObject);

    gpuErrchk(cudaMalloc((void**)&d_b,sizeof(Detect2dObject)*count));
    gpuErrchk(cudaMemcpy(d_b, bb.data(),sizeof(Detect2dObject)*count, cudaMemcpyHostToDevice));

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
    NMS_GPU_64_shared_upper_triangle<<<gridSize, blockSize>>>(d_b,d_res);
    
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