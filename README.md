# nms_cpu_gpu
nms的cpu与gpu实现

# make

```shell
cd cpu && make && ./nms
cd gpu && make && ./nms_2d
```

# code
三个函数一步一步加速，不需要排序，函数内部没有包含边界判断（不过不影响代码执行）
 - NMS_GPU_64
 - NMS_GPU_64_shared
 - NMS_GPU_64_shared_upper_triangle
