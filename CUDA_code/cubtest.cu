#include<iostream>
#include<chrono>
#include<curand.h>
#include<curand_kernel.h>
#include<string>
#include<cub/cub.cuh>
#include"kernel.cuh"
#include"myKernel.cuh"

#define ARRAY_SIZE 1e2
#define BLOCK_SIZE 256
#define SEED 42

typedef unsigned int datatype;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
        if (abort) exit(code);
    }
}


__global__ void init_arr(datatype* data, unsigned long seed, int array_length){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < array_length){
      curandState state;
      curand_init(seed, idx, 0, &state);
      data[idx] = (datatype)curand(&state) % 10000;
    }
}


void print_cuda_array(datatype* array, size_t array_length){
  datatype* cpu_data = (datatype*)malloc(array_length*sizeof(datatype));
  cudaMemcpy(cpu_data, array,array_length*sizeof(datatype), cudaMemcpyDeviceToHost);

  std::cout << "[";
  for(size_t i = 0; i < array_length; i++) {
    std::string str = std::to_string(cpu_data[i]);
    std::cout << str;
    if( i < array_length - 1) std::cout << ", ";
  }
  std::cout << "]\n";
}


int main() {
  //Init data
  const size_t N = ARRAY_SIZE;

  bool run_cub = false;
  bool run_my  = true;
  bool run_mm  = true;

  bool print_cub = false;
  bool print_my  = true;
  bool print_mm  = true;
  
  
  
  datatype *data_in;
  datatype *data_out;
  datatype *data_out_cub;
  datatype *data_in_my;
  datatype *data_out_my;
  datatype *data_in_mm;
  datatype *data_out_mm;

  cudaMalloc(&data_in, N * sizeof(datatype));
  cudaMalloc(&data_out, N * sizeof(datatype));
  cudaMalloc(&data_out_cub, N * sizeof(datatype));
  cudaMalloc(&data_in_my, N * sizeof(datatype));
  cudaMalloc(&data_out_my, N * sizeof(datatype));
  cudaMalloc(&data_in_mm, N * sizeof(datatype));
  cudaMalloc(&data_out_mm, N * sizeof(datatype));
  

  init_arr<<<ARRAY_SIZE / BLOCK_SIZE + 1,BLOCK_SIZE>>>(data_in, SEED, N);
  //print_cuda_array(data_in, N);

  cudaMemcpy(data_in_my,data_in, N * sizeof(datatype), cudaMemcpyDeviceToDevice);
  cudaMemcpy(data_in_mm,data_in, N * sizeof(datatype), cudaMemcpyDeviceToDevice);

  if (run_cub) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data_in, data_out_cub, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaDeviceSynchronize();
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data_in, data_out_cub, N);
    cudaDeviceSynchronize();
    if (print_cub) {
      std::cout << "Post Process cub\n";
      std::cout << "Input Array\n";
      print_cuda_array(data_in, N);
      std::cout << "Output Array\n";
      print_cuda_array(data_out_cub, N);
    }
  } 

  if (run_my) {
    Kernels::radix_sort_my(data_in_my, data_out_my, N);
    if (print_my){  
      std::cout << "My implementation\n";
      std::cout << "Input Array\n";
      print_cuda_array(data_in_my, N);
      std::cout << "Output Array\n";
      print_cuda_array(data_out_my, N);
    }
  }

  if (run_mm){
    radix_sort((unsigned int*)data_out_mm, (unsigned int*)data_in_mm, N);
    if(print_mm){
      std::cout << "Post Process M&M's\n";
      std::cout << "Input Array\n";
      print_cuda_array(data_in_mm, N);
      std::cout << "Output Array\n";
      print_cuda_array(data_out_mm, N);
    }
  } 
  
  cudaFree(data_in); 
  cudaFree(data_out);
  cudaFree(data_out_cub);

  return 0;
}