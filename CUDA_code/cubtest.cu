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

  datatype *data_in;
  datatype *data_out;
  datatype *data_out_cub;

  cudaMalloc(&data_in, N * sizeof(datatype));
  cudaMalloc(&data_out, N * sizeof(datatype));
  cudaMalloc(&data_out_cub, N * sizeof(datatype));
  
  init_arr<<<ARRAY_SIZE / BLOCK_SIZE + 1,BLOCK_SIZE>>>(data_in, SEED, N);
  //print_cuda_array(data_in, N);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data_in, data_out_cub, N);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cudaDeviceSynchronize();
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data_in, data_out_cub, N);
  cudaDeviceSynchronize();
  std::cout << "Post Process cub\n";
  std::cout << "Input Array\n";
  //print_cuda_array(data_in, N);
  std::cout << "Output Array\n";
  //print_cuda_array(data_out_cub, N);

  Kernels::radix_sort(data_in, data_out, N);

  radix_sort((unsigned int*)data_out, (unsigned int*)data_in, N);
  std::cout << "Post Process M&M's\n";
  std::cout << "Input Array\n";
  //print_cuda_array(data_in, N);
  std::cout << "Output Array\n";
  //print_cuda_array(data_out, N);
  
  cudaFree(data_in); 
  cudaFree(data_out);
  cudaFree(data_out_cub);

  return 0;
}