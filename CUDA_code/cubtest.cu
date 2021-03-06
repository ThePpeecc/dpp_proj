#include<iostream>
#include<chrono>
#include<curand.h>
#include<curand_kernel.h>
#include<string>
#include<chrono>
#include"marc-marko-cuda-radix-code/kernel.cuh"
#include"myKernel.cuh"

#define ARRAY_SIZE 100000000
#define BLOCK_SIZE 256
#define SEED 42
#define RUNS 10

typedef unsigned int datatype;

__global__ void init_arr(datatype* data, unsigned long seed, int array_length){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < array_length){
      curandState state;
      curand_init(seed, idx, 0, &state);
      data[idx] = (datatype)curand(&state);
    }
}

template<class T>
bool compare_arrays(T* arr_1, T* arr_2, size_t array_length) {
  for(size_t i = 0; i < array_length; i++){
    if (arr_1[i] != arr_2[i]) return false;
  }

  return true;
}


int main() {
  //Init data
  const size_t N = ARRAY_SIZE;

  const size_t block_num = (N % BLOCK_SIZE == 0) ? 
      N / BLOCK_SIZE : N / BLOCK_SIZE + 1;
  const size_t d_block_sums_len = WINDOW_SIZE * block_num; // 16-way split
    


  bool run_cub = true;
  bool run_my  = true;
  bool run_mm  = true;
  
  datatype *data_in;
  datatype *data_out;
  datatype *data_out_cub;
  datatype *data_out_my;
  datatype *data_in_mm;
  datatype *data_out_mm;
  datatype* prefixes;
  datatype* d_block_sums;
  datatype* scan_block_sums;


  int64_t* runs_cub = new int64_t[RUNS];
  int64_t* runs_my  = new int64_t[RUNS];
  int64_t* runs_mm  = new int64_t[RUNS];
  
  cudaMallocManaged(&data_in,         N * sizeof(datatype));
  cudaMallocManaged(&data_out,        N * sizeof(datatype));
  cudaMallocManaged(&data_out_cub,    N * sizeof(datatype));
  cudaMallocManaged(&data_out_my,     N * sizeof(datatype));
  cudaMallocManaged(&data_in_mm,      N * sizeof(datatype));
  cudaMallocManaged(&data_out_mm,     N * sizeof(datatype));
  cudaMallocManaged(&prefixes,        sizeof(datatype) * N);
  cudaMallocManaged(&d_block_sums,    sizeof(datatype) * d_block_sums_len);
  cudaMallocManaged(&scan_block_sums, sizeof(datatype) * d_block_sums_len);

  

  init_arr<<<ARRAY_SIZE / BLOCK_SIZE + 1, BLOCK_SIZE>>>(data_in, SEED, N);
  //print_cuda_array(data_in, N);

  cudaMemcpy(data_in_mm, data_in, N * sizeof(datatype), cudaMemcpyDeviceToDevice);
  
  if (run_my) {
    for (int i = 0; i < RUNS; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      Kernels::radix_sort_my(data_in, data_out_my, prefixes, d_block_sums, scan_block_sums, N);
      cudaDeviceSynchronize();
      auto elapsed = std::chrono::high_resolution_clock::now() - start;
      runs_my[i] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }
    std::cout << "My Implentations runtimes: \n";
    Kernels::print_cuda_array(runs_my, RUNS);
    
    
  }

  if (run_cub) {
    // INIT CUB
    std::cout << "Cup Version:" << CUB_VERSION << "\n";
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data_in, data_out_cub, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < RUNS; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data_in, data_out_cub, N);
      cudaDeviceSynchronize();
      auto elapsed = std::chrono::high_resolution_clock::now() - start;
      runs_cub[i] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }
    std::cout << "Cubs runtimes: \n";
    Kernels::print_cuda_array(runs_cub, RUNS);
  }


  if (run_mm){
    for (int i = 0; i < RUNS; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      radix_sort((unsigned int*)data_out_mm, (unsigned int*)data_in_mm, N);
      cudaDeviceSynchronize();
      auto elapsed = std::chrono::high_resolution_clock::now() - start;
      runs_mm[i] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
      // Because Marc & Marko solution overwrites the input array
      cudaMemcpy(data_in_mm, data_in, N * sizeof(datatype), cudaMemcpyDeviceToDevice);
    }
    std::cout << "MM runtimes: \n";
    Kernels::print_cuda_array(runs_mm, RUNS);
  }
   

  if (run_my && run_cub){
    std::cout << "My implementation and cubs agree on data: " << compare_arrays(data_out_cub, data_out_my, N) << "\n";
  }
  
  cudaFree(data_in); 
  cudaFree(data_out);
  cudaFree(data_out_cub);

  return 0;
}