#include<cuda.h>
//#include"kernel.cuh"
#define BLOCK_SIZE 256
#define LOG2_BLOCK_SIZE 8
#define WINDOW_SIZE 16
#define WINDOW_BITS 4

namespace Kernels {
  
  template<class T>
  void print_cuda_array(T* array, size_t array_length){
    // Note the cuda array must instanciated with cudaManagedMalloc 
    // This ensures it's part of the Unified Memory space and can be accessed
    // by the Host.

    std::cout << "[";
    for(size_t i = 0; i < array_length; i++) {
      std::string str = std::to_string(array[i]);
      std::cout << str;
      if( i < array_length - 1) std::cout << ", ";
    }
    std::cout << "]\n";
  }


  template<class T>
  __global__ void block_radix_sort_my(
    T* data,
    T* prefixes,
    T* d_block_sums,
    unsigned int shift_width,
    size_t data_length
  ){
    // INIT Shared memory
    
    __shared__ T shared_memory_data[BLOCK_SIZE];
    __shared__ T mask[BLOCK_SIZE];
    __shared__ T merged_scan_mask[BLOCK_SIZE];
    __shared__ T mask_sums[WINDOW_SIZE];
    __shared__ T histogram[WINDOW_SIZE];

    const uint32_t window_size = 1 << WINDOW_BITS;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    shared_memory_data[threadIdx.x] = (idx < data_length) ? data[idx] : 0; 

    /* // Performance test
    if (idx < data_length) {
      data_array[threadIdx.x] = data_in[idx];
    } else {
      data_array[threadIdx.x] = 0;
    }
    */

    T write_data = shared_memory_data[threadIdx.x];
    unsigned int radix = (write_data >> shift_width) & (window_size - 1);
    __syncthreads();
    for(int i = 0; i < WINDOW_SIZE; i++){
      // Init The mask
      bool val_equals_i = false;
      if (idx < data_length) {
        val_equals_i = radix == i;
        mask[threadIdx.x] = val_equals_i;
      } else {
        mask[threadIdx.x] = 0;
      }
      __syncthreads();
      // Scan over the mask

      //This is a register value mask[threadIdx.x]
      unsigned int sum = 0;
      for(unsigned int d = 0; d < LOG2_BLOCK_SIZE; d++){
        unsigned int offset = 1 << d;
        if (threadIdx.x < offset){
          sum = mask[threadIdx.x];
        } else {
          sum = mask[threadIdx.x] + mask[threadIdx.x - offset];
        }
        __syncthreads();
        mask[threadIdx.x] = sum;
        __syncthreads();
      }
      if(threadIdx.x == BLOCK_SIZE - 1){
        mask_sums[i] = sum;
        //I think there might be an error here with regards to 
        d_block_sums[i * gridDim.x + blockIdx.x] = sum;

      }
      __syncthreads();
      if (val_equals_i && idx < data_length){ // This is entered once, then radix == i
        if(threadIdx.x != 0){
          merged_scan_mask[threadIdx.x] = mask[threadIdx.x - 1];
        } else {
          merged_scan_mask[threadIdx.x] = 0;
        }
      }
      __syncthreads();
    }
      
    // this block is not very long, but can still be optimized.
    if (threadIdx.x == 0) {
        //Turn inclusive to exclusive scan
        unsigned int mask_sum = 0;
        for (unsigned int i = 0; i < window_size; i++)
        {
            histogram[i] = mask_sum;
            mask_sum += mask_sums[i];
        }
    }
    __syncthreads();
    if (idx < data_length) {
      unsigned int target_prefix_sum = merged_scan_mask[threadIdx.x];
      unsigned int new_position = target_prefix_sum + histogram[radix];

      __syncthreads();
      shared_memory_data[new_position] = write_data;
      merged_scan_mask[new_position] = target_prefix_sum;

      __syncthreads();
      prefixes[idx] = merged_scan_mask[threadIdx.x];
      data[idx] = shared_memory_data[threadIdx.x];
    }
  }  
  
  template<class T>
  __global__ void block_shuffle_my(
    T* data,
    T* scan_block_sums,
    T* prefixes,
    unsigned int window,
    size_t data_length
  ){
    uint64_t idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int window_size = 1 << WINDOW_BITS;

    if (idx < data_length){
      T target_data = data[idx];
      uint32_t global_radix = ((target_data >> window) & (window_size - 1)) * gridDim.x + blockIdx.x;
      uint64_t target_global_position = scan_block_sums[global_radix] + prefixes[idx];
      __syncthreads();
      data[target_global_position] = target_data;
    }
  }

  template<class T>
  void radix_sort_my(T* data_in, T* data_out, T* prefixes, T* d_block_sums, T* scan_block_sums,  size_t data_length) {
    const size_t block_num = (data_length % BLOCK_SIZE == 0) ? 
      data_length / BLOCK_SIZE : data_length / BLOCK_SIZE + 1;
    const size_t d_block_sums_len = WINDOW_SIZE * block_num; // 16-way split
    cudaMemcpy(data_out, data_in, sizeof(T)*data_length, cudaMemcpyDeviceToDevice);    
    
    // This codes doesn't run an ExclusiveSum, just sets up allocations
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_block_sums, scan_block_sums, d_block_sums_len);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for(unsigned int window = 0; window < 32; window += WINDOW_BITS){
      block_radix_sort_my<<<block_num, BLOCK_SIZE>>>(
        data_out,
        prefixes,
        d_block_sums,
        window,
        data_length
      );
    
      cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_block_sums,
        scan_block_sums,
        d_block_sums_len
      );  

      block_shuffle_my<<<block_num, BLOCK_SIZE>>>(
        data_out,
        scan_block_sums,
        prefixes,
        window,
        data_length
      );
    }
    

    cudaFree(scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(prefixes);
    cudaFree(d_temp_storage);

  }

}
