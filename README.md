

This repository contains the work of Marcus (tkz347) & Chirstoffers (jvf734) final project in DPP. 

Remember to sync the libraries by running

```
futhark pkg sync
```

To run test:

```
make test

futhark test test.fut
```

To run benchmarks:

```
futhark bench --backend=opencl bench.fut
# With cuda
futhark bench --backend=cuda bench.fut


# Benchmarks for cuda kernels
cd ./CUDA_code && make
```

To clean:

```
make clean
```




