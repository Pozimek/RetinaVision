#ifndef CUDAHELPERS__CUH
#define CUDAHELPERS__CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

#define cudaCheckErrors(msg) \
		do { \
			cudaError_t __err = cudaGetLastError(); \
			if (__err != cudaSuccess) { \
				fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
						msg, cudaGetErrorString(__err), \
						__FILE__, __LINE__); \
						fprintf(stderr, "*** FAILED - ABORTING\n"); \
						exit(1); \
			} \
		} while (0)


__device__ double atomicAdd(double* address, double val);
template <class T>
__host__ void setPointerToNull(T **d_ptr);


class CUDAHelper {
public:
	CUDAHelper() {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		_maxNumberOfThreadsPerBlock = prop.maxThreadsPerBlock;
		_maxNumberOBlocks = prop.maxGridSize[0];
	}
	int2 calcThreadPerBlock(int numOfThreads) {
		int2 r;
		r.x = 0;
		r.y = 0;
		return r;
	}

private:
	int _maxNumberOfThreadsPerBlock;
	int _maxNumberOBlocks;
};

#endif //CUDAHELPERS__CUH
