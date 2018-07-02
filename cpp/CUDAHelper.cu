#include "CUDAHelper.cuh"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
						__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

template <class T>
__host__ void setPointerToNull(T **d_ptr) {
	if (*d_ptr != nullptr){
		cudaFree(*d_ptr);
		cudaCheckErrors("ERROR");
		*d_ptr = nullptr;
	}
}
