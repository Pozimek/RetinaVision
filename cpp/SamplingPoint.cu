#include <cmath>
#include "SamplingPoint.cuh"
#include "CUDAHelper.cuh"

SamplingPoint::SamplingPoint(const SamplingPoint &other) : _x(other._x), _y(other._y), _i (other._i),
			_s(other._s), _t(other._t), _fo(other._fo), _kernelSize(other._kernelSize), _kernel(nullptr), d_kernel(nullptr) {
	if (other._kernel != nullptr) {
		_kernel = new double[_kernelSize * _kernelSize];
		memcpy(_kernel, other._kernel,  sizeof(double) * _kernelSize * _kernelSize);
	}
}

SamplingPoint::~SamplingPoint() {
	if (_kernel != nullptr)
		delete [] _kernel;
}

double* SamplingPoint::setKernel(std::vector<double> kernel, bool overrideSize) {
	if (!overrideSize && kernel.size() != _kernelSize * _kernelSize) {
		return nullptr;
	} else {
		_kernelSize = sqrt(kernel.size());
		if (_kernel != nullptr)
			delete [] _kernel;

		_kernel = new double[_kernelSize * _kernelSize];
		for (int i = 0; i != _kernelSize * _kernelSize; ++i) {
			_kernel[i] = kernel.at(i);
		}

		if (d_kernel != nullptr) {
			cudaFree(d_kernel);
			cudaMalloc((void**)&d_kernel, sizeof(double) * _kernelSize * _kernelSize);
			cudaMemcpy(d_kernel, _kernel, sizeof(double) * _kernelSize * _kernelSize, cudaMemcpyHostToDevice);
		}

		return _kernel;
	}
}

void SamplingPoint::copyToDevice() {
	if (_kernel == nullptr)
		return;
	if (d_kernel != nullptr) {
		cudaFree(d_kernel);
	}

	cudaMalloc((void**)&d_kernel, sizeof(double) * _kernelSize * _kernelSize);
	cudaMemcpy(d_kernel, _kernel, sizeof(double) * _kernelSize * _kernelSize, cudaMemcpyHostToDevice);
	cudaCheckErrors("ERROR");
}

void SamplingPoint::removeFromDevice() {
	if (d_kernel == nullptr)
		return;
	cudaFree(d_kernel);
	d_kernel = nullptr;
	cudaCheckErrors("ERROR");
}

