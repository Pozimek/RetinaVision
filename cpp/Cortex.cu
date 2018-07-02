#include "Cortex.cuh"
#include <iostream>
#include "sm_60_atomic_functions.h"
#include "CUDAHelper.cuh"

__global__ void cort_image_kernel(double *d_img, double *d_img_vector, SamplingPoint *d_fields,
		uint2 cortImgSize, size_t locSize, size_t vecLen, bool rgb) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (locSize <= globalIndex)
		return;

	int channel = globalIndex / (locSize / (rgb ? 3 : 1));
	int offset = channel * cortImgSize.x * cortImgSize.y;
	int index = globalIndex % (locSize / (rgb ? 3 : 1));
	int vecOffset = channel * vecLen;

	SamplingPoint *point = &d_fields[index];
	int kernelSize = point->_kernelSize;
	double *kernel = point->d_kernel;

	int X = point->_x - (float)kernelSize/2.0 + 0.5;
	int Y = point->_y - (float)kernelSize/2.0 + 0.5;

	double value = d_img_vector[vecOffset + d_fields[index]._i];
	for (int i = 0; i != kernelSize; ++i) {
		for (int j = 0; j != kernelSize; ++j) {
			if (X + j >= 0 && Y + i >= 0 && X + j < cortImgSize.x && Y + i < cortImgSize.y)
				atomicAdd(&d_img[offset + (Y + i) * cortImgSize.x + X + j], value * kernel[i * kernelSize + j]);
		}
	}
}

__global__ void normalise(uchar *d_norm, double *d_image, float *normaliser, size_t size, bool rgb) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= globalIndex)
		return;

	int index = globalIndex % (size / (rgb ? 3 : 1));
	d_norm[globalIndex] = normaliser[index] == 0.0 ? 0 : (int)(d_image[globalIndex] / normaliser[index]);
}

template <class T>
void setPointerToNull(T **d_ptr) {
	if (*d_ptr != nullptr){
		cudaFree(*d_ptr);
		cudaCheckErrors("ERROR");
		*d_ptr = nullptr;
	}
}

Cortex::~Cortex() {
	removeCortexFields(&d_leftFields, _leftCortexSize);
	removeCortexFields(&d_rightFields, _rightCortexSize);
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
}

int Cortex::cortImage(double *h_imageVector, size_t vecLen, float **d_norm, uchar *h_result,
			size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector,
			SamplingPoint *d_fields, size_t locSize) {
	if (!isReady())
		return ERRORS::uninitialized;
	if ((h_imageVector == nullptr && d_imageVector == nullptr) || h_result == nullptr)
		return ERRORS::invalidArguments;
	if (cortImgX != _cortImgSize.x || cortImgY != _cortImgSize.y || rgb != _rgb ||
			vecLen != _channels * (_leftCortexSize + _rightCortexSize))
		return ERRORS::imageParametersDidNotMatch;

	double *d_img;
	cudaMalloc((void**)&d_img, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(double));
	cudaMemset(d_img, 0.0, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(double));
	double *_d_imageVector;
	if (d_imageVector != nullptr)
		_d_imageVector = d_imageVector;
	else {
		cudaMalloc((void**)&_d_imageVector, _channels * (_leftCortexSize + _rightCortexSize) * sizeof(double));
		cudaMemcpy(_d_imageVector, h_imageVector, _channels * (_leftCortexSize + _rightCortexSize) * sizeof(double), cudaMemcpyHostToDevice);
	}

	cort_image_kernel<<<ceil(_channels * locSize / 512.0), 512>>>(
			d_img, _d_imageVector, d_fields, _cortImgSize, _channels * locSize,
			_leftCortexSize + _rightCortexSize, _rgb);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	uchar *d_normalised;
	cudaMalloc((void**)&d_normalised, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(uchar));
	normalise<<<ceil(_channels * _cortImgSize.x * _cortImgSize.y / 512.0), 512>>>(
			d_normalised, d_img, *d_norm, _channels * _cortImgSize.x * _cortImgSize.y, rgb);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	cudaMemcpy(h_result, d_normalised, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");

	cudaFree(d_normalised);
	if (d_imageVector == nullptr)
		cudaFree(_d_imageVector);
	cudaFree(d_img);
	return 0;
}

int Cortex::cortImageLeft(double *h_imageVector, size_t vecLen, uchar *h_result,
							size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector) {
	return cortImage(h_imageVector, vecLen, &d_leftNorm, h_result, cortImgX, cortImgY, rgb,
					 d_imageVector, d_leftFields, _leftCortexSize);
}

int Cortex::cortImageRight(double *h_imageVector, size_t vecLen, uchar *h_result,
							size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector) {
	return cortImage(h_imageVector, vecLen, &d_rightNorm, h_result, cortImgX, cortImgY, rgb,
					 d_imageVector, d_rightFields, _rightCortexSize);
}

bool Cortex::isReady() const {
	return _leftCortexSize != 0 && _rightCortexSize != 0 &&
		    d_leftFields != nullptr && d_rightFields != nullptr &&
		    d_leftNorm != nullptr && d_rightNorm != nullptr &&
			_cortImgSize.x != 0 && _cortImgSize.y != 0;
}

void Cortex::setRGB(const bool rgb) {
	if (rgb == _rgb)
		return;
	_rgb = rgb;
	_channels = _rgb ? 3 : 1;
}

void Cortex::setCortImageSize(uint2 cortImgSize) {
	if (cortImgSize.x == _cortImgSize.x && cortImgSize.y == _cortImgSize.y)
		return;
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
	_cortImgSize = cortImgSize;
}

error Cortex::setLeftCortexFields(SamplingPoint *h_leftFields, size_t leftSize) {
	return setCortexFields(h_leftFields, &d_leftFields, leftSize, _leftCortexSize);
}

error Cortex::setRightCortexFields(SamplingPoint *h_rightFields, size_t rightSize) {
	return setCortexFields(h_rightFields, &d_rightFields, rightSize, _rightCortexSize);
}

error Cortex::setLeftNorm(const float *h_leftNorm, size_t leftNormSize) {
	size_t tmp;
	return setOnDevice(h_leftNorm, leftNormSize, &d_leftNorm, tmp);
}

error Cortex::setRightNorm(const float *h_rightNorm, size_t rightNormSize){
	size_t tmp;
	return setOnDevice(h_rightNorm, rightNormSize, &d_rightNorm, tmp);
}

error Cortex::setCortexFields(SamplingPoint *h_fields, SamplingPoint **d_fields, const size_t &h_size, size_t &d_size) {
	if (h_fields == nullptr)
		return ERRORS::invalidArguments;
	removeCortexFields(d_fields, d_size);
	for (int i = 0; i != h_size; ++i) {
		h_fields[i].copyToDevice();
	}
	cudaMalloc((void**)d_fields, sizeof(SamplingPoint) * h_size);
	cudaMemcpy(*d_fields, h_fields, sizeof(SamplingPoint) * h_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("ERROR");
	d_size = h_size;
	return 0;
}

error Cortex::removeCortexFields(SamplingPoint **d_fields, size_t &d_size) {
	if (*d_fields != nullptr) {
		SamplingPoint *h_fields = (SamplingPoint*)malloc(sizeof(SamplingPoint) * d_size);
		cudaMemcpy(h_fields, *d_fields, sizeof(SamplingPoint) * d_size, cudaMemcpyDeviceToHost);
		for (int i = 0; i != d_size; ++i)
			h_fields[i].removeFromDevice();
		free(h_fields);
		setPointerToNull(d_fields);
		cudaCheckErrors("ERROR");
	}
	return 0;
}


template <class T>
error Cortex::getFromDevice(T *h_ptr, const size_t h_size, const T *d_ptr, const size_t d_size) const {
	if (h_ptr == nullptr || h_size == 0)
		return ERRORS::invalidArguments;
	if (h_size != d_size)
		return ERRORS::cortexSizeDidNotMatch;
	if (d_ptr == nullptr)
		return ERRORS::uninitialized;
	cudaMemcpy(h_ptr, d_ptr, sizeof(T) * d_size, cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");
	return 0;
}

template <class T>
error Cortex::setOnDevice(const T *h_ptr, size_t h_size, T **d_ptr, size_t &d_size) {
	if (h_ptr == nullptr || h_size == 0)
		return ERRORS::invalidArguments;

	setPointerToNull(d_ptr);
	cudaMalloc((void**)d_ptr, sizeof(T) * h_size);
	cudaMemcpy(*d_ptr, h_ptr, sizeof(T) * h_size, cudaMemcpyHostToDevice);
	d_size = h_size;
	cudaCheckErrors("ERROR");
	return 0;
}
