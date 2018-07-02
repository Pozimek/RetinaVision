#ifndef CORTEX__CUH
#define CORTEX__CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "SamplingPoint.cuh"

typedef unsigned short ushort;
typedef unsigned int  uint;
typedef unsigned char uchar;
typedef int error;

class Cortex {
	enum ERRORS {
		invalidArguments = -1,
		uninitialized = 1,
		cortexSizeDidNotMatch,
		imageParametersDidNotMatch
	};

public:
	Cortex() : _rgb(false), _channels(1), _leftCortexSize(0), _rightCortexSize(0),
				d_leftFields(nullptr), d_rightFields(nullptr),
				d_leftNorm(nullptr), d_rightNorm(nullptr), _cortImgSize(make_uint2(0,0)) {}
	Cortex(const Cortex&) = delete;
	~Cortex();

	error cortImageLeft(double *h_imageVector, size_t vecLen, uchar *h_result,
					size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = nullptr);
	error cortImageRight(double *h_imageVector, size_t vecLen, uchar *h_result,
					size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = nullptr);

	bool getRGB() const { return _rgb; }
	void setRGB(const bool rgb);

	uint2 getCortImageSize() const { return _cortImgSize; }
	void setCortImageSize(uint2 cortImgSize);

	size_t getLeftSize() const { return _leftCortexSize; }
	error setLeftCortexFields(SamplingPoint *h_leftFields, size_t leftSize);

	size_t getRightSize() const { return _rightCortexSize; }
	error setRightCortexFields(SamplingPoint *h_rightFields, size_t rightSize);

	error setLeftNorm(const float *h_leftNorm, size_t leftNormSize);
	error setRightNorm(const float *h_rightNorm, size_t rightNormSize);

private:
	bool isReady() const;
	error cortImage(double *h_imageVector, size_t vecLen, float **d_norm, uchar *h_result,
			size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector,
			SamplingPoint *d_fields, size_t size);
	error setCortexFields(SamplingPoint *h_fields, SamplingPoint **d_fields, const size_t &h_size, size_t &d_size);
	error removeCortexFields(SamplingPoint **d_fields, size_t &d_size);
	template <class T>
	error getFromDevice(T *h_fields, const size_t h_size, const T *d_fields, const size_t d_size) const;
	template <class T>
	error setOnDevice(const T *h_fields, const size_t h_size, T **d_fields, size_t &d_size);

	bool _rgb;
	ushort _channels;
	uint2 _cortImgSize;

	size_t _leftCortexSize;
	size_t _rightCortexSize;
	SamplingPoint *d_leftFields;
	SamplingPoint *d_rightFields;
	float *d_leftNorm;
	float *d_rightNorm;
};

#endif //CORTEX__CUH
