#ifndef SAMPLINGPOINT_H
#define SAMPLINGPOINT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

class SamplingPoint {
public:
	SamplingPoint() :_x(0), _y(0), _i(0), _s(0), _t(0), _fo(0), _kernelSize(0), _kernel(nullptr), d_kernel(nullptr) {}
	SamplingPoint(float x, float y, int i, int s, int t, int fo, int kernelSize)
		: _x(x), _y(y), _i(i), _s(s), _t(t), _fo(fo), _kernelSize(kernelSize), _kernel(nullptr), d_kernel(nullptr) {}
	SamplingPoint(const SamplingPoint &other);
	~SamplingPoint();

	double* setKernel(std::vector<double> kernel, bool overrideSize = false);
	void copyToDevice();
	void removeFromDevice();

	float _x;
	float _y;
	int _i;
	int _s;
	int _t;
	int _fo;
	int _kernelSize;
	double *_kernel;
	double *d_kernel;
};

#endif //SAMPLINGPOINT_H
