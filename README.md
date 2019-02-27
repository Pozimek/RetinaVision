# RetinaVision
A Python implementation of the software retina being developed by the CVAS team at the University of Glasgow.

[Link to lastest paper](http://eprints.gla.ac.uk/148797/7/148797.pdf)


## Requirements
	Pip
	Numpy
	Scipy
	Python 3.6 (2.7 no longer supported)
	cPickle
	opencv (for running examples only, might be replaced by imageio soon)
	CUDA
	
	
## Download
Depending on how you download the package, the line endings of .pkl files can be changed causing the following error when running the demo:

	ModuleNotFoundError: No module named 'numpy.core.multiarray\r'

On Windows you have to download the package as a .zip and unpack it to prevent this issue. Using *git clone* might be safe on Ubuntu, but hasn't been tested.

## Installation
After installing all of the requirements, navigate to the RetinaVision directory and run:

	pip install -e .

If using Anaconda make sure to be inside the correct environment when installing.
For your convenience, [here are the instructions on how to install CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

Windows might block the RetinaCUDA.dll (unindentified developer). 
If this happens, try enabling the dll in it's properties. If the dll still 
cannot be loaded, run the scripts in a terminal with administrator privileges.