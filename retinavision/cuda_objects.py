"""
This file wraps the shared libraries compiled from the GPU code into
Python objects, serving as proxies to the main classes.
"""
import sys
import numpy as np
from os import path
np.seterr(divide='ignore', invalid='ignore')

import ctypes

if sys.platform.startswith('linux'):
    lib = ctypes.cdll.LoadLibrary(path.join(path.dirname(path.dirname(__file__)), 'bin/Linux/libRetinaCUDA.so'))
elif sys.platform.startswith('win'):
    lib = ctypes.cdll.LoadLibrary(path.join(path.dirname(path.dirname(__file__)),'bin\\Windows\\RetinaCUDA.dll'))


def convert_from_gpu(rgb_image_vector):
    '''
    Reshape flat RGB image vector from GPU to fit original implementation.\n
    Parameters
    ----------
    rgb_image_vector : np.ndarray
        Must be flat, with length of retina_size * 3\n
    Returns
    -------
    rgb_image_vector : np.ndarray
        image vector shaped [retina_size, 3]
    '''
    retina_size = int(len(rgb_image_vector) / 3)
    return np.hstack(\
        (np.resize(rgb_image_vector[0:retina_size], (retina_size,1)),\
        np.resize(rgb_image_vector[retina_size:2*retina_size], (retina_size,1)), \
        np.resize(rgb_image_vector[2*retina_size:3*retina_size], (retina_size,1))))

def convert_to_gpu(rgb_image_vector):
    '''
    Flattens RGB image vector to become compatible with GPU computation.\n
    Parameters
    ----------
    rgb_image_vector : np.ndarray
        must have shape of [retina_size, 3]\n
    Returns
    -------
    rgb_image_vector : np.ndarray
        flattened image vector
    '''
    retina_size = rgb_image_vector.shape[0]
    return np.append(\
        np.resize(rgb_image_vector[:,0], (1, retina_size))[0],\
        [np.resize(rgb_image_vector[:,1], (1, retina_size))[0],\
        np.resize(rgb_image_vector[:,2], (1, retina_size))[0]])


class CudaRetina(object):
    def resolveError(self, err):
        if err == -1:
            raise Exception("Invalid arguments")
        elif err == 1:
            raise Exception("Retina was not initialized properly")
        elif err == 2:
            raise Exception("Retina size did not match the parameter")
        elif err == 3:
            raise Exception("Image parameteres did not match")

    def __init__(self):
        lib.Retina_new.argtypes = []
        lib.Retina_new.restype = ctypes.c_void_p
        lib.Retina_delete.argtypes = [ctypes.c_void_p]
        lib.Retina_delete.restype = ctypes.c_void_p

        lib.Retina_setSamplingFields.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Retina_setSamplingFields.restype = ctypes.c_int
        
        lib.Retina_setGaussNormImage.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.Retina_setGaussNormImage.restype = ctypes.c_int
        
        lib.Retina_getGaussNormImage.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.Retina_getGaussNormImage.restype = ctypes.c_int
        
        lib.Retina_sample.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_bool]
        lib.Retina_sample.restype = ctypes.c_int

        lib.Retina_inverse.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8), \
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool]
        lib.Retina_inverse.restype = ctypes.c_int
        
        lib.Retina_getRetinaSize.argtypes = [ctypes.c_void_p]
        lib.Retina_getRetinaSize.restype = ctypes.c_int

        lib.Retina_getImageHeight.argtypes = [ctypes.c_void_p]
        lib.Retina_getImageHeight.restype = ctypes.c_int
        lib.Retina_setImageHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setImageHeight.restype = ctypes.c_void_p

        lib.Retina_getImageWidth.argtypes = [ctypes.c_void_p]
        lib.Retina_getImageWidth.restype = ctypes.c_int
        lib.Retina_setImageWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setImageWidth.restype = ctypes.c_void_p

        lib.Retina_getRGB.argtypes = [ctypes.c_void_p]
        lib.Retina_getRGB.restype = ctypes.c_bool
        lib.Retina_setRGB.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Retina_setRGB.restype = ctypes.c_void_p

        lib.Retina_getCenterX.argtypes = [ctypes.c_void_p]
        lib.Retina_getCenterX.restype = ctypes.c_int
        lib.Retina_setCenterX.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setCenterX.restype = ctypes.c_void_p
        
        lib.Retina_getCenterY.argtypes = [ctypes.c_void_p]
        lib.Retina_getCenterY.restype = ctypes.c_int
        lib.Retina_setCenterY.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setCenterY.restype = ctypes.c_void_p

        self.obj = lib.Retina_new()

    def __del__(self):
        '''Calls the C++ destructor on self'''
        lib.Retina_delete(self.obj)

    @property
    def retina_size(self):
        '''int, number of sampling fields in the retina'''
        return lib.Retina_getRetinaSize(self.obj)

    @property
    def image_height(self):
        '''int, height of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getImageHeight(self.obj)
    @image_height.setter
    def image_height(self, value):
        '''int, height of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setImageHeight(self.obj, value)

    @property
    def image_width(self):
        '''int, width of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getImageWidth(self.obj)
    @image_width.setter
    def image_width(self, value):
        '''int, width of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setImageWidth(self.obj, value)

    @property
    def rgb(self):
        '''bool, whether the retina can process rgb images (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getRGB(self.obj)
    @rgb.setter
    def rgb(self, value):
        '''bool, whether the retina can process rgb images (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setRGB(self.obj, value)

    @property
    def center_x(self):
        '''int, X coordinate of the retina center
        Note: in openCV this is [1]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getCenterX(self.obj)
    @center_x.setter
    def center_x(self, value):
        '''int, X coordinate of the retina center
        Note: in openCV this is [1]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setCenterX(self.obj, value)

    @property
    def center_y(self):
        '''int, Y coordinate of the retina center
        Note: in openCV this is [0]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getCenterY(self.obj)
    @center_y.setter
    def center_y(self, value):
        '''int, Y coordinate of the retina center
        Note: in openCV this is [0]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setCenterY(self.obj, value)
    
    def set_samplingfields(self, loc, coeff):
        '''
        Sets the sampling fields of the retina\n
        Parameters
        ----------
        loc : np.ndarray
            shape [retina_size, 7], 7 values each line, locations of the fields (from matlab)
        coeff : np.ndarray
            kernels of the sampling
        '''
        if loc.shape[0] != len(coeff.flatten()):
            print("Number of locs and coeffs must be the same")
            return
        loc1D = loc.flatten()
        coeff1D = []
        for i in coeff.flatten():
            coeff1D += i.flatten().tolist()

        #self.__retina_size = loc.shape[0]
        err = lib.Retina_setSamplingFields(self.obj, (ctypes.c_float * len(loc1D))(*loc1D),
                (ctypes.c_double * len(coeff1D))(*coeff1D), loc.shape[0])
        self.resolveError(err)

    def set_gauss_norm(self, gauss_norm=None):
        '''
        Sets the gaussian matrix to normalise with on backprojection\n
        Parameters
        ----------
        guass_norm : np.ndarray, optional
            shape must be [image_height, image_width]
            if None, CUDA will generate the gauss norm
            if not None, height and width must match with retina's 
            (3rd dimension is handled by the function)
        '''
        if gauss_norm is None:
            lib.Retina_setGaussNormImage(self.obj, None, 0, 0, 0)
        else:
            gauss_channels = 1
            gauss_norm_p = gauss_norm.flatten()
            if self.rgb:
                gauss_channels = 3#gauss_norm.shape[2]
                gauss_norm_p = np.vstack((gauss_norm[:,:].flatten(), gauss_norm[:,:].flatten(), gauss_norm[:,:].flatten()))

            err = lib.Retina_setGaussNormImage(self.obj, \
                    gauss_norm_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    gauss_norm.shape[0], gauss_norm.shape[1], gauss_channels)
            self.resolveError(err)

    def sample(self, image):
        '''
        Sample image\n
        Parameters
        ----------
        image : np.ndarray
            height, width and rgb must match the retina parameters\n
        Returns
        -------
        image_vector : np.ndarray
            sampled flat image vector
            if rgb, must be reshaped to become compatible (convert_from_gpu)
        '''
        image_vector = np.empty(self.retina_size * (3 if self.rgb else 1), dtype=ctypes.c_double)

        image_channels = 1
        image_p = image.flatten()
        if self.rgb:
            image_channels = image.shape[2]
            image_p = np.vstack((image[:,:,0].flatten(), image[:,:,1].flatten(), image[:,:,2].flatten()))
        
        err = lib.Retina_sample(self.obj, image_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), \
                image.shape[0], image.shape[1], image_channels, \
                image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                image_vector.shape[0], False)
        self.resolveError(err)

        return convert_from_gpu(image_vector) if self.rgb else image_vector

    def backproject(self, image_vector):
        '''
        Backprojects image from image vector\n
        Parameters
        ----------
        image_vector : np.ndarray
            length must match retina size\n
        Returns
        -------
        image : np.ndarray
            Backprojected image
        '''
        if len(image_vector.shape) > 1:
            image_vector = convert_to_gpu(image_vector)

        channels = (3 if self.rgb else 1)
        image = np.empty(self.image_height * self.image_width * channels, dtype=ctypes.c_uint8)
            
        err = lib.Retina_inverse(self.obj, \
            image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
            self.retina_size * channels, image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), \
            self.image_height, self.image_width, channels, False)
        self.resolveError(err)
        
        if self.rgb:
            flat_length = self.image_height * self.image_width
            out = np.dstack(\
            (np.resize(image[0:flat_length], (self.image_height, self.image_width)),\
            np.resize(image[flat_length:2*flat_length], (self.image_height, self.image_width)),\
            np.resize(image[2*flat_length:3*flat_length], (self.image_height, self.image_width))))
        else:
            out = np.resize(image, (self.image_height, self.image_width))
        return out

class CudaCortex(object):
    def resolveError(self, err):
        if err == -1:
            raise Exception("Invalid arguments")
        elif err == 1:
            raise Exception("Cortex was not initialized properly")
        elif err == 2:
            raise Exception("Cortex size did not match the parameter")
        elif err == 3:
            raise Exception("Image parameteres did not match")

    def __init__(self):
        lib.Cortex_new.argtypes = []
        lib.Cortex_new.restype = ctypes.c_void_p
        lib.Cortex_delete.argtypes = [ctypes.c_void_p]
        lib.Cortex_delete.restype = ctypes.c_void_p
    
        lib.Cortex_getRGB.argtypes = [ctypes.c_void_p]
        lib.Cortex_getRGB.restype = ctypes.c_bool
        lib.Cortex_setRGB.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Cortex_setRGB.restype = ctypes.c_void_p

        lib.Cortex_getCortImageX.argtypes = [ctypes.c_void_p]
        lib.Cortex_getCortImageX.restype = ctypes.c_uint
        lib.Cortex_getCortImageY.argtypes = [ctypes.c_void_p]
        lib.Cortex_getCortImageY.restype = ctypes.c_uint
        lib.Cortex_setCortImageSize.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        lib.Cortex_setCortImageSize.restype = ctypes.c_void_p

        lib.Cortex_getLeftSize.argtypes = [ctypes.c_void_p]
        lib.Cortex_getLeftSize.restype = ctypes.c_size_t
        lib.Cortex_setLeftCortexFields.argtypes = [ctypes.c_void_p, \
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Cortex_setLeftCortexFields.restype = ctypes.c_int

        lib.Cortex_getRightSize.argtypes = [ctypes.c_void_p]
        lib.Cortex_getRightSize.restype = ctypes.c_size_t
        lib.Cortex_setRightCortexFields.argtypes =  [ctypes.c_void_p, \
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Cortex_setRightCortexFields.restype = ctypes.c_int

        lib.Cortex_setLeftNorm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        lib.Cortex_setLeftNorm.restype = ctypes.c_int

        lib.Cortex_setRightNorm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        lib.Cortex_setRightNorm.restype = ctypes.c_int

        lib.Cortex_cortImageLeft.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,\
                                             ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, \
                                             ctypes.c_bool, ctypes.POINTER(ctypes.c_double)]
        lib.Cortex_cortImageLeft.restype = ctypes.c_int

        lib.Cortex_cortImageRight.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,\
                                             ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, \
                                             ctypes.c_bool, ctypes.POINTER(ctypes.c_double)]
        lib.Cortex_cortImageRight.restype = ctypes.c_int

        self.obj = lib.Cortex_new()

    def __del__(self):
        lib.Cortex_delete(self.obj)
    
    @property
    def left_size(self):
        '''int, size of the left cortex.'''
        return lib.Cortex_getLeftSize(self.obj)

    @property
    def right_size(self):
        '''int, size of the right cortex.'''
        return lib.Cortex_getRightSize(self.obj)
        
    @property
    def cort_image_size(self):
        '''
        pair of int, [cort_img_height,cort_img_width], size of the cortical image.
        Setting this property invalidates the cortical map.
        '''
        return [lib.Cortex_getCortImageY(self.obj), lib.Cortex_getCortImageX(self.obj)]
    @cort_image_size.setter
    def cort_image_size(self, size):
        '''
        pair of int, [cort_img_height,cort_img_width], size of the cortical image.
        Setting this property invalidates the cortical map.
        '''
        lib.Cortex_setCortImageSize(self.obj, size[1], size[0])

    @property
    def rgb(self):
        '''bool, whether the cortex can process rgb images (2D image vector)'''
        return lib.Cortex_getRGB(self.obj)
    @rgb.setter
    def rgb(self, value):
        '''bool, whether the cortex can process rgb images (2D image vector)'''
        lib.Cortex_setRGB(self.obj, value)

    def set_cortex(self, Lloc, Rloc, Lcoeff, Rcoeff, Lnorm, Rnorm, hemishape):
        if Lloc.shape[0] != len(Lcoeff.flatten()):
            print("Number of Llocs and Lcoeffs must be the same")
            return
        Lloc1D = Lloc.flatten()
        Lcoeff1D = []
        for i in Lcoeff.flatten():
            Lcoeff1D += i.flatten().tolist()

        err = lib.Cortex_setLeftCortexFields(self.obj, (ctypes.c_float * len(Lloc1D))(*Lloc1D),
                (ctypes.c_double * len(Lcoeff1D))(*Lcoeff1D), Lloc.shape[0])
        self.resolveError(err)

        if Rloc.shape[0] != len(Rcoeff.flatten()):
            print("Number of Llocs and Lcoeffs must be the same")
            return
        Rloc1D = Rloc.flatten()
        Rcoeff1D = []
        for i in Rcoeff.flatten():
            Rcoeff1D += i.flatten().tolist()

        err = lib.Cortex_setRightCortexFields(self.obj, (ctypes.c_float * len(Rloc1D))(*Rloc1D),
                (ctypes.c_double * len(Rcoeff1D))(*Rcoeff1D), Rloc.shape[0])
        self.resolveError(err)

        lib.Cortex_setCortImageSize(self.obj, hemishape[1], hemishape[0])

        Lnorm1D = Lnorm.flatten()
        err = lib.Cortex_setLeftNorm(self.obj, (ctypes.c_float * len(Lnorm1D))(*Lnorm1D), Lnorm1D.shape[0])
        self.resolveError(err)

        Rnorm1D = Rnorm.flatten()
        err = lib.Cortex_setRightNorm(self.obj, (ctypes.c_float * len(Rnorm1D))(*Rnorm1D), Rnorm1D.shape[0])
        self.resolveError(err)
   
    def cort_image_left(self, image_vector):
        '''
        Generates the left cortical image from the image_vector\n
        Parameters
        ----------
        image_vector : np.ndarray of float64
            sampled image vector
        Returns
        -------
        cort_image_left : np.ndarray of uint8
            shape of [cort_img_size[1], cort_img_size[0]]
        '''
        if len(image_vector.shape) > 1:
            image_vector = convert_to_gpu(image_vector)

        image = np.empty(self.cort_image_size[0] * self.cort_image_size[1] * (3 if self.rgb else 1), dtype=ctypes.c_uint8)

        err = lib.Cortex_cortImageLeft(self.obj, \
            image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(image_vector), \
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.cort_image_size[1], \
            self.cort_image_size[0], self.rgb, None)
        self.resolveError(err)

        if self.rgb:
            flat_length = self.cort_image_size[0] * self.cort_image_size[1]
            out = np.dstack(\
                (np.resize(image[0:flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[flat_length:2*flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[2*flat_length:3*flat_length], (self.cort_image_size[0], self.cort_image_size[1]))))
        else:
           out = np.resize(image, (self.cort_image_size[0], self.cort_image_size[1]))
        return out
    
    def cort_image_right(self, image_vector):
        '''
        Generates the right cortical image from the image_vector\n
        Parameters
        ----------
        image_vector : np.ndarray of float64
            sampled image vector
        Returns
        -------
        cort_image_right : np.ndarray of uint8
            shape of [cort_img_size[1], cort_img_size[0]]
        '''
        if len(image_vector.shape) > 1:
            image_vector = convert_to_gpu(image_vector)
        
        image = np.empty(self.cort_image_size[0] * self.cort_image_size[1] * (3 if self.rgb else 1), dtype=ctypes.c_uint8)

        err = lib.Cortex_cortImageRight(self.obj, \
            image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(image_vector), \
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.cort_image_size[1], \
            self.cort_image_size[0], self.rgb, None)
        self.resolveError(err)

        if self.rgb:
            flat_length = self.cort_image_size[0] * self.cort_image_size[1]
            out = np.dstack(\
                (np.resize(image[0:flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[flat_length:2*flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[2*flat_length:3*flat_length], (self.cort_image_size[0], self.cort_image_size[1]))))
        else:
            out = np.resize(image, (self.cort_image_size[0], self.cort_image_size[1]))
        return out
