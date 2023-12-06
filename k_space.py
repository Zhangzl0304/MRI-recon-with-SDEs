import numpy as np
sqrt = np.sqrt
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift


def fftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def ifftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def fft2c(x):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def ifft2c(x):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res
