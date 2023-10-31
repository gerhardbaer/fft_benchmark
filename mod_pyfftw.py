# Copyright (C) 2023 Gerhard Baer
#
# SPDX-License-Identifier: MIT

import pyfftw


class PyFFTW(object):
    '''
    Bridge to PyFFTW FFT module.
    '''
    def __init__(self):
        self.__obj = None
        self.__obj_config = None

    def __build_config(self, a):
        return {'shape':a.shape, 'dtype':a.dtype, 'strides':a.strides,
                'nbytes':a.nbytes}

    def __build_kwargs(self, n, dim, norm, overwrite_x, workers):
        d = {'norm':norm, 'overwrite_input':overwrite_x, 'threads':workers}
        if isinstance(dim, int):
            d.update({'n':n, 'axis':dim})
        else:
            d.update({'s':n, 'axes':dim})
        return d

    def __invoke(self, func_name, a, n, dim, norm, overwrite_x, workers):
        config = {
            'func': func_name,
            'input': self.__build_config(a),
            'kwargs': self.__build_kwargs(n, dim, norm, overwrite_x, workers),
        }
        if self.__obj_config is None or self.__obj_config != config:
            self.__obj_config = config
            self.__obj = self.func(func_name)(a, **config['kwargs'])
        return self.__obj(a)

    def fft(self, a, n=None, dim=-1, norm=None, overwrite_x=False, workers=None):
        return self.__invoke('fft', a, n, dim, norm, overwrite_x, workers)
    def rfft(self, a, n=None, dim=-1, norm=None, overwrite_x=False, workers=None):
        return self.__invoke('rfft', a, n, dim, norm, overwrite_x, workers)
    def fft2(self, a, s=None, dim=(-2, -1), norm=None, overwrite_x=False, workers=None):
        return self.__invoke('fft2', a, s, dim, norm, overwrite_x, workers)
    def rfft2(self, a, s=None, dim=(-2, -1), norm=None, overwrite_x=False, workers=None):
        return self.__invoke('rfft2', a, s, dim, norm, overwrite_x, workers)
    def fftn(self, a, s=None, dim=None, norm=None, overwrite_x=False, workers=None):
        return self.__invoke('fftn', a, s, dim, norm, overwrite_x, workers)
    def rfftn(self, a, s=None, dim=None, norm=None, overwrite_x=False, workers=None):
        return self.__invoke('rfftn', a, s, dim, norm, overwrite_x, workers)


class PyFFTW_Builders(PyFFTW):
    '''
    Bridge to PyFFTW FFT module (Builders backend).
    '''
    def func(self, func_name: str):
        return getattr(pyfftw.builders, func_name)
