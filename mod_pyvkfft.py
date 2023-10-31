import pyopencl as cl
import pyopencl.array as cla
import pyvkfft
import pyvkfft.fft


class VkFft(object):
    '''
    Bridge to PyVkFFT FFT module.
    '''
    def fft(self, a, n=None, dim=-1):
        return self.__invoke(pyvkfft.fft.fftn, a, ndim=1, axes=(dim,))
    def rfft(self, a, n=None, dim=-1):
        return self.__invoke(pyvkfft.fft.rfftn, a, ndim=1)
    def fft2(self, a, s=None, dim=(-2, -1)):
        return self.__invoke(pyvkfft.fft.fftn, a, ndim=2, axes=dim)
    def rfft2(self, a, s=None, dim=(-2, -1)):
        return self.__invoke(pyvkfft.fft.rfftn, a, ndim=2)
    def fftn(self, a, s=None, dim=None):
        return self.__invoke(pyvkfft.fft.fftn, a, ndim=len(a.shape), axes=dim)
    def rfftn(self, a, s=None, dim=None):
        return self.__invoke(pyvkfft.fft.rfftn, a, ndim=len(a.shape))

    def __invoke(self, func, a, **kwargs):
        return self.download(func(self.upload(a), **kwargs))


class VkFft_OpenCL(VkFft):
    '''
    Bridge to PyVkFFT FFT module (OpenCL backend).
    '''
    def __init__(self, device_type):
        super().__init__()
        self.__device = self.get_device(device_type)
        self.__context = cl.Context(devices=(self.__device,))
        self.__queue = cl.CommandQueue(self.__context)

    def upload(self, a):
        return cla.to_device(self.__queue, a)
    def download(self, a):
        return a.get()

    @staticmethod
    def get_device(device_type):
        for p in cl.get_platforms():
            for d in p.get_devices():
                if d.type == device_type:
                    return d
        return None


class VkFft_OpenCL_CPU(VkFft_OpenCL):
    device_type = cl.device_type.CPU

    def __init__(self):
        super().__init__(VkFft_OpenCL_CPU.device_type)

    @staticmethod
    def available():
        return VkFft_OpenCL.get_device(VkFft_OpenCL_CPU.device_type) is not None


class VkFft_OpenCL_GPU(VkFft_OpenCL):
    device_type = cl.device_type.GPU

    def __init__(self):
        super().__init__(VkFft_OpenCL_GPU.device_type)

    @staticmethod
    def available():
        return VkFft_OpenCL.get_device(VkFft_OpenCL_GPU.device_type) is not None
