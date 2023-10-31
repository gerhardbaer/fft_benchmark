import torch.fft


class TorchFft(object):
    '''
    Bridge to PyTorch FFT module.
    '''
    def fft(self, a, n=None, dim=-1, norm=None):
        return self.download(torch.fft.fft(self.upload(a), n, dim, norm))
    def rfft(self, a, n=None, dim=-1, norm=None):
        return self.download(torch.fft.rfft(self.upload(a), n, dim, norm))
    def fft2(self, a, s=None, dim=(-2, -1), norm=None):
        return self.download(torch.fft.fft2(self.upload(a), s, dim, norm))
    def rfft2(self, a, s=None, dim=(-2, -1), norm=None):
        return self.download(torch.fft.rfft2(self.upload(a), s, dim, norm))
    def fftn(self, a, s=None, dim=None, norm=None):
        return self.download(torch.fft.fftn(self.upload(a), s, dim, norm))
    def rfftn(self, a, s=None, dim=None, norm=None):
        return self.download(torch.fft.rfftn(self.upload(a), s, dim, norm))


class TorchFft_CPU(TorchFft):
    '''
    Bridge to PyTorch FFT module (CPU backend).
    '''
    def upload(self, a):
        return torch.from_numpy(a)
    def download(self, a):
        return a.numpy()
