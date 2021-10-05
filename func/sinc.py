import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
from torch import Tensor
    
class Conv_0(nn.Module):
    """Convolutional block as comparision with sinc filters"""
    def __init__(self, out_channels, kernel_size, stride=1, padding=2, dilation=1, bias=False, groups=1, is_mask=False):
        super(Conv_0, self).__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride, padding, dilation, groups)
        self.channel_number = out_channels
        self.is_mask = is_mask
    
    def forward(self, x, is_training):
        x = self.conv(x)
        if is_training and self.is_mask:
            v = self.channel_number
            f = np.random.uniform(low=0.0, high=16)
            f = int(f)
            f0 = np.random.randint(0, v-f)
            x[:, f0:f0+f, :] = 0

        return x

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=2, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50,
                 freq_scale='mel', is_trainable=False, is_mask=False):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels+4
        self.kernel_size = kernel_size
        self.is_mask = is_mask
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        # low_hz = 30
        # high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # self.min_low_hz = 300
        # self.min_band_hz = 300
        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        if freq_scale == 'mel':
            mel = np.linspace(self.to_mel(low_hz),
                            self.to_mel(high_hz),
                            self.out_channels + 1)
            hz = self.to_hz(mel)
        elif freq_scale == 'lem':
            mel = np.linspace(self.to_mel(low_hz),
                            self.to_mel(high_hz),
                            self.out_channels + 1)
            hz = self.to_hz(mel)
            hz=np.abs(np.flip(hz)-1)
        elif freq_scale == 'linear':
            hz = np.linspace(low_hz,
                            high_hz,
                            self.out_channels + 1)
        
        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1), requires_grad=is_trainable)

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1), requires_grad=is_trainable)

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms, is_training=False):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)
        self.filters = self.filters[:self.out_channels-4, :, :]
        
        if is_training and self.is_mask:
            v = self.filters.shape[0]
            f = np.random.uniform(low=0.0, high=16)
            f = int(f)
            f0 = np.random.randint(0, v-f)
            self.filters[f0:f0+f, :, :] = 0
        output = F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 
        return output

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, freq_scale='mel', is_mask=False):

        super(SincConv,self).__init__()


        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels+1
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.is_mask = is_mask

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate/2) * np.linspace(0,1,int(NFFT/2)+1)

        # using mel scale
        if freq_scale == 'mel':
            print('***Initialising Mel scale Sinc Layer...***')
            f_mel = self.to_mel(f) # Hz to mel conversion
            f_mel_max = np.max(f_mel)
            f_mel_min = np.min(f_mel)
            filband_widths_mel = np.linspace(f_mel_min, f_mel_max, self.out_channels+2)
            filbandwidthsf = self.to_hz(filband_widths_mel) # Mel to Hz conversion
            self.freq = filbandwidthsf[:self.out_channels]

        # using Inverse-mel scale
        elif freq_scale == 'lem':
            print('***Initialising Inverse-Mel scale Sinc Layer...***')
            f_mel = self.to_mel(f) # Hz to mel conversion
            f_mel_max = np.max(f_mel)
            f_mel_min = np.min(f_mel)
            filband_widths_mel = np.linspace(f_mel_min, f_mel_max, self.out_channels+2)
            filbandwidthsf = self.to_hz(filband_widths_mel) # Mel to Hz conversion
            self.mel = filbandwidthsf[:self.out_channels]
            self.freq = np.abs(np.flip(self.mel)-1) ## invert mel scale

        # using linear scale
        elif freq_scale == 'linear':
            print('***Initialising Linear scale Sinc Layer...***')
            f_mel_max = np.max(f)
            f_mel_min = np.min(f)
            filband_widths_mel = np.linspace(f_mel_min, f_mel_max, self.out_channels+2)
            self.freq = filband_widths_mel[:self.out_channels]
        
        self.hsupp = torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass = torch.zeros(self.out_channels-1, self.kernel_size)
        self.freq_ = nn.Parameter(torch.Tensor(self.freq), requires_grad=True)
    
       
        
    def forward(self, waveforms, is_training=False):
        self.sp = torch.tensor(self.sample_rate).to(waveforms.device)
        self.hsupp = self.hsupp.to(waveforms.device)
        self.window = torch.hamming_window(self.kernel_size).to(waveforms.device)
        print(self.freq_.grad)
        for i in range(len(self.freq)-1):
            fmin = self.freq[i]
            fmax = self.freq[i+1]
            hHigh=(2*fmax/self.sp) * torch.sinc(2*fmax*self.hsupp/self.sp)
            hLow=(2*fmin/self.sp) * torch.sinc(2*fmin*self.hsupp/self.sp)
            hideal = hHigh-hLow
            
            self.band_pass[i,:] = self.window*hideal
        
        band_pass_filter = self.band_pass.to(waveforms.device)
        

        self.filters = (band_pass_filter).view(self.out_channels-1, 1, self.kernel_size)
        if is_training and self.is_mask:
            v = self.filters.shape[0]
            f = np.random.uniform(low=0.0, high=int(v*0.75))
            f = int(f)
            f0 = np.random.randint(0, self.out_channels-f)
            self.filters[f0:f0+f, :, :] = 0
        
        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
