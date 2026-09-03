
import numpy as np
import numpy.matlib
from band_pass_filter import butter_bandpass_filter,lowpass_filter
import pandas

from scipy import interpolate,fftpack
from scipy.fftpack import fft2, ifft2
from scipy import signal,fftpack

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from scipy.stats import norm
import math

from scipy.signal import find_peaks
import pywt

sns.set()
sns.set_style('whitegrid')
sns.set_palette("Set2")


#%%
class pre_processing:    
    
    def __init__(self, cfg):
        self.cfg = cfg
      
    #%%
    def pre_processing(self, dat, rejectFlg=False):
        """
        Input:    
            dat  - 
            cfg  -
        Output:
        Example:
        """
      
        #%% if data is list
        if isinstance(dat, list):
            rejectNum=[]
            out_y = []
            
            for i,p in enumerate(dat):
                
                timeWin = len(p)
                
                ##### Smoothing
                if len(self.cfg['windowL']) > 0:
                    y = pandas.Series(p).rolling(window=int(np.array(self.cfg['windowL']))).mean()
                    y = np.array(y)
                
                    timeWin = len(y)
                else:
                    y = np.array(p).copy()
                    
                x = np.arange(timeWin)*(1/self.cfg["RESAMPLING_RATE"]) + self.cfg["TIME_START"]
                # x = np.arange(timeWin)*(1/cfg["RESAMPLING_RATE"]) + cfg["TIME_START"]
                # x = np.linspace(self.cfg['TIME_START'][i],self.cfg['TIME_END'][i],timeWin)
               
                ##### filtering
                # if len(filt) > 0:
                #     ave = np.nanmean(y)
                #     y = y - ave
                #     y = butter_bandpass_filter(y, filt[0], filt[1], fs, order=4)
                #     y = y + ave
            
                ##### baseline correction
                baselineData = np.array([getNearestValue(x,self.cfg['WID_BASELINE'][0][0]),
                                         getNearestValue(x,self.cfg['WID_BASELINE'][0][1]),
                                         getNearestValue(x,self.cfg['WID_ANALYSIS'])])
                
                baselinePLR = y[np.arange(baselineData[0],baselineData[1])]
                baselinePLR_std = np.std(baselinePLR)
                # baselinePLR_std = np.tile(baselinePLR_std, (1,timeWin)).reshape(1,timeWin).T
                baselinePLR = np.nanmean(baselinePLR)
                # baselinePLR = np.tile(baselinePLR, (1,timeWin)).reshape(timeWin,dat.shape[0]).T   
                
                
                if self.cfg['METHOD'] == 1:
                    y = y - baselinePLR
                elif self.cfg['METHOD'] == 2:
                    y = (y - baselinePLR) / baselinePLR_std
                else:
                    y = y 
                    
                ##### reject artifact
                tmp_rejectNum = reject_trials(y.reshape(1,len(y)),self.cfg['THRES_DIFF'],baselineData)
                
                if len(tmp_rejectNum) > 0:
                    rejectNum.append(i)
                # fx = np.diff(y)
                # ind = np.argwhere(abs(fx[np.arange(baselineData[0],baselineData[2])]) > cfg['THRES_DIFF'])
                # # ind = np.argwhere(abs(fx) > thres)
            
                # if len(ind) > 0:
                #     rejectNum.append(i)
                #     # continue
                
                # ## reject trials when number of 0 > 50#
                # if sum(np.argwhere(y[np.arange(baselineData[0],baselineData[2])] == 0)) > y.shape[0] / 2:
                #     rejectNum.append(i)
                #     # continue
                    
                # ## reject trials when the NAN includes
                # if len(np.argwhere(np.isnan(y[self.cfg['windowL']:]) == True)) > 0:
                #     rejectNum.append(i)
                    
                out_y.append(y)
           
            # print(baselineData)
            rejectNum = np.unique(rejectNum).tolist()
            # set(rejectNum)
            y = out_y
            
       
            
        #%% if data is array
        else:
            
            ##### Smoothing
            if self.cfg['windowL'] > 0:
                y = moving_avg(dat.copy(),int(np.array(self.cfg['windowL'])))
            else:
                y = dat.copy()
                
            ##### baseline(-200ms - 0ms)
            x = np.linspace(self.cfg['TIME_START'],self.cfg['TIME_END'],y.shape[1])
          
            ###### filtering
            if len(self.cfg['WID_FILTER']) > 0:
                ave = np.mean(y,axis=1)
                y = y - np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                y = butter_bandpass_filter(y, self.cfg['WID_FILTER'][0], self.cfg['WID_FILTER'][1], self.cfg['SAMPLING_RATE'], order=4)
                y = y + np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
        
            ##### baseline correction
            if len(self.cfg['WID_BASELINE']) > 2:
                tmp_baseline = np.zeros((y.shape[0],y.shape[1]))
                for iTrial in np.arange(len(self.cfg['WID_BASELINE'])):
                    baselineData = np.array([getNearestValue(x,self.cfg['WID_BASELINE'][iTrial][0]),getNearestValue(x,self.cfg['WID_BASELINE'][iTrial][1]),getNearestValue(x,self.cfg['WID_ANALYSIS'])])
                    baselinePLR = y[iTrial,np.arange(baselineData[0],baselineData[1])]
                    # baselinePLR_std = np.std(baselinePLR,axis=1)
                    # baselinePLR_std = np.tile(baselinePLR_std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                    baselinePLR = np.mean(baselinePLR)
                    # baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
                    tmp_baseline[iTrial,:] = np.tile(baselinePLR, (1,y.shape[1]))
                baselinePLR = tmp_baseline
            else:
                baselineData = np.array([getNearestValue(x,self.cfg['WID_BASELINE'][0][0]),
                                         getNearestValue(x,self.cfg['WID_BASELINE'][0][1]),
                                         getNearestValue(x,self.cfg['WID_ANALYSIS'])])
                baselinePLR = y[:,np.arange(baselineData[0],baselineData[1])]
                baselinePLR_std = np.std(baselinePLR,axis=1)
                baselinePLR_std = np.tile(baselinePLR_std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                baselinePLR = np.mean(baselinePLR,axis=1)
                baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
            
            if self.cfg['METHOD'] == 1: # subtraction
                y = y - baselinePLR
            elif self.cfg['METHOD'] == 2: # ratio
                # y = (y - baselinePLR) / baselinePLR_std
                y = y / baselinePLR
            else: 
                ave = np.mean(y,axis=1)
                ave = np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                std = np.std(y,axis=1)
                std = np.tile(std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T           
                y = (y - ave) / std
                
                baselinePLR = y[:,np.arange(baselineData[0],baselineData[1])]
                baselinePLR = np.mean(baselinePLR,axis=1)
                baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
                y = y - baselinePLR
            
            ##### low-pass filter
            if self.cfg['FLAG_LOWPASS']:
                y = lowpass_filter(y, self.cfg['TIME_END']-self.cfg['TIME_START'])
           
            ##### reject artifact
            rejectNum = reject_trials(y,self.cfg['THRES_DIFF'],baselineData)
            # print(baselineData)
            
        return y,rejectNum
    
     
    #%% Hippus
    def getHippus(self, dat, sec = 3, overlap = 90, zeropad = 2 ** 11):
        '''        
        tmp_y = np.array(datHash["PDR"])
        hippus,FFT_FREQ,hippus_peak = pp.getHippus(tmp_y, (cfg["TIME_END"] - cfg["TIME_START"]), 90, 2 ** 13)
        '''
        
        ave = np.mean(dat,axis=1)
        dat = dat - np.tile(ave, (1,dat.shape[1])).reshape(dat.shape[1],dat.shape[0]).T
        
        amp = []
        for iBP in tqdm(dat.tolist()):
            
            # time_array, N_ave = ov(iBP, self.cfg["SAMPLING_RATE"], overlap, self.cfg["SAMPLING_RATE"]*sec)
            # time_array, acf = hanning(time_array, N_ave, self.cfg["SAMPLING_RATE"]*sec)
            time_array, N_ave = ov(iBP, self.cfg["SAMPLING_RATE"], overlap)
            time_array, acf = hanning(time_array, N_ave)
            time_array = zero_padding(np.array(time_array.copy()), zeropad)
            
            spectrum, fft_array, phase, freq = getfft(time_array, self.cfg["SAMPLING_RATE"])
            amp.append(fft_array.mean(axis=0).tolist())
    
        amp = np.array(amp)[:,getNearestValue(freq, 0):getNearestValue(freq, 3)].tolist()
        freq = freq[getNearestValue(freq, 0):getNearestValue(freq, 3)].tolist()
        
        # plt.plot(freq,np.array(amp).mean(axis=0))
        # plt.plot(freq,np.array(amp).T)
        indPeak = np.argmax(np.array(amp).mean(axis=0))

        return amp,freq,np.array(amp)[:,indPeak].tolist()

    #%% Blink
    def getBlink(self, blink, ev_t1, jitter=[], fs=1000):
        
        ##### blink detection
        # downSampleRate = 100        
        buffer = int(0.05*fs)

        windowLen = self.cfg['WID_ANALYSIS']*2
        rate = np.zeros((len(blink),windowLen*fs))
        rate_tmp = np.zeros((len(blink),windowLen*fs))

        for iTrial,dTrial in enumerate(blink):
            for iBlink in dTrial:
                if iBlink[1] < (windowLen/2)*fs:
                    rate[iTrial,np.arange(iBlink[0],iBlink[1])+(self.cfg["WID_ANALYSIS"]*fs)] = 1
                    
                    if iBlink[1]+buffer < (self.cfg["WID_ANALYSIS"]*fs):
                        rate_tmp[iTrial,np.arange(iBlink[0]-buffer,iBlink[1]+buffer)+(self.cfg["WID_ANALYSIS"]*fs)] = 1
                    else:
                        rate_tmp[iTrial,(iBlink[0]-buffer)+(self.cfg["WID_ANALYSIS"]*fs):] = 1
                   
        # rate_tmp = re_sampling(rate_tmp,int((windowLen)*fs))
        rate_tmp[rate_tmp==-1]=0
        rate_tmp[rate_tmp==2]=1
        
        if len(jitter) > 0:
            for iTrial,lag in enumerate(jitter):
                rate_tmp[iTrial,:] = np.r_[rate_tmp[iTrial,int(lag):],np.zeros(int(lag))]

        ##### reject if MS overlap with blink
        coeff = 1-rate_tmp.copy()

        sTimeOfMS = []
        for iTrial,dTrial in enumerate(ev_t1):
            rejectMS = []
            empty = np.zeros(coeff.shape[1])
                
            for ims,dms in enumerate(dTrial):
                t = coeff[iTrial,dms["onset"]:dms["offset"]]
                if len(np.argwhere(t==0)):
                    rejectMS.append(ims)
                
                if dms["onset"]>0:
                    if dms["amplitude"] > 0:
                        empty[dms["onset"]] = 1
                    else:
                        empty[dms["onset"]] = -1
                        
            ev_t1[iTrial] = [d for i,d in enumerate(dTrial) if not i in rejectMS]

            sTimeOfMS.append(empty.tolist())

        sTimeOfMS = np.multiply(np.array(sTimeOfMS),coeff)
       
        ##### moving average
        tmp = moving_avg(sTimeOfMS,int(fs/10))
        sTimeOfMS = re_sampling(tmp, int((self.cfg['TIME_END']-self.cfg['TIME_START'])*self.cfg["RESAMPLING_RATE"]))
    
        dat_MS = {"amplitude":[],
                     "theta":[],
                     "onset":[],
                     "trial":[]
                     }
        
        for iTrial,dTrial_t1 in enumerate(ev_t1):
            rejectMS = []
            empty = np.zeros(coeff.shape[1])
            
            for ims,dms in enumerate(dTrial_t1):
                dat_MS["amplitude"].append(dms["amplitude"])
                dat_MS["theta"].append(dms["theta"])
                dat_MS["onset"].append(dms["onset"]/fs)
                dat_MS["trial"].append(iTrial)
      
        rate = moving_avg(rate.copy(),100)
        rate = re_sampling(rate,int((windowLen)*self.cfg["RESAMPLING_RATE"]))
        
        return rate, dat_MS, sTimeOfMS
    

#%% Add vertical lines for mean age on each plot
def vertical_line(x, **kwargs):
    
    plt.axvline(x.mean(), linestyle = '--', color ='g')
    
    if 'col' in kwargs.keys():
        plt.axvline(x.mean(), linestyle = '--', color = kwargs['col'])
        # print('kwargs: ', kwargs["col"])

#%% Add vertical lines for mean age on each plot
def horizontal_line(x, **kwargs):
    plt.axhline(x.mean(), linestyle = '--', color = 'g')

#%% 
def scaleFigs(**kwargs):
    plt.axis('scaled')

#%% overlap
def ov(data, fs, overlap, frame = 2**12):
    
    """
    Input:
    Output:
    Example:
    """
    
    N = len(data)  
    Ts = N / fs         #data
    Fc = frame / fs     #frame
    x_ol = frame * (1 - (overlap/100)) #overlap
    
    N_ave = int((Ts - (Fc * (overlap/100))) / (Fc * (1-(overlap/100))))

    array = []

    for i in range(N_ave):
        ps = int(x_ol * i) 
        # dat = zero_padding(data[ps:ps+frame:1],zeropad)
        # array.append(dat) 
        array.append(data[ps:ps+frame:1])
        
    return array, N_ave

#%% hanning
def hanning(data, N_ave, frame = 2**12):
    """
    Input:    
    Output:
    Example:
    """
    #haning
    # han = signal.hann(frame) 
    han = signal.windows.kaiser(frame,2)
    # han = signal.blackman(frame)
    
    acf = 1 / (sum(han) / frame)
    
    data_pad = []
    for i in range(N_ave):
        data[i] = data[i] * han
        data_pad.append(data[i])
        
    return data_pad, acf

#%% FFT
def fft_ave(data, fs, N_ave, acf, frame = 2**12):
    
    """
    Input:    
    Output:
    Example:
    """
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(fftpack.fft(data[i])/(frame/2)))
        
    fft_axis = np.linspace(0, fs, frame)
    fft_array = np.array(fft_array)
    fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))
    
    return fft_array, fft_mean, fft_axis

#%% zero_padding
def zero_padding(data, len_pad):
    """
    Input:    
    Output:
    Example:
    """
    if data.ndim == 1:
        pad = np.zeros(len_pad)
        data_pad = np.hstack((np.insert(data, 0, pad), pad))
        acf = (sum(np.abs(data)) / len(data)) / (sum(np.abs(data_pad)) / len(data_pad))
        return data_pad * acf
    else:    
        pad = np.zeros((data.shape[0],len_pad))
        data_pad = np.hstack([np.hstack([pad,data]),pad])
        acf = (np.abs(data[0,:]).sum() / data.shape[1]) / (np.abs(data_pad[0,:]).sum() / data_pad.shape[1])
        
        return data_pad

#%% split_list2
def split_list2(l, n):
    windowL = np.round(np.linspace(0, len(l), n+1))
    windowL = [int(windowL[i]) for i in np.arange(len(windowL))]
    for idx in np.arange(len(windowL)-1):
        yield np.arange(windowL[idx],windowL[idx+1]).tolist(),l[windowL[idx]:windowL[idx+1]]

#%% split_list
def split_list(l, n):
    windowL = np.round(np.linspace(0, len(l), n+1))
    windowL = [int(windowL[i]) for i in np.arange(len(windowL))]
    for idx in np.arange(len(windowL)-1):
        yield l[windowL[idx]:windowL[idx+1]]
  
#%% perform fourier transform
# def getfft(data, fs):
    
#     if (type(data) is pd.core.frame.DataFrame):
        
        
#         tmp_dat = data[data["Time[s]"]>=0]["PDR"].values
#         tmp_dat[np.isnan(tmp_dat)]=0
#         spectrum = np.fft.fft(tmp_dat.reshape(1,len(tmp_dat)), axis=1, norm=None)   
#         # spectrum = np.array(spectrum)
        
#         N = len(data[0])
#         amp = abs(spectrum)
#         amp = amp / (N / 2)
#         # plt.plot(amp.T)
#         # plt.xlim([0.1,1.5])
#         phase = np.arctan2(spectrum.imag, spectrum.real)
#         phase = np.degrees(phase)
#         freq = np.linspace(0, fs, N)
    
#     else:
#         # spectrum = fftpack.fft(data)
#         # spectrum = []
#         # for d in data:
#         #     spectrum.append(fftpack.fft(d))
#         data[np.isnan(data)]=0
#         spectrum = np.fft.fft(data, axis=1, norm=None)   
#         spectrum = np.array(spectrum)
                         
#         N = len(data[0])
#         amp = abs(spectrum)
#         amp = amp / (N / 2)
#         phase = np.arctan2(spectrum.imag, spectrum.real)
#         phase = np.degrees(phase)
#         freq = np.linspace(0, fs, N)
    
#         return spectrum[:,:round(N / 2)], amp[:,:round(N / 2)], phase[:,:round(N / 2)], freq[:round(N / 2)]
def pad_to_length(x, N):
    out = np.zeros(N)
    out[:len(x)] = x
    return out

def getfft(x, fs, detrend=True, window="hann"):

    x = np.asarray(x, dtype=float)
    n = x.size

    if detrend:
        x = x - np.mean(x)

    # window
    if window is None:
        w = np.ones(n)
    elif window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    else:
        raise ValueError("window must be None/'hann'/'hamming'")

    xw = x * w

    # rfft for real signal
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1/fs)

    # amplitude correction (rough): compensate window power & single-sided scaling
    # coherent gain for amplitude
    cg = np.sum(w) / n
    amp = (2.0 / n) * np.abs(X) / cg
    amp[0] *= 0.5  # DC component not doubled

    # simple power spectrum (not strict PSD)
    psd = (np.abs(X) ** 2) / (fs * np.sum(w**2))

    return f, amp.reshape(-1), psd.reshape(-1)

def getfft_multi(y, fs, maxfs, detrend=True, window="hamming"):
    
    amps, psds,freqs = [], [], []
    
    # max_len = max([len(yy[0]) for yy in y])
    max_len = max([len(yy) for yy in y])

    for yy in y:
        # x_pad = pad_to_length(yy[0], max_len)
        x_pad = pad_to_length(yy, max_len)
        f, amp, psd = getfft(x_pad, fs, detrend=detrend, window=window)
        
        amps.append(amp[f<maxfs])
        psds.append(psd[f<maxfs])
        freqs.append(f[f<maxfs])
   
    return freqs, amps, psds

def morlet_cwt(x, fs, freqs, w=6.0, wavelet="cmor1.5-1.0"):

    # x = np.asarray(x, dtype=float)
    # t = np.arange(x.size) / fs

    # widths = (w * fs) / (2*np.pi*freqs)

    # cwtm = signal.cwt(x, signal.morlet2, widths, w=w)
    # power = np.abs(cwtm) ** 2
    dt = 1 / fs
    fc = pywt.central_frequency(wavelet)
    scales = fc / (freqs * dt)

    coef, _ = pywt.cwt(np.array(x), scales, wavelet, sampling_period=dt)
    power = np.abs(coef) ** 2

    t = (np.arange(np.array(x).size) / fs).tolist()

    return t, power

def morlet_cwt_multi(y,fs,freqs):
    
    ts,powers = [], []
    
    # max_len = max([len(yy[0]) for yy in y])

    for yy in tqdm(y):
        
        t, power= morlet_cwt(yy[0], fs, freqs)
        
        ts.append(t)
        powers.append(powers)
   
    return ts, powers


#%%
def getTimeFreq(dat,fs,fft_size,hop_length,zeropad = 2 ** 13):
    
    # zeropad = 0
    time_array = []
    for i in np.arange((len(dat)-fs)/hop_length):
        if -fs*fft_size + i*hop_length < 0:
            time_array.append(np.r_[np.zeros(abs(int(-fs*fft_size + i*hop_length))), dat[np.arange(0,int(fs*fft_size+i*hop_length))]])
        
        elif fs*fft_size + i*hop_length > len(dat):
            time_array.append(np.r_[dat[np.arange(int(-fs*fft_size + i*hop_length), len(dat))], np.zeros(int(fs*fft_size + i*hop_length)-len(dat)) ])
            
        else:
            time_array.append(dat[np.arange(int(-fs*fft_size + i*hop_length),int(fs*fft_size+i*hop_length))])
            
    time_array, acf = hanning(np.array(time_array), len(time_array), len(time_array[0]))
    
    time_array = zero_padding(np.array(time_array.copy()), zeropad)
    
    spectrum, fft_array, phase, freq = getfft(time_array, fs)

    return fft_array[:,(freq>=0.3)&(freq<2)], freq[(freq>=0.3)&(freq<2)]

    # plt.figure()
    # plt.plot(freq,fft_array.mean(axis=0))
    # plt.xlim([0,2]) 
    
    # plt.pcolormesh(np.linspace(0,7,int((len(dat)-fs)/hop_length)), freq, fft_array.T, cmap = 'jet')
    # plt.ylim([0,2])
    
#%% rejectDat
def rejectDat(dat,rejectNum,*args):
    for mm in list(dat.keys()):
        dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNum]
    
    if len(args) > 0:
        tmp = []
        for i,arg in enumerate(args):
            tmp.append(np.delete(arg,rejectNum,axis=0))
    
        return dat, *tuple(tmp)

    else: 
        return dat
    
#%% zscore
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def downsample_binary_max(dat, num):
    
    re_sampled = []
    for iTrial in np.arange(len(dat)):
        t = np.array(dat[iTrial])

        edges = np.linspace(0, len(t), num + 1).astype(int)

        y = np.zeros(num)
        
        for i in range(num):
            if np.sum(t[edges[i]:edges[i+1]])>0:
                y[i] = 1
        
        re_sampled.append(y.tolist())
    
    return re_sampled

#%% re_sampling
def re_sampling(dat,num):
    """
    Input:    
        dat  - list of data
        num  - list or int of re-sampling rate
    Output:
    Example:
    """
    
    if not isinstance(dat, list):
        if dat.ndim == 1:
            ndimFlag = True
            dat = dat.reshape(1,len(dat))
        else:
            ndimFlag = False
    else:
        ndimFlag = False
        
    re_sampled = []
    for iTrial,d in enumerate(dat):
        t = np.array(d)
        
        if t.ndim == 2:
            tmp_out = []
            for iEyes in np.arange(t.shape[0]):
                ind = np.argwhere(np.isnan(t[iEyes,]) == False).reshape(-1)
                yy = interpolate.PchipInterpolator(ind, t[iEyes,ind])
                if isinstance(num, list):
                    t_resample = np.linspace(0, len(t[iEyes,]), num[iTrial])
                else:
                    t_resample = np.linspace(0, len(t[iEyes,]), num)
            
                tmp_out.append(yy(t_resample).tolist())
     
            re_sampled.append(tmp_out)
                
        else:
          
            ind = np.argwhere(np.isnan(t) == False).reshape(-1)
            
            if len(ind) == 0:
                if isinstance(num, list):
                    tmp = np.zeros(num[iTrial])
                    
                else:
                    tmp = np.zeros(num)
                tmp[:] = np.nan
                re_sampled.append(tmp.tolist())
            else:
                
                t[np.isnan(t)]=0
                
                # ind = np.argwhere(np.isnan(t) == False).reshape(-1)
                
                # yy = interpolate.PchipInterpolator(ind, t[ind])
                yy = interpolate.PchipInterpolator(np.arange(len(t)), t)
                
                if isinstance(num, list):
                    t_resample = np.linspace(0, len(t), num[iTrial])
                else:
                    t_resample = np.linspace(0, len(t), num)
                    
                out = yy(t_resample)
                
                # binary option
                if len(np.unique(t))==2:
                    out[out>=0.5] = 1
                    out[out<0.5] = 0
                    
                re_sampled.append(out.tolist())
                
    if ndimFlag:
        re_sampled = np.array(re_sampled[0])
        
    return re_sampled


#%% getNearestValue
def getNearestValue(in_y, num):
    idx = np.abs(np.asarray(in_y) - num).argmin()
    return idx

#%% moving_avg
def moving_avg(y,windowL):
    
    """
    Input:    
        dat      - list of data
        windowL  - list or int of re-sampling rate
    Output:
    Example:
    """
    
    tmp_y = y.copy()
    if isinstance(y, list):
        
        tmp_y = y.copy()
        for trial,d in enumerate(tmp_y):
            s = pandas.Series(np.array(d))
            tmp_y[trial] = np.array(s.rolling(window=windowL,center=True).mean())
        
            for i in np.arange(int(windowL/2)):
                tmp_y[trial][i] = np.array(y[trial][:int(i+windowL/2)]).mean()
    
                if i>0:
                    tmp_y[trial][-i] = np.array(y[trial][-int(i+windowL/2):]).mean()
         
        out_y = [yy.tolist() for yy in tmp_y]
        
        # out_y = []
        # for iTrials,d in enumerate(y):
        #     out_y.append(np.r_[np.array(tmp_y[iTrials])[np.arange(windowL)],
        #                  np.array(d)[np.arange(windowL,len(d))]].tolist())
            
    else:
    
        if y.ndim == 1:
            y = y.reshape(1,len(y))
            tmp_y = y.copy()
            
        out_y = []
        for trial in np.arange(len(y)):
            s = pandas.Series(y[trial,])
            tmp_y[trial,] = s.rolling(window=windowL,center=True).mean()
        
            # tmp_y[trial,0] = y[trial,:int(windowL/2)].mean()
            for i in np.arange(int(windowL/2)):
                # aa
                tmp_y[trial,i] = y[trial,:int(i+windowL/2)].mean()

                if i>0:
                    tmp_y[trial,-i] = y[trial,-int(i+windowL/2):].mean()
                
            out_y.append(tmp_y[trial,:])
       
        out_y = np.array(out_y)
            # out_y.append(tmp_y[trial,windowL:])
            
        if y.shape[0] == 1:
            out_y = out_y.reshape(-1)
        
    return out_y

#%% reject_trials
def reject_trials(y,thres,baselineData):
  
    ############## reject trials 
    ############## when the velocity of pupil change is larger than threshold
    rejectNum=[]
    fx = abs(np.diff(y, n=1))
    fx = fx[:,np.arange(baselineData[0],baselineData[2])]
    
    ind = np.argwhere(fx > thres)
    
    rejectNum = np.unique(ind[:,0])
    
    # for trials in np.arange(y.shape[0]):
    #     ind = np.argwhere(abs(fx[trials,np.arange(baselineData[0],baselineData[2])]) > thres)
    #     # ind = np.argwhere(abs(fx[trials,np.arange(50,baselineData[2])]) > thres)
        
    #     if len(ind) > 0:
    #         plt.plot(fx[trials,:])
    #         rejectNum.append(trials)
    #         continue
            
    #     if sum(np.isnan(y[trials,np.arange(baselineData[0],baselineData[2])])) > 0:
    #         rejectNum.append(trials)
    #         continue
    
    ############## when number of 0 > 50
    rejectNum_zero = []
    for i in np.arange(y.shape[0]):
        if sum(np.argwhere(y[i,np.arange(baselineData[0],baselineData[2])] == 0)) > y.shape[1] / 2:
            rejectNum_zero.append(int(i))
            # continue
    rejectNum = np.r_[rejectNum, np.array(rejectNum_zero)]

    ############## when the NAN includes
    rejectNum_nan = []
    for i in np.arange(y.shape[0]):
        if len(np.argwhere(np.isnan(y[i,np.arange(baselineData[0],baselineData[2])]) == True)) > y.shape[1] / 2:
            rejectNum_nan.append(int(i))
    
    rejectNum = np.r_[rejectNum, np.array(rejectNum_nan)]
     
    rejectNum = np.unique(rejectNum).astype("int64") 
    set(rejectNum)
    
    return rejectNum.tolist()

#%% rejectedByOutlier
def rejectedByOutlier(dat, y_bp=[],factorName = "BaselinePD", figFlag=False):
    
    if type(dat) is pd.core.frame.DataFrame:
        
        tmp = dat.copy()

    else:
        if y_bp.ndim != 1:
            y_bp = y_bp.mean(axis=1)
            # y_bp = np.max(np.abs(y_bp),axis=1)
        
        tmp = pd.DataFrame()
        tmp["sub"] = np.array(dat["sub"])
        tmp[factorName] = y_bp
         
        # df_std = tmp.groupby(["sub"]).agg(np.std)
        # df_ave = tmp.groupby(["sub"]).agg(np.median)
            
        # df_std[factorName] = df_std[factorName]*3
    
        # q75 = tmp.groupby(["sub"],as_index=False).quantile(.95)
        # q25 = tmp.groupby(["sub"],as_index=False).quantile(.5)
        
    q75 = tmp.groupby(["sub"],as_index=False).quantile(.75, numeric_only=True)
    q25 = tmp.groupby(["sub"],as_index=False).quantile(.25, numeric_only=True)
    # df_ave = tmp.groupby(["sub"]).agg(np.median)
 
    IQR = q75[factorName] - q25[factorName]

    lower = q25
    # lower[factorName] = lower[factorName] - IQR*1.5
    lower[factorName] = lower[factorName] - IQR*3
    # lower[factorName] = lower[factorName]*3

    # lower[factorName] = lower[factorName] - IQR*6

    upper = q75
    # upper[factorName] = upper[factorName] + IQR*1.5
    upper[factorName] = upper[factorName] + IQR*3
    # upper[factorName] = upper[factorName]*3
    
    # df_ave["lower"] = df_ave[factorName] - df_std[factorName] 
    # df_ave["upper"] = df_ave[factorName] + df_std[factorName] 

    tmp["lower"] = 0
    tmp["upper"] = 0
    tmp["a"] = 145
    tmp["b"] = -145
    # for iSub, lo, up in zip(np.unique(dat["sub"]),list(df_ave["lower"]), list(df_ave["upper"])):
    for iSub in np.unique(tmp["sub"]):
        tmp.loc[tmp["sub"]==iSub,"lower"] = float(lower[factorName][lower["sub"]==iSub].values)
        tmp.loc[tmp["sub"]==iSub,"upper"] = float(upper[factorName][lower["sub"]==iSub].values)
        
    reject = tmp[(tmp[factorName] < tmp["lower"]) | (tmp[factorName] > tmp["upper"])].index.values
    # reject = tmp[(tmp[factorName] < tmp["b"]) | (tmp[factorName] > tmp["a"])].index.values
    if figFlag:
        grid = sns.FacetGrid(tmp, col="sub", hue="sub", col_wrap=5)
        grid.map(sns.distplot, factorName)
    
        grid.map(vertical_line, 'lower') 
        grid.map(vertical_line, 'upper') 
      
        # plt.xlim([tmp["lower"],tmp["upper"]])
    # grid.map(vertical_line, 'a',col='r') 
    # grid.map(vertical_line, 'b',col='r') 

    
    # dat, y, y2, y_bp = rejectDat(dat,reject["BP"].tolist(),y,y2,y_bp)
  
    return reject.tolist()

#%% rejectedSubByOutlier
def rejectedSubByOutlier(dat, y_bp, factorName = "BaselinePD"):
    
    if y_bp.ndim != 1:
        y_bp = y_bp.mean(axis=1)

    tmp = pd.DataFrame()
    tmp["sub"] =np.array(dat["sub"])
    tmp[factorName] = y_bp
     
    df_std = np.std(y_bp)
    df_ave = np.median(y_bp)
        
    df_std = df_std*3

    tmp["lower"] = df_ave - df_std
    tmp["upper"] = df_ave + df_std
        
    reject = tmp[(tmp[factorName] < tmp["lower"]) | (tmp[factorName] > tmp["upper"])]["sub"].values

    sns.histplot(data=tmp,x=factorName)
    plt.axvline(tmp["lower"][0],color="red")
    plt.axvline(tmp["upper"][0],color="red")

    return reject

#%% SDT
def SDT(hit_rate, fa_rate):
   
    """ 
    returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections
    
    """
    
    hit_rate = np.clip(hit_rate, 1e-5, 1 - 1e-5)
    fa_rate = np.clip(fa_rate, 1e-5, 1 - 1e-5)

    # z score
    zH = norm.ppf(hit_rate)
    zFA = norm.ppf(fa_rate)

    # d'
    d_prime = zH - zFA

    # beta
    beta = np.exp((zFA**2 - zH**2) / 2)

    # c（criterion）
    c = -(zH + zFA) / 2

    # A-prime
    if hit_rate >= fa_rate:
        A_prime = 0.5 + ((hit_rate - fa_rate) * (1 + hit_rate - fa_rate)) / (4 * hit_rate * (1 - fa_rate))
    else:
        A_prime = 0.5 - ((fa_rate - hit_rate) * (1 + fa_rate - hit_rate)) / (4 * fa_rate * (1 - hit_rate))

    return pd.DataFrame({
        'dprime': d_prime,
        'beta': beta,
        'c': c,
        'Aprime': A_prime
        },index=[0])
    
#%%
def MPCL(sub, data_y, cfg, pcpd="PD", time_min = 0, time_max = 3,pltFlg=False):
    
    min_idx = []
    min_xval = []
    min_yval = []
    x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],data_y.shape[1])
   
    mpcl = pd.DataFrame()
    mpcl["sub"] = np.unique(sub)
    for mmName in ["idx","xval","yval"]:
        mpcl[mmName] = 0
    
    if pltFlg:
        plt.figure(figsize=(10,10))
    
    for i,iSub in enumerate(np.unique(sub)):
        
        if isinstance(time_min, list):
            x_t = x[np.argwhere((x > time_min[i]) & (x <= time_max)).reshape(-1)]
            tmp_y = data_y[:,np.argwhere((x > time_min[i]) & (x <= time_max)).reshape(-1)]
        else:
            x_t = x[np.argwhere((x >= time_min) & (x <= time_max)).reshape(-1)]
            tmp_y = data_y[:,np.argwhere((x > time_min) & (x <= time_max)).reshape(-1)]
           
        ind = np.argwhere(sub == iSub).reshape(-1)
         
        tmp_p = tmp_y[ind,:].copy()
        tmp_p = tmp_p.mean(axis=0)
         
        # second-order accurate central differences 
        pv = np.gradient(tmp_p)
        
        # find inflection point
        indices = np.where(np.diff(np.sign(pv)))[0]
        # plt.plot(x_t[indices],tmp_p[indices],'bo')  
        
        ev = []
        for itr in np.arange(len(indices)):
            if pv[indices[itr]] - pv[indices[itr]+1] > 0:
                ev.append(1)
            else:
                ev.append(0)
        
        if len(indices) > 0:  # peak pupil constriction/dilation(PC/PD)
            if pcpd == "PD":
                indices = indices[np.argwhere(np.array(ev) == 1).reshape(-1)]
            else:
                indices = indices[np.argwhere(np.array(ev) == 0).reshape(-1)]
        
            if len(indices) > 0: # if there is no PC/PD peak
                
                # PC/PD within 200ms will be rejected
                ini = 0
                rej = []
                for iPeak in np.arange(len(indices)):
                    if (indices[iPeak] - ini) < 0.2*cfg["RESAMPLING_RATE"]:
                        rej.append(iPeak)
                    else:
                        ini = indices[iPeak]
                 
                indices = [d for i,d in enumerate(indices) if not i in rej]
         
        if pltFlg:
            plt.subplot(12,round(len(np.unique(sub))/10),int(iSub))
            plt.plot(x_t,tmp_p,'-')
            
        if len(indices) > 0:  # peak pupil constriction/dilation(PC/PD)
            if pltFlg:
                plt.plot(x_t[indices],tmp_p[indices],'bo')
            
            if pcpd == "PD":
                indices = indices[np.argmax(tmp_p[indices])]
            else:
                indices = indices[np.argmin(tmp_p[indices])]
               
            mpcl["idx"][mpcl["sub"]==iSub] = int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[indices]))
            mpcl["xval"][mpcl["sub"]==iSub] = float(x_t[indices])
            mpcl["yval"][mpcl["sub"]==iSub] = float(tmp_p[indices])
            ## min_xval.append(float(x_t[indices]))
            # min_yval.append(float(tmp_p[indices]))
            # min_idx.append(int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[indices])))
            if pltFlg:
                plt.plot(x_t[indices],tmp_p[indices],'ro')
            
        else:
            if pcpd == "PD":
                mpcl["idx"][mpcl["sub"]==iSub] = int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[0]))
                mpcl["xval"][mpcl["sub"]==iSub] = float(x_t[0])
                mpcl["yval"][mpcl["sub"]==iSub] = float(tmp_p[0])
               
                # plt.plot(x_t[0],tmp_p[0],'ro')
            else:
                mpcl["idx"][mpcl["sub"]==iSub] = int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[-1]))
                mpcl["xval"][mpcl["sub"]==iSub] = float(x_t[-1])
                mpcl["yval"][mpcl["sub"]==iSub] = float(tmp_p[-1])
            
            if pltFlg:            
                plt.plot(x_t[-1],tmp_p[-1],'ro')
        
    return mpcl

def getPCPDevents(y,minHeight=None):
    
    #%% ############## PC/PD  ###################################
    # test_y = moving_avg(y.copy(),int(cfg["RESAMPLING_RATE"]))
    # test_y = moving_avg(y.copy(),int(cfg["RESAMPLING_RATE"]*10))

    if not isinstance(y, list):
        if y.ndim == 1:
            y = y.reshape(1,len(y))
        
        len_y = y.shape[1]
    
    else:
        len_y = np.max([yy.shape[1] for yy in y])
        # len_y = np.max([len(yy) for yy in y])
    #     y = [yy.tolist() for yy in y]
        
    test_y = y.copy()
    # fs = cfg["RESAMPLING_RATE"]*5

    events = {
        'peaks':[],
        'troughs':[]
        }
    # events = {'indices':[],'event':[]}
    for iTrial in np.arange(len(test_y)):
        # aa
        
        if not minHeight is None:
            peaks, _ = find_peaks(test_y[iTrial].reshape(-1),prominence=minHeight)
            troughs, _ = find_peaks(-test_y[iTrial].reshape(-1),prominence=minHeight)

        else:
            peaks, _ = find_peaks(test_y[iTrial].reshape(-1))
            troughs, _ = find_peaks(-test_y[iTrial].reshape(-1))

        # events['indices'].append(np.sort(np.r_[peaks,troughs]).tolist())
        events["peaks"].append(peaks.tolist())
        events["troughs"].append(troughs.tolist())
        
    if isinstance(y, list):
        events["summary"] = []
        for iRun, (peak, trough) in enumerate(zip(events["peaks"],events["troughs"])):
        
            tmp_summary = np.zeros((test_y[iRun].shape[1])).astype(int)
            
            tmp_summary[peak] = 1
            tmp_summary[trough] = -1
            events["summary"].append(tmp_summary.tolist())
    
    else:
        events["summary"] = np.zeros((len(test_y),len_y)).astype(int)
        for iRun, (peak, trough) in enumerate(zip(events["peaks"],events["troughs"])):
            events["summary"][iRun,peak] = 1
            events["summary"][iRun,trough] = -1
        
        # ev = []
        # for tmp_ind in events['indices'][-1]:
        #     if tmp_ind in peaks:
        #         ev.append(0)
        #     else:
        #         ev.append(1)
        # events['event'].append(ev)
            
        ##################################################
        '''
        pv = np.gradient(test_y[iTrial])
        indices = np.where(np.diff(np.sign(pv)))[0]
        
        # xv = np.r_[0,np.diff(indices)]
        xv = np.gradient(indices)
        
        indices = indices[xv > fs*minWindow] # < 300ms
        
        events['indices'].append(indices.tolist())
        
        ev = []
        for itr in np.arange(len(indices)):
            if pv[indices[itr]] - pv[indices[itr]+1] > 0:
                ev.append(1)
            else:
                ev.append(0)
        events['event'].append(ev)
        '''
        ##################################################
    # if iTrial == 3:  
    #     # plt.plot(pv*fs)
        # plt.figure()
        # plt.plot(test_y[iTrial])
        # pcpd = np.array(events['indices'][iTrial])
        # plt.plot(pcpd[np.array(ev)==1],test_y[iTrial][pcpd[np.array(ev)==1]],'ro')    
        # plt.plot(pcpd[np.array(ev)==0],test_y[iTrial][pcpd[np.array(ev)==0]],'bo')    
    #     # plt.xlim([10000,20000])
    #     plt.xlim([0,10000])
    
    return events
        

