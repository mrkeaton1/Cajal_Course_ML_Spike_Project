
# In[3]:


### imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio # first conda install scipy
import scipy.signal as sig 
import ghostipy as gsp # first pip install ghostipy


### load data

# load dict of spike times in seconds
#spike_dict = np.load('data/' + 'spike_dict.npy', allow_pickle=True) 

# load DLC lables (one label per frame)
fl_paw_speed = np.load('data/' + 'front_left_paw_speed.npy')
fr_paw_speed = np.load('data/' + 'front_right_paw_speed.npy')
bl_paw_speed = np.load('data/' + 'rear_left_paw_speed.npy')
br_paw_speed = np.load('data/' + 'rear_right_paw_speed.npy')

# load camera TTL times (seconds) corresponding to each frame
cam_ttl_times = np.load('data/' + 'camera_ttl_times.npy') 

cam_ttl_times = cam_ttl_times[:1795025]
fl_paw_speed = fl_paw_speed[:1795025]
fr_paw_speed = fr_paw_speed[:1795025]
bl_paw_speed = bl_paw_speed[:1795025]
br_paw_speed = br_paw_speed[:1795025]


step = 0.005#seconds
normalized_time = np.arange(0, np.max(dlc_time), step)#np.linspace(dlc_time[int(lift_times[ee])]- window_s, dlc_time[int(lift_times[ee])]+window_s, num =  199)#
binned_spike_dict = []
for ii in range(0, np.size(spike_dict)):
    binned_spike_dict.append( np.histogram(spike_dict[ii], bins = normalized_time)[0])



#binned_spike_dict = binned_spike_dict[:,:1768047]

# fig1    = plt.figure()
# #plt.hist( np.diff(camera_ttl_times),1000, range = [-.1,.1])
# plt.plot(np.diff(camera_ttl_times[:1796482]))#np.diff
# plt.show()

end_time = camera_ttl_times[int(1795025)]

#1795025 = the split in the video aka session one end time in indexes 
binned_spike_dict = np.array(binned_spike_dict)[:,:int(end_time/step)]

smooth_spike_dict = []
for ii in tqdm(range(0, np.shape(binned_spike_dict)[0])):
    kernel = np.ones(4)
    smooth_spike_dict.append(scipy.signal.fftconvolve(binned_spike_dict[ii], kernel, mode='same'))


### get sampling rate of the paw files

# get the interval between each camera TTL
cam_ttl_intervals = np.diff(cam_ttl_times)

# print the unique intervals and plot a histogram (excludes outliers)
print('Intervals between camera TTLs (s):')
print(np.unique(cam_ttl_intervals))
plt.hist(cam_ttl_intervals, range=[8e-3, 9e-3], bins=100)
plt.xlabel('Interval between camera TTLs (s)')
plt.ylabel('Count');


# In[21]:


# use the mean interval between camera TTLs to get sampling rate
print('Mean interval between camera TTLs (s): {}'.format(np.mean(cam_ttl_intervals)))
fs_paw_speed = 1/np.mean(cam_ttl_intervals)
print('True average sampling rate of paw speed files: {} Hz'.format(fs_paw_speed))

# make it an integer for convenience 
fs_paw_speed = int(np.round(fs_paw_speed))
print('Rounding to an int for convenience: {} Hz'.format(fs_paw_speed))

# how much data is there?
print('Recording length (mins): {}'.format(len(fl_paw_speed)/fs_paw_speed/60))





#import wavefun

# get spectral content via continuous wavelet transform 

from tqdm import tqdm
# cwt parameters
gamma = 3 
beta = 100 # increasing beta increases freq resolution and decreases temporal resolution
freq_lims = [0.5, 30] 

paws = [fl_paw_speed, fr_paw_speed, bl_paw_speed, br_paw_speed]
labels = ['front left', 'front right', 'back left', 'back right']
fl_paw_freq = []
fr_paw_freq = []
bl_paw_freq = []
br_paw_freq = []

for i in tqdm(range(len(paws))):
    data = paws[i]#[fs_paw_speed*6:fs_paw_speed*bin]
    n_samps = len(data) # number of samples
    T = n_samps/fs_paw_speed # total time
    t = np.linspace(0, T, n_samps, endpoint=False) # new time vector

    # apply cwt
    cwtcoefs, _, freq, cwtts, _ = gsp.cwt(
        data, timestamps=t, freq_limits=freq_lims, fs=fs_paw_speed,
        wavelet=gsp.MorseWavelet(gamma=gamma, beta=beta))
    cwt = cwtcoefs.imag**2 + cwtcoefs.real**2
    cwt = cwt/np.max(cwt) # normalize
    if i == 0:
        fl_paw_freq.append(cwt)
    if i == 1:
        fr_paw_freq.append(cwt)
    if i == 2:
        bl_paw_freq.append(cwt)
    if i == 3:
        br_paw_freq.append(cwt)

    # # plot spectrogram
    # fig, ax = plt.subplots()
    # fig.set_size_inches((12,3)) 
    # t_ax, f_ax = np.meshgrid(t, freq)
    # ax.pcolormesh(t_ax, f_ax, cwt, shading='gouraud', 
    #             cmap=plt.cm.viridis, vmin=0, vmax=0.6)
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_xlabel("Time (s)")
    # ax.set_title(labels[i])
    # plt.show()



relevant_freq_idx = np.where(np.logical_and(freq>=4, freq<=7))[0]

fl_paw_freq = np.vstack(fl_paw_freq)
fr_paw_freq = np.vstack(fr_paw_freq)
bl_paw_freq = np.vstack(bl_paw_freq)
br_paw_freq = np.vstack(br_paw_freq)

avg_freq = np.mean([fl_paw_freq[relevant_freq_idx], fr_paw_freq[relevant_freq_idx], bl_paw_freq[relevant_freq_idx],br_paw_freq[relevant_freq_idx]], axis = 0)
avg_freq = np.mean(avg_freq, axis = 0)

still_freq_idx = np.where(freq>=3)[0]
still_freq = np.mean([fl_paw_freq[still_freq_idx], fr_paw_freq[still_freq_idx], bl_paw_freq[still_freq_idx],br_paw_freq[still_freq_idx]], axis = 0)
still_freq = np.mean(still_freq, axis = 0)

behave = []#np.zeros(np.shape(fl_paw_freq*2)[2])
window_size = int(fs_paw_speed/6)#
slide = int(fs_paw_speed/8)#
walk_threshold = [0.0007, 10]
still_threshold = 0.000008
behave = []
avg_val = []

def find_behave(avg_freq, window_size, walk_threshold, still_threshold):
    num_windows = int(len(avg_freq) / window_size * (window_size/slide))
    for i in range(num_windows):
        start = i*slide
        window = avg_freq[start:start+window_size]
        still_pow = still_freq[start:start+window_size]

        if np.mean(window) > walk_threshold[0]:#, np.mean(window) < walk_threshold[1]):
            if np.all(window)< 15:
                behave.append(1)#walk
        elif np.mean(still_pow) < still_threshold:
            behave.append(-1)#still
        else: 
            behave.append(0)
        avg_val.append(np.mean(window))


    return np.hstack(behave), np.hstack(avg_val) #half second bins slid by a fourth of a second 1 = walk, -1 = still, 0 = too noisy to determine


matplotlib.rcParams.update({'font.size': 18})

find_behave(avg_freq, window_size, walk_threshold, still_threshold)

xaxis1 = np.linspace(0,cam_ttl_times[-1], np.size(fl_paw_speed))
xaxis2 = np.linspace(0,cam_ttl_times[-1], np.size(behave))
plt.subplots()
plt.plot(xaxis1,fl_paw_speed, color = 'b', label = 'front left paw', alpha = 0.5)
plt.plot(xaxis1,fr_paw_speed, color = 'indianred', label = 'front right paw', alpha = 0.5)
plt.plot(xaxis1,bl_paw_speed, color = 'c', label = 'back left paw', alpha = 0.5)
plt.plot(xaxis1,br_paw_speed, color = 'm', label = 'back right paw', alpha = 0.5)

plt.plot(xaxis2,behave, color = '0.2', label = 'behavior type')
plt.plot(xaxis2,np.array(avg_val)*1000, color = '0.3', label = 'badpassed power (4-7Hz)')
plt.xlabel('time (s)')
plt.ylabel('paw speed (pixels/second')
plt.legend()
plt.show()

#np.save('run_frequency_3to7Hz_vl6_230418_', avg_val)
#np.save('run_still_bianarized_vl6_230418_winsz20_slide15frames', behave)



# interpolate back too camera_ttl_times size 

#end_time = camera_ttl_times[int(1795025)]
#xaxis2 = np.linspace(0,cam_ttl_times[:1795025], np.size(behave))

#normalized_time
#cut_norm_time = np.arange(0, end_time, step)
#cut_norm_time = cut_norm_time[:-1]

rythem = np.interp(cam_ttl_times,xaxis2,avg_val)



