### Transforms paw speed data into power, a proxy for speed
### Returns binned and smoothed spike and speed data

#TODO: clean this

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import scipy.io as sio # first conda install scipy
import scipy.signal as sig 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import ghostipy as gsp # first pip install ghostipy
from tqdm import tqdm



def generate_binned_data(data_path, step=0.005, kernel_size=4, ):
    first_video_end = 1795025
    # load DLC labels (one label per frame)
    fl_paw_speed = np.load(join(data_path, 'front_left_paw_speed.npy'))
    fr_paw_speed = np.load(join(data_path, 'front_right_paw_speed.npy'))
    bl_paw_speed = np.load(join(data_path, 'rear_left_paw_speed.npy'))
    br_paw_speed = np.load(join(data_path, 'rear_right_paw_speed.npy'))

    # load camera TTL times (seconds) corresponding to each frame
    cam_ttl_times = np.load(join(data_path, 'camera_ttl_times.npy'))
    spike_dict = np.load(join(data_path, 'spike_dict.npy'), allow_pickle=True)

    cam_ttl_times = cam_ttl_times[:first_video_end]
    fl_paw_speed = fl_paw_speed[:first_video_end]
    fr_paw_speed = fr_paw_speed[:first_video_end]
    bl_paw_speed = bl_paw_speed[:first_video_end]
    br_paw_speed = br_paw_speed[:first_video_end]

    normalized_time = np.arange(np.min(cam_ttl_times), np.max(cam_ttl_times), step)#np.linspace(dlc_time[int(lift_times[ee])]- window_s, dlc_time[int(lift_times[ee])]+window_s, num =  199)#
    binned_spike_dict = []
    for ii in tqdm(range(0, np.size(spike_dict)), desc='binning spikes'):
        binned_spike_dict.append( np.histogram(spike_dict[ii], bins = normalized_time)[0])

    #binned_spike_dict = binned_spike_dict[:,:1768047]

    # plt.figure()
    # #plt.hist( np.diff(camera_ttl_times),1000, range = [-.1,.1])
    # plt.plot(np.diff(cam_ttl_times[:1796482]))#np.diff
    # plt.show()

    start_time = cam_ttl_times[0]
    end_time = cam_ttl_times[-1]

    #1795025 = the split in the video aka session one end time in indexes 
    binned_spike_dict = np.array(binned_spike_dict)[:,:int(end_time/step)]

    smooth_spike_dict = []
    for ii in tqdm(range(0, np.shape(binned_spike_dict)[0]), desc='smoothing spikes'):
        kernel = np.ones(kernel_size)
        smooth_spike_dict.append(sig.fftconvolve(binned_spike_dict[ii], kernel, mode='same'))


    ### get sampling rate of the paw files

    # get the interval between each camera TTL
    cam_ttl_intervals = np.diff(cam_ttl_times)

    # print the unique intervals and plot a histogram (excludes outliers)
    # print('Intervals between camera TTLs (s):')
    # print(np.unique(cam_ttl_intervals))
    # plt.figure()
    # plt.hist(cam_ttl_intervals, range=[8e-3, 9e-3], bins=100)
    # plt.xlabel('Interval between camera TTLs (s)')
    # plt.ylabel('Count');
    # plt.show()



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

    plt.figure()
    for i in tqdm(range(len(paws)), desc='applying cwt to paw speed data'):
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

        # plot spectrogram
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


    # plt.figure()
    # plt.rcParams.update({'font.size': 18})

    find_behave(avg_freq, window_size, walk_threshold, still_threshold)

    xaxis1 = np.linspace(0,cam_ttl_times[-1], np.size(fl_paw_speed))
    xaxis2 = np.linspace(0,cam_ttl_times[-1], np.size(behave))
    # plt.subplots()
    # plt.plot(xaxis1,fl_paw_speed, color = 'b', label = 'front left paw', alpha = 0.5)
    # plt.plot(xaxis1,fr_paw_speed, color = 'indianred', label = 'front right paw', alpha = 0.5)
    # plt.plot(xaxis1,bl_paw_speed, color = 'c', label = 'back left paw', alpha = 0.5)
    # plt.plot(xaxis1,br_paw_speed, color = 'm', label = 'back right paw', alpha = 0.5)

    # plt.plot(xaxis2,behave, color = '0.2', label = 'behavior type')
    # plt.plot(xaxis2,np.array(avg_val)*1000, color = '0.3', label = 'badpassed power (4-7Hz)')
    # plt.xlabel('time (s)')
    # plt.ylabel('paw speed (pixels/second')
    # plt.legend()
    # plt.show()

    #np.save('run_frequency_3to7Hz_vl6_230418_', avg_val)
    #np.save('run_still_bianarized_vl6_230418_winsz20_slide15frames', behave)

    # interpolate back too camera_ttl_times size 

    #end_time = camera_ttl_times[int(1795025)]
    #xaxis2 = np.linspace(0,cam_ttl_times[:1795025], np.size(behave))

    #normalized_time
    #cut_norm_time = np.arange(0, end_time, step)
    #cut_norm_time = cut_norm_time[:-1]

    # rhythm = np.interp(cam_ttl_times,xaxis2,avg_val)
    neuron_time = np.arange(start_time, end_time, step)
    neuron_time = neuron_time[:-1]
    rhythm_interp = np.interp(neuron_time,xaxis2,avg_val)
    behave = np.interp(neuron_time,xaxis2,behave)
    behave = np.round(behave).astype(int)

    paw_diff = fl_paw_speed - fr_paw_speed
    neuron_time = np.arange(start_time, end_time, step)
    #neuron_time = neuron_time[:-1]
    diff_interp = np.interp(neuron_time,xaxis1,paw_diff)

    #walk_idx = np.where(np.logical_and(rythem_interp > 0.001 , rythem_interp < 0.007))[0]
    acceleration = gaussian_filter1d(np.diff(diff_interp),2)

    return smooth_spike_dict, rhythm_interp, acceleration, behave


def obtain_binned_rhythm(speeds):
    percentiles = np.percentile(speeds, [0, 25, 50, 75, 98])
    print(f'Percentiles: {percentiles}')
    thresholded_speeds = np.array([i if i < percentiles[4] else percentiles[4] for i in speeds])

    # plt.figure()
    # plt.yscale("log")
    # plt.plot(thresholded_speeds)
    # plt.plot(percentiles[0] * np.ones_like(speeds), label = '0th percentile')
    # plt.plot(percentiles[1] * np.ones_like(speeds), label = '25th percentile')
    # plt.plot(percentiles[2] * np.ones_like(speeds), label = '50th percentile')
    # plt.plot(percentiles[3] * np.ones_like(speeds), label = '75th percentile')
    # plt.plot(percentiles[4] * np.ones_like(speeds), label = '98th percentile')
    # plt.legend()
    # plt.show()

    thresholded_speeds.shape
    binned_speeds = np.digitize(thresholded_speeds, percentiles[:-1])
    bins, counts = np.unique(binned_speeds, return_counts=True)
    print(f'Bins: {bins}')
    print(f'Counts: {counts}')
    return binned_speeds


def obtain_binned_acceleration(behavior, acceleration, spikes, just_running):
    running_acceleration = acceleration[behavior == 1]
    if just_running:
        spikes = spikes[behavior == 1]
    percentiles = np.percentile(running_acceleration, [0, 25, 50, 75, 100])
    print(f'Percentiles: {percentiles}')
    if just_running:
        binned_acceleration = np.digitize(running_acceleration, percentiles[:-1])
    else:
        binned_acceleration = np.digitize(acceleration, percentiles[:-1])
    bins, counts = np.unique(binned_acceleration, return_counts=True)
    print(f'Bins: {bins}')
    print(f'Counts: {counts}')

    # set acceleration to 0 when behavior is -1, 1 when behavior is 0, and acceleration when behavior is 1
    # acceleration = np.where(behavior == -1, 0, acceleration)
    if not just_running:
        binned_acceleration = np.where(behavior < 1, behavior, binned_acceleration)
    bins, counts = np.unique(binned_acceleration, return_counts=True)
    print(f'Acceleration: {bins}')
    print(f'Counts: {counts}')
    return binned_acceleration, spikes


if __name__ == "__main__":
    generate_binned_data('sleep_data')
    # first_video_end = 1795025
    # data_path = 'sleep_data'

    # spikes, speed = generate_binned_data(data_path)

    # spikes = np.array(spikes).transpose()
