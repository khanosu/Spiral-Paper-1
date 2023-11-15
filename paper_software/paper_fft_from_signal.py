#######################################################################
#                  From Discrete Signal to FFT
#
# STEP 1: Read Discrete Signal from csv file
# STEP 2: Determine the Real FFT of the Discrete Signal
# STEP 3: Determine the Inverse FFT of the truncated FFT
# STEP 4: Save the coefficients of the  truncated FFT to Pandas data frames 
#
#######################################################################
import matplotlib.pyplot as plt
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.signal import savgol_filter, butter, sosfilt
import math
import statistics

#######################################################################
#                            STEP 1 
# Read Discrete Signal from csv file
# Apply Low Pass Filter to eliminate high spatial frequency noise (spikes)
#######################################################################

# input csv filename containing the discrete signal
# input_file_name = 'spiral42_disc_signal'
# input_file_name = 'spiral_web_flattened1_disc_signal'
input_file_name = 'S6_disc_signal'
cropping_ratio =  0.8234368428774023 # ratio of the time to draw the cropped spirral vs. the time to draw the whole spiral
time_to_draw = 11.7
  # time it took for the patient to draw the spiral in seconds
time_to_draw_measured_flag = True
estimated_tremor_freq = 5.12 # Hz 

spatial_cutoff_wrt_nyquest = .40
# spatial_cutoff_wrt_nyquest = 0.25
# spatial_cutoff_wrt_nyquest = 0.35

# N_FFT_truncate: Number of truncated coefficients of the spectrum
N_FFT_truncate = 150


tremor_freq_limit = 12

T = time_to_draw * cropping_ratio # time it took for the patient to draw the cropped part of the spiral in seconds

with open(input_file_name+"_fft_log", 'w') as log:
    pr = f"STEP 1 -- Read Discrete Signal and apply LP Filter"
    print('\n'+pr); log.write('\n'+pr+'\n')

    # log = open(input_file_name+"_fft_log", 'w')
    pr = f"input_filename = {input_file_name}" 
    print(pr); log.write(pr+'\n')

    pr = f"log_filename = {input_file_name}_fft_log"
    print(pr); log.write(pr+'\n')

    # Read signal from the csv file
    df1 = pd.read_csv(input_file_name + ".csv") 
    # Need to flatten to change shape froom (n, 1) to (n, )
    df2 = df1.to_numpy().flatten() 

    # subtract the dc (mean) value
    aver = np.average(df2)
    df3 = df2 - aver

    #---------- Appy LP FILTER ---------------
    # filter LPF signal (Butterworth low)
    filter_order = 5
    filter_type = 'low'
    # spatial_cutoff_wrt_nyquest = 0.4
    
    spatial_sampling_rate = 1 # samples/pixel
    spatial_nyquest_freq = spatial_sampling_rate/2.0 # samples/pixel
    spatial_cutoff = spatial_cutoff_wrt_nyquest*spatial_nyquest_freq #samples/pixel

    pr = f"Spatial Sampling Rate: {spatial_sampling_rate} sample/pixel"
    print('\n'+pr); log.write('\n'+pr+'\n')
    pr = f"Spatial Nyquest freq: {spatial_nyquest_freq} samples/pixel" 
    print(pr); log.write(pr+'\n')
    pr = f"Spatial cutoff freq: {spatial_cutoff} samples/pixel"
    print(pr); log.write(pr+'\n')
    pr = f"Spatial_cutoff_freq/spatial_nyquest_freq: {spatial_cutoff/spatial_nyquest_freq}\n"
    print(pr); log.write(pr)

    # Apply LP filter
    sos = butter(filter_order, spatial_cutoff, filter_type, fs=spatial_sampling_rate, output='sos')
    filtered_signal = sosfilt(sos, df3)
    # df3 = filtered_signal
    
    # Calculate the RMS of the discrete signal before normalization. Recall that the mean value 
    # has already been subtracted from the signal, therefore rms and std deviation are the same.
    rms_pre_normalization = statistics.stdev(df3)
    pr = f"Signal rms before normalization: {rms_pre_normalization:.2f} pixel values"
    print('\n'+pr); log.write('\n'+pr+'\n')
    
    # Normalize to 10000
    df = np.int16((filtered_signal / np.abs(filtered_signal).max()) * 10000)

    plt.plot(df)
    # plt.plot(filtered_signal)
    plt.title('Normalized Signal', fontsize='22')
    plt.xlabel('samples')
    plt.ylabel('signal [a.u.]')
    plt.show()

    #######################################################################
    #                            STEP 2 
    # Determine the Real FFT of the Discrete Signal
    #######################################################################
    pr = f"STEP 2 -- Determine FFT"
    print('\n'+pr); log.write('\n'+pr+'\n')
    
    # rfft stands for real fft
    # Number of sample points, N
    N = df.shape[0]
    pr = f"Number of sample points: N = {N}"
    print(pr); log.write(pr+'\n')

    yf = rfft(df)
    # xf contains the frequencies 
    xf = rfftfreq(N, 1)
    # Number of coefficients in rfft will equal N/2 + 1
    pr = f"Number of coefficients in the transform, N/2 + 1: {np.shape(yf)[0]}"
    print(pr); log.write(pr+'\n')

    # Determine parameters based on the time, T, it took a patient to draw the cropped part of the spiral
    Sampling_rate = N/T
    Nyquist_freq = (Sampling_rate/2.0)
    LP_filter_cutoof_Hz =( spatial_cutoff/spatial_nyquest_freq)*Nyquist_freq 
    
    pr = f"\nTime it took the patient to draw the whole spiral: {time_to_draw:.2f} sec"
    print(pr); log.write(pr) 
    pr = f"\nTime it took the patient to draw the cropped part of the spiral: {T:.2f} sec"
    print(pr); log.write(pr+'\n')
    pr = f"Sampling Rate: {Sampling_rate:.2f} Hz"
    print(pr); log.write(pr+'\n')
    pr = f"Nyquest Frequency: {Nyquist_freq:.2f} Hz"
    print(pr); log.write(pr+'\n')
    pr = f"Low pass filter cutoff in Hz: { LP_filter_cutoof_Hz:.2f} Hz\n"
    print(pr); log.write(pr+'\n')
 
    # Apply Savitzky-Golay filter to visualize any peaks better
    yhat = savgol_filter(np.abs(yf)/N, 51, 3) # window size 51, polynomial order 3 
    N_to_plot = int(N/2)+1
    
    pr = f"Number of coefficients in the LP band: {int(spatial_cutoff_wrt_nyquest*N_to_plot)}"
    print(pr); log.write(pr+'\n')
    
    def freq2N(f):
        return f * (N_to_plot/Nyquist_freq)

    def N2freq(N):
        return (N * (Nyquist_freq/N_to_plot))

    N_LP_filter_cutoff = int(spatial_cutoff_wrt_nyquest*N_to_plot)
    if time_to_draw_measured_flag == True:
        fig, ax = plt.subplots()
          
        # xf0 contains the frequencies 
        xf0 = rfftfreq(N, 1)*Sampling_rate

        ax.plot(xf0, np.abs(yhat[:N_to_plot]), label = 'Magnitute FFT')
        # plt.stem(np.abs(yhat[:N_to_plot]), markerfmt=" ")
        # plt.stem(np.abs(yf))
        
        # for dual secondary x-axis
        secax = ax.secondary_xaxis('top', functions=(freq2N, N2freq))
        secax.set_xlabel('\ncoefficients')
        
        truncation_freq = N2freq(N_FFT_truncate)
        ax.axhline(y = 0.0, color = 'k')
        ax.axvline(x = estimated_tremor_freq, color = 'r', linestyle = 'dashed', label = f'Estimated tremor freq: {estimated_tremor_freq:.2f} Hz')
        ax.axvline(x = tremor_freq_limit, color = 'g', label = f'Tremor freq limit: {tremor_freq_limit:.2f} Hz')
        ax.axvline(x = truncation_freq, color = 'b', linestyle = 'dashed', label = f'Truncated coefficients: {N_FFT_truncate}\nTruncation freq: {truncation_freq:.2f} Hz')
        ax.axvline(x = LP_filter_cutoof_Hz, color = 'k', linestyle = 'dashed', label = f'LP Filter cutoff coefficients: {N_LP_filter_cutoff}\nLP Filter cutoff freq: {LP_filter_cutoof_Hz:.2f}  Hz')
        ax.axvline(x = Nyquist_freq, color = 'g', linestyle = 'dashed', label = f'Nyquist freq coefficients: {int(N/2)+1}\nNyquest freq: {Nyquist_freq:.2f}  Hz')
        ax.legend(loc='upper right')

        plt.title('Magnitude(FFT)', fontsize='22')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('magnitude [a.u.]')
        plt.show() 
        
    else:
        fig, ax = plt.subplots()
    
        ax.plot(np.abs(yhat[:N_to_plot]), label = 'Magniture FFT')
        # plt.stem(np.abs(yhat[:N_to_plot]), markerfmt=" ")
        # plt.stem(np.abs(yf))
        ax.axhline(y = 0.0, color = 'k')
        ax.axvline(x = N_FFT_truncate, color = 'b', linestyle = 'dashed', label = f'Truncated coefficients: {N_FFT_truncate}')
        ax.axvline(x = N_LP_filter_cutoff, color = 'k', linestyle = 'dashed', label = f'LP Filter cutoff: {N_LP_filter_cutoff}') 

        ax.legend(loc='upper right')

        plt.title('Magnitude(FFT)', fontsize='22')
        plt.xlabel('coefficeinets')
        plt.ylabel('magnitude [a.u.]')
        plt.show()  

    # inverse rfft using irfft
    iyf = irfft(yf)
    plt.plot(df, label="Signal", c="blue")
    plt.plot(iyf, linestyle = 'dashed', c="red", linewidth="2", label="Inverse FFT")
    plt.title('Inverse FFT vs. Signal', fontsize='22')
    plt.xlabel('samples')
    plt.ylabel('signal [a.u.]')
    plt.legend(loc='upper right')
    plt.show()
    
    #######################################################################
    #                            STEP 3 
    # Determine the Inverse FFT of the truncated FFT
    # Determine the Truncation Error
    #######################################################################
    pr = f"STEP 3 -- Truncate FFT and calculate Trunction Error"
    print('\n'+pr); log.write('\n'+pr+'\n')
    
    pr = f"Number of coefficients in the transform after truncation: {N_FFT_truncate}"
    print(pr); log.write(pr+'\n')

    yf_trunc = []
    for i in range(N_FFT_truncate):
        yf_trunc.append(yf[i])

    # plt.stem(np.abs(yf_trunc))
    # plt.title('Magnitude(Truncated FFT)', fontsize='22')
    # plt.xlabel('coefficients')
    # plt.ylabel('magnitude')
    # plt.show() 

    # plt.stem(np.angle(yf_trunc))
    # plt.title('Phase(Truncated FFT)', fontsize='22')
    # plt.xlabel('coefficients')
    # plt.ylabel('phase')
    # plt.show() 

    # Use irfft to determine the inverse transform from the truncated spectrum
    N_inv = (N_FFT_truncate-1)*2
    iyf_trunc = irfft(yf_trunc, n=N_inv)

    scale = N/N_inv
    ixf = []
    for i in range(iyf_trunc.shape[0]):
        # ixf.append(int(i*scale))
        ixf.append(i*scale)

    # Plot the inverse transform from the truncated spectrum on top of original signal
    plt.plot(df, label='Signal', c='blue')
    # plt.plot(ixf, iyf_trunc/scale, linestyle = 'dotted', marker='.', c="red", label='Inverse FFT')
    plt.plot(ixf, iyf_trunc/scale, linestyle = 'dashed', c="red", label='Inverse FFT', linewidth="2")
    # plt.plot(ixf, iyf_trunc/scale, c="red", label='Inverse FFT')
    plt.title('Inverse Truncated FFT vs. Signal', fontsize='22')
    plt.xlabel('samples')
    plt.ylabel('signal [a.u.]')
    plt.legend(loc='upper right')
    plt.show()

    # Determine the relative error bettween the input signal df and the truncated 
    # inverse transform iff_trunc

    # print(f"df.shape[0]: {df.shape[0]}")
    # print(f"iyf_trunc.shape[0]: {iyf_trunc.shape[0]}")
    
    # Note that the time dependent discrete function iff_trunc has fewer sample points than
    # the time dependent df function, therefore the samples of the two functions fall on 
    # differnt points on the time axis. To calculate the rms error the loop below runs through 
    # each sample point of ifft_trunc and useses linear interpolation to determine the value 
    # of df at the point.

    sum1 = 0
    sum2 = 0 
    for jj in range(iyf_trunc.shape[0]-2):
        j = jj+1
        x = j*scale
        k = int(x)
        # kk = math.ceil(x) # this will also work
        kk = k+1
        
        dfx = df[k] + (df[kk]-df[k])*(x-k)
        sum1 = sum1 + ((iyf_trunc[j]/scale) - dfx)**2
        sum2 = sum2 + (iyf_trunc[j]/scale)**2

    error = math.sqrt(sum1/sum2)
    pr = f"Truncation Error: {error*100:.2f}%\n"
    print(pr); log.write(pr)

    #######################################################################
    #                            STEP 4 
    # Save the coefficients of the truncated FFT to Pandas data frames
    #######################################################################
    pr = f"STEP 4 -- Save Coefficients"
    print('\n'+pr); log.write('\n'+pr)
    
    # Save cropped coeffiecien to pandas data frame and save to disc
    dict = {'Magnitude': np.abs(yf_trunc), "Phase": np.angle(yf_trunc)}        
    df = pd.DataFrame(dict) 

    df.to_csv(input_file_name+'_trunc_spectrum.csv', index=False) 
    pr = f"Truncated spectrum has been saved as {input_file_name+'_trunc_spectrum.csv'}"
    print(pr); log.write('\n'+pr+'\n')

    df2 = df.drop(['Phase'], axis = 1)
    df2 = df2.T
    column_names = []
    for i in range(N_FFT_truncate):
        column_names.append("m"+str(i))
    df2.columns = column_names

    df3 = df.drop(['Magnitude'], axis = 1)
    df3 = df3.T 
    column_names = []
    for i in range(N_FFT_truncate):
        column_names.append("p"+str(i))
    df3.columns = column_names

    df4 = pd.concat([df2.reset_index(drop=True), df3.reset_index(drop=True)], axis=1)
    df4.to_csv(input_file_name+'_trunc_spectrum_features.csv', index=False) 
    pr = f"Transpose of truncated spectrum as features has been saved as {input_file_name+'_trunc_spectrum_feaatures.csv'}\n"
    print(pr); log.write(pr+'\n')

    log.close()
    
    #######################################################################
    #                            END 
    #######################################################################