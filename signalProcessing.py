import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
from tools import get_session_split_and_grouped_by_markers

NOTION_ELECTRODES = ["CP6","F6","C4","CP4","CP3","F5","C3","CP5"]
MUSE_NOTION_ELECTRODES = ['TP9', 'AF7', 'AF8', 'TP10']
MUSE_FREQUENCY = 256
NOTION_FREQUENCY = 250


ALPHA_MARKERS = ["eyes-open", "eyes-closed"]
SMR_MARKERS = ["rest","visualisation","movement"]
N170_MARKERS = ['house','face']
P300_MARKERS = ['red', 'blue' ,'press']

def getBandPSD(f1, f2, frequencies, power_spectral_densities):
    band_dsp = []
    band_frequencies = []
    all_dsp = []
    all_frequencies = []
    for i in range(len(frequencies)):
        if frequencies[i] >= 5 and frequencies[i] < 20:
            all_dsp.append(power_spectral_densities[i])
            all_frequencies.append(frequencies[i])
            if frequencies[i] >= f1 and frequencies[i] < f2:
                band_dsp.append(power_spectral_densities[i])
                band_frequencies.append(frequencies[i])

    return band_dsp, band_frequencies, all_dsp, all_frequencies


def calculatePSD(signal, Fe):
    nfft = getNextPow2(len(signal))
    fft_data = np.fft.fft(signal, n=nfft, axis=0) / (len(signal))
    power_spectral_densities = np.abs(fft_data[0:int(nfft / 2)])
    frequencies = Fe / 2 * np.linspace(0, 1, int(nfft / 2))
    return power_spectral_densities, frequencies


def calculateSimpleSNR(signal):
    mean = np.mean(signal)
    sd = np.std(signal)
    return abs(mean/sd)


def getNextPow2(num):
    x = 2
    while x <= num:
        x *= 2
    return x


def alphaRhythmDetection(chann_data, chann, Fe, n_good_result, total, plot=True):
    signal_len = len(chann_data['eyes-open'][0])
    x = np.linspace(0, 4*signal_len, signal_len, endpoint=False)

    means_alpha_open = []
    means_alpha_closed = []

    # Eyes open
    if plot:
        print("-----------------------EYES OPEN--------------------------")
    for exp in range(len(chann_data['eyes-open'])):
        total += 1
        signal = chann_data['eyes-open'][exp].values
        snr = calculateSimpleSNR(signal)
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        alpha_dsp, alpha_frequencies, all_dsp, all_frequencies = getBandPSD(
            8, 12, frequencies, power_spectral_densities)
        means_alpha_open.append(np.mean(alpha_dsp))
        alphaRdetected = max(alpha_dsp) / max(all_dsp) > 0.9
        color = 'red' if alphaRdetected else 'green'
        n_good_result = n_good_result + 1 if color == 'green' else n_good_result
        #print("[Eyes O-exp {0}][channel {1}] Signal-to-noise ratio: ".format(exp,chann), snr)
        if plot:
            print("[Eyes O-exp {0}][channel {1}] Alpha Rhythm detected: ".format(exp,chann), colored(alphaRdetected,color))

    # Eyes closed
    if plot:
        print("-----------------------EYES CLOSED--------------------------")
    for exp in range(len(chann_data['eyes-closed'])):
        total += 1
        signal = chann_data['eyes-closed'][exp].values
        snr = calculateSimpleSNR(signal)
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        alpha_dsp, alpha_frequencies, all_dsp, all_frequencies = getBandPSD(
            8, 12, frequencies, power_spectral_densities)
        means_alpha_closed.append(np.mean(alpha_dsp))
        alphaRdetected = max(alpha_dsp) / max(all_dsp) > 0.9
        color = 'green' if alphaRdetected else 'red'
        n_good_result = n_good_result + 1 if color == 'green' else n_good_result
        #print("[Eyes C-exp {0}][channel {1}] Signal-to-noise ratio: ".format(exp,chann), snr)
        if plot:
            print("[Eyes C-exp {0}][channel {1}] Alpha Rhythm detected: ".format(exp,chann), colored(alphaRdetected,color))
    return means_alpha_open,means_alpha_closed,chann, n_good_result, total

def plotAlphaDetectionStats(means_alpha_open, means_alpha_closed,chann):
    plt.boxplot([means_alpha_open, means_alpha_closed],showmeans=True)
    plt.title(chann)
    plt.ylabel("Alpha psd")
    plt.xticks([1, 2], labels=['Eyes Open', 'Eyes Closed'])
    

def smrDetection(chann_data, chann, figure, Fe):
    signal_len = len(chann_data['eyes open data'][0])
    x = np.linspace(0, 4*signal_len, signal_len, endpoint=False)
    print(len(x))

    means_Smr_rest = []
    means_Smr_visualisation = []
    means_Smr_movement = []

    # Rest
    for sig in range(len(chann_data['Rest'])):
        signal = chann_data['Rest'][sig]
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        smr_dsp, smr_frequencies, all_dsp, all_frequencies = getBandPSD(
            12, 15, frequencies, power_spectral_densities)
        means_Smr_rest.append(np.mean(smr_dsp))

    # Visualisation
    for sig in range(len(chann_data['Visualisation'])):
        signal = chann_data['Visualisation'][sig]
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        smr_dsp, smr_frequencies, all_dsp, all_frequencies = getBandPSD(
            12, 15, frequencies, power_spectral_densities)
        means_Smr_visualisation.append(np.mean(smr_dsp))

    # Movement
    for sig in range(len(chann_data['Movement'])):
        signal = chann_data['Movement'][sig]
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        smr_dsp, smr_frequencies, all_dsp, all_frequencies = getBandPSD(
            12, 15, frequencies, power_spectral_densities)
        means_Smr_movement.append(np.mean(smr_dsp))

    return means_Smr_rest, means_Smr_visualisation, means_Smr_movement, chann

def plotsmrDetection(means_Smr_rest, means_Smr_visualisation, means_Smr_movement, chann):
    plt.boxplot([means_Smr_rest, means_Smr_visualisation, means_Smr_movement],showmeans=True)
    plt.title(chann)
    plt.ylabel("SMR psd")
    plt.xticks([1, 2, 3], labels=['Rest', 'Visualisation', 'Movement'])

def plotSignals(chann_data, index_chann, Fe):
    colors = ['blue','green','red','cyan','magenta','yellow']
    signal_len = len(chann_data['eyes-open'][0])
    plt.subplot(8,2,index_chann*2+1)
    plt.title("{} - eyes open".format(NOTION_ELECTRODES[index_chann]))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Psd amplitude')
    for exp in range(len(chann_data['eyes-open'])):
        signal = chann_data['eyes-open'][exp].values
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        alpha_dsp, alpha_frequencies, all_dsp, all_frequencies = getBandPSD(
            8, 13, frequencies, power_spectral_densities)
        
        plt.plot(alpha_frequencies,alpha_dsp,colors[exp])
        
    plt.subplot(8,2,index_chann*2+2)
    plt.title("{} - eyes closed".format(NOTION_ELECTRODES[index_chann]))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Psd amplitude')
    for exp in range(len(chann_data['eyes-closed'])):
        signal = chann_data['eyes-closed'][exp].values
        power_spectral_densities, frequencies = calculatePSD(signal, Fe)
        alpha_dsp, alpha_frequencies, all_dsp, all_frequencies = getBandPSD(
            8, 13, frequencies, power_spectral_densities)
        
        plt.plot(alpha_frequencies,alpha_dsp,colors[exp])

def alphaAnalysis(sessionData,plot=True):
    ''' Run the alpha Analysis for a session '''
    
    dataset = get_session_split_and_grouped_by_markers(sessionData, ALPHA_MARKERS)
    if plot:
        fig1 = plt.figure(1,figsize=(15,10))
        fig1.suptitle('Statistics of the mean amplitude of the alpha band')
    total = 0
    n_good_result = 0
    for electrode in range(len(NOTION_ELECTRODES)):
        if plot:
            plt.subplot(4,2,electrode+1)
        means_alpha_open, means_alpha_closed, chann, n_good_result, total = alphaRhythmDetection(dataset[NOTION_ELECTRODES[electrode]],NOTION_ELECTRODES[electrode],250,n_good_result,total,plot)
        if plot:
            plotAlphaDetectionStats(means_alpha_open, means_alpha_closed, chann)
    percentage = n_good_result/total
    color = 'green' if percentage >= 0.8 else 'red'
    print("Percentage of success: ", colored("{}%".format(percentage*100),color))
    if plot:
        fig2 = plt.figure(2,figsize=(15,20))
        fig2.suptitle("Fourier Analysis")
        for electrode in range(len(NOTION_ELECTRODES)):
            plotSignals(dataset[NOTION_ELECTRODES[electrode]], electrode, 250)
    return percentage >= 0.8

def smrAnalysis(sessionData,plot=True):
    ''' Run the smr Analysis for a session '''
    
    dataset = get_session_split_and_grouped_by_markers(sessionData, SMR_MARKERS)
    fig = plt.figure(1,figsize=(15,20))
    for electrode in range(len(NOTION_ELECTRODES)):
        plt.subplot(4,2,electrode+1)
        means_Smr_rest, means_Smr_visualisation, means_Smr_movement, chann = smrDetection(dataset[NOTION_ELECTRODES[electrode]],NOTION_ELECTRODES[electrode],NOTION_FREQUENCY,plot)
        if plot:
            plotsmrDetection(means_Smr_rest, means_Smr_visualisation, means_Smr_movement, chann)

