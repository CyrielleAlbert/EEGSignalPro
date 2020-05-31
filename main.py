import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io as sio

locations = ['CP6', 'F6', 'C4', 'CP4', 'CP3', 'C3', 'F5', 'CP5']
def openCSVfile(path):
    data = pd.read_csv(path)
    return data

def openMATfile(path):
    mat = sio.loadmat(path)
    print(mat['SIGNAL'].shape)
    data = mat['SIGNAL'][:,1:]
    print(data.shape)
    timestamp = mat['SIGNAL'][:,0]
    return data,timestamp   

def openTXTfile(path):
    return 0

def openJSONfile(path):
    return 0

def alphaRhythmDetection(signal,time,figure):
    Fe = 512
    nfft = getNextPow2(len(signal))

    fft_data = np.fft.fft(signal, n=nfft, axis=0) / (len(signal))
    power_spectral_densities = np.abs(fft_data[0:int(nfft / 2)])
    frequencies = Fe / 2 * np.linspace(0, 1, int(nfft / 2)) 
    alpha_dsp = []
    alpha_frequencies = []
    all_dsp = []
    all_frequencies = []
    for i in range(len(frequencies)): 
        if frequencies[i] >= 5 and frequencies[i]< 20:
            all_dsp.append(power_spectral_densities[i])
            all_frequencies.append(frequencies[i])
            if frequencies[i] >= 8 and frequencies[i]< 12 :
                alpha_dsp.append(power_spectral_densities[i])
                alpha_frequencies.append(frequencies[i])
    alphaRdetected = max(alpha_dsp)/ max(all_dsp) > 0.9
    print(alphaRdetected)
    plt.figure(figure)
    plt.subplot(2,1,1)
    plt.plot(alpha_frequencies,alpha_dsp)
    plt.subplot(2,1,2)
    plt.plot(all_frequencies,all_dsp)

def calculateSimpleSNR(signal):
    mean = np.mean(signal)
    sd = abs(np.std(signal))
    return mean/sd

def getNextPow2(num):
    x=2
    while x<=num:
        x *=2
    return x

def plotData(data):
    startTime = data['timestamp'][0]
    time = []
    for tstamp in data['timestamp'] :
        time.append(tstamp-startTime)
    time = np.array(time)
    plt.figure(1)
    for chann in locations :
        plt.plot(time[1000:],data[chann][1000:])

    plt.xlabel('time')
    plt.ylabel('signal')
    plt.figure(2)
    print('next pow 2')
    next_pow = getNextPow2(len(data['CP6']))
    for chann in locations :
        print("psd")
        plt.psd(data[chann],Fs=256,NFFT=next_pow, pad_to=next_pow)

def run_neurosity_data():
    path = './session4.csv'
    data = openCSVfile(path)
    plotData(data)
    startTime = data['timestamp'][0]
    print(startTime)
    time = []
    for tstamp in data['timestamp'] :
        time.append(tstamp-startTime)
    time = np.array(time)
    eyes_closed_t = 30*250
    eyes_open_t = 60*250
    eyes_closed_t2 = 90*250
    eyes_open_t2 = 120*250
    eyes_closed_t3 = 150*250
    for chann in locations :
        #alphaRhythmDetection(data[chann][eyes_closed_t2:eyes_open_t2],time[eyes_closed_t:eyes_open_t2],3)
        #alphaRhythmDetection(data[chann][eyes_open_t2:eyes_closed_t3],time[eyes_open_t2:eyes_closed_t3],4)
        snr = calculateSimpleSNR(data[chann])
        print("snr {} :".format(chann),snr)
        if chann == 'C3' or chann == 'C4':
            alphaRhythmDetection(data[chann][eyes_closed_t2:eyes_open_t2],time[eyes_closed_t:eyes_open_t2],3)
            alphaRhythmDetection(data[chann][eyes_open_t2:eyes_closed_t3],time[eyes_open_t2:eyes_closed_t3],4)

    plt.show()

def run_openS_data():
    #openMATfile('subject_00.mat')
    data,timestamp = openMATfile('subject_01.mat')
    startTime = timestamp[0]
    print(startTime)
    print(timestamp)
    time = []
    for tstamp in timestamp :
        time.append(tstamp-startTime)
    time = np.array(time)
    print(time)
    eyes_open=[]
    eyes_closed=[]
    print(data.shape,len(time))

    for i in range(len(time)):
        if data[i,-2] == 1:
            eyes_open.append((time[i]))
        if data[i,-1] == 1:
            eyes_closed.append((time[i]))
    print(eyes_open,eyes_closed)
    for i in range(len(eyes_open)):
        for chann in range(16) :
            print(data[:,chann])
            snr = calculateSimpleSNR(data[:,chann])
            print("snr {} :".format(chann),snr)
            try :
                alphaRhythmDetection(data[int(eyes_closed[i]*512):int(eyes_open[i+1]*512),chann],time[int(eyes_closed[i]*512):int(eyes_open[i]*512)],i)
            except :
                print('end')
            #alphaRhythmDetection(data[chann][eyes_open_t2:eyes_closed_t3],time[eyes_open_t2:eyes_closed_t3],4)
    plt.show()
    
if __name__ == "__main__":
    #run_neurosity_data()
    run_openS_data()
    





    
