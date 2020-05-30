Python script for checking EEG signal quality

## Detection of the Alpha Rhythm 
The alpha rhythm is a pattern of slow brain waves (alpha waves) in normal persons at rest with closed eyes, thought by some to be associated with an alert but daydreaming mind. 
The alpha rhythm can be detected when closing eyes. A peak can we seen on the FFT representation when eyes closed but not when eyes open. The frequency depends on the subject but usually stand between 8Hz and 12Hz. 

### Data format : 
**.CSV** : [index,CP6,F6,C4,CP4,CP3,C3,F5,CP5,,timestamp]

**.mat** : {'SIGNAL': [timestamp, channel 1, ... , channel 16, eyes open trig, eyes closed trig] }

### Libraries used : 
 - Matplotlib
 - Numpy
 - Pandas
 - Scipy

# TO DO :
## Detection of the Sensorimotor Rhythm (SMR)
The SMR (sensorimotor rhythm) is the idling rhythm for the motor strip. As SMR increases, a person’s body becomes more relaxed.
The Sensorimotor can be detected when imagining a movment or moving a part of the body. A drop in the area of the 13Hz to 15Hz can be seen of the FFT representation.
The frequency depends on the subject.

## Signal-to-noise Ratio 
Signal-to-noise ratio (abbreviated SNR or S/N) is a measure that compares the level of a desired signal to the level of background noise. SNR is defined as the ratio of signal power to the noise power, often expressed in decibels.




# REFERENCES :
https://www.researchgate.net/publication/271643878_Alpha_rhythm_onset_detector_based_on_localized_EEG_sensor
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4082720/



