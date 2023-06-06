import numpy as np
import scipy
from scipy.signal import firwin
import matplotlib.pyplot as plt

# Plot 3 images with different titles
def plotImagesTitles(img1, img2, img3, t1, t2, t3):

    fig, ax = plt.subplots(1, 3, figsize=(8, 6))

    ax[0].imshow(img1)
    ax[0].set_title(t1, fontsize = 10)

    ax[1].imshow(img2)
    ax[1].set_title(t2, fontsize = 10)

    ax[2].imshow(img3)
    ax[2].set_title(t3, fontsize = 10)

    plt.show()

# Plot RGB channels
def plotRGBchannels(redChannel, greenChannel, blueChannel, title = ""):

    rx = range(len(redChannel))
    gx = range(len(greenChannel))
    bx = range(len(blueChannel))

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(8, 6))

    ax[0].plot(rx, redChannel, color = 'red')
    ax[0].set_ylim(min(redChannel)-10,max(redChannel)+10)
    ax[0].grid(True)

    ax[1].plot(gx, greenChannel, color = 'green')
    ax[1].set_ylim(min(greenChannel)-10,max(greenChannel)+10)
    ax[1].grid(True)

    ax[2].plot(bx, blueChannel, color = 'blue')
    ax[2].set_ylim(min(blueChannel)-10,max(blueChannel)+10)
    ax[2].grid(True)

    plt.suptitle(title)
    plt.show()

# Plot Heart Rate and Breath Rate Signals
def plotHearthBreathSignals(heartSig, breathSig):

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(9, 7))

    ax[0].plot(range(len(heartSig)), heartSig, color = 'red')
    ax[0].set_ylim(min(heartSig), max(heartSig))
    ax[0].set_title("Computed Hearth Signal")
    ax[0].grid(True)

    ax[1].plot(range(len(breathSig)), breathSig, color = 'blue')
    ax[1].set_ylim(min(breathSig), max(breathSig))
    ax[1].set_title("Computed Breath Signal")
    ax[1].grid(True)

    plt.show()

# Reconstructs images with the computed ROIs in order to check what pixels have been stored
def reconstructImages(startingImages, computedROIs):

    # Here we reconstruct the images starting from the pixels we stored for each ROI and
    # after that we try to plot some of them to check the results
    reconstructedImages = []

    # Calculate maximum dimensions
    maxH = max([frame.shape[0] for frame in startingImages])
    maxW = max([frame.shape[1] for frame in startingImages])

    for roi in computedROIs:

        # this produces some errors when reconstructing images
        # height, width, channels = fingerTips[0].shape
        height, width, channels = maxH, maxW, startingImages[0].shape[2]
        # height, width, channels = maxH, maxW, 3

        reconstructedImage = np.zeros((height, width, channels), dtype=np.uint8)

        for row_idx, row in enumerate(roi):
            for col_idx, (pixel1, pixel2) in enumerate(zip(row[::2], row[1::2])):

                if col_idx * 2 < width:
                    reconstructedImage[row_idx, col_idx * 2] = pixel1

                if col_idx * 2 + 1 < width:
                    reconstructedImage[row_idx, col_idx * 2 + 1] = pixel2

        reconstructedImages.append(reconstructedImage)

    print("%d images have been reconstructed!\nPlotting three of them.\n" % len(reconstructedImages))
    return reconstructedImages

# Computes the mean of a list of images
def meanComputation(imagesList):
    means = []

    for img in imagesList:
        total_sum = 0
        count = 0

        for row in img:
            for pixel in row:
                total_sum += pixel
                count += 1

        mean = total_sum / count
        means.append(mean)

    return means

# This function computes the oxygen saturation value
def oxygen_saturation(mean_signal_blue, mean_signal_red):
    
    # Standard Deviation of both red and blue channel's means
    std_blue = np.std(mean_signal_blue)
    std_red = np.std(mean_signal_red)

    # Spo2 - Beer/Lambert Law Formula
    rOr = (std_red/mean_signal_red)/(std_blue/mean_signal_blue)

    # Anemic Children
    spo2 = 101.6 - 5.834 * rOr
    meanSpo2 = np.mean(spo2)

    return (round(meanSpo2, 0))

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

# This function apply the chrome process on channels and return the heart and breath signals
def alpha_function(red, green, blue, fps):
    
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)

    xS = 3*red-2*green
    yS = 1.5*red+green-1.5*blue

    fs = fps/0.5
    ntaps = 512

    #Frequency cut for the breathing signal
    br_lowcut = 0.17
    br_highcut = 0.67

    # Frequency cut for the heart signal
    hr_lowcut = 0.6
    hr_highcut = 4.0

    #-------------------------------- HR -----------------------------------------------------------------
    hanning_hr_filter = bandpass_firwin(ntaps, hr_lowcut, hr_highcut, fs, window='hann')
    xF = scipy.signal.lfilter(hanning_hr_filter, 1, xS)
    yF = scipy.signal.lfilter(hanning_hr_filter, 1, yS)
    redF = scipy.signal.lfilter(hanning_hr_filter, 1, red)
    greenF = scipy.signal.lfilter(hanning_hr_filter, 1, green)
    blueF = scipy.signal.lfilter(hanning_hr_filter, 1, blue)
    alpha = np.std(xF)/np.std(yF)

    #signal_final=xF-(alpha*yF) raggruppamento
    signal_hr_final = 3*(1-alpha/2)*redF-2*(1+alpha/2)*greenF+3*(alpha/2)*blueF
    #-------------------------------------------------------------------------------------------------------

    #--------------------------------------BR---------------------------------------------------------------
    hanning_br_filter = bandpass_firwin(ntaps, br_lowcut, br_highcut, fs, window='hann')
    xF = scipy.signal.lfilter(hanning_br_filter, 1, xS)
    yF = scipy.signal.lfilter(hanning_br_filter, 1, yS)
    redF = scipy.signal.lfilter(hanning_br_filter, 1, red)
    greenF = scipy.signal.lfilter(hanning_br_filter, 1, green)
    blueF = scipy.signal.lfilter(hanning_br_filter, 1, blue)
    alpha = np.std(xF) / np.std(yF)

    # signal_final=xF-(alpha*yF) raggruppamento
    signal_br_final = 3 * (1 - alpha / 2) * redF - 2 * (1 + alpha / 2) * greenF + 3 * (alpha / 2) * blueF
    #-----------------------------------------------------------------------------------------------------

    return signal_hr_final, signal_br_final

# Computation of heart and breath rate given the signals computed with the alpha function
def high_peak(Fs, n, component, low, high):
    
    raw = np.fft.rfft(component)
    fft = np.abs(raw)

    freqs = float(Fs) / n * np.arange(n / 2 + 1)
    freqs = 60. * freqs
    idx = np.where((freqs > low) & (freqs < high))

    pruned = fft[idx]
    pfreq = freqs[idx]

    freqs = pfreq
    fft = pruned

    idx2 = np.argmax(fft)
    bpm = freqs[idx2]
    idx += (1,)

    return bpm