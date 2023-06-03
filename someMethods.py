import numpy as np
import scipy
from sklearn import preprocessing
import matplotlib.pyplot as plt

def plotImagesTitles(img1, img2, img3, t1, t2, t3):

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(img1)
    ax[0].set_title(t1, fontsize = 10)

    ax[1].imshow(img2)
    ax[1].set_title(t2, fontsize = 10)

    ax[2].imshow(img3)
    ax[2].set_title(t3, fontsize = 10)

    plt.show()

def plotRGBchannels(redChannel, greenChannel, blueChannel):

    rx = range(len(redChannel))
    gx = range(len(greenChannel))
    bx = range(len(blueChannel))

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(12, 6))

    ax[0].plot(rx, redChannel, color = 'red')
    ax[0].set_ylim(min(redChannel)-10,max(redChannel)+10)
    ax[0].grid(True)

    ax[1].plot(gx, greenChannel, color = 'green')
    ax[1].set_ylim(min(greenChannel)-10,max(greenChannel)+10)
    ax[1].grid(True)

    ax[2].plot(bx, blueChannel, color = 'blue')
    ax[2].set_ylim(min(blueChannel)-10,max(blueChannel)+10)
    ax[2].grid(True)
    plt.show()

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
        height, width, channels = maxH, maxH, startingImages[0].shape[2]
        # height, width, channels = maxH, maxW, 3

        reconstructedImage = np.zeros((height, width, channels), dtype=np.uint8)

        for row_idx, row in enumerate(roi):
            for col_idx, (pixel1, pixel2) in enumerate(zip(row[::2], row[1::2])):

                if col_idx * 2 < width:
                    reconstructedImage[row_idx, col_idx * 2] = pixel1

                if col_idx * 2 + 1 < width:
                    reconstructedImage[row_idx, col_idx * 2 + 1] = pixel2

        reconstructedImages.append(reconstructedImage)

    print("\n%d images reconstructed!\n" % len(reconstructedImages))
    return reconstructedImages


# This function extract blood oxygen saturation value
def oxygen_saturation(self,signal_blue,signal_red,times,fps):
    
    #ratio of ratios
    #mean red/blue channel
    mean_blue = np.mean(signal_blue)
    mean_red = np.mean(signal_red)
    #standard deviation red/blue channel
    std_blue = np.std(signal_blue)
    std_red = np.std(signal_red)
    #formula Spo2 - Beer/Lambert Law
    rOr = (std_red/mean_red)/(std_blue/mean_blue)
    spo2 = 101.6 - 5.834 * rOr #bambini anemici

    return round(spo2,0)

# This function apply the chrome process on channels and return the heart and breath signals
def alpha_function(self,red,green,blue,fps):
    
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
    hanning_hr_filter = preprocessing.bandpass_firwin(ntaps, hr_lowcut, hr_highcut, fs, window='hanning')
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
    hanning_br_filter = preprocessing.bandpass_firwin(ntaps, br_lowcut, br_highcut, fs, window='hanning')
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