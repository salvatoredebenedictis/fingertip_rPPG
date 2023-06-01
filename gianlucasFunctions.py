
    def oxygen_saturation(self,signal_blue,signal_red,times,fps):
        """This function extract blood oxygen saturation value"""

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

def alpha_function(self,red,green,blue,fps):
        """This function apply the chrome process on channels and return the heart and breath signals"""

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