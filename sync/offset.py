'''
 Original author of this utility: Allison Deal
 Code adapted from her VideoSync project at:
 https://github.com/allisonnicoledeal/VideoSync
 
 Adapted by Greg Kramida for calib_video_opencv project.
 
 Copyright 2012-2014 Allison Deal
'''

import scipy.io.wavfile
import numpy as np
from subprocess import Popen, PIPE
import math
import re


# Extract audio from video file, save as wav auido file
# INPUT: Video file
# OUTPUT: Does not return any values, but saves audio as wav file
def extract_audio(folder, video_file):
    track_name = video_file.split(".")
    audio_output_name = track_name[0] + "WAV.wav"  # !! CHECK TO SEE IF FILE IS IN UPLOADS DIRECTORY
    audio_output_path = folder + audio_output_name
    #ffmpeg was accepted back into Debian, and libav is now a 2nd choice, so, what gives? -Greg
    #Now piping the output back to this process (to reduce verbosity and potentially be able to use
    #the output).-Greg
    #TODO: detect whether user has avconv or ffmpeg, and use the appropriate call
    args = ["ffmpeg", "-y", "-i", folder+video_file, "-vn", "-ac", "1", "-f", "wav", audio_output_path]
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    output, err = process.communicate()
    exit_code = process.wait()
    return exit_code, str(err), audio_output_path, str(output)

def parse_frame_rate(ffmpeg_output):
    '''
    Parse the frame rate (fps) that ffmpeg prints to stdout during video->audio conversion
    @type ffmpeg_output str
    @param ffmpeg_output string printed by ffmpeg process during conversion of input video to raw WAVE file
    @rtype float
    @return frame rate of input video
    '''
    return float(re.search('(\d\d\.\d\d)\sfps', ffmpeg_output).group(1))

# Read file
# INPUT: Audio file
# OUTPUT: Sets sample rate of wav file, Returns data read from wav file (numpy array of integers)
def read_audio(audio_file):
    #Return the sample rate (in samples/sec) and data from a WAV file
    rate, data = scipy.io.wavfile.read(audio_file)  #@UndefinedVariable 
    return data, rate


def make_horiz_bins(data, fft_bin_size, overlap, box_height):
    horiz_bins = {}
    # process first sample and set matrix height
    sample_data = data[0:fft_bin_size]  # get data for first sample
    if (len(sample_data) == fft_bin_size):  # if there are enough audio points left to create a full fft bin
        intensities = fourier(sample_data)  # intensities is list of fft results
        for i_intensity in range(len(intensities)):
            #have to force int, since int by float division yields float in Python3
            #original causes mem overflow due to too many bins -Greg
            box_y = int(i_intensity/box_height)
            if box_y in horiz_bins:
                horiz_bins[box_y].append((intensities[i_intensity], 0, i_intensity))  # (intensity, x, y)
            else:
                horiz_bins[box_y] = [(intensities[i_intensity], 0, i_intensity)]
                
    # process remainder of samples
    x_coord_counter = 1  # starting at second sample, with x index 1
    for i_bin in range(int(fft_bin_size - overlap), len(data), int(fft_bin_size-overlap)):
        sample_data = data[i_bin:i_bin + fft_bin_size]
        if (len(sample_data) == fft_bin_size):
            intensities = fourier(sample_data)
            for i_intensity in range(len(intensities)):
                #have to force int, since int by float division yields float in python 3
                #original causes mem overflow due to too many bins -Greg
                box_y = int(i_intensity/box_height) 
                if box_y in horiz_bins:
                    horiz_bins[box_y].append((intensities[i_intensity], x_coord_counter, i_intensity))  # (intensity, x, y)
                else:
                    horiz_bins[box_y] = [(intensities[i_intensity], x_coord_counter, i_intensity)]
        x_coord_counter += 1

    return horiz_bins


# Compute the one-dimensional discrete Fourier Transform
# INPUT: list with length of number of samples per second
# OUTPUT: list of real values len of num samples per second
def fourier(sample):  #, overlap):
    mag = []
    fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
    for i in range(int(len(fft_data)/2)): # Python 3 defaults to float division, convert to int -Greg 
        r = fft_data[i].real**2
        j = fft_data[i].imag**2
        mag.append(round(math.sqrt(r+j),2))

    return mag


def make_vert_bins(horiz_bins, box_width):
    boxes = {}
    for key in horiz_bins.keys():
        for i_bin in range(len(horiz_bins[key])):
            # Python 3 defaults to float division, convert to int -Greg 
            box_x = int(horiz_bins[key][i_bin][1] / box_width)
            if (box_x,key) in boxes:
                boxes[(box_x,key)].append((horiz_bins[key][i_bin]))
            else:
                boxes[(box_x,key)] = [(horiz_bins[key][i_bin])]

    return boxes


def find_bin_max(boxes, maxes_per_box):
    freqs_dict = {}
    for key in boxes.keys():
        max_intensities = [(1,2,3)]
        for i_box in range(len(boxes[key])):
            if boxes[key][i_box][0] > min(max_intensities)[0]:
                if len(max_intensities) < maxes_per_box:  # add if < number of points per box
                    max_intensities.append(boxes[key][i_box])
                else:  # else add new number and remove min
                    max_intensities.append(boxes[key][i_box])
                    max_intensities.remove(min(max_intensities))
        for i_intensity in range(len(max_intensities)):
            if max_intensities[i_intensity][2] in freqs_dict:
                freqs_dict[max_intensities[i_intensity][2]].append(max_intensities[i_intensity][1])
            else:
                freqs_dict[max_intensities[i_intensity][2]] = [max_intensities[i_intensity][1]]

    return freqs_dict


def find_freq_pairs(freqs_dict_orig, freqs_dict_sample):
    time_pairs = []
    for key in freqs_dict_sample.keys():  # iterate through freqs in sample
        if key in freqs_dict_orig:        # if same sample occurs in base
            for i_sample_freq in range(len(freqs_dict_sample[key])):  # determine time offset
                for i_orig_freq in range(len(freqs_dict_orig[key])):
                    time_pairs.append((freqs_dict_sample[key][i_sample_freq], freqs_dict_orig[key][i_orig_freq]))

    return time_pairs


def find_delay(time_pairs):
    t_diffs = {}
    for i_time_pair in range(len(time_pairs)):
        delta_t = time_pairs[i_time_pair][0] - time_pairs[i_time_pair][1]
        if delta_t in t_diffs:
            t_diffs[delta_t] += 1
        else:
            t_diffs[delta_t] = 1
    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
    print(t_diffs_sorted)
    time_delay = t_diffs_sorted[-1][0]

    return time_delay


# Find time offset between two video files
def find_time_offset(video_filenames, folder, audio_delays, fft_bin_size=512, overlap=0, box_height=512, box_width=43, samples_per_box=7):
    '''
    Find time offset between two video files and the frame rate 
    (requires for frame and bit rates of the two videos to be the same)
    @type video_filenames: list[str] 
    @param video_filenames: filenames of the two videos 
    @param folder: absolute or relative directory wherein the two videos are located
    @type audio_delays: list[float]
    @param audio_delays: delay between video and audio, in seconds, in the first and second video files respectively 
    @param fft_bin_size size of the FFT bins, i.e. segments of audio, in beats, for which a separate 
    peak is found
    @param overlap overlap between each bin
    @param box_height height of the boxes in frequency histograms
    @param box_width width of the boxes in frequency histograms
    @param samples_per_box # of frequency samples within each constellation
    @return time offset between the two videos and their frame rate, 
    in format ((offset1, offset2) frame_rate), where one of the offsets is zero, 
    the other is how much you need to skip in that video to get to the corresponding point in the other video. 
    @rtype tuple[float]
    '''
    orig_duration = 120
    sample_duration = 60
    
    # Process first file (orig)
    retcode, err_text, wavfile1, output  = extract_audio(folder, video_filenames[0])#@UnusedVariable
    if(retcode != 0):
        raise RuntimeError("ffmpeg error:\n{0:s}".format(err_text))    
    frame_rate1 = parse_frame_rate(err_text)#turns out ffmpeg prints its regular output to stderr (?)
    raw_audio1, rate1 = read_audio(wavfile1)
    bins_dict1 = make_horiz_bins(raw_audio1[:rate1*orig_duration], fft_bin_size, overlap, box_height) #bins, overlap, box height
    boxes1 = make_vert_bins(bins_dict1, box_width)  # box width
    ft_dict1 = find_bin_max(boxes1, samples_per_box)  # samples per box

    # Process second file (sample)
    retcode, err_text, wavfile2, output = extract_audio(folder, video_filenames[1])#@UnusedVariable
    if(retcode != 0):
        raise RuntimeError("ffmpeg error:\n{0:s}".format(err_text))
    raw_audio2, rate2 = read_audio(wavfile2)
    frame_rate2 = parse_frame_rate(err_text)
    #perform sanity checks
    if(rate1 != rate2):
        raise ValueError("The bitrates of two provided files do not match.")
    if(frame_rate1 != frame_rate2):
        raise ValueError("The framerates of two provided files do not match.")
    bins_dict2 = make_horiz_bins(raw_audio2[:rate2*sample_duration], fft_bin_size, overlap, box_height)
    boxes2 = make_vert_bins(bins_dict2, box_width)
    ft_dict2 = find_bin_max(boxes2, samples_per_box)
    


    # Determie time delay
    pairs = find_freq_pairs(ft_dict1, ft_dict2)
    delay = find_delay(pairs)
    samples_per_sec = rate1 / (fft_bin_size-(overlap/2))

    seconds = delay / samples_per_sec
    
    #(manually compute difference in delay between video and audio for both videos as diff1 and diff2,
    # correction = diff1 - diff2)
    correction = audio_delays[0] - audio_delays[1]
    seconds = round(seconds, 4)
 
    if seconds > 0:
        return ((0, seconds + correction), frame_rate1)
    else:
        return ((abs(seconds) + correction, 0), frame_rate1)






