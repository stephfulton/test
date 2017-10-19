# 5/19/2016
# Contributors: Charles Horn, David Rosenberg, Bastien Maurice, Michael Sciullo
###########################################################

# write a gitignore file to ignore ".smr", ".s2r" and ".h5" files
###########################################################
try:
	# This will create a new file or **overwrite an existing file**.
	f = open(".gitignore", "w")
	try:
		f.write("*.smr \n*.s2r \n*.h5") # Write a string to a file
	finally:
		f.close()
except IOError:
	pass
# ==> run "git add -A" and then "git commit -m "created ignore file" at command line <==

# load modules
##########################################################
#import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import neo
import numpy as np
from OpenElectrophy.spikesorting import (generate_block_for_sorting, SpikeSorter)
import pandas as pd
from PIL import Image
import pip
import pylab
import quantities as pq
import scipy.signal as sg
import scipy.stats as st
from sklearn import mixture
import sys
import warnings
import webbrowser
# hide warnings when importing data (optional)
warnings.filterwarnings("ignore")
# use a plotting style in the notebook that is similar to ggplot2, see http://ggplot2.org/
plt.style.use("ggplot")
# which version of Python is installed?
print("Python version: {}\n\nPackages versions: ".format(sys.version))
# which package versions are installed?
all_packages = pip.get_installed_distributions()
used_packages = ["cv2", "math", "matplotlib", "neo", "numpy", "OpenElectrophy", "os", "pandas", "PIL", "pip", "pylab", "quantities", "scipy", "sklearn", "sys", "warnings", "webbrowser"]
for entry in used_packages:
	for p in all_packages:
		if entry in str(p):
			print(str(p))

# define global variables
##########################################################
sample_rate = 20000.0 # Sample rate (Hz)

# load the stomach templates for the heatmap
##########################################################
stomach_coord = np.genfromtxt("outline_heatmap.csv", delimiter = ' ')

# Use the "butter_bandpass" functions
##########################################################
"""
lowcut = low pass filter level in Hz
highcut = high pass filter level in Hz
sample_rate = sampling rate of the recording
order = order of the Butterworth filter, one by default
data = set of data to filter
"""
def butter_bandpass(lowcut, highcut, sample_rate, order = 1):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sg.butter(order, [low, high], btype = "band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order = 1):
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order = order)
    y = sg.lfilter(b, a, data)
    return y

# Use the "importSpike2" function
###########################################################
"""
filename = name of Spike2 file
newfilename = name of new HDF5 file
gain = amplification correction to convert to uV (this depends on your I/O setttings)
... ours is 50 = multiplication factor for 20K amp
........... 25 = multiplication factor for 10K amp
........... 12.5 = multipication factor for 5K amp
save_start = start time for saving data to file (sec)
save_end = end time for saving data to file (sec)
display_start = start time for displaying data in a plot (sec)
display_end = end time for displaying data in a plot (sec)
graph = True if you want to plot it, False if not
"""
def importSpike2(filename, newfilename, gain, save_start, save_end, display_start, display_end, graph):
	"Function that loads a .smr file and converts it to a .h5 file"
	global sample_rate
	# print todays date and time
	import time
	## dd/mm/yyyy format
	print (time.strftime("%d/%m/%Y"))
	## 12 hour format ##
	print (time.strftime("%I:%M:%S"))
	print ("\n====================")
	# import a Spike2 file (CED)
	r = neo.Spike2IO(filename)
	bl = r.read()[0]
	asig = bl.segments[0].analogsignals[0]
	# keep the signal as a 16 bit float
	asig = np.float16(asig)
	asig = asig * gain
	datatype = asig.dtype
	pts = round(float(np.prod(asig.shape)), 0)
	secs = round(pts/sample_rate, 2)
	mins = round(secs/60.0, 2)
	hrs = round(mins/60, 3)
	print("{}:\n====================".format(filename.split("/")[-1]))
	print("{} data points\n{} sec\n{} min\n{} hr\n20,000 Hz".format(pts, secs, mins, hrs))
	print (datatype)
	# plot data if wanted
	if graph: # only if graph == True
		fig, ax = plt.subplots(1, 1, figsize=(8, 3))
		start = display_start * sample_rate
		end = display_end * sample_rate
		asig_display = asig[start:end]
		time = np.arange(display_start, display_end, 1.0/sample_rate)
		ax.plot(time, asig_display, "b") # b = blue
		ax.set_xlabel("seconds", fontsize = 12)
		ax.set_ylabel("microvolts", fontsize = 12)
		plt.grid()
		plt.tight_layout()
		# save the figure as a SVG image
		plt.savefig("(4)-Publication_Graphics/sample_" + str(display_start)  + "-" + str(display_end) + ".svg", bbox_inches = "tight")
		plt.show()
	# save data to an HDF5 file type to share
	# first remove any previously created files from running this cell
	try:
		os.remove(newfilename)
	except OSError:
		pass
	# convert sec to pts using the sampling rate
	start1 = save_start * sample_rate
	end1 = save_end * sample_rate
	# create the new array
	new_asig = asig[start1:end1]
	# Write a new Block>Segment>AnalogSignal Hierarchy from the ground up
	# make a segment
	seg = neo.Segment()
	seg.name = 'New segment'
	seg.description = 'Sample'
	seg.index = 0
	seg.analogsignals = []
	seg.analogsignals.append(new_asig)
	# make a recording channel
	rec = neo.RecordingChannel()
	rec.analogsignals = []
	rec.analogsignals.append(new_asig)
	# make a recording channel group
	recg = neo.RecordingChannelGroup()
	recg.recordingchannels = []
	recg.recordingchannels.append(rec)
	recg.name = 'New group'
	# make a block ... finally!
	b = neo.Block()
	b.name = 'New block'
	b.segments = []
	b.segments.append(seg)
	b.recordingchannelgroups = []
	b.recordingchannelgroups.append(recg)
	# Write the block to a new HDF5 file:
	w = neo.io.NeoHdf5IO(newfilename)
	w.write_block(b)
	w.close()
	print ('\n====================')
	print ("wrote %s" % newfilename)
"""
Alternative way to create a simple HDF5, without the complicated hierarchy, but doesn't work with OpenElectrophy (might work with tridesclous)
	# write the new file
	h5f = h5py.File(newfilename, "w") 
	# write the dataset
	h5f.create_dataset("nerve", data = new_asig)
	h5f.close() # close the file
	return
"""

# Use the "chamber_images" function
##########################################################
"""
grid_off = name of the image of the chamber without the grid
grid_on = name of the image of the chamber with the grid on
"""
def chamber_images(grid_off, grid_on):
	"Function that loads the images of the chamber with and without the grid"
	img1 = Image.open(grid_off)
	img2 = Image.open(grid_on)
	#img2 = img2.rotate(-90) depending on the original image
	plt.figure(figsize=(15, 8))
	plt.subplot(121)
	plt.axis("off")
	plt.imshow(img1)
	plt.title("Preparation", fontsize = 20)
	plt.subplot(122)
	plt.imshow(img2)
	plt.title("Grid on", fontsize = 20)
	plt.axis("off")
	plt.show()
	return

# Use the "signal_stability" function
##########################################################
"""
filename = name of the .h5 file
filter_level = value for the Butterworth low pass filter
t_start = time of the first mechanical stimulation
t_stop = time of the last electrical stimulation
graph_start = beginning of the graph
graph_end = end of the graph
"""
def signal_stability(filename, filter_level, t_start, t_stop, graph_start, graph_end):
	"Function that prints the amplitude of the signal through the recording"
	global sample_rate
	# read the HDF5 file
	r = neo.io.NeoHdf5IO(filename)
	bl = r.read()[0]
	rcg = bl.recordingchannelgroups[0]
	spikesorter = SpikeSorter(rcg)
	# filter the data
	spikesorter.DerivativeFilter()
	spikesorter.ButterworthFilter(f_low = filter_level)
	# print the raw and filtered signals
	plt.figure(figsize = (18, 6))
	plt.subplot(121)
	plt.plot(np.arange(graph_start, graph_end, 1.0/sample_rate), spikesorter.full_band_sigs[0][0][graph_start * sample_rate : graph_end * sample_rate])
	plt.title("Original Signal", fontsize = 20)
	plt.xlabel("Time (sec)", fontsize = 12, color = "black")
	plt.ylabel("microvolts", fontsize = 12, color = "black")
	plt.xticks(fontsize = 12, color = "black")
	plt.yticks(fontsize = 12, color = "black")
	plt.subplot(122)
	plt.plot(np.arange(graph_start, graph_end, 1.0/sample_rate), spikesorter.filtered_sigs[0][0][graph_start * sample_rate : graph_end * sample_rate], color = "b") # blue
	plt.title("Filtered Signal", fontsize = 20)
	plt.xlabel("Time (sec)", fontsize = 12, color = "black")
	plt.ylabel("microvolts", fontsize = 12, color = "black")
	plt.xticks(fontsize = 12, color = "black")
	plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/signalfilter" + str(graph_start) + "-" + str(graph_end) + ".svg", bbox_inches = "tight") # save figure as a SVG image
	plt.show()
	# determine the evolution of the firing rate during the recording
	firing = [] # initialize
	while t_stop > t_start:
		y = spikesorter.filtered_sigs[0][0][t_start * sample_rate : (t_start + 10) * sample_rate] # consider bins of 10s
		indexes = peakutils.indexes(y, thres = 0.8, min_dist = 0.03 * sample_rate) # detect the spikes
		firing = firing + 10 * [len(indexes) / 10] # add the firing rate
		t_start = t_start + 10
	# plot the evolution of the firing rate
	xvalues = range(t_start, t_stop)
	plt.figure(figsize = (13, 5))
	plt.bar(xvalues, firing, 1, color = "darkblue") # x, y, width, color
	plt.title("Evolution of the firing rate", fontsize = 20)
	plt.xlabel("Time (sec)", fontsize = 12, color = "black")
	plt.ylabel("Firing rate (Hz)", fontsize = 12, color = "black")
	plt.xticks(fontsize = 12, color = "black")
	plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/firingrate_evolution" + str(t_start) + "-" + str(t_stop) + ".svg", bbox_inches = "tight") # save figure as a SVG image
	plt.show()
	return

''' -- V1 --
	# determine the amplitude of the signal during the recording
	amplitude = [] # initialize
	amplitudefilter = [] # initialize
	xvalues = range(t_start, t_stop)
	# calculate the signal amplitude for each 1s period
	while t_stop > t_start:
		piece = spikesorter.full_band_sigs[0][0][t_start * sample_rate : (t_start + 1) * sample_rate]
		piecefilter = spikesorter.filtered_sigs[0][0][t_start * sample_rate : (t_start + 1) * sample_rate]
		amplitude = amplitude + [max(piece) - min(piece)]
		amplitudefilter = amplitudefilter + [max(piecefilter) - min(piecefilter)]
		t_start = t_start + 1
	# plot it for the raw and filtered signal
	plt.figure(figsize = (18, 6))
	plt.subplot(121)
	plt.plot(xvalues, amplitude, "gd") # green diamond
	plt.title("Amplitude for the raw signal", fontsize = 20)
	plt.xlabel("Time (sec)", fontsize = 12, color = "black")
	plt.ylabel("microvolts", fontsize = 12, color = "black")
	plt.xticks(fontsize = 12, color = 'black')
	plt.yticks(fontsize = 12, color = 'black')
	plt.subplot(122)
	plt.plot(xvalues, amplitudefilter, "rd") # red diamond
	plt.title("Amplitude for the filtered signal", fontsize = 20)
	plt.xlabel("Time (sec)", fontsize = 12, color = "black")
	plt.ylabel('microvolts', fontsize = 12, color = "black")
	plt.xticks(fontsize = 12, color = "black")
	plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/signalamplitude" + str(t_start) + "-" + str(t_stop) + ".svg") # save figure as a SVG image
	plt.show()
	# determine the evolution of the firing rate during the recording
	firing = [] # initialize
	while t_stop > t_start2:
		spike = spike_detection(filename, filter_level, 6, t_start2, t_start2 + 10, False, False) # filename, filter level, threshold, start, end, graph
		firing = firing + 10 * [spike] # consider bins of 10s
		t_start2 = t_start2 + 10
	# plot the evolution of the firing rate
	xvalues2 = range(t_start2, t_stop)
	plt.figure(figsize = (13, 5))
	plt.bar(xvalues2, firing, 1, color = "darkblue") # x, y, width, color
	plt.title("Evolution of the firing rate", fontsize = 20)
	plt.xlabel("Time (sec)", fontsize = 12, color = "black")
	plt.ylabel("Firing rate (Hz)", fontsize = 12, color = "black")
	plt.xticks(fontsize = 12, color = "black")
	plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/firingrate_evolution" + str(t_start2) + "-" + str(t_stop) + ".svg") # save figure as a SVG image
	plt.show()
	return
'''

# Use the "estimate_length" function
##########################################################
"""
photo = name of the image of the chamber without the grid
csvfile = csv file that contains the coordinates of the region points
scalevalue = measure of the scale in mm

On ImageJ, open the image, use the multi-point tool to point the 10 regions, the top of the stomach (11), the recording electrode (12) and then 2 points for the scale (13 & 14).
File > Save as > XY Coordinates to build a text file with the XY values
Plugins > Utilities > Capture Image to save the image with the points on it
"""
def estimate_length(photo, csvfile, scalevalue):
	"Function that estimates the total vagus length (in mm) between the region stimulated and the recording electrode"
	# show the image with the different regions
	plt.rcParams["figure.figsize"] = (11.0, 7.0) # Change it to the dimensions you want, here : 11 x 7 inches
	img = matplotlib.image.imread(photo)
	plt.title("Estimate the vagus length", fontsize = 20)
	plt.axis("off") # don't show the axis
	plt.imshow(img)
	# import the CSV file of the coordinates
	coord = pd.read_csv(csvfile)
	# set the scale
	sx1 = coord.values[12][1] # x(start of the scale)
	sx2 = coord.values[13][1] # x(end of the scale)
	sy1 = coord.values[12][2] # y(start of the scale)
	sy2 = coord.values[13][2] # y(end of the scale)
	scale = np.sqrt((sx2 - sx1)**2 + (sy2 - sy1)**2) # represents scalevalue mm
	# calculate distance between top of the stomach and rec electrode (px)
	rx1 = coord.values[10][1] # x(top of the stomach)
	rx2 = coord.values[11][1] # x(rec electrode)
	ry1 = coord.values[10][2] # y(top of the stomach)
	ry2 = coord.values[11][2] # y(rec electrode)
	stomrec = np.sqrt((rx2 - rx1)**2 + (ry2 - ry1)**2)
	# initialize the list that will keep all the vagus lengths (mm)
	vagus_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stomrec / scale * scalevalue, 0, "/", "/"]
	# calculte the vagus lengths
	for i in range(10):
		calcx1 = coord.values[i][1] # x(i)
		calcy1 = coord.values[i][2] # y(i)
		dist = (np.sqrt((calcx1 - rx1)**2 + (calcy1 - ry1)**2) + stomrec) / scale * scalevalue
		vagus_length[i] = dist
	# add a column to display the vagus length
	coord["vagus length (mm)"] = vagus_length
	return coord, vagus_length 

# Use the "load_CAP" function
###########################################################
"""
filename = name of the .h5 file
name = description of the experiment (region stimulated, current used for the stimulation...)
stim_start = beginning of the stimulation (s)
stim_end = end of the stimulation
vagus = length of the vagus (mm)
time_cutoff = time after stimulation when the first unit appears (ms)
data_compression = value of the compression for the median filter, 1 by default which means no compression
lowcut = low pass filter level, 0 by default
highcut = high pass filter level, 10000 by default
butter_order = order of the Butterworth filter, 0 by default which means no Butterworth filter
"""
def load_CAP(filename, name, stim_start, stim_end, vagus, time_cutoff, data_compression = 1, lowcut = 0, highcut = 10000, butter_order = 0):
	"Function that generates the compound action potentials"
	global sample_rate
	plt.rcParams.update({'xtick.labelsize': 15})
	plt.rcParams.update({'ytick.labelsize': 15})
	# print stim times and convert to a numpy array
	stims = stim_on.loc[(stim_on["Time"] > stim_start) & (stim_on["Time"] < stim_end) & (stim_on["Note"] == "stim-on")]
	print(stims)
	stims1 = np.array(stims["Time"])
	# Select timestamp for each pulse and select 10 CAPs | consider 10000ms before and 150ms (3000 points) after the pulse 
	s1 = stims1[0]; s2 = stims1[1]; s3 = stims1[2]; s4 = stims1[3]; s5 = stims1[4]; s6 = stims1[5]; s7 = stims1[6]; s8 = stims1[7]; s9 = stims1[8]; s10 = stims1[9]
	start1 = np.rint((s1 * sample_rate)); end1 = np.rint((start1 + 0.15 * sample_rate)); start2 = np.rint((s2 * sample_rate)); end2 = np.rint((start2 + 0.15 * sample_rate)); start3 = np.rint((s3 * sample_rate)); end3 = np.rint((start3 + 0.15 * sample_rate)); start4 = np.rint((s4 * sample_rate)); end4 = np.rint((start4 + 0.15 * sample_rate)); start5 = np.rint((s5 * sample_rate)); end5 = np.rint((start5 + 0.15 * sample_rate)); start6 = np.rint((s6 * sample_rate)); end6 = np.rint((start6 + 0.15 * sample_rate)); start7 = np.rint((s7 * sample_rate)); end7 = np.rint((start7 + 0.15 * sample_rate)); start8 = np.rint((s8 * sample_rate)); end8 = np.rint((start8 + 0.15 * sample_rate)); start9 = np.rint((s9 * sample_rate)); end9 = np.rint((start9 + 0.15 * sample_rate)); start10 = np.rint((s10 * sample_rate)); end10 = np.rint((start10 + 0.15 * sample_rate))
	# read the HDF5 file
	r = neo.io.NeoHdf5IO(filename) # read it thanks to neo
	bl = r.read()[0] # read the block
	r.close()
	seg = bl.segments[0] # read the segment
	raw_asig = np.array(seg.analogsignals[0]) # read the analogsignal
	med_asig = raw_asig; asig = raw_asig # initialize and then only take the part of interest to filter (so that we don't process the entire file for each region)
	# plot raw and filtered signal side by side for the first CAP
	time = np.arange(-10, 150, 1000.0/sample_rate) # make a time variable in ms
	wholetime = np.arange(-10000, 150, 1000.0/sample_rate)
	fig = plt.figure(figsize=(18, 6))
	sub1 = fig.add_subplot(131)
	sub1.set_title("Raw signal - CAP 1", fontsize = 20)
	sub1.set_ylabel("Amplitude (mV)", fontsize = 18)
	sub1.set_xlabel("Time (ms)", fontsize = 18)
	sub1.plot(time, raw_asig[start1-0.01*sample_rate:end1])
	sub1.plot([time_cutoff, time_cutoff], [-250, 250], "--b") # plot the cutoff
	sub1.set_ylim(-250, 250)
	# filter the signal with a median filter
	med_asig[start1-10*sample_rate:end10] = sg.medfilt(volume = raw_asig[start1-10*sample_rate:end10], kernel_size = data_compression)
	sub2 = fig.add_subplot(132)
	sub2.set_title("Median filtered signal - CAP 1", fontsize = 20)
	sub2.set_ylabel("Amplitude (mV)", fontsize = 18)
	sub2.set_xlabel("Time (ms)", fontsize = 18)
	sub2.plot(time, med_asig[start1-0.01*sample_rate:end1])
	sub2.plot([time_cutoff, time_cutoff], [-250, 250], "--b") # plot the cutoff
	sub2.set_ylim(-250, 250)
	# filter the signal with a Butterworth filter
	asig[start1-10*sample_rate:end10] = butter_bandpass_filter(data = med_asig[start1-10*sample_rate:end10], lowcut = lowcut, highcut = highcut, sample_rate = sample_rate, order = butter_order)
	# Select the 10 CAPs in the filtered signal
	T1 = asig[start1-10*sample_rate:end1]; T2 = asig[start2-10*sample_rate:end2]; T3 = asig[start3-10*sample_rate:end3]; T4 = asig[start4-10*sample_rate:end4]; T5 = asig[start5-10*sample_rate:end5]; T6 = asig[start6-10*sample_rate:end6]; T7 = asig[start7-10*sample_rate:end7]; T8 = asig[start8-10*sample_rate:end8]; T9 = asig[start9-10*sample_rate:end9]; T10 = asig[start10-10*sample_rate:end10]
	# plot raw and filtered signal side by side for the first CAP
	sub3 = fig.add_subplot(133)
	sub3.set_title("Butterworth filtered signal - CAP 1", fontsize = 20)
	sub3.set_ylabel("Amplitude (mV)", fontsize = 18)
	sub3.set_xlabel("Time (ms)", fontsize = 18)
	sub3.plot(time, asig[start1-0.01*sample_rate:end1])
	sub3.plot([time_cutoff, time_cutoff], [-250, 250], "--b") # plot the cutoff
	sub3.set_ylim(-250, 250)
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_filters.svg", bbox_inches = "tight")
	# Plot each individual CAP
	fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(nrows = 10, sharex = True, sharey = True, figsize = (14, 6))
	ax1.plot(time, T1[9.99*sample_rate::])
	ax1.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax1.set_title("10 Compound action potentials (CAPs)", fontsize = 20)
	ax1.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax2.plot(time, T2[9.99*sample_rate::])
	ax2.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax2.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax3.plot(time, T3[9.99*sample_rate::])
	ax3.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax3.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax4.plot(time, T4[9.99*sample_rate::])
	ax4.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax4.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax5.plot(time, T5[9.99*sample_rate::])
	ax5.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax5.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax6.plot(time, T6[9.99*sample_rate::])
	ax6.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax6.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax7.plot(time, T7[9.99*sample_rate::])
	ax7.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax7.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax8.plot(time, T8[9.99*sample_rate::])
	ax8.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax8.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax9.plot(time, T9[9.99*sample_rate::])
	ax9.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax9.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax10.plot(time, T10[9.99*sample_rate::])
	ax10.plot([time_cutoff, time_cutoff], [-100, 100], "--b") # plot the cutoff
	ax10.tick_params(axis = "both", which = "both", bottom = "on", top = "off", right = "off" )
	ax10.set_xlabel("Time (ms)", fontsize = 18)
	fig.text(0.04, 0.5, "Amplitude (mV)", ha = "center", fontsize = 18, rotation = "vertical")
	ax10.set_yticks(range(-100, 110, 100))
	# save the figure as a SVG file
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_individual_CAP.svg", bbox_inches = "tight")
	plt.show()
	# compute the average CAP
	ave1 = (T1+T2+T3+T4+T5+T6+T7+T8+T9+T10) / 10.0
	# plot the average CAP
	fig = plt.figure(figsize=(18, 6))
	sub1 = fig.add_subplot(121)
	sub1.set_title("Average CAP", fontsize = 25)
	sub1.set_ylabel("Amplitude (mV)", fontsize = 25)
	sub1.set_xlabel("Time (ms)", fontsize = 25)
	sub1.plot(time, ave1[9.99*sample_rate::])
	sub1.plot([time_cutoff, time_cutoff], [-150, 250], "--b") # plot the cutoff
	plt.ylim(-150, 250) # focus on the zone of interest
	# compute rectification
	ave_rect1 = np.absolute(ave1)
	# plot the rectified CAP
	sub2 = fig.add_subplot(122)
	sub2.set_title("Average and Rectified CAP", fontsize = 25)
	sub2.set_ylabel("Amplitude (mV)", fontsize = 25)
	sub2.set_xlabel("Time (ms)", fontsize = 25)
	sub2.plot(time, ave_rect1[9.99*sample_rate::])
	sub2.plot([time_cutoff, time_cutoff], [-0.5, 250], "--b") # plot the cutoff
	plt.ylim(-0.5, 250) # focus on the zone of interest
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_average_CAP.svg", bbox_inches = "tight")
	# export data into a CSV file
	FINAL1 = pd.DataFrame({"rect":ave_rect1})
	FINAL1.to_csv("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_analog.csv")
	# compute the velocity
	time2 = np.arange(0, 150, 1000/sample_rate)
	velo = vagus / time2 # m/s
	# plot the average CAP according to the velocity
	fig = plt.figure(figsize=(18, 6))
	sub1 = fig.add_subplot(121)
	sub1.set_title("Average CAP", fontsize = 25)
	sub1.set_ylabel("Amplitude (mV)", fontsize = 25)
	sub1.set_xlabel("Velocity (m/s)", fontsize = 25)
	plt.xscale("log") # set a logarithmic scale
	plt.grid(True, which = "both") # display the grid
	sub1.plot(velo, ave1[10*sample_rate::]) # consider the zone between 0 ms and 150 ms
	sub1.plot([vagus / time_cutoff, vagus / time_cutoff], [min(ave1[10*sample_rate::]), max(ave1[10*sample_rate::])], "--b") # plot the cutoff
	plt.xlim(30, velo[len(velo) - 1])  # reverse the x axis, go from 30 m/s (max value) to the min value which is the last
	# plot the rectified CAP according to the velocity
	sub2 = fig.add_subplot(122)
	sub2.set_title("Average and Rectified CAP", fontsize = 25)
	sub2.set_ylabel("Amplitude (mV)", fontsize = 25)
	sub2.set_xlabel("Velocity (m/s)", fontsize = 25)
	plt.xscale("log") # set a logarithmic scale
	plt.grid(True, which = "both") # display the grid
	sub2.plot(velo, ave_rect1[10*sample_rate::]) # consider the zone between 0 ms and 150 ms
	sub2.plot([vagus / time_cutoff, vagus / time_cutoff], [0, max(ave_rect1[10*sample_rate::])], "--b") # plot the cutoff
	plt.xlim(30, velo[len(velo) - 1])  # reverse the x axis, go from 30 m/s (max value) to the min value which is the last
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_averaged_CAP_velo.svg", bbox_inches = "tight")
	# Compute the area under the curve using the composite trapezoidal rule, see http://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html
	AUC = np.trapz(y = ave_rect1[(10000+time_cutoff)/1000.0 * sample_rate : 10150/1000.0 * sample_rate], x = wholetime[(10000+time_cutoff)/1000.0 * sample_rate : 10150/1000.0 * sample_rate]) / (150.0 - time_cutoff) # don't take the 10 ms before the stim and the first ms with the artefact
	# Consider the baseline activity during the 10000 ms before the stim, and express the AUC according to it
	correction = np.trapz(y = ave_rect1[0 : 10000/1000.0 * sample_rate], x = wholetime[0 : 10000/1000.0 * sample_rate]) / 10000
	AUC2 = AUC / correction * 100 # percentage of baseline activity
	'''
	# determine latency in ms for each velocity starting with 0.25 m/s or 0.25 mm/ms, e.g. 0.3 = 40 mm / 133.3 ms ... which is 133.3 ms = 40 / 0.3
	ms40 = vagus / 0.3 #0.30 m/s
	ms39 = ms40 * (1-1/39) #0.31 m/s
	ms38 = ms40 * (1-1/38) #0.32 m/s
	ms37 = ms40 * (1-1/37) #0.32 m/s
	ms36 = ms40 * (1-1/36) #0.33 m/s
	ms35 = ms40 * (1-1/35) #0.34 m/s
	ms34 = ms40 * (1-1/34) #0.35 m/s
	ms33 = ms40 * (1-1/33) #0.36 m/s
	ms32 = ms40 * (1-1/32) #0.38 m/s
	ms31 = ms40 * (1-1/31) #0.39 m/s
	ms30 = ms40 * (1-1/30) #0.40 m/s
	ms29 = ms40 * (1-1/29) #0.41 m/s
	ms28 = ms40 * (1-1/28) #0.43 m/s
	ms27 = ms40 * (1-1/27) #0.44 m/s
	ms26 = ms40 * (1-1/26) #0.46 m/s
	ms25 = ms40 * (1-1/25) #0.48 m/s
	ms24 = ms40 * (1-1/24) #0.50 m/s
	ms23 = ms40 * (1-1/23) #0.52 m/s
	ms22 = ms40 * (1-1/22) #0.55 m/s
	ms21 = ms40 * (1-1/21) #0.57 m/s
	ms20 = ms40 * (1-1/20) #0.60 m/s
	ms19 = ms40 * (1-1/19) #0.63 m/s
	ms18 = ms40 * (1-1/18) #0.67 m/s
	ms17 = ms40 * (1-1/17) #0.71 m/s
	ms16 = ms40 * (1-1/16) #0.75 m/s
	ms15 = ms40 * (1-1/15) #0.80 m/s
	ms14 = ms40 * (1-1/14) #0.86 m/s
	ms13 = ms40 * (1-1/13) #0.92 m/s
	ms12 = ms40 * (1-1/12) #1.00 m/s
	ms11 = ms40 * (1-1/11) #1.09 m/s
	ms10 = ms40 * (1-1/10) #1.20 m/s
	ms9 = ms40 * (1-1/9) #1.33 m/s
	ms8 = ms40 * (1-1/8) #1.50 m/s
	ms7 = ms40 * (1-1/7) #1.71 m/s
	ms6 = ms40 * (1-1/6) #2.00 m/s
	ms5 = ms40 * (1-1/5) #2.40 m/s
	ms4 = ms40 * (1-1/4) #3.00 m/s
	ms3 = ms40 * (1-1/3) #4.00 m/s
	ms2 = ms40 * (1-1/2) #6.00 m/s
	ms1 = ms40 * (1-1/1) #12.00 m/s
	# make an array for the latencies
	Lat = np.around([ms1,ms2,ms3,ms4,ms5,ms6,ms7,ms8,ms9,ms10,ms11,ms12,ms13,ms14,
		ms15,ms16,ms17,ms18,ms19,ms20,ms21,ms22,ms23,ms24,ms25,ms26,ms27,ms28,
		ms29,ms30,ms31,ms32,ms33,ms34,ms35,ms36,ms37,ms38,ms39,ms40], decimals=2)
	# make an array for the velocities
	# divide averaged rectified array into 40 parts
	size = int((ms40 / 40) * 40) # multiply by 40 because there are 20 pts in every ms ... and convert to an integer because it gives a warning
	# set up some parameters for the plot
	bins = 40
	index = np.arange(bins)
	bar_width = 1
	opacity = 0.4
	ticks = ("12.00","6.00","4.00","3.00","2.40","2.00","1.71","1.50","1.33","1.20",
		"1.09","1.00","0.92","0.86","0.80","0.75","0.71","0.67","0.63","0.60","0.57",
		"0.55","0.52","0.50","0.48","0.46","0.44","0.43","0.41","0.40","0.39","0.38",
		"0.36","0.35","0.34","0.33","0.32","0.32","0.31","0.30")
	# select the 40 parts from each averaged and rectified signal
	# essentially this is just a conversion of a numpy array from 1D to 2D
	ave_rect1bin = np.reshape(ave_rect1, (bins,75))
	# compute the areas for each of the 20 parts using the composite trapezoidal rule. http://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html
	areas1 = np.trapz(ave_rect1bin, dx = 1)
	# plot the areas
	plotname4 = name + "-area.svg"
	sub3 = fig.add_subplot(133)
	sub3.bar(index, areas1, bar_width, alpha = opacity, color = "black")
	sub3.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	sub3.set_xticks(range(1, 40))
	sub3.set_xticklabels(ticks, rotation = "vertical")
	sub3.set_title("Area under the curve (AUC)", fontsize = 20)
	sub3.set_ylabel("AUC", fontsize = 18)
	sub3.set_xlabel("Conduction Velocity (m/s)", fontsize = 18)
	plt.show()
	# export AUC data into a CSV file
	FINAL2 = pd.DataFrame({"area": areas1})
	name2 = name + "_AUC.csv"
	FINAL2.to_csv("(4)-Publication_Graphics/Compound_Action_Potentials/" + name2)
	# save the previous figures individually
	fig, (ax1) = plt.subplots(nrows = 1, sharex = True, sharey = True, figsize = (8, 8))
	ax1.set_title("Average CAP", fontsize = 20)
	ax1.plot(time, ave1)
	ax1.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax1.set_yticks(np.arange(-100, 120, 100))
	ax1.set_xlabel("Time (ms)", fontsize = 18)
	fig.text(0.04, 0.5, "microvolts", ha = "center", fontsize = 18, rotation = "vertical")
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_average.svg", bbox_inches = "tight") # save the figure as a SVG file
	plt.close() # but don't show it
	fig, (ax1) = plt.subplots(nrows = 1,sharex = True,sharey = True,figsize = (8, 8))
	ax1.set_title("Averaged and Rectified CAP", fontsize = 20)
	ax1.plot(time, ave_rect1)
	ax1.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax1.set_xlabel("Time (ms)", fontsize = 18)
	fig.text(0.04, 0.5, "microvolts", ha = "center", fontsize = 18, rotation = "vertical")
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_rectified.svg", bbox_inches = "tight") # save the figure as a SVG file
	plt.close() # but don't show it
	fig, (ax1) = plt.subplots(nrows = 1, sharex = True, sharey = True, figsize = (8, 8))
	ax1.set_title("Average CAP", fontsize = 20)
	ax1.set_ylabel("microvolts", fontsize = 18)
	ax1.set_xlabel("Velocity (m/s)", fontsize = 18)
	plt.xscale("log") # set a logarithmic scale
	plt.grid(True, which = "both") # display the grid
	ax1.plot(velo, ave1[200::]) # consider the zone between 0 ms and 150 ms
	plt.xlim(30, velo[len(velo) - 1])  # reverse the x axis, go from 30 m/s (max value) to the min value which is the last
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_average_velocity.svg", bbox_inches = "tight") # save the figure as a SVG file
	plt.close() # but don't show it
	fig, (ax1) = plt.subplots(nrows = 1, sharex = True, sharey = True, figsize = (8, 8))
	ax1.set_title("Average and Rectified CAP", fontsize = 20)
	ax1.set_ylabel("microvolts", fontsize = 18)
	ax1.set_xlabel("Velocity (m/s)", fontsize = 18)
	plt.xscale("log") # set a logarithmic scale
	plt.grid(True, which = "both") # display the grid
	ax1.plot(velo, ave_rect1[200::]) # consider the zone between 0 ms and 150 ms
	plt.xlim(30, velo[len(velo) - 1])  # reverse the x axis, go from 30 m/s (max value) to the min value which is the last
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + name + "_rectified_velocity.svg", bbox_inches = "tight") # save the figure as a SVG file
	plt.close() # but don't show it
	fig, (ax1) = plt.subplots(nrows = 1, sharex = True, sharey = True, figsize = (8, 8))
	ax1.bar(index, areas1, bar_width, alpha = opacity, color = "black")
	ax1.tick_params(axis = "both", which = "both", bottom = "off", top = "off", right = "off")
	ax1.set_xticks(range(1, 40))
	ax1.set_xticklabels(ticks, rotation = "vertical")
	fig.text(0.5, 0.04, "Conduction Velocity (m/s)", ha = "center", fontsize = 18)
	fig.text(0.04, 0.5, "AUC", ha = "center", fontsize = 18, rotation = "vertical")
	ax1.set_title("Area under the curve (AUC)", fontsize=20)
	plt.savefig("(4)-Publication_Graphics/Compound_Action_Potentials/" + plotname4) # save the figure as a SVG file
	plt.close() # but don't show it
	'''
	return round(AUC2, 3)

# Use the "heatmap" function
###########################################################
"""
name = description of the heatmap
intensity = list of the response's intensity for each region
surface = mucosal or serosal
"""
def heatmap(name, intensity, surface):
	"Function that represents the intensity of the response for each region according to a heatmap"
	# plot a 3 x 4 grid with the appropriate regions
	if surface == "mucosal":
		x = [0, 0.798, 2.1, 3]
		y = [0, 1, 2, 3, 4]
		xstomach = stomach_coord[:, 0] # extract the x coordinates
		ystomach = stomach_coord[:, 1] # extract the y coordinates
	if surface == "serosal":
		x = [0, 0.76, 2, 3]
		y = [0, 1, 2, 3, 4]
		xstomach = stomach_coord[:, 2] # extract the x coordinates
		ystomach = stomach_coord[:, 3] # extract the y coordinates
	# convert the intensity list under the necessary form for the heatmap
	intens = [[8, 6, 10], [2, 5, 9], [1, 4, 8], [0, 3, 7],]
	intens[0][1] = intensity[5] # region 6
	intens[0][2] = intensity[9] # region 10
	intens[1][0] = intensity[1] # region 2
	intens[1][1] = intensity[4] # region 5
	intens[1][2] = intensity[8] # region 9
	intens[2][0] = intensity[0] # region 1
	intens[2][1] = intensity[3] # region 4
	intens[2][2] = intensity[7] # region 8
	intens[3][1] = intensity[2] # region 3
	intens[3][2] = intensity[6] # region 7
	# plot the heatmap
	plt.rcParams["figure.figsize"] = (7, 7) # Dimension of the figure, here 7 x 7 inches
	plt.title(name, fontsize = 18)
	plt.axis("off")
	plt.pcolormesh(x, y, intens)
	plt.colorbar() # display a colorbar to show the intensity scale
	# plot the stomach outline
	plt.plot(xstomach, ystomach, "k.") # k. for black points
	plt.savefig("(4)-Publication_Graphics/heatmap_" + name + surface + ".png", bbox_inches = "tight") # save figure as a SVG image
	plt.show() # show the graph
	return

# Use the "confidence" function
###########################################################
"""
The confidence interval is generated assuming that C has a Poisson distribution, as suggested by the following article : 
Abeles M. Quantification, smoothing, and confidence limits for single-units histograms. Journal of Neuroscience Methods. 5(4):317-25, 1982.

filename = name of the .h5 file
relative_threshold = How many times * deviation to detect a spike
stim_start = beginning of the stimulation
stim_end = end of the stimulation
alpha = statistical risk
filter_level = value for the Butterworth low pass filter, no filter by default
"""
def confidence(filename, relative_threshold, stim_start, stim_end, alpha, filter_level = 0):
	"Function that gives the confidence interval for the amount of spikes that you should see"
	# 10s period before the stim
	F = float(spike_detection(filename, relative_threshold, start = (stim_start - 10), end = stim_start, showstats = True, graph = False, filter_level = 0)) # returns the spike frequency for the 10s period before the stimulation
	C = F * (stim_end - stim_start + 0.5) # number of spikes expected in the (period of stimulation + 0.5s) if there is no effect
	# generate the confidence interval for the Poisson distribution using the with the chi-squared distribution in this period
	if C == 0: 
		i1 = 0.0
	i1 = float(st.chi2.ppf(alpha/2, 2*C) / 2) # lower bound
	i2 = float(st.chi2.ppf(1 - alpha/2, 2*C + 2) / 2) # higher bound
	# 2s after the end of the stimulation	
	F2 = float(spike_detection(filename, relative_threshold, start = stim_start, end = stim_end + 0.5, showstats = True, graph = False, filter_level = 0)) # spike frequency in the stim period
	compare = F2 * (stim_end - stim_start + 0.5) # value to compare to the confidence interval
	if compare > i2 or compare < i1:
		statdiff = True
	else:
		statdiff = False
	print ("Interval of confidence = [" + str(i1) + " : " + str(i2) + "]" + "\n" + "Value for the sample = " + str(compare) + "\n" +"***** Statistical difference for " + str(alpha * 100) + " % = " + str(statdiff) + " *****")
	# compute the percentage of baseline spike frequency
	if F == 0: # to avoid division by zero | no spike in the baseline activity
		coef = 100.000
	else:
		coef = F2 / F * 100 
	return round(coef, 3)

# Use the "spike_detection" function
###########################################################
"""
filename = name of the .h5 file
relative_threshold = How many times * deviation to detect a spike
start = beginning of the stimulation
end = end of the stimulation
showstats = do you want to show the summary statistics
graph = do you want to plot the spikes on a figure, hidden by default
filter_level = value for the Butterworth low pass filter, no filter by default
"""
def spike_detection(filename, relative_threshold, start, end, showstats, graph = False, filter_level = 0):
	"Function that detects and displays spikes based on a specific threshold"
	global duration, frame, rcg, sample_rate, spikesorter, threshold
	# read the HDF5 file
	r = neo.io.NeoHdf5IO(filename) # read it thanks to neo
	bl = r.read()[0] # read the block
	r.close()
	seg = bl.segments[0] # read the segment
	asig = seg.analogsignals[0] # read the analogsignal
	sample = asig[start * sample_rate : end * sample_rate] # select the piece of interest from the recording
	seg.analogsignals[0] = sample # only consider the piece of interest in the HDF5 (in the segment)
	bl.recordingchannelgroups[0].recordingchannels[0].analogsignals[0] = sample # only consider the piece of interest in the HDF5 (in the recordingchannel)
	rcg = bl.recordingchannelgroups[0] # take the recording channel
	# sort the spikes
	spikesorter = SpikeSorter(rcg)
	# filter the signal
	spikesorter.ButterworthFilter(f_low = filter_level)
	# detect the spikes
	spikesorter.RelativeThresholdDetection(sign = "-", relative_thresh = relative_threshold) # Spike detection: negative relative threshold (Median absolute deviation of filtered signal * thresh)
	spikesorter.AlignWaveformOnPeak(sign = "-", left_sweep = 1.5 * pq.ms, right_sweep = 1.5 * pq.ms) # Align detected waveforms on peak, with 1.5 ms before and 1.5 ms after detection
	# Generate summary statistics for detected spikes and print it if wanted
	threshold = spikesorter.detection_thresholds[0][0]
	spikes = spikesorter.nb_spikes
	wavelength = float(spikesorter.left_sweep + spikesorter.right_sweep + 1)/sample_rate * 1000
	duration = np.array(spikesorter.seg_t_stop - spikesorter.seg_t_start)
	spike_freq = spikes / duration
	frame = float(spikesorter.left_sweep + spikesorter.right_sweep + 1) # the number of points used in the spike length
	if showstats:
		print("\nSummary Statistics:\n"+ "=" * 20)
		print("Recording duration: {} sec\nThreshold: {}\nNumber of Spikes: {} spikes\nSpike Length: {} ms\nMulti-unit Freq: {} Hz\n".format(float(duration), threshold, spikes, wavelength, float(spike_freq)))
	# plot the spikes if wanted
	if graph:
		plt.figure(figsize= (10,6));
		plt.title("Detected Spikes", fontsize = 20);
		step = int(math.floor(spikes / duration)) # used to randomly select a spike from each 1 sec bin to plot
		for i in range(0, duration):
			plt.plot(np.arange(0.0, frame) / 20, spikesorter.spike_waveforms[i * step].T, color = "darkred");
		plt.xlabel("Time (msec)", fontsize = 16, color = "black");
		plt.ylabel("microvolts", fontsize = 16, color = "black")
		plt.xticks(fontsize = 12, color = "black"); plt.yticks(fontsize = 12, color= "black")
		plt.savefig("(4)-Publication_Graphics/Spike_Sorting/aligned_waveforms_" + str(start) + "-" + str(end) +".svg", bbox_inches = "tight") # save figure as a SVG image
	return spike_freq

# Use the "detect_peaks" function from Marcos Duarte
###########################################################
"""
x : 1D array_like data
mph : {None, number}, optional (default = None) detect peaks that are greater than minimum peak height
mpd : positive integer, optional (default = 1) detect peaks that are at least separated by minimum peak distance (in number of data)
threshold : positive number, optional (default = 0) detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors
edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising') for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), 
both edges ('both'), or don't detect a flat peak (None)
kpsh : bool, optional (default = False) keep peaks with same height even if they are closer than `mpd`
valley : bool, optional (default = False) if True (1), detect valleys (local minima) instead of peaks
show : bool, optional (default = False) if True (1), plot data in matplotlib figure
ax : a matplotlib.axes.Axes instance, optional (default = None)
"""
def detect_peaks(x, mph = None, mpd = 1, threshold = 0, edge = "rising", kpsh = False, valley = False, show = False, ax = None):
    "Detect peaks in data based on their amplitude and other features"
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        plt.show()

# Use the "find_threshold" function
###########################################################
"""
filename = name of the .h5 file
start = beginning of the sample (s)
end = end of the sample (s)
relative_thresh = 
mph = minimal peak height in terms of amplitude, None by default
mpd = minimal peak distance, i.e the minimum distance between two peaks, None by default
valley_nb = number of the valley that you want to take as threshold, takes the first by default
"""
def find_threshold(filename, start, end, relative_thresh, mph = None, mpd = None, valley_nb = 0):
	"Function that determines a threshold to detect spikes from noise and artifacts based on the peak amplitude distribution"
	global sample_rate
	plt.rcParams.update({'xtick.labelsize': 20})
	plt.rcParams.update({'ytick.labelsize': 20})
	# load raw data and prepare the sample of interest
	r = neo.io.NeoHdf5IO(filename) # read the HDF5 version of the recording
	bl = r.read()[0] # read the block
	r.close()
	seg = bl.segments[0] # read the segment
	asig = seg.analogsignals[0] # read the analogsignal
	sample = asig[start * sample_rate : end * sample_rate] # select the piece of interest from the recording
	seg.analogsignals[0] = sample # only consider the piece of interest in the HDF5 (in the segment)
	bl.recordingchannelgroups[0].recordingchannels[0].analogsignals[0] = sample # only consider the piece of interest in the HDF5 (in the recordingchannel)
	rcg = bl.recordingchannelgroups[0] # take the recording channel
	# detect the peaks
	spikesorter = SpikeSorter(rcg) # apply the Spikesorter tool on it
	spikesorter.RelativeThresholdDetection(sign = "-", relative_thresh = relative_thresh)
	spikesorter.AlignWaveformOnPeak(sign = "-", left_sweep = 1.5 * pq.ms, right_sweep = 1.5 * pq.ms)
	# plot the peak amplitude distribution
	amplitude = [max(spikesorter.spike_waveforms[i][0]) for i in range(len(spikesorter.spike_waveforms))] # list of the amplitudes
	fig = plt.figure(figsize=(18, 6))
	sub1 = fig.add_subplot(121)
	values, bounds, patches = sub1.hist(amplitude, bins = 100, normed = True) # histogram to run the valley detection on it but don't plot it for now
	plt.close(fig)
	bounds2 = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds) - 1)] # take the mean between the bounds of the histogram to generate a curve
	peaks = detect_peaks(values, mph = None, mpd = 0, valley = True, show = True) # detect the local minimum of the histogram to determine the threshold
	fig = plt.figure(figsize=(18, 6))
	sub2 = fig.add_subplot(122)
	sub2.set_title("Peak amplitude distribution", fontsize = 25)
	sub2.set_xlabel("Amplitude (mV)", fontsize = 25)
	sub2.set_ylabel("Density", fontsize = 25)
	sub2.hist(amplitude, bins = 100, normed = True)
	sub2.plot(bounds2, values, "b") # plot a curve that fits the histogram form
	sub2.annotate("Cutoff", xy = (bounds2[peaks[valley_nb]], values[peaks[valley_nb]]), xytext = (bounds2[peaks[valley_nb]-3], 0), arrowprops = dict(facecolor="black", shrink = 0.05)) # point the cutoff with an arrow
	plt.savefig("(4)-Publication_Graphics/Spike_Sorting/peak_distribution.svg", bbox_inches = "tight") # save figure as a SVG image
	plt.show()
	cutoff_mv = bounds2[peaks[valley_nb]]
	cutoff_sd = cutoff_mv / (np.sort(amplitude)[0]/relative_thresh)
	print("Number of spikes in the sample = " + str(len(spikesorter.spike_waveforms))) # plot the amount of spikes in the sample
	print ("Minimum amplitude = " + str(min(np.sort(amplitude))) + " mV") # plot the minimum amplitude for this threshold
	print("Cutoff for valley " + str(valley_nb) + " = " + str(cutoff_mv) + " mV or " + format(cutoff_sd, ".3f") + " x median absolute deviation")
	return 

# Use the "PCA_plotting" function
###########################################################
"""
n = number of clusters expected
relative_threshold = How many times * deviation to detect a spike
start = beginning of the stimulation
end = end of the stimulation
"""
def PCA_plotting(n, relative_threshold, start, end):
	"Function that detects spikes and plots them into n clusters according to a PCA"
	global clusters, spikesorter
	from mpl_toolkits.mplot3d import Axes3D
	spikesorter.PcaFeature(n_components = 6) # Use Principle Component Analysis (PCA) to generate features for spikes
	spikesorter.SklearnGaussianMixtureEm(n_cluster = n, n_iter = 1000) # Use Gaussian Mixture Model Expectation Maximization algorithm to cluster spikes
	spikesorter.refresh_cluster_names()
	colors = ["darkred", "darkblue", "darkgreen", "purple", "darkorange", "darkcyan", "gold"] # add more colors as needed
	c = matplotlib.colors.ColorConverter() # initialize color converter
	for i in range(0, n):
		spikesorter.cluster_colors[i] = c.to_rgb(colors[i])
	counts = [] # Create counts list to store spike-cluster assignments, create features variable to contain the PCA features associated, with each spike
	features = []
	for i in range (0, len(spikesorter.cluster_names.items())):
		counts.append(np.where(spikesorter.spike_clusters == i)[0])
		features.append(spikesorter.waveform_features[counts[i],:])
	spikesorter.cluster_names.keys() # New reorganize our clusters by increasing size (i.e., smallest cluster is Cluster 0, etc.)
	clusters = []
	for i in range (0,n):
		clusters.append((str(spikesorter.cluster_names.keys()[i]), counts[i].size))
	dtype = [("name", "S10"), ("size", int)]
	clusters = np.array(clusters, dtype = dtype)
	clusters = np.sort(clusters, order = "size")
	new_features = [0]*n
	for i in range(0, n):
		new_features[i] = features[int(clusters[i][0])]
	features = new_features
	plt.figure(figsize=(18,6))
	plt.subplot(131)
	for i in range (0, len(features)):
		plt.scatter(features[i][:,0], features[i][:,1], c= spikesorter.cluster_colors[i], edgecolor = spikesorter.cluster_colors[i])
		plt.xlabel("PCA 1", fontsize = 18, color = "black"); plt.ylabel("PCA 2", fontsize = 18, color = "black")
		plt.xticks(fontsize = 14, color = "black"); plt.yticks(fontsize = 14, color = "black")
	plt.subplot(132)
	for i in range (0, len(features)):
		plt.scatter(features[i][:,0], features[i][:,2], c = spikesorter.cluster_colors[i], edgecolor = spikesorter.cluster_colors[i])
		plt.title("Wilson Feature Plots", fontsize = 20)
		plt.xlabel("PCA 1", fontsize = 18, color = "black"); plt.ylabel("PCA 3", fontsize = 18, color = "black")
		plt.xticks(fontsize = 14, color = "black"); plt.yticks(fontsize = 14, color = "black")
	plt.subplot(133)
	for i in range (0, len(features)):
		plt.scatter(features[i][:,2], features[i][:,1], c = spikesorter.cluster_colors[i], edgecolor = spikesorter.cluster_colors[i])
		plt.xlabel("PCA 3", fontsize = 18, color = "black"); plt.ylabel("PCA 1", fontsize = 18, color = "black")
		plt.xticks(fontsize = 14, color = "black"); plt.yticks(fontsize = 14, color = "black")
	plt.savefig("(4)-Publication_Graphics/Spike_Sorting/features_clusters_" + str(start) + "-" + str(end) + ".svg", bbox_inches = "tight") # save figure as a SVG image
	fig = plt.figure(figsize = (7,7))
	ax = fig.add_subplot(111, projection = "3d")
	for i in range(0, len(features)):
		ax.scatter(features[i][:,0], features[i][:,1], np.array(features[i][:,2]), c = spikesorter.cluster_colors[i], edgecolor = spikesorter.cluster_colors[i])
	ax.set_title("PCA Feature Space", fontsize = 20)
	ax.set_xlabel("PCA 1", fontsize = 14, color = "black"); ax.set_ylabel("PCA 2", fontsize = 14, color = "black")
	ax.set_zlabel("PCA 3", fontsize = 14, color = "black")
	ax.tick_params(axis = "x", labelsize = 10, labelcolor = "black")
	ax.tick_params(axis = "y", labelsize = 10, labelcolor = "black")
	ax.tick_params(axis = "z", labelsize = 10, labelcolor = "black")
	plt.savefig("(4)-Publication_Graphics/Spike_Sorting/3Dfeatures_clusters_" + str(start) + "-" + str(end) + ".svg", bbox_inches = "tight") # save figure as a SVG image
	return

# Use the "spike_interval" function
###########################################################
"""
n = number of clusters expected
start = beginning of the stimulation
end = end of the stimulation
"""
def spike_interval(n, start, end):
	"Function that plots information about each cluster"
	global cluster, frame, rcg, spikesorter, times, waveforms
	rcg = spikesorter.populate_recordingchannelgroup() # repopulate
	for u, unit in enumerate(rcg.units):
		print (int(clusters[u][0]), "unit name", spikesorter.cluster_names.values()[int(clusters[u][0])])
		for s, seg in enumerate(rcg.block.segments):
			sptr = seg.spiketrains[u]
			print (" in Segment", s, "has SpikeTrain with", sptr.size)
	spikesorter.refresh_cluster_names()
	colors = ["darkred", "darkblue", "darkgreen", "purple", "darkorange", "darkcyan", "gold"] # add more colors as needed
	c = matplotlib.colors.ColorConverter() # initialize color converter
	for i in range(0, n):
 		spikesorter.cluster_colors[i] = c.to_rgb(colors[i])
	times = [] # Extract times from each unit into a list called timestamps
	for i in range (0,len(rcg.units)):
		u = rcg.units[i].spiketrains[0]
		times.append(np.array(u.times))
	waveforms = []
	for i in range (0,len(rcg.units)):
		u = rcg.units[i].spiketrains[0]
		waveforms.append(np.squeeze(u.waveforms))
	new_waveforms = [0]*n; new_times = [0]*n
	for i in range(0, n):
		new_waveforms[i] = waveforms[int(clusters[i][0])]
		new_times[i] = times[int(clusters[i][0])]
	waveforms = new_waveforms
	times = new_times
	bounds = np.arange(int(math.floor(spikesorter.seg_t_start)), int(math.floor(spikesorter.seg_t_stop)))
	# plot the waveforms for each cluster
	plt.figure(figsize = (15, 4))
	for i in range(0,n):
		plt.subplot(1, n, i+1)
		plt.plot(np.arange(0.0,frame)/20, waveforms[i].T, color = spikesorter.cluster_colors[i])
		plt.xlabel("Time (msec)", fontsize = 16, color = "black")
		plt.ylabel("microvolts", fontsize = 14, color = "black")
		plt.title("Cluster {}".format(i), fontsize = 20)
		plt.xticks(fontsize = 12, color = "black"); plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/Spike_Sorting/cluster_waveforms" + str(start) + "-" + str(end) +".svg", bbox_inches = "tight") # save figure as a SVG image
	# plot the firing rate for each cluster
	plt.figure(figsize = (15, 4))
	for i in range (0, n):
		plt.subplot(1, len(waveforms), i+1)
		plt.hist(times[i], bins= bounds, facecolor = spikesorter.cluster_colors[i], edgecolor=spikesorter.cluster_colors[i]);
		plt.xlabel("Time (sec)", fontsize = 16, color = "black")
		plt.ylabel("Spikes/s", fontsize = 16, color = "black")
		plt.xticks(fontsize = 12, color = "black"); plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/Spike_Sorting/cluster_firing" + str(start) + "-" + str(end) +".svg", bbox_inches = "tight") # save figure as a SVG image
	# plot the interspike interval
	diff = []
	for i in range(0, len(times)):
		diff.append(np.diff(times[i]))
	bounds = np.arange(0, 0.01, 0.001)
	plt.figure(figsize = (16, 4))
	for i in range (0,n):
		plt.subplot(1, n, i+1)
		plt.hist(diff[i], bins= bounds, facecolor = spikesorter.cluster_colors[i], edgecolor=spikesorter.cluster_colors[i]);
		plt.xticks(np.arange(0, 0.01, 0.001), np.arange(10))
		plt.xlabel("ISI (msec)", fontsize = 16, color = "black")
		plt.ylabel("Count", fontsize = 16, color = "black")
		plt.title("Cluster {}".format(i), fontsize = 20)
		plt.xticks(fontsize = 12, color = "black"); plt.yticks(fontsize = 12, color = "black")
	plt.savefig("(4)-Publication_Graphics/Spike_Sorting/ISI_clusters" + str(start) + "-" + str(end) +".svg") # save figure as a SVG image
	# write timestamps and waveforms of each cluster to a text file
	for i in range(0, n): 
		filename = "(4)-Publication_Graphics/Spike_Sorting/times_unit" + str(i) + ".txt"
		np.savetxt(filename, times[i])
	for i in range(0,n):
		filename = "(4)-Publication_Graphics/Spike_Sorting/waveform_unit" + str(i) + ".txt"
		np.savetxt(filename, waveforms[i])
	return

"""
This code loads all the functions for quality metrics, required for the "errors" function.

For each algorithm, we assume the following data structures : all unit timestamps into a list called 'times',  all waveforms into a list called 'waveforms'  
  
For example:  
> times[0] is a numpy array containing all of the timestamps for unit 0  
> waveforms[2] is a numpy matrix containing all waveforms for unit 2  (Events x Samples)
"""

# Refractory Violation : Calculates proportion of spikes that are false negative contaminations due to refractory violations
def refractory(duration, refract_time, censor_time, unitnum):
	global times
	N = times[unitnum].size                                    # number of spikes in cluster
	rpvt = 2 * (refract_time - censor_time)* N                 # amount of refractory time surrounding each spike
	T = duration                                               # length of recording (s)
	diff = np.diff(times[unitnum])                             # inter-spike intervals
	refract_violate = np.where(diff < refract_time)            # location of refractory violations
	rpv = refract_violate[0].size                              # number of refractory violations
	f = float(N) / float(T)                                    # firing frequency of unit
	f_r = float(rpv) / rpvt                                    # refractory frequency
	p = f_r / f
	if np.isnan(float(p)):
		p = 0
	if p > 1:
		p = 1
	return p

# Censored Spikes : Estimates proportion of spikes that were missed due to the censor period of the spike detection algorithm combined during the activity of other units
def censor(duration, censor_time, unitnum):
	global times
	spkts = times[unitnum]
	spikes = 0
	for i in range(0, len(times)):
		spikes += times[i].shape[0]
	spikes -= spkts.shape[0] # the total number of spikes in all other units
	cens = (spikes * censor_time) / float(duration)
	return cens

# Threshold Cutoff : Estimates proportion of false negative spikes in a cluster due to the spikes falling under the detection threshold voltage
def thresh_cut(unitnum, thresh, graph):
	global waveforms
	bins = 75.0;
	spkts = waveforms[unitnum] / thresh
	spkts = np.matrix(spkts)
	criteria = np.squeeze(np.asarray(spkts.max(1)))
	global_max = np.max(criteria)
	lims = np.linspace(1.0, global_max, bins + 1.0)
	x = lims + (lims[1] - lims[0])/2
	n, b = np.histogram(criteria, lims)
	n = np.insert(n, 74,0)
	num_samples = float(criteria.size) # find the mode with an approximation
	shift = np.round(num_samples * 0.05)
	k = np.sort(criteria)
	o = k[shift:criteria.size] - k[0:criteria.size - shift]
	w = np.argmin(o)
	m = k[np.round(w + np.float(shift)/2) - 1]
	num = 20 # now approximate the stdev and mu from the mode
	init = np.sqrt(np.mean((m - criteria[criteria >= m])**2))
	st_guesses = np.linspace(init/2, init*2, num)
	m_guesses = np.linspace(m - init, max(m + init, 1), num)
	error = np.zeros((m_guesses.size, st_guesses.size))
	for i in range(m_guesses.size):
		for j in range(st_guesses.size):
			b = st.norm.pdf(x,m_guesses[i], st_guesses[j])
			a= b * np.sum(n) / np.sum(b)
			error[i, j] = np.sum(abs(a[:] - n[:]))
	pos = np.argmin(error) # selecting the least error
	jpos = pos % num
	if jpos == 0:
		jpos = num - 1
	kpos = np.ceil(float(pos) / num)
	stdev = st_guesses[kpos]
	mu = m_guesses[jpos]
	p = st.norm.cdf(1.0, mu, stdev)
	return p

# Quantitative Overlap : Uses multivariate Gaussians to approximate two given clusters as two separate data distributions. Posterior probabilities are then used to approximate amount of overlap between clusters in a feature space
def overlap(unitnum1, unitnum2, graph):
	global waveforms
	spkts1 = waveforms[unitnum1]
	spkts2 = waveforms[unitnum2]
	N1 = spkts1.shape[0]
	N2 = spkts2.shape[0]
	spkts = np.vstack((spkts1, spkts2))
	spkts = np.matrix(spkts)
	data = sg.detrend(spkts, type = "constant", axis = 0)
	u , s, v = np.linalg.svd(data, full_matrices = 0, compute_uv = 1)
	proj = data * np.matrix(v.T)
	cumvals = np.cumsum(s)
	threshold = np.sum(s) * 0.98
	num_dims = np.where(cumvals < threshold)[-1][-1]
	w1 = proj[0:N1, 0:num_dims]
	w2 = proj[N1:N1+N2, 0:num_dims]
	g = mixture.GMM(n_components=2, covariance_type= "full")
	g.weights_ = np.array([float(N1)/(N1 + N2), float(N2)/(N1 + N2)])
	g.fit(np.vstack((w1, w2)))
	g.covars_[0] = np.cov(w1, rowvar = 0); g.covars_[1] = np.cov(w2, rowvar = 0)
	g.means_[0,:] = np.mean(w1, axis = 0); g.means_[1,:] = np.mean(w2, axis = 0)
	g.fit(np.vstack((w1, w2)))
	pr1 = g.predict_proba(w1)
	pr2 = g.predict_proba(w2)
	if np.mean(pr1[:,0]) + np.mean(pr2[:,1]) < 1:
		pr1 = pr1[:,np.arange(1,-1,-1)]
		pr2 = pr2[:,np.arange(1,-1,-1)]
	confusion = np.zeros((2,2))
	confusion[0,0] = np.mean(pr1[:,1])
	confusion[0,1] = np.sum(pr2[:,0]) / N1
	confusion[1,1] = np.mean(pr2[:,0])
	confusion[1,0] = np.sum(pr1[:,1]) / N2
	if graph:
		prob = np.concatenate((pr1[:,0], pr2[:,0]))
		r = prob.max() - prob.min()
		prob = (prob - prob.min()) / r
		plt.figure(figsize=(8,6))
		features_total  = np.vstack((w1, w2))
		s= plt.scatter(features_total[:,0], features_total[:,1], c = prob,edgecolors = "none", cmap = "jet")
		plt.colorbar(s)
		plt.legend()
		plt.title("Quantitative Overlap")
		plt.show()
	return confusion

# Use the "errors" function
###########################################################
"""In Python, we must reproduce the following quality metrics from Hill et al. 2011.

Hill, D. N., Mehta, S. B. and Kleinfeld, D. 
Quality Metrics to Accompany Spike Sorting of Extracellular Signals. J. Neurosci. 31, 8699-8705 (2011).

    Negative Error:
        Censored Spikes
        Threshold Cutoff
        Quantitative Overlap
    Positive Error:
        Quantitative Overlap
        Refractory Overlap

n = number of clusters
censor_time = censor time of the spike detection
refract_time = refractory time
"""
def errors(n, censor_time, refract_time):
	"Function that gives you the different errors for your analysis"
	global duration, threshold
	metrics = dict()
	for i in range(n):
		metrics["cluster" + str(i)] = dict()
		metrics["cluster" + str(i)]["Threshold Negative"] = 0
		metrics["cluster" + str(i)]["Overlap Negative"] = 0
		metrics["cluster" + str(i)]["Censored Spikes"] = 0
		metrics["cluster" + str(i)]["Overlap Positive"] = 0
		metrics["cluster" + str(i)]["Refractory Violation"] = 0
		metrics["cluster" + str(i)]["Total Positive Error"] = 0
		metrics["cluster" + str(i)]["Total Negative Error"] = 0
	for i in range(n):
		metrics["cluster" + str(i)]["Censored Spikes"] = censor(duration, censor_time, i)
		metrics["cluster" + str(i)]["Refractory Violation"] = refractory(duration, refract_time, censor_time , i)
		metrics["cluster" + str(i)]["Threshold Negative"] = thresh_cut(i, threshold, False)
	clus = list(range(n))
	for i in range(n):
		other_clus = clus[:]
		other_clus.remove(i)
	for j in other_clus:
		confusion = overlap(i, j, False)
		metrics["cluster" + str(i)]["Overlap Positive"] += confusion[0,0]
		metrics["cluster" + str(i)]["Overlap Negative"] += confusion[0,1]
	for i in range(n): # sum the positive and negative errors
		metrics["cluster"+ str(i)]["Total Positive Error"] = metrics["cluster" + str(i)]["Overlap Positive"] + metrics["cluster" + str(i)]["Refractory Violation"]
		metrics["cluster"+ str(i)]["Total Negative Error"] = metrics["cluster" + str(i)]["Overlap Negative"] + metrics["cluster" + str(i)]["Censored Spikes"] + metrics["cluster" + str(i)]["Threshold Negative"]
	m = pd.DataFrame(metrics)
	y1 = np.round(m, decimals = 4)
	y2 = pd.DataFrame(y1.values * 100, columns = y1.columns, index = y1.index) # displays as %
	for i in range(n):
		name = "cluster" + str(i)
		y2[name] = y2[name].astype(str) + "%"
	return y2