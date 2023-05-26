import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import sys
import os
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, RectangleSelector
from matplotlib.widgets import LassoSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib import path as matplotlib_path
import matplotlib.patches as patches
import pandas as pd
import basic_functions as bf
c_kms = 299792.458 # Speed in Light in Km/s


def remove_common(a, b):
	for i in a[:]:
		if i in b:
			a.remove(i)
			b.remove(i)
	return (a, b)

def find_nearest_idx(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

def vel_prof(x, centre):
    xnew = c_kms * ((x-centre)/x)
    return (xnew)

def wave_prof(vel_center, centre):
    xnew = (centre*c_kms) / (c_kms-vel_center)
    return (xnew)

def gaus_prof_wave(wave, wave_rest, log_amp, center, sigma):
	vel_array = vel_prof(wave, wave_rest)
	prof = (10**amp)*np.exp(-(vel_array-center)**2./(2.*sigma**2.))
	return (prof)

def gaus_prof_wave_array(wave, wave_rest_array, log_amp_array, center_array, sigma_array, **kwargs):
	cont_array = kwargs.get('continuum', np.zeros_like(wave))  # Sleep Time
	prof = np.zeros_like(wave)
	for i in range(len(wave_rest_array)):
		vel_array = vel_prof(wave, wave_rest_array[i])
		prof += (10**log_amp_array[i])*np.exp(-(vel_array-center_array[i])**2./(2.*sigma_array[i]**2.))
	prof+=cont_array
	return (prof)

def gaus_prof_wave_array_rev(wave_obs, wave_redshifted_array, log_amp_array, sigma_array, **kwargs):
	cont_array = kwargs.get('continuum', np.zeros_like(wave_obs))  # continuum
	prof = np.zeros_like(wave_obs)
	for i in range(len(wave_redshifted_array)):
		vel_array = vel_prof(wave_obs, (wave_redshifted_array[i]))
		prof += (10**log_amp_array[i])*np.exp(-(vel_array)**2./(2.*sigma_array[i]**2.))
	prof+=cont_array
	return (prof)


def gaus_prof_vel(vel_array, amp, center, sigma):
	prof = (10**amp)*np.exp(-(vel_array-center)**2./(2.*sigma**2.))
	return (prof)
	

emission_line_data = pd.read_csv('default_emission_lines.dat', sep="\t", encoding=None, header=0)

def creating_additional_figure_new(wave_array_1, grouped_1d_data_array_1, grouped_1d_error_array_1, redshift):
	print ('\n')
	print ('Creating secondary figure...')
	print ('\n')
	sys.stdout.flush()
	#redshift = kwargs.get('redshift', 0.0)  # redshift
	fig_secondary_1d_data, (ax_secondary_1d_data) = plt.subplots(1)
	line4, = ax_secondary_1d_data.plot([], [], 'm.', zorder=5)
	line_custom_1, = ax_secondary_1d_data.plot(wave_array_1, grouped_1d_data_array_1, color='tab:blue', alpha=0.8, label='Spec', zorder=2)
	axdata_smooth = fig_secondary_1d_data.add_axes([0.15, 0.92, 0.1, 0.03])
	sdata_smooth = Slider(axdata_smooth, 'Smoothing', 1, 100, valinit=1)

	axdata_amp = fig_secondary_1d_data.add_axes([0.32, 0.92, 0.1, 0.03])
	sdata_amp = Slider(axdata_amp, 'Amp', np.nanmin(grouped_1d_data_array_1), np.nanmax(grouped_1d_data_array_1), valinit=(np.nanmedian(grouped_1d_data_array_1)*2))
	axdata_vel = fig_secondary_1d_data.add_axes([0.6, 0.92, 0.34, 0.03])
	sdata_vel = Slider(axdata_vel, 'z', 1.0, 3.0, valinit=redshift)

	cont_func = bf.continuum_fitClass()
	cont_func.flux = grouped_1d_data_array_1
	fitted_cont = cont_func.median_filter_continuum(wave_array_1)
	redshift_val = float(sdata_vel.val)
	emission_line_data_rev = emission_line_data['wavelength']*(1.+redshift_val)
	mask = ( (emission_line_data_rev > np.nanmin(wave_array_1)) & (emission_line_data_rev < np.nanmax(wave_array_1)) )
	log_amp_array = np.log10(float(sdata_amp.val)*emission_line_data['flux_ratio'][mask].to_numpy())
	sigma_array = np.full_like(emission_line_data['flux_ratio'][mask].to_numpy(), fill_value=100.)
	fiducial_profile = gaus_prof_wave_array_rev(wave_array_1, emission_line_data_rev[mask].to_numpy(), log_amp_array, sigma_array, continuum=fitted_cont)
	line3, = ax_secondary_1d_data.plot(wave_array_1, fitted_cont, 'g--', zorder=3)
	line2, = ax_secondary_1d_data.plot(wave_array_1, fiducial_profile, 'r--', zorder=4)
	ax_secondary_1d_data.set_ylim((0.9*np.nanmin(grouped_1d_data_array_1)), (1.1*np.nanmax(grouped_1d_data_array_1)))
	ax_secondary_1d_data.set_xlim(np.nanmin(wave_array_1), np.nanmax(wave_array_1))

	def update(val):
		kmos_ob = bf.kmos_cube_analysis()
		kmos_ob.box_pts = int(sdata_smooth.val)
		ydata_updated = kmos_ob.spectral_smooth(grouped_1d_data_array_1)
		line_custom_1.set_ydata(ydata_updated)
		redshift_val = float(sdata_vel.val)
		emission_line_data_rev = emission_line_data['wavelength']*(1.+redshift_val)
		mask = ( (emission_line_data_rev > np.nanmin(wave_array_1)) & (emission_line_data_rev < np.nanmax(wave_array_1)) )
		log_amp_array = np.log10(float(sdata_amp.val)*emission_line_data['flux_ratio'][mask].to_numpy())
		sigma_array = np.full_like(emission_line_data['flux_ratio'][mask].to_numpy(), fill_value=100.)
		fiducial_profile_rev = gaus_prof_wave_array_rev(wave_array_1, emission_line_data_rev[mask].to_numpy(), log_amp_array, sigma_array, continuum=fitted_cont)
		fitted_cont_rev = kmos_ob.spectral_smooth(fitted_cont)
		fiducial_profile_rev2 = kmos_ob.spectral_smooth(fiducial_profile_rev)
		line3.set_ydata(fitted_cont)
		line2.set_ydata(fiducial_profile_rev2)
		ax_secondary_1d_data.set_xlabel(r"Wavelength ($\rm \AA$)")
		ax_secondary_1d_data.set_ylabel(r"Relative Flux")
		fig_secondary_1d_data.canvas.draw()
	sdata_smooth.on_changed(update)
	sdata_amp.on_changed(update)
	sdata_vel.on_changed(update)

	
	#Defining the selection function
	str1 = ''
	global mask_selection
	mask_selection = []
	def line_select_callback(eclick, erelease):
		global str1, x1, y1, x2, y2, mask_selection
		global idx_xaxis_min, idx_yaxis_min, idx_xaxis_max, idx_yaxis_max
		x1, y1 = eclick.xdata, eclick.ydata
		x2, y2 = erelease.xdata, erelease.ydata
		x1_idx = find_nearest_idx(wave_array_1, min(x1, x2))
		x2_idx = find_nearest_idx(wave_array_1, max(x1, x2))
		for i in range(x1_idx, x2_idx):
			if (min(y1, y2) < grouped_1d_data_array_1[i] < max(y1,y2)):
				if (str1 == 'add'):
					mask_selection.append(i)
				elif (str1 == 'rem'):
					if (i in mask_selection[:]):
						mask_selection.remove(i)
				else:
					print ("Function Inactive.....")
		line4.set_data(wave_array_1[mask_selection], grouped_1d_data_array_1[mask_selection])
		fig_secondary_1d_data.canvas.draw()

	def toggle_selector(event):
		global str1
		print(' Key pressed.')
		if event.key in ['T', 't'] and toggle_selector.RS.active:
				print(' RectangleSelector deactivated.')
				toggle_selector.RS.set_active(False)
		if event.key in ['Y', 'y'] and not toggle_selector.RS.active:
				print(' RectangleSelector activated.')
				toggle_selector.RS.set_active(True)
		if event.key in ['H', 'h'] and toggle_selector.RS.active:
				print('Add function activated')
				str1 = 'add'
				toggle_selector.RS.set_active(True)
		if event.key in ['J', 'j'] and toggle_selector.RS.active:
				print('Remove function activated')
				str1 = 'rem'
				toggle_selector.RS.set_active(True)

	toggle_selector.RS = RectangleSelector(ax_secondary_1d_data, line_select_callback, drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
	plt.connect('key_press_event', toggle_selector)

	add_sky_button_axes = plt.axes([0.1, 0.01, 0.11, 0.04])
	add_sky_button = Button(add_sky_button_axes, 'add_sky')
	def func_add_sky(event):
		global mask_selection
		mask_selection = list(np.unique(mask_selection))
		np.savetxt('tmp_sky_spec.dat', mask_selection, fmt='%d')
		#with open('tmp_sky_spec.dat', 'ab') as f:
		#	np.savetxt(f, mask_selection, fmt='%d')
		print ('sky spectral index saved to file')
	add_sky_button.on_clicked(func_add_sky)

	load_sky_button_axes = plt.axes([0.25, 0.01, 0.11, 0.04])
	load_sky_button = Button(load_sky_button_axes, 'load_sky')
	def func_load_sky(event):
		global mask_selection
		mask_selection = list(np.loadtxt('tmp_sky_spec.dat').astype(np.int32))
		line4.set_data(wave_array_1[mask_selection], grouped_1d_data_array_1[mask_selection])
		fig_secondary_1d_data.canvas.draw()
		print ('sky loaded...')
	load_sky_button.on_clicked(func_load_sky)


	set_gal_button_axes = plt.axes([0.4, 0.01, 0.11, 0.04])
	set_gal_button = Button(set_gal_button_axes, 'set_gal')
	def func_set_gal(event):
		global set_gal_selection, mask_selection
		set_gal_selection = list(np.unique(mask_selection))
		np.savetxt('set_gal_spec.dat', set_gal_selection, fmt='%d')
		print ('galaxy spectral index saved to file')
	set_gal_button.on_clicked(func_set_gal)


	plt.show()

