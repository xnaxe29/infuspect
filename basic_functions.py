import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from progress.bar import IncrementalBar
import re


def load_two_column_data_from_file(filename):
	column1_data = []
	column2_data = []
	with open(filename, 'r') as file:
		for line in file:
			columns = line.split('\t')  # Assumes columns are separated by a tab
			# Extract data from column 1
			column1_value = re.findall(r'\d+', columns[0])
			column1_value = int(column1_value[0]) if column1_value else None
			column1_data.append(column1_value)
			# Extract data from column 2
			column2_value = re.findall(r'\d+', columns[1])
			column2_value = int(column2_value[0]) if column2_value else None
			column2_data.append(column2_value)
	final_array = np.zeros([len(column1_data), 2])
	final_array[:,0] = column1_data
	final_array[:,1] = column2_data
	final_array_rev = np.unique(final_array, axis=0)
	return final_array_rev



def add_colorbar(mappable):
	last_axes = plt.gca()
	ax = mappable.axes
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax1 = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(mappable, cax=cax1)
	cbar.set_ticks(ticker.LogLocator(), update_ticks=True)
	cbar.ax.tick_params(size=0)
	return cbar

def add_colorbar_lin(mappable):
	last_axes = plt.gca()
	ax = mappable.axes
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax1 = divider.append_axes("right", size="5%", pad=0.55)
	cbar = plt.colorbar(mappable, cax=cax1)
	cbar.set_ticks(ticker.LinearLocator(), update_ticks=True)
	cbar.ax.tick_params(size=0)
	return cbar

def clean_data(data, **kwargs):
	data_type = kwargs.get('type_of_data', 'data')  # Default Data type
	data_val_min = kwargs.get('val_data', np.nanmin(data))  # Default minimum value
	data_val_max = kwargs.get('val_data', np.nanmax(data))  # Default minimum value
	data = data.astype(np.float64)
	data_1 = np.nan_to_num(data, nan=data_val_min, posinf=data_val_max, neginf=data_val_min)
	mask = np.where(data_1[:]=='')
	data_1[mask] = data_val_min
	mask2 = np.where(data_1[:]==' ')
	data_1[mask2] = data_val_min
	if 'err' in data_type:
		data_1[data_1<=0.]=np.abs(data_val_min)*10.
	return (data_1)




##################################GET_CONTINUUM##################################


class continuum_fitClass:
	def __init__(self):
		self.plot=False
		self.print=False
		self.legend=False
		self.ax=False
		self.label=False
		self.color='gray'
		self.default_smooth=5
		self.default_order=8
		self.default_allowed_percentile=75
		self.default_filter_points_len=10
		self.data_height_upscale=2
		self.default_poly_order=3
		self.default_window_size_default=999
		self.default_fwhm_galaxy_min=10
		self.default_noise_level_sigma=10
		self.default_fwhm_ratio_upscale=10
		#self.spline_smoothing_factor = 1
		self.pca_component_number = 1
		self.lowess_cont_frac = 0.05
		self.gaussian_cont_fit = 200
		self.peak_prominence = 0.05
		self.peak_width = 10
		self.median_filter_window = 101
		self.continuum_finding_method = 'custom'
		pass
	def smooth(self, y, box_pts):
		box = np.ones(box_pts)/box_pts
		y_smooth = np.convolve(y, box, mode='same')
		return y_smooth
	def gaussian(self, x, amp, mu, sig):
		return amp*(1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2))
	#This function has been created taking inference from Martin+2021 (2021MNRAS.500.4937M)
	def continuum_finder(self, wave, *pars, **kwargs):
		flux = self.flux
		#n_smooth=5, order=8, allowed_percentile=75, poly_order=3, window_size_default=None
		if pars:
			#n_smooth, allowed_percentile, poly_order, window_size_default, fwhm_galaxy_min = pars
			#print (pars)
			n_smooth, allowed_percentile, filter_points_len, data_height_upscale, poly_order, window_size_default, fwhm_galaxy_min, noise_level_sigma, fwhm_ratio_upscale = pars
		else:
			#n_smooth, allowed_percentile, poly_order, window_size_default, fwhm_galaxy_min = np.array([self.default_smooth, self.default_allowed_percentile, self.default_poly_order, self.default_window_size_default, self.default_fwhm_galaxy_min])
			n_smooth, allowed_percentile, filter_points_len, data_height_upscale, poly_order, window_size_default, fwhm_galaxy_min, noise_level_sigma, fwhm_ratio_upscale = np.array([self.default_smooth, self.default_allowed_percentile, self.default_filter_points_len, self.data_height_upscale, self.default_poly_order, self.default_window_size_default, self.default_fwhm_galaxy_min, self.default_noise_level_sigma, self.default_fwhm_ratio_upscale])

		n_smooth = int(n_smooth)
		#order = int(order)
		allowed_percentile = int(allowed_percentile)
		filter_points_len = int(filter_points_len)
		poly_order = int(poly_order)
		window_size_default = int(window_size_default)
		fwhm_galaxy_min = int(fwhm_galaxy_min)
		noise_level_sigma = int(noise_level_sigma)
		fwhm_ratio_upscale = int(fwhm_ratio_upscale)

		pick = np.isfinite(flux) #remove NaNs
		flux = flux[pick] #remove NaNs
		#smoothed_data = scp.ndimage.convolve1d(flux, np.asarray([1.]*n_smooth)/n_smooth) #smooth data
		smoothed_data = self.smooth(flux, int(n_smooth))
		local_std = np.median([ np.std(s) for s in np.array_split(flux, int(n_smooth)) ]) #find local standard deviation
		mask_less = argrelextrema(smoothed_data, np.less)[0] #find relative extreme points in absorption
		mask_greater = argrelextrema(smoothed_data, np.greater)[0] #find relative extreme points in emission
		mask_less_interpolate_func = interpolate.interp1d(wave[mask_less], flux[mask_less], kind='cubic', fill_value="extrapolate") #interpolate wavelength array like function from relative extreme points in absorption
		mask_greater_interpolate_func = interpolate.interp1d(wave[mask_greater], flux[mask_greater], kind='cubic', fill_value="extrapolate") #interpolate wavelength array like function from relative extreme points in emission
		absolute_array = mask_greater_interpolate_func(wave)-mask_less_interpolate_func(wave) #obtain the absolute array for find_peaks algorithm
		filter_points = np.array([int(i*len(absolute_array)/filter_points_len) for i in range(1,filter_points_len)])
		noise_height_max_default = noise_level_sigma*np.nanmin(np.array([np.nanstd(absolute_array[filter_points[i]-10:filter_points[i]+10]) for i in range(len(filter_points))]))
		data_height_max_default = data_height_upscale*np.nanmax(np.array([np.abs(np.nanmax(absolute_array)), np.abs(np.nanmin(absolute_array))]))
		noise_height_max = kwargs.get('noise_height_max', noise_height_max_default)  # Maximal height for noise
		data_height_max = kwargs.get('data_height_max', data_height_max_default)  # Maximal height for data
		peaks = find_peaks(absolute_array, height=[noise_height_max, data_height_max], prominence=(local_std*3.), width = [fwhm_galaxy_min, int(fwhm_ratio_upscale*fwhm_galaxy_min)]) #run scipy.signal find_peaks algorithm to find peak points
		edges = np.int32([np.round(peaks[1]['left_ips']), np.round(peaks[1]['right_ips'])]) #find edges of peaks
		d = (np.diff(flux, n=1))
		w = 1./np.concatenate((np.asarray([np.median(d)]*1),d))
		w[0] = np.max(w)
		w[-1] = np.max(w)
		for edge in edges.T:
			#print (wave[pick][edge[0]], wave[pick][edge[1]])
			diff_tmp = int((edge[1] - edge[0])/2)
			w[edge[0]-diff_tmp:edge[1]+diff_tmp] = 1./10000.
		w = np.abs(w)
		pick_2 = np.where(w > np.percentile(w, allowed_percentile * (float(len(flux)) / float(len(wave)))))[0]
		#fit = np.poly1d(np.polyfit(tsteps[pick][pick_2], a[pick_2], order))
		#print (wave[pick][pick_2])
	
		if len(wave[pick][pick_2])>3:
			xx = np.linspace(np.min(wave[pick][pick_2]), np.max(wave[pick][pick_2]), 1000)
			itp = interpolate.interp1d(wave[pick][pick_2], flux[pick_2], kind='linear')

		else:
			mask = np.ones_like(wave, dtype=np.bool8)
			ynew = np.abs(np.diff(flux[mask], prepend=1e-10))
			ynew2 = np.percentile(ynew, allowed_percentile)
			xx = wave[mask][ynew < ynew2]
			y_rev = flux[mask][ynew < ynew2]
			itp = interpolate.interp1d(xx, y_rev, axis=0, fill_value="extrapolate", kind='linear')
			#y_rev2 = f_flux(wave)

		#window_size = int(((1.0 / (step_to_t[pick][pick_2][-1] - step_to_t[pick][pick_2][0])) * 1000.))
		#window_size = window_size_default
		window_size = int(fwhm_ratio_upscale*fwhm_galaxy_min)
		if window_size % 2 == 0:
			window_size = window_size + 1
		fit_savgol = savgol_filter(itp(xx), window_size, poly_order)
		fit = interpolate.interp1d(xx, fit_savgol, kind='cubic', fill_value="extrapolate")
		#std_cont = np.std((a[pick_2] - fit(tsteps[pick][pick_2])) / fit(tsteps[pick][pick_2]))
		std_cont = np.std(flux[pick_2] - fit(wave[pick][pick_2]))
		#r = (a[pick_2] - fit(tsteps[pick][pick_2])) / fit(tsteps[pick][pick_2])
		#r = flux[pick_2] - fit(wave[pick][pick_2])
		return fit, std_cont, flux, pick, pick_2, peaks, std_cont


	# Function for fitting a polynomial continuum
	def poly_fit_continuum(self, wave):
		degree = self.default_poly_order
		flux = self.flux
		wavelength = wave
		coefficients = np.polyfit(wavelength, flux, degree)
		continuum = np.polyval(coefficients, wavelength)
		return continuum



	# Function for fitting a spline continuum
	def spline_fit_continuum(self, wave):
		flux = self.flux
		wavelength = wave
		#smoothing_factor = self.spline_smoothing_factor
		continuum = interp1d(wavelength, flux, kind='cubic')
		return continuum(wavelength)



	# Function for estimating the continuum using PCA
	def pca_continuum(self, wave):
		flux = self.flux
		wavelength = wave
		pca_component_number = self.pca_component_number
		pca = PCA(n_components = pca_component_number)
		X = flux.reshape(-1, 1)
		pca.fit(X)
		continuum = pca.inverse_transform(pca.transform(X)).flatten()
		return continuum



	# Function for estimating the continuum using a lowess smoother
	def lowess_continuum(self, wave):
		wavelength = wave
		flux = self.flux
		lowess_continuum_fraction = self.lowess_cont_frac
		continuum = lowess(flux, wavelength, frac=lowess_continuum_fraction)[:, 1]
		return continuum


	# Function for estimating the continuum using a Gaussian fit
	def gaussian_func(self, x, a, b, c, d):
		return a * np.exp(-((x-b)/c)**2) + d

	def gaussian_fit_continuum(self, wave):
		wavelength = wave
		flux = self.flux
		window = self.gaussian_cont_fit
		continuum = np.zeros_like(flux)
		for i in range(len(flux)):
			low = max(0, i-window//2)
			high = min(len(flux), i+window//2)
			x = wavelength[low:high]
			y = flux[low:high]
			try:
				popt, _ = curve_fit(self.gaussian_func, x, y, p0=[1, wavelength[i], 5, 0])
				continuum[i] = self.gaussian_func(wavelength[i], *popt)
			except RuntimeError:
				continuum[i] = np.nan
		mask = np.isnan(continuum)
		continuum[mask] = np.interp(wavelength[mask], wavelength[~mask], continuum[~mask])
		return continuum


	# Function for estimating the continuum using peak finding
	def peak_find_continuum(self, wave):
		wavelength = wave
		flux = self.flux
		prominence = self.peak_prominence
		width = self.peak_width
		peaks, _ = find_peaks(flux, prominence=prominence, width=width)
		troughs, _ = find_peaks(-flux, prominence=prominence, width=width)
		indices = np.concatenate([peaks, troughs, [0, len(flux)-1]])
		continuum = np.interp(wavelength, wavelength[indices], flux[indices])
		return continuum


	# Function for estimating the continuum using median filtering
	def median_filter_continuum(self, wave):
		wavelength = wave
		flux = self.flux
		window = self.median_filter_window
		continuum = np.zeros_like(flux)
		for i in range(len(flux)):
			low = max(0, i-window//2)
			high = min(len(flux), i+window//2)
			continuum[i] = np.nanmedian(flux[low:high])
		mask = np.isnan(continuum)
		continuum[mask] = np.interp(wavelength[mask], wavelength[~mask], continuum[~mask])
		return continuum

	# Define distance metric function
	def dist_metric(self, x, y):
		return np.abs(x - y)

	def continuum_using_fof(self, wave):
		wavelength = wave
		flux = self.flux
		# Calculate distance matrix
		X = wavelength.reshape(-1, 1)
		D = cdist(X, X, self.dist_metric)
		# Find clusters using DBSCAN
		db = DBSCAN(eps=3, min_samples=3, metric='precomputed').fit(D)
		labels = db.labels_
		# Calculate continuum
		mask = (labels == 0)
		continuum = np.median(flux[mask])
		return continuum


	def continuum_finder_flux(self, wave, *pars, **kwargs):
		cont_find_method = self.continuum_finding_method
		if (cont_find_method=='custom'):
			fit, std_cont, flux, pick, pick_2, peaks, std_cont = self.continuum_finder(wave, *pars)
			cont_flux = fit(wave[pick])
		elif (cont_find_method=='poly'):
			cont_flux = self.poly_fit_continuum(wave)
		elif (cont_find_method=='spline'):
			cont_flux = self.spline_fit_continuum(wave)
		elif (cont_find_method=='pca'):
			cont_flux = self.pca_continuum(wave)
		elif (cont_find_method=='lowess'):
			cont_flux = self.lowess_continuum(wave)
		elif (cont_find_method=='gauss'):
			cont_flux = self.gaussian_fit_continuum(wave)
		elif (cont_find_method=='peak_find'):
			cont_flux = self.peak_find_continuum(wave)
		elif (cont_find_method=='median_filtering'):
			cont_flux = self.median_filter_continuum(wave)
		elif (cont_find_method=='fof'):
			continuum_fof = self.continuum_using_fof(wave)
			cont_flux = continuum_fof*np.ones_like(wave)
		else:
			print(f'Continuum finding method: {cont_find_method} not found. Reverting back to custom method.')
			fit, std_cont, flux, pick, pick_2, peaks, std_cont = self.continuum_finder(wave, *pars)
			cont_flux = fit(wave[pick])
		return (cont_flux)

##################################GET_CONTINUUM##################################


##################################KMOS_DATA_ANALYSIS######################

class kmos_cube_analysis:
	def __init__(self):
		self.data_cube=np.ones([10,10,100])
		self.sky_reduced_data_cube = self.data_cube
		self.data_err_cube=np.zeros([10,10,100])
		self.ra = np.arange(1, 11, 1)
		self.x_axis = np.arange(0, 10, 1).astype(np.int32)
		self.dec = np.arange(1, 11, 1)
		self.y_axis = np.arange(0, 10, 1).astype(np.int32)
		self.wave = np.arange(1001, 1101, 1)
		self.wave_obs = np.arange(1001, 1101, 1)
		self.sky_spectra = np.ones_like(self.wave)
		self.wave_axis = np.arange(0, len(self.wave), 1).astype(np.int32)
		self.spectral_mask_for_sky = np.arange(0, len(self.wave), 1).astype(np.int32)
		self.redshift=0.0
		self.spectral_smoothing_factor=1
		self.spatial_smoothing_factor=1
		self.binning_type = 'median'
		self.box_pts = 0

	def spectral_smooth(self, y):
		ynew = clean_data(y)
		box = np.ones(self.box_pts)/self.box_pts
		y_smooth = np.convolve(y, box, mode='same')
		return y_smooth

	def spatial_smooth(self):
		self.updated_im_data = gaussian_filter(self.updated_im_data, sigma=int(self.spatial_smoothing_factor))
		#return smoothed_data

	def gaussian(self, x, amp, mu, sig):
		return amp*(1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2))

	def get_data_from_file(self, filename):
		assert os.path.exists(filename), f"File: {filename} not found..."
		with fits.open(str(filename)) as hdul:
			header_original = hdul[1].header
			header_original_err = hdul[2].header
			data_original_file = hdul[1].data
			err_original_file = hdul[2].data

		data_file = clean_data(data_original_file)
		data_err_file = clean_data(err_original_file, type_of_data='err')
		kmos_wcs_celestial = WCS(header_original).celestial
		#self.kmos_wcs_celestial = kmos_wcs_celestial
		kmos_wcs_spectral = WCS(header_original).spectral
		axis_physical_x = np.zeros([kmos_wcs_celestial._naxis[1]])
		axis_physical_y = np.zeros([kmos_wcs_celestial._naxis[0]])
		axis_physical_z = np.zeros([kmos_wcs_spectral._naxis[0]])
		for i in range(len(axis_physical_x)):
			axis_physical_x[i] = kmos_wcs_celestial.array_index_to_world_values([i],[0])[1][0]
		for j in range(len(axis_physical_y)):
			axis_physical_y[j] = kmos_wcs_celestial.array_index_to_world_values([0],[j])[0][0]
		for z in range(len(axis_physical_z)):
			axis_physical_z[z] = kmos_wcs_spectral.array_index_to_world_values([z])[0]*1e10

		self.wave_obs = axis_physical_z
		axis_physical_z_rev = axis_physical_z / (1.+self.redshift)
		self.data_cube = data_file
		self.sky_reduced_data_cube = data_file
		self.data_err_cube = data_err_file
		self.ra = axis_physical_x
		self.dec = axis_physical_y
		self.x_axis = np.arange(0, len(axis_physical_x))
		self.y_axis = np.arange(0, len(axis_physical_y))
		self.wave = axis_physical_z_rev
		self.wave_axis = np.arange(0, len(axis_physical_z_rev))
		self.wcs_celestial = kmos_wcs_celestial
		self.wcs_spectral = kmos_wcs_spectral

	def make_image_fits(self, filename_old, filename_new, data_im_new, data_err_im_new):
		assert os.path.exists(filename_old), f"File: {filename_old} not found..."
		#assert os.path.exists(filename_new), f"File: {filename_new} not found..."
		with fits.open(filename_old) as hdu_list:
			hdu_list.writeto(filename_new)  # to write all HDUs, including the updated one, to a new file
		with fits.open(filename_new, mode='update', output_verify='fix') as hdu_list:
			hdu_list[1].data = data_im_new
			hdu_list[2].data = data_err_im_new
			#del hdu_list[1].header['NAXIS3']; del hdu_list[1].header['CRPIX3']; del hdu_list[1].header['CD3_3']; del hdu_list[1].header['CRVAL3']; del hdu_list[1].header['CTYPE3']
			#del hdu_list[2].header['NAXIS3']; del hdu_list[2].header['CRPIX3']; del hdu_list[2].header['CD3_3']; del hdu_list[2].header['CRVAL3']; del hdu_list[2].header['CTYPE3']
			




	def prepare_2d_image(self, spatial_smooth_bool=False):
		if (self.binning_type=='mean'):
			self.updated_im_data = np.nanmean(self.sky_reduced_data_cube, axis=0)
		elif (self.binning_type=='sum'):
			self.updated_im_data = np.nansum(self.sky_reduced_data_cube, axis=0)
		else:
			self.updated_im_data = np.nanmedian(self.sky_reduced_data_cube, axis=0)
		if (spatial_smooth_bool==True):
			self.spatial_smooth()

	def prepare_3d_cube(self):
		result = np.all(self.sky_spectra == self.sky_spectra[0])
		if (result):
			print ('No new sky found')
		else:
			flattened_data_cube = self.data_cube.reshape(self.data_cube.shape[0], -1)
			flattened_tmp_array = flattened_data_cube
			tmp_bar1 = IncrementalBar('Countdown', max = int(flattened_data_cube.shape[1]))
			if os.path.exists('tmp_sky_spec.dat'):
				self.spectral_mask_for_sky = list(np.loadtxt('tmp_sky_spec.dat').astype(np.int32))
			print ('\n')
			for i in range(flattened_data_cube.shape[1]):
				tmp_bar1.next()
				flattened_tmp_array[self.spectral_mask_for_sky[:], i] = flattened_data_cube[self.spectral_mask_for_sky[:], i] - self.sky_spectra[self.spectral_mask_for_sky[:]]
			print ('\n')
			self.sky_reduced_data_cube = flattened_tmp_array.reshape(len(self.wave), self.data_cube.shape[1], self.data_cube.shape[2])


##################################KMOS_DATA_ANALYSIS######################

