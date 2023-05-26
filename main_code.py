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
import re

import basic_functions as bf
import secondary_spectral_window as ssw



##################################LASSO_FUNCTIONS##################################

def onselect(verts):
	print(verts)

def retrieve_lasso_details(data_array, err_array, indices, grouping_string):
	indices_new = indices.reshape(data_array.shape[1], data_array.shape[2])
	new_data_array_median = np.full_like(data_array, np.nan)
	new_data_array = np.zeros([data_array.shape[0]])
	temp_err_array = np.zeros([data_array.shape[0]])
	count = 0
	for i in range(data_array.shape[1]):
		for j in range(data_array.shape[2]):
			if indices_new[i,j] == True:
				new_data_array_median[:,i,j] = data_array[:,i,j]
				new_data_array = new_data_array + data_array[:,i,j]
				temp_err_array = temp_err_array + (err_array[:,i,j]**2)
				count+=1
	if (grouping_string=='sum'):
		revised_data_array = new_data_array
		revised_err_array = np.sqrt(temp_err_array)
	elif (grouping_string=='mean'):
		revised_data_array = new_data_array / (count)
		revised_err_array = np.sqrt(temp_err_array) / (count)
	elif (grouping_string=='median'):
		revised_data_array = np.nanmedian(new_data_array_median, axis=(1,2))
		revised_err_array = np.zeros([len(revised_data_array)])
	else:
		print ("Grouping string not specified. Returning Sum.")
		revised_data_array = new_data_array
		revised_err_array = np.sqrt(temp_err_array)
	return (revised_data_array, revised_err_array)

def retrieve_grouped_data(data_array, err_array, grouping_string):
	global ind, str2
	string_lasso_check = str2
	if (string_lasso_check == 'las'):
		grouped_1d_data, grouped_1d_error  = retrieve_lasso_details(data_array, err_array, ind, grouping_string)
	elif (string_lasso_check == 'delas'):
		data_array_test_sum = data_array[:, idx_yaxis_min:idx_yaxis_max, idx_xaxis_min:idx_xaxis_max]
		err_array_sum_temp_1 = err_array[:, idx_yaxis_min:idx_yaxis_max, idx_xaxis_min:idx_xaxis_max]**2
		count = (idx_yaxis_max-idx_yaxis_min)*(idx_xaxis_max-idx_xaxis_min)
		final_data_array_sum = np.nansum(data_array_test_sum, axis=(1,2))
		err_array_sum_temp_2 = np.nansum(err_array_sum_temp_1, axis=(1,2))
		err_array_sum_temp_3 = np.sqrt(err_array_sum_temp_2)
		if (grouping_string=='sum'):
			grouped_1d_data = final_data_array_sum
			grouped_1d_error = err_array_sum_temp_3
		elif (grouping_string=='mean'):
			grouped_1d_data = final_data_array_sum / count
			grouped_1d_error = err_array_sum_temp_3 / count
		elif (grouping_string=='median'):
			grouped_1d_data = np.nanmedian(data_array_test_sum, axis=(1,2))
			grouped_1d_error = np.zeros([len(grouped_1d_data)])
	return (grouped_1d_data, grouped_1d_error)
	
##################################LASSO_FUNCTIONS##################################








kmos_ob = bf.kmos_cube_analysis()
kmos_ob.redshift = 1.5906735751295336

tmp_filename = "/Users/adarshranjan/Desktop/main_projects/asiaa_projects/In-FU-spect/KMOS_TC/calibrated/P1-combined_OB/reflex_end_products/2023-04-15T20:39:47/KMOS.2021-01-24T04:20:32.711_combine_OBs/P1-OB3_COMBINED_CUBE_CSMS15-698840.fits"
kmos_ob.get_data_from_file(tmp_filename)
kmos_ob.prepare_2d_image()
cmap_type_preferred = 'viridis'
vmin_from_file = 0.0
#vmax_from_file = np.nanmax(data_2d)
fig, ax = plt.subplots()
ax.cla()
im1 = ax.imshow(kmos_ob.updated_im_data, cmap=str(cmap_type_preferred), aspect='equal', origin='lower')
im_cl1 = bf.add_colorbar_lin(im1)

#Defining smoothening parameter
axdata_smooth = fig.add_axes([0.6, 0.04, 0.15, 0.03])
sdata_smooth = Slider(axdata_smooth, 'Smoothing', 0, int(kmos_ob.sky_reduced_data_cube.shape[0]), valinit=0)



####################################RADIO_BUTTONS_FOR_MEAN_MEDIAN_SUM################################
binning_type_buttons_axes = plt.axes([0.01, 0.2, 0.11, 0.2])
radio_binning_type_buttons = RadioButtons(binning_type_buttons_axes, ('median', 'mean', 'sum'), active=0)
def func_binning_type_buttons(label_binning_type_buttons):
	print('Binning set to ', label_binning_type_buttons)
radio_binning_type_buttons.on_clicked(func_binning_type_buttons)
####################################RADIO_BUTTONS_FOR_MEAN_MEDIAN_SUM################################

####################################RADIO_BUTTONS_FOR_MEAN_MEDIAN_SUM################################
display_type_buttons_axes = plt.axes([0.01, 0.05, 0.11, 0.14])
radio_display_type_buttons = RadioButtons(display_type_buttons_axes, ('lin', 'log'), active=0)
def func_display_type_buttons(label_display_type_buttons):
	print('Display set to ', label_display_type_buttons)
radio_display_type_buttons.on_clicked(func_display_type_buttons)
####################################RADIO_BUTTONS_FOR_MEAN_MEDIAN_SUM################################



#updating values in GUI in realtime
def update(val):
	global im_cl1
	kmos_ob.spatial_smoothing_factor = int(sdata_smooth.val)
	kmos_ob.binning_type = str(radio_binning_type_buttons.value_selected)
	kmos_ob.display_type = str(radio_display_type_buttons.value_selected)
	kmos_ob.prepare_2d_image(spatial_smooth_bool=True)
	ax.cla()
	im_cl1.remove()
	if (kmos_ob.display_type=='log'):
		im1 = ax.imshow(np.log10(kmos_ob.updated_im_data), cmap=str(cmap_type_preferred), aspect='equal', origin='lower')
	else:
		im1 = ax.imshow(kmos_ob.updated_im_data, cmap=str(cmap_type_preferred), aspect='equal', origin='lower')
	im_cl1 = bf.add_colorbar_lin(im1)
	fig.canvas.draw()
sdata_smooth.on_changed(update)
radio_binning_type_buttons.on_clicked(update)
radio_display_type_buttons.on_clicked(update)




#######################################################################################################################
################################SELECT_DATA_CHUNKS_USING_RECTANGULAR_OR_LASSO_SELECTION################################
#######################################################################################################################

xv, yv = np.meshgrid(kmos_ob.y_axis, kmos_ob.x_axis)
pix = np.vstack((xv.flatten(), yv.flatten())).T

def updateArray(array, indices):
	lin = np.arange(array.size)
	newArray = array.flatten()
	newArray[lin[indices]] = 1
	return newArray.reshape(array.shape)
	
def onselect(verts):
    global array_las, pix, ind
    p = matplotlib_path.Path(verts)
    ind = p.contains_points(pix, radius=1)
    array_las = updateArray(kmos_ob.updated_im_data, ind)
    if (str2 == 'las'):
        ax.patches = []
        line2 = ax.imshow(array_las, cmap=str(cmap_type_preferred), aspect='equal', origin='lower', alpha=0.1)
    fig.canvas.draw()

#Defining the selection function
def line_select_callback(eclick, erelease):
	global str1, str2
	global x1, y1, x2, y2
	#global data_selected_region1
	global idx_xaxis_min, idx_yaxis_min, idx_xaxis_max, idx_yaxis_max
	y1, x1 = eclick.xdata, eclick.ydata
	y2, x2 = erelease.xdata, erelease.ydata
	idx_yaxis_min = int(np.searchsorted(kmos_ob.x_axis, min(x1, x2)))
	idx_yaxis_max = int(np.searchsorted(kmos_ob.x_axis, max(x1, x2)))
	idx_xaxis_min = int(np.searchsorted(kmos_ob.y_axis, min(y1, y2)))
	idx_xaxis_max = int(np.searchsorted(kmos_ob.y_axis, max(y1, y2)))
	data_selected_region1 = []
	rect_region = []
	# Create a Rectangle patch
	heigth_rect_region = idx_yaxis_max - idx_yaxis_min
	width_rect_region = idx_xaxis_max - idx_xaxis_min
	rect1 = patches.Rectangle((idx_xaxis_min,idx_yaxis_min),width_rect_region,heigth_rect_region,linewidth=1,edgecolor='y',facecolor='white', alpha=0.7)
	data_selected_region1 = np.append(data_selected_region1, kmos_ob.updated_im_data[idx_yaxis_min:idx_yaxis_max,idx_xaxis_min:idx_xaxis_max])
	if (str1 == 'add' and str2 == 'delas'):
		# Add the patch to the Axes
		ax.patches = []
		ax.add_patch(rect1)
	elif (str1 == 'rem' and str2 == 'delas'):
		# Remove the patch to the Axes
		ax.patches = []
	else:
		print ("Function Inactive.....")
		sys.stdout.flush()
	fig.canvas.draw()

def toggle_selector(event):
	global str1, str2
	print(' Key pressed.')
	if event.key in ['T', 't'] and toggle_selector.RS.active:
		print(' RectangleSelector deactivated.')
		sys.stdout.flush()
		str2 = 'delas'
		toggle_selector.RS.set_active(False)
	if event.key in ['Y', 'y'] and not toggle_selector.RS.active:
		print(' RectangleSelector activated.')
		sys.stdout.flush()
		str2 = 'delas'
		toggle_selector.RS.set_active(True)
	if event.key in ['H', 'h'] and toggle_selector.RS.active:
		print('Add function activated')
		sys.stdout.flush()
		str1 = 'add'
		str2 = 'delas'
		toggle_selector.RS.set_active(True)
	if event.key in ['J', 'j'] and toggle_selector.RS.active:
		print('Remove function activated')
		sys.stdout.flush()
		str1 = 'rem'
		str2 = 'delas'
		toggle_selector.RS.set_active(True)
	if event.key in ['U', 'u']:
		print('lasso function activated')
		sys.stdout.flush()
		str2 = 'las'
		toggle_selector.RS.set_active(False)
	else:
		str2 = 'delas'
toggle_selector.RS = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
plt.connect('key_press_event', toggle_selector)


#######################################################################################################################
################################SELECT_DATA_CHUNKS_USING_RECTANGULAR_OR_LASSO_SELECTION################################
#######################################################################################################################


   


add_sky_button_axes = plt.axes([0.01, 0.5, 0.11, 0.04])
add_sky_button = Button(add_sky_button_axes, 'add_sky')
def func_add_sky(event):
	global str2, idx_xaxis_min, idx_yaxis_min, idx_xaxis_max, idx_yaxis_max, X_sky, Y_sky, loaded_sky_data
	if (str2 == 'delas'):
		x_array = np.arange(idx_xaxis_min, idx_xaxis_max)
		y_array = np.arange(idx_yaxis_min, idx_yaxis_max)
		X, Y = np.meshgrid(x_array, y_array)
		# Flatten and reshape the meshgrid arrays
		X_flat = X.flatten().reshape((-1, 1))
		Y_flat = Y.flatten().reshape((-1, 1))
		# Concatenate X and Y arrays
		combined_data = np.concatenate([X_flat, Y_flat], axis=1)
		# Custom formatting function
		def format_data(x):
			return str(x).strip('[]')
		# Format the combined data
		formatted_data = np.array2string(combined_data, separator='\t', formatter={'int': format_data})
		# Save the formatted data to a text file
		with open('tmp_sky.dat', 'a') as file:
			file.write(formatted_data[1:-1] + '\n')  # Exclude the enclosing square brackets and add a newline character
		print ('sky index saved to file')
	else:
		print ('Please activate rectangular selection.')
add_sky_button.on_clicked(func_add_sky)

load_sky_button_axes = plt.axes([0.01, 0.55, 0.11, 0.04])
load_sky_button = Button(load_sky_button_axes, 'load_sky')
def func_load_sky(event):
	global loaded_sky_data
	# Load the sky data
	loaded_sky_data = bf.load_two_column_data_from_file('tmp_sky.dat')
	updated_sky_array = np.zeros_like(kmos_ob.updated_im_data)
	updated_sky_array[loaded_sky_data[:,1].astype(np.int32), loaded_sky_data[:,0].astype(np.int32)] = kmos_ob.updated_im_data[loaded_sky_data[:,1].astype(np.int32), loaded_sky_data[:,0].astype(np.int32)]
	ax.patches = []
	line2 = ax.imshow(updated_sky_array, cmap=str(cmap_type_preferred), aspect='equal', origin='lower', alpha=0.1)
	fig.canvas.draw()
	print ('sky loaded...')
load_sky_button.on_clicked(func_load_sky)


remove_sky_button_axes = plt.axes([0.01, 0.6, 0.11, 0.04])
remove_sky_button = Button(remove_sky_button_axes, 'rem_sky')
def func_remove_sky(event):
	global loaded_sky_data, updated_kmos_cube, im_cl1
	# Load the sky data
	assert os.path.exists(str('tmp_sky.dat')), f"File: {str('tmp_sky.dat')} not found..."
	loaded_sky_data = bf.load_two_column_data_from_file('tmp_sky.dat')
	sky_data_array_cube = kmos_ob.data_cube[:, loaded_sky_data[:,1].astype(np.int32), loaded_sky_data[:,0].astype(np.int32)]
	sky_data_array_1d = np.nanmedian(sky_data_array_cube, axis=(1))
	kmos_ob.spatial_smoothing_factor = int(sdata_smooth.val)
	kmos_ob.binning_type = str(radio_binning_type_buttons.value_selected)
	kmos_ob.display_type = str(radio_display_type_buttons.value_selected)
	kmos_ob.sky_spectra = sky_data_array_1d
	kmos_ob.prepare_3d_cube()
	kmos_ob.prepare_2d_image()
	ax.cla()
	im_cl1.remove()
	if (kmos_ob.display_type=='log'):
		im1 = ax.imshow(np.log10(kmos_ob.updated_im_data), cmap=str(cmap_type_preferred), aspect='equal', origin='lower')
	else:
		im1 = ax.imshow(kmos_ob.updated_im_data, cmap=str(cmap_type_preferred), aspect='equal', origin='lower')
	im_cl1 = bf.add_colorbar_lin(im1)
	fig.canvas.draw()
remove_sky_button.on_clicked(func_remove_sky)

make_gal_im_ax = plt.axes([0.01, 0.75, 0.11, 0.04])
button_make_gal_im = Button(make_gal_im_ax, 'im_gal')
def func_select_make_gal_im(event):
	print ('Creating figure for summation of selected spectra')
	assert os.path.exists(str('set_gal_spec.dat')), f"File: {str('set_gal_spec.dat')} not found..."
	loaded_gal_spec_idx = list(np.loadtxt('set_gal_spec.dat').astype(np.int32))
	gal_im = np.nansum(kmos_ob.sky_reduced_data_cube[loaded_gal_spec_idx, :, :], axis=0)
	kmos_wcs = kmos_ob.wcs_celestial
	fig2 = plt.figure(3, figsize=[10,8])
	#ax2_save = fig2.add_subplot(111, projection=kmos_wcs)
	ax2_save = fig2.add_subplot(111)
	im_save = ax2_save.imshow(gal_im, origin='lower', cmap='viridis')
	ax2_save.set_title('test.pdf', fontsize=20, fontweight='bold')
	im2_cl_save = bf.add_colorbar_lin(im_save)
	fig2.savefig('test.pdf', dpi=100)
	plt.close(fig2)
	print ('Figure test.pdf Saved...')
button_make_gal_im.on_clicked(func_select_make_gal_im)


save_gal_im_ax = plt.axes([0.01, 0.68, 0.11, 0.04])
button_save_gal_im = Button(save_gal_im_ax, 'save_gal')
def func_select_save_gal_im(event):
	print ('Saving figure for summation of selected spectra')
	assert os.path.exists(str('set_gal_spec.dat')), f"File: {str('set_gal_spec.dat')} not found..."
	loaded_gal_spec_idx = list(np.loadtxt('set_gal_spec.dat').astype(np.int32))
	gal_cube = kmos_ob.sky_reduced_data_cube[loaded_gal_spec_idx, :, :]
	gal_err_cube = kmos_ob.data_err_cube[loaded_gal_spec_idx, :, :]**2
	gal_im = np.nansum(kmos_ob.sky_reduced_data_cube[loaded_gal_spec_idx, :, :], axis=0)
	gal_err_im = np.sqrt(np.nansum(kmos_ob.sky_reduced_data_cube[loaded_gal_spec_idx, :, :], axis=0))
	filename_new = 'test.fits'
	kmos_ob.make_image_fits(tmp_filename, filename_new, gal_im, gal_err_im)
	print ('File saved')
button_save_gal_im.on_clicked(func_select_save_gal_im)













##########################################SELECTED_1D_SPECTRA#########################################

select_new_region_ax = plt.axes([0.010, 0.41, 0.11, 0.03])
button_select_new = Button(select_new_region_ax, 'spec')
def select_new_region_ax(event):
	kmos_ob.binning_type = str(radio_binning_type_buttons.value_selected)
	print (f'Aggregating {kmos_ob.binning_type} of selected spaxels...')
	grouped_1d_data_1, grouped_1d_error_1 = retrieve_grouped_data(kmos_ob.data_cube, kmos_ob.data_err_cube, kmos_ob.binning_type)
	ssw.creating_additional_figure_new(kmos_ob.wave_obs, grouped_1d_data_1, grouped_1d_error_1, kmos_ob.redshift)
button_select_new.on_clicked(select_new_region_ax)

##########################################SELECTED_1D_SPECTRA#########################################




lasso = LassoSelector(ax, onselect)
plt.show()























quit()
