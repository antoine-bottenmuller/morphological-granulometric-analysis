# Demo for Curvature Analysis Method (CAM) 
# which allows to detect circular shapes 
# in 2D grayscale images of any size 
# through the analysis of edge curvature

# Libraries

from utils import import_2D_image, show, show_circles, build_density_histogram
from CAM import CAM_circles_2D


#%% import an image from given folder and crop it

num_image = 1

folder_path = "../sample/"
image_path = folder_path + str(num_image) + ".bmp"

image = import_2D_image(image_path, crop_image=[[-1,960],[-1,-1]])

show(image)


#%% test: detect cercles with CAM in a real SEM image of maltodextrin particles

# please set appropriate parameters (see function description)
l_centers, l_radii = CAM_circles_2D(
    image, 
    resolution = 0.5, 
    cleaning_power = 1.0, 
    curvature_sensitivity = 1.0, 
    circle_merging_power = 1.0, 
    circle_accuracy_power = 1.0, 
    min_radius_ratio = 0.01, 
    max_radius_ratio = 0.50
)

show_circles(image, l_centers, l_radii)


#%% build a quick histogram of resulting Particle Size Distribution (PSD)

# figures can be vectorized and exported by setting "export" argument to "True"
build_density_histogram(l_radii, nb_intervals=20, 
                        lim_x=None, lim_y=None, 
                        volume=False, 
                        line_curve=True, interpol=False, kernel_density=True, #export=True, 
                        title="Particle Size Distribution (PSD) with CAM")

