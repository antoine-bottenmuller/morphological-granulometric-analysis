# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 2022
@author: Antoine BOTTENMULLER
"""



## Libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage
from scipy.interpolate import make_interp_spline
from sklearn.neighbors import KernelDensity
from pathlib import Path
from matplotlib.figure import Figure
from numba import jit





#%% Classical functions in image processing

# Function to import and crop image

def import_image(image_path:str, crop_image:bool=None) -> np.ndarray:
    img = cv.imread(image_path)
    while len(img.shape) >= 3:
        img = img.mean(2)
    img_shape = img.shape
    if crop_image is None:
        crop_image = [[-1,-1],[-1,-1]]
    for i in range(2):
        for j in range(2):
            if crop_image[i][j] < 0:
                crop_image[i][j] = j * ( (1-i) * img_shape[0] + i * img_shape[1] )
    img = img[min(min(img_shape[0],crop_image[0][1]),crop_image[0][0]):max(min(img_shape[0],crop_image[0][1]),crop_image[0][0]),
              min(min(img_shape[1],crop_image[1][1]),crop_image[1][0]):max(min(img_shape[1],crop_image[1][1]),crop_image[1][0])]
    return img


## Function to show an image or a set of images

def show(image, title="", out=False):
    if(type(image) == np.ndarray):
        if(len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2]==3)):
            # Convertion into adapted image
            if(np.min(image)<0 or np.max(image)>255):
                if(np.max(image) > 0):
                    converted_image = np.array( image/np.max(image) * 255 , dtype = np.uint8 )
                else:
                    converted_image = np.zeros( shape = image.shape , dtype = np.uint8 )
            elif(np.max(image) <= 1):
                converted_image = np.array( image * 255 , dtype = np.uint8 )
            else:
                converted_image = np.array( image , dtype = np.uint8 )
            # Showing image OUT or IN console
            if(out):
                cv.imshow(title,converted_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            else:
                if(len(image.shape) == 2):
                    plt.imshow(cv.cvtColor(converted_image,cv.COLOR_GRAY2RGB))
                else:
                    plt.imshow(converted_image)
                plt.title(title)
                plt.show()
        elif(len(image.shape) == 3):
            if(image.shape[2] > 3):
                for i in range(image.shape[0]):
                    show(image[i], title+" | {}".format(i+1), out)
            else:
                show(np.sum(image,axis=2)/image.shape[2], title, out)
        elif(len(image.shape) > 3):
            for i in range(image.shape[0]):
                show(image[i], title+" | {}".format(i+1), out)
        else:
            print("Error: image or array of images must have more than 1 dimension!")
    elif(type(image) == list):
        array_image = np.array(image) # Warning returned if the convertion is not possible
        if(array_image.dtype != object):
            show(array_image, title, out)
        else:
            print("Error: the given list can not be converted into numpy ndarray!")
    else:
        print("Error: image object must be a numpy ndarray or a list of numpy ndarrays!")


## Nomalization of array

def normalized(array, output_range=(0.0,1.0)):
    if(type(array)==np.ndarray):
        try:
            a_min = np.min(array)
            a_max = np.max(array)
            if(a_min!=a_max):
                norm = np.array( (array-a_min)/(a_max-a_min) * (1+254*(array.dtype==np.uint8)) , dtype = array.dtype ) * (output_range[1]-output_range[0]) + output_range[0]
            else:
                norm = np.zeros(shape = array.shape, dtype = array.dtype) + output_range[0]
            return norm
        except:
            print("Warning: can not process input array (data type must be real)!")
            return array
    elif(type(array)==list or type(array)==tuple):
        try:
            arraylist = np.array(array, dtype=type(array[0]))
            return normalized(arraylist)
        except:
            print("Warning: can not convert input list or tuple into array (wrong data type or homogeneity)!")
            return array
    else:
        print("Warning: wrong type for input object (must be array, list or tuple)!")
        return array


## Borders detection

def gradient_magnitude(image, norm=1):
    if(type(image)==np.ndarray and (type(norm)==int or type(norm)==float)):
        Jx = ndimage.sobel(image.astype(np.float64), axis=1) # Horizontal gradient (X)
        Jy = ndimage.sobel(image.astype(np.float64), axis=0) # Vertical gradient (Y)
        J = ( np.abs(Jx)**norm + np.abs(Jy)**norm )**(1/norm)
        gradient = normalized(J)
        return gradient
    else:
        print("Error: wrong types for input objects!")
        return None


## Reduce size of image

def resized(image,ratio):
    if(type(image)==np.ndarray and (type(ratio)==int or type(ratio)==float)):
        if(ratio>0):
            new_shape = np.array([image.shape[1]//ratio,image.shape[0]//ratio], dtype=np.uint32)
            new_image = cv.resize(image, new_shape, interpolation = cv.INTER_AREA)
            return new_image.astype(image.dtype)
        else:
            print("Error: 'ratio' must be >0 !")
            return None
    else:
        print("Error: Wrong input objects' type!")
        return None


# Classical disk

def disk(diameter, data_type=bool) :
    # defines a circular structuring element with its diameter
    radius = (diameter-1)/2
    x = np.arange(diameter)-radius
    xx, yy = np.meshgrid(x, x)
    distance = xx**2 + yy**2
    d = distance <= radius**2 + (radius-int(radius))**2
    return d.astype(data_type)



#%% Showing image with drawn circles

def draw_circle(image,center,radius,thickness=1,color=(0,1,0)):
    diameter_ext = int( 2*radius+1 + thickness )
    diameter_int = int( 2*radius+1 - thickness )
    circle = disk(diameter_ext, np.uint8)
    struct_int = disk(diameter_int, np.uint8)
    circle[int(thickness):diameter_ext-int(thickness),int(thickness):diameter_ext-int(thickness)] -= struct_int
    inf_0 = int(np.round(center[0]))-int(diameter_ext/2)
    sup_0 = int(np.round(center[0]))+int(np.ceil(diameter_ext/2))
    inf_1 = int(np.round(center[1]))-int(diameter_ext/2)
    sup_1 = int(np.round(center[1]))+int(np.ceil(diameter_ext/2))
    if(len(image.shape)==2):
        if(type(color)==int or type(color)==float):
            image[inf_0:sup_0,inf_1:sup_1] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1] + circle*color
        elif(type(color)==tuple or type(color)==list or type(color)==np.ndarray):
            image[inf_0:sup_0,inf_1:sup_1] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1] + circle*np.mean(color)
        else:
            print("WARNING: Wrong object type for 'color' parameter!")
    elif(len(image.shape)==3):
        if(type(color)==int or type(color)==float):
            image[inf_0:sup_0,inf_1:sup_1,0] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1,0] + circle*color
            image[inf_0:sup_0,inf_1:sup_1,1] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1,1] + circle*color
            image[inf_0:sup_0,inf_1:sup_1,2] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1,2] + circle*color
        elif(type(color)==tuple or type(color)==list or type(color)==np.ndarray):
            image[inf_0:sup_0,inf_1:sup_1,0] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1,0] + circle*color[0]
            image[inf_0:sup_0,inf_1:sup_1,1] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1,1] + circle*color[1]
            image[inf_0:sup_0,inf_1:sup_1,2] = (1-circle)*image[inf_0:sup_0,inf_1:sup_1,2] + circle*color[2]
        else:
            print("WARNING: Wrong object type for 'color' parameter!")
    else:
        print("WARNING: Wrong array size for 'image' parameter!")

@jit(nopython=True)
def draw_circle_pixel(I,i,j,x,y,r,color):
    rayon = np.sqrt( (i-x)**2 + (j-y)**2 )
    if( int(rayon)==r ):
        I[i,j,0] = color[0]
        I[i,j,1] = color[1]
        I[i,j,2] = color[2]
@jit(nopython=True)
def draw_circle_border(I,center,r,color=(0,1,0)):
    x,y = int(center[0]), int(center[1])
    r = int(r)
    for i in range(max(0,x-r),min(I.shape[0]-1,x+r+1)):
        for j in range(max(0,y-r),min(I.shape[1]-1,y+r+1)):
            draw_circle_pixel(I,i,j,x,y,r,color)


def show_circles(image, lc, lr, color_circle=(0,1,0), color_center=(1,0,0), thickness=3, title="Circles", out=False):
    if(image.dtype==np.uint8):
        img = np.array( image/255 , dtype = np.float64 )
    elif(np.min(image)>=0 and np.max(image)<=1):
        img = image.astype(np.float64)
    elif(np.min(image)>=0):
        img = image.astype(np.float64)/np.max(image)
    else:
        img = normalized(image.astype(np.float64))
    h,l = img.shape
    nimg = np.zeros([h,l,3])
    nimg[:,:,0] += img
    nimg[:,:,1] += img
    nimg[:,:,2] += img
    for i in range(len(lc)):
        x,y = int(lc[i,0]),int(lc[i,1])
        if(x>=0 and x<h and y>=0 and y<l):
            nimg[x,y] = color_center # draw center
        try:
            draw_circle(nimg,lc[i],lr[i],thickness,color_circle) # draw circle
        except:
            try:
                for j in range(int(thickness)):
                    draw_circle_border(nimg,lc[i],lr[i]+j-int(thickness/2),color_circle) # draw circle
            except:
                print("WARNING: Cannot draw circle {}.".format(i))
    show(nimg,title,out)



#%% function to show or export histogram of distribution

def build_density_histogram(l_radius, nb_intervals=200, lim_x=None, lim_y=None, volume=False, line_curve=False, interpol=False, kernel_density=False, color=None, linestyle='solid', title=None, build_in_console=True, export=False, end=True, multi=1, label=None, line_width=1):
    
    L_RAD = np.array(np.sort(l_radius))
    if(type(lim_x)!=type(None)):
        L_RAD = L_RAD - (L_RAD - lim_x[1]*0.97) * (L_RAD>lim_x[1]*0.97) # limit sup
        L_RAD = L_RAD - (L_RAD - lim_x[0]*1.03) * (L_RAD<lim_x[0]*1.03) # limit inf
    
    if(volume):
        X_weight = 4/3*np.pi*L_RAD**3
    else:
        X_weight = np.ones([len(L_RAD)])
    
    if(type(lim_x)!=type(None)):
        limx = lim_x
    else:
        limx = [np.min(L_RAD), np.max(L_RAD)]
    band_width = (limx[1] - limx[0]) / nb_intervals
    
    if(kernel_density):
        X_edges = np.linspace(limx[0], limx[1], 1000)[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=band_width).fit(L_RAD[:, np.newaxis], sample_weight=X_weight)
        log_dens = kde.score_samples(X_edges)
        Y_out = np.exp(log_dens)
        X_edges = X_edges[:,0]
    else:
        Y_out, X_edges = np.histogram(L_RAD, nb_intervals, range=limx, weights=X_weight, density=True)
        X_edges = ( X_edges[:-1] + X_edges[1:] ) / 2
    
    if(interpol):
        if(kernel_density):
            X_edges = np.linspace(limx[0], limx[1], nb_intervals)
            lin = np.linspace(0, len(Y_out)-1, nb_intervals)
            lin_do , lin_up = lin.astype(np.uint32) , np.ceil(lin).astype(np.uint32)
            Y_out = Y_out[ lin_do ] * (1-lin+lin_do) + Y_out[ lin_up ] * (lin-lin_do)
        model = make_interp_spline(X_edges, Y_out)
        X_edges = np.linspace(limx[0], limx[1], 1000)
        Y_out = model(X_edges)
    
    
    Y_out *= multi
    
    
    if(type(color)==type(None)):
        c = (0,0,0)
    else:
        c = color
    
    if(build_in_console):
        
        if(line_curve):
            plt.plot(X_edges, Y_out, color=c, linestyle=linestyle, linewidth=line_width)
        else:
            plt.bar(X_edges, Y_out, width=band_width, color=c)
        
        plt.xlabel("radius (pxl)")
        plt.ylabel("{} (density)".format("number"*(not(volume))+"volume"*volume))
        
        plt.xlim(limx)
        plt.ylim(lim_y)
        
        if(type(title)==type(None)):
            plt.title("Particle size distribution")
        else:
            plt.title(title)
        if(type(end)==bool and end):
            plt.show()
        
    if(export):
        
        if(type(title)==type(None)):
            gtitle = "particle_size_distribution"
        else:
            gtitle = title
        
        if(type(end)==type(Figure())): # do not end
            ax = end.gca()
        else: # build and end
            fn = Path("~/{}.svg".format(gtitle)).expanduser()
            fg = Figure()
            ax = fg.gca()
        
        if(line_curve):
            ax.plot(X_edges, Y_out, color=c, linestyle=linestyle, label=label, linewidth=line_width)
        else:
            ax.bar(X_edges, Y_out, width=band_width, color=c)
        
        # legend
        if(type(label)!=type(None)):
            ax.legend()
        
        # here to set title of figure and of axis
        if(type(title)!=type(None)):
            ax.set_title(gtitle)
        ax.set_xlabel("radius (pxl)")
        ax.set_ylabel("{} (density)".format("number"*(not(volume))+"volume"*volume))
        
        # here to set xlim and ylim
        if(type(lim_x)!=type(None)):
            ax.set_xlim(limx[0],limx[1])
        if(type(lim_y)!=type(None)):
            ax.set_ylim(lim_y[0],lim_y[1])
        
        if(type(end)!=type(Figure())):
            fg.savefig(fn, bbox_inches='tight')
            print("File '{}' saved!".format("{}.svg".format(gtitle)))
        
    if(not(build_in_console or export)):
        print("WARNING: No action performed!")


