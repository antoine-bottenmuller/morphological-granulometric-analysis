#@title: Utility Functions
#@description: Set of basic functions for CAM algorithm
#@author: Antoine BOTTENMULLER
#@created: Dec 2022
#@updated: Jun 2024

import numpy as np
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Optional, Any
from scipy.interpolate import make_interp_spline
from sklearn.neighbors import KernelDensity
from matplotlib.figure import Figure
from pathlib import Path


# Import and crop image

def import_2D_image(image_path:str, crop_image:bool=None) -> np.ndarray:
    image = io.imread(image_path)
    while image.ndim >= 3:
        image = image.mean(2)
    img_shape = image.shape
    if crop_image is None:
        crop_image = [[-1,-1],[-1,-1]]
    for i in range(2):
        for j in range(2):
            if crop_image[i][j] < 0:
                crop_image[i][j] = j * ( (1-i) * img_shape[0] + i * img_shape[1] )
    image = image[min(min(img_shape[0],crop_image[0][1]),crop_image[0][0]):max(min(img_shape[0],crop_image[0][1]),crop_image[0][0]), 
                  min(min(img_shape[1],crop_image[1][1]),crop_image[1][0]):max(min(img_shape[1],crop_image[1][1]),crop_image[1][0])]
    return image


# Get an adapted image to show as uint8 in [|0,255|], regarding a minimum value (v_min) and a miximum one (v_max)

def get_show(img:np.ndarray, v_min:Optional[float]=None, v_max:Optional[float]=None) -> np.ndarray:
    out_dtype = np.uint8
    if img.dtype == out_dtype and v_min is None and v_max is None:
        return img.copy()
    if img.ndim < 1:
        raise ValueError(f"Array must be of dimension >0.")
    if v_min is not None or v_max is not None:
        img = img.copy()
    if img.ndim == 1:
        img = img[np.newaxis]
    elif not(img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)):
        x_show = tuple([int(s/2) for s in img.shape[:-2]])
        img = img[x_show]
    if v_min is None:
        v_min = img.min()
    else:
        img[img < v_min] = v_min
    if v_max is None:
        v_max = img.max()
    else:
        img[img > v_max] = v_max
    if v_min == v_max:
        return np.zeros(shape=img.shape, dtype=out_dtype)
    out = np.empty(shape=img.shape, dtype=out_dtype)
    return np.multiply(img-v_min, 255/(v_max-v_min), out=out, casting='unsafe')


# Show an image or a set of images

def show(image:np.ndarray, title:str="") -> None:
    if np.iterable(image):
        if type(image) is not np.ndarray:
            image = np.asarray(image)
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3):
            image = get_show(image)
            if image.ndim == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)
            plt.title(title)
            plt.show()
        elif image.ndim >= 3:
            for i in range(image.shape[0]):
                show(image[i], title+" | {}".format(i+1))
        else:
            print("WARNING: image must be of dimension >1!")
    else:
        print("WARNING: image must be iterable!")


# Nomalize array

def normalized(array:np.ndarray, output_range:tuple|None=None, output_dtype:np.dtype|None=None) -> np.ndarray:
    if type(array) is not np.ndarray:
        array = np.asarray(array)
    if output_dtype is None:
        output_dtype = array.dtype
    if output_range is None:
        if np.issubdtype(output_dtype, np.integer):
            output_range = (np.iinfo(output_dtype).min, np.iinfo(output_dtype).max)
        else:
            output_range = (0., 1.)
    a_min, a_max = array.min(), array.max()
    if a_min != a_max:
        mul = (output_range[1] - output_range[0]) / (a_max - a_min)
        return ((array - a_min) * mul + output_range[0]).astype(output_dtype)
    return np.full(array.shape, fill_value=output_range[0], dtype=output_dtype)


# Compute gradient

def sobel_norm(ndim:int) -> int:
    return 2**(2*ndim-1)

def sobel_weights(shape:tuple, axis:int, dtype:type=np.uint16) -> np.ndarray:
    ndim = len(shape)
    sobel_n = sobel_norm(ndim)
    sobel_w = np.full(shape, dtype(sobel_n/2), dtype=dtype)
    mid_grad = tuple(slice(int(i==axis), shape[i]-int(i==axis)) for i in range(ndim))
    sobel_w[mid_grad] = dtype(sobel_n)
    return sobel_w

def gradient(image:np.ndarray, axis:int|tuple|None=None, mode:int=3) -> np.ndarray:
    """
    Gradient computation mode:
    * 0 : computes literal img[1:]-img[:-1] on given axis, filled from 1 to n ;
    * 1 : computes literal img[1:]-img[:-1] on given axis, filled from 0 to n-1 ;
    * 2 : weighted Sobel filter, which averages gradient on a 3x3 weighted filter ;
    * 3+: numpy.gradient, which averages gradient on two grad neighbour pixels.
    """
    grad_mode = int(np.round(mode))
    if grad_mode == 0 or grad_mode == 1: # img[1:] - img[:-1]
        if np.issubdtype(type(axis), np.integer):
            grad = np.insert(image[tuple(slice(int(i==axis), image.shape[i]) for i in range(image.ndim))] - image[tuple(slice(0, image.shape[i]-int(i==axis)) for i in range(image.ndim))],obj=(image.shape[axis]-1)*grad_mode,values=0,axis=axis)
        else: # axis is iterable or None
            if axis is None: axis = np.arange(image.ndim)
            grad = np.asarray([np.insert(image[tuple(slice(int(i==ax), image.shape[i]) for i in range(image.ndim))] - image[tuple(slice(0, image.shape[i]-int(i==ax)) for i in range(image.ndim))],obj=(image.shape[ax]-1)*grad_mode,values=0,axis=ax) for ax in axis], dtype=image.dtype)
    elif grad_mode == 2: # Sobel
        if np.issubdtype(type(axis), np.integer):
            grad = ndimage.sobel(image, axis=axis)
            sobel_w = sobel_weights(image.shape, axis=axis)
            np.divide(grad, sobel_w, out=grad, casting='unsafe')
        else: # axis is iterable or None
            if axis is None: axis = np.arange(image.ndim)
            grad = []
            for ax in axis:
                gradx = ndimage.sobel(image, axis=ax)
                sobel_w = sobel_weights(image.shape, axis=ax)
                np.divide(gradx, sobel_w, out=gradx, casting='unsafe')
                del sobel_w
                grad.append(gradx)
            grad = np.asarray(grad, dtype=image.dtype)
    else: # numpy.gradent
        if np.issubdtype(type(axis), np.integer):
            grad = np.gradient(image, axis=axis)
        elif np.iterable(axis) and len(axis) == 1:
            grad = np.gradient(image, axis=axis)[np.newaxis]
        else: # axis is iterable of length !=1 or None
            grad = np.asarray(np.gradient(image, axis=axis), dtype=image.dtype)
    return grad

def gradient_magnitude(image:np.ndarray, norm:float=2) -> np.ndarray:
    grad = gradient(image)
    grad = np.sum(np.abs(grad)**norm, axis=0)**(1/(norm+(norm==0)))
    return grad


# Change size of image

def resized(image:np.ndarray, ratio:float|tuple, interp_order:int|None=None) -> np.ndarray:
    if interp_order is None:
        if np.issubdtype(image.dtype, np.floating):
            interp_order = 1
        else:
            interp_order = 0
    if np.iterable(ratio):
        if len(ratio) != image.ndim:
            raise ValueError(f"Parameter 'ratio', if iterable, must be of length image's dim. Given length: {len(ratio)} ; image dim: {image.ndim}.")
        if not( np.issubdtype(type(ratio[0]), np.integer) or np.issubdtype(type(ratio[0]), np.floating) ):
            raise ValueError(f"Parameter 'ratio', if iterable, must be of dtype 'floating' or 'integer'. Given datatype: {type(ratio[0])}.")
        if (np.asarray(ratio) == 1).prod() == 1:
            return image
        shape = image.shape
        ratio = [ ratio[i] * (ratio[i]*shape[i]>=1) + 1/shape[i] * (ratio[i]*shape[i]<1) for i in range(image.ndim) ]
    elif np.issubdtype(type(ratio), np.integer) or np.issubdtype(type(ratio), np.floating):
        if ratio <= 0:
            raise ValueError(f"Parameter 'ratio', if is float or int, must be greater than 0. Given value: {ratio}.")
        if ratio == 1:
            return image
        ratio = [ ratio * (ratio*s>=1) + 1/s * (ratio*s<1) for s in image.shape ]
    else:
        raise ValueError(f"Parameter 'ratio' must be either of type 'float', 'int', or iterable. Given type: {type(ratio)}")
    return ndimage.zoom(image, ratio, order=interp_order, mode="reflect")


# Classical ndim ball

def to_diameter(radius:int|float) -> int|float:
    return 2*radius+1

def to_radius(diameter:int|float) -> int|float:
    res = (diameter-1)/2
    if int(res) == res:
        return int(res)
    return res

def binary_ball(diameter:int|float, ndim:int=2, dtype:type=bool) -> np.ndarray:
    radius = to_radius(diameter)
    x = np.arange(diameter)-radius
    xs = [x]*ndim
    grids = np.meshgrid(*xs)
    distance = np.sum(np.asarray(grids)**2, axis=0)
    d = distance <= radius**2 + (radius-int(radius))**2
    if np.sum(d)==0 and np.prod(d.shape)>0:
        return np.ones(d.shape, dtype=dtype)
    return d.astype(dtype)


# Draw binary ball

def draw_binary_ball(image:np.ndarray, center:tuple[int], diameter:int|float, value:float=1) -> None:
    d = binary_ball(diameter, ndim=image.ndim, dtype=image.dtype)
    ctr = np.round(center).astype(int)
    radius = to_radius(diameter)
    avg_res = center - ctr > 0
    rad_inf = int(np.ceil(radius)) * ~avg_res + int(np.floor(radius)) * avg_res
    rad_sup = np.asarray(d.shape) - rad_inf
    a_inf = np.max([ctr - rad_inf, [0]*image.ndim], axis=0)
    a_sup = np.min([ctr + rad_sup, image.shape], axis=0)
    d_inf = a_inf - ctr + rad_inf
    d_sup = a_sup - ctr + rad_inf
    a_range = ()
    d_range = ()
    for i in range(image.ndim):
        a_range += (slice(a_inf[i], a_sup[i]),)
        d_range += (slice(d_inf[i], d_sup[i]),)
    image[a_range] = d[d_range] * value


# Compute ball volume and surface

def gamma(n:float) -> float:
    # n est un reel >=1 multiple de 1/2
    if int(np.mean(2*n)) == 2*int(np.mean(n)): # n est un entier
        m = int(np.mean(n-1))
        gam = (1+np.arange(m)).prod()
    else: # n n'est pas un entier
        m = int(np.mean(2*(n-1)))
        gam = (1+np.arange(m)).prod() * np.sqrt(np.pi) / ( 2 ** m * (1+np.arange(int(n-1))).prod() )
    return gam

def hypersphere_volume(radius:float, n:int) -> float:
    return np.pi**(n/2) * radius**n / gamma(n/2+1)

def hypersphere_surface(radius:float, n:int) -> float:
    return n / radius * hypersphere_volume(radius, n)


# Plot 2D image with circles

def draw_circle(image:np.ndarray, center:tuple, radius:float, thickness:float=1, color:tuple|float=(0,1,0)) -> None:
    ndim = len(center)
    if np.iterable(color) and image.shape[-1] != len(color):
        if image.ndim == ndim:
            image = np.tile(image, (len(color),)+(1,)*image.ndim).transpose(*tuple(np.arange(1, image.ndim+1)), 0)
        else:
            color = np.mean(color)
    thickness = int(np.round(thickness))
    diameter_ext = int(to_diameter(radius) + thickness)
    diameter_int = int(to_diameter(radius) - thickness)
    circle = binary_ball(diameter_ext, ndim=ndim, dtype=np.uint8)
    ballin = binary_ball(diameter_int, ndim=ndim, dtype=np.uint8)
    circle[(slice(thickness, diameter_ext-thickness),)*ndim] -= ballin
    if np.sum(circle) > 0:
        where_circle = np.argwhere(circle) + np.round(center).astype(int) - int(np.round(radius + thickness/2))
        out_of_bounds = ((where_circle < 0) + (where_circle >= image.shape[:ndim])).sum(axis=1, dtype=bool)
        if np.prod(out_of_bounds) == 0:
            where_circle = where_circle[~out_of_bounds]
            image[tuple(where_circle.T)] = color

def show_circles(
        image:np.ndarray, 
        lc:list, lr:list, 
        circle_color:tuple|float=(0,1,0), 
        center_color:tuple|float=(1,0,0), 
        border_color:tuple|float=(0,0,0), 
        circle_thickness:float=4, 
        center_thickness:float=2, 
        border_thickness:float=2, 
        title:str="Circles"
    ) -> None:

    if len(lc) != len(lr):
        raise ValueError("Lists 'lr' and 'lc' must be of same length.")
    
    nb_channels = 0
    if np.iterable(circle_color) and np.iterable(center_color):
        if len(circle_color) != len(center_color):
            raise ValueError("Tuples 'circle_color' and 'center_color' must be of same length.")
        nb_channels = len(circle_color)
    elif np.iterable(circle_color):
        nb_channels = len(circle_color)
    elif np.iterable(center_color):
        nb_channels = len(center_color)
    
    nlc = len(lc)
    
    if nlc > 0:

        ndim = len(lc[0])

        if nb_channels > 0 and image.shape[-1] != nb_channels:
            if image.ndim == ndim:
                image = np.tile(image, (nb_channels,)+(1,)*image.ndim).transpose(*tuple(np.arange(1, image.ndim+1)), 0)
            else:
                circle_color = np.mean(circle_color)
                center_color = np.mean(center_color)
        
        image = get_show(image, v_max=image.max()*1.1)

        circle_color = np.uint8(np.float64(circle_color) * 255)
        center_color = np.uint8(np.float64(center_color) * 255)

        circle_thickness = max(int(np.round(circle_thickness)), 1)
        border_thickness = max(int(np.round(border_thickness)), int(border_thickness>0))
        half_c_thickness = int(np.round(center_thickness/2))

        lc = np.round(lc).astype(int)

        for i in range(nlc):

            # Draw center of circle
            if center_thickness > 0:

                ball = binary_ball(diameter=to_diameter(half_c_thickness), ndim=ndim)
                where_ball = np.argwhere(ball) + np.round(lc[i]).astype(int) - half_c_thickness
                out_of_bounds = ((where_ball < 0) + (where_ball >= image.shape[:ndim])).sum(axis=1, dtype=bool)

                if np.prod(out_of_bounds) == 0:
                    where_ball = where_ball[~out_of_bounds]
                    image[tuple(where_ball.T)] = center_color # draw center
            
            # Draw border of circle
            if border_thickness > 0:
                draw_circle(image, lc[i], lr[i], circle_thickness+2*border_thickness, border_color) # draw border

            # Draw circle itself
            draw_circle(image, lc[i], lr[i], circle_thickness, circle_color) # draw circle

    show(image, title)


# Plot or export histogram of particle size distribution

def build_density_histogram(
        l_radius:list, nb_intervals:int=200, 
        lim_x:tuple=None, lim_y:tuple=None, 
        volume:bool=False, line_curve:bool=False, 
        interpol:bool=False, kernel_density:bool=False, 
        color:tuple=None, linestyle:str='solid', title:str=None, 
        build_in_console:bool=True, export:bool=False, end:bool|Figure=True, 
        proportion:float=1, label:bool|Any=None, line_width:float=1
    ) -> None:
    
    L_RAD = np.asarray(np.sort(l_radius))
    if lim_x is not None:
        L_RAD = L_RAD - (L_RAD - lim_x[1]*0.97) * (L_RAD>lim_x[1]*0.97) # limit sup
        L_RAD = L_RAD - (L_RAD - lim_x[0]*1.03) * (L_RAD<lim_x[0]*1.03) # limit inf
    
    if volume:
        X_weight = 4/3*np.pi*L_RAD**3
    else:
        X_weight = np.ones([len(L_RAD)])
    
    if lim_x is not None:
        limx = lim_x
    else:
        limx = [np.min(L_RAD), np.max(L_RAD)]
    band_width = (limx[1] - limx[0]) / nb_intervals
    
    if kernel_density:
        X_edges = np.linspace(limx[0], limx[1], 1000)[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=band_width).fit(L_RAD[:, np.newaxis], sample_weight=X_weight)
        log_dens = kde.score_samples(X_edges)
        Y_out = np.exp(log_dens)
        X_edges = X_edges[:,0]
    else:
        Y_out, X_edges = np.histogram(L_RAD, nb_intervals, range=limx, weights=X_weight, density=True)
        X_edges = ( X_edges[:-1] + X_edges[1:] ) / 2
    
    if interpol:
        if kernel_density:
            X_edges = np.linspace(limx[0], limx[1], nb_intervals)
            lin = np.linspace(0, len(Y_out)-1, nb_intervals)
            lin_do , lin_up = lin.astype(np.uint32) , np.ceil(lin).astype(np.uint32)
            Y_out = Y_out[ lin_do ] * (1-lin+lin_do) + Y_out[ lin_up ] * (lin-lin_do)
        model = make_interp_spline(X_edges, Y_out)
        X_edges = np.linspace(limx[0], limx[1], 1000)
        Y_out = model(X_edges)
    
    Y_out *= proportion
    
    if color is None:
        c = (0,0,0)
    else:
        c = color
    
    if build_in_console:
        
        if line_curve:
            plt.plot(X_edges, Y_out, color=c, linestyle=linestyle, linewidth=line_width)
        else:
            plt.bar(X_edges, Y_out, width=band_width, color=c)
        
        plt.xlabel("radius (pxl)")
        plt.ylabel("{} (density)".format("number"*(not(volume))+"volume"*volume))
        
        plt.xlim(limx)
        plt.ylim(lim_y)
        
        if title is None:
            plt.title("Particle size distribution")
        else:
            plt.title(title)
        if type(end)==bool and end:
            plt.show()
        
    if export:
        
        if title is None:
            gtitle = "particle_size_distribution"
        else:
            gtitle = title
        
        if type(end)==type(Figure()): # do not end
            ax = end.gca()
        else: # build and end
            fn = Path("./{}.svg".format(gtitle)).expanduser()
            fg = Figure()
            ax = fg.gca()
        
        if line_curve:
            ax.plot(X_edges, Y_out, color=c, linestyle=linestyle, label=label, linewidth=line_width)
        else:
            ax.bar(X_edges, Y_out, width=band_width, color=c)
        
        # legend
        if label is not None:
            ax.legend()
        
        # here to set title of figure and of axis
        if title is not None:
            ax.set_title(gtitle)
        ax.set_xlabel("radius (pxl)")
        ax.set_ylabel("{} (density)".format("number"*(not(volume))+"volume"*volume))
        
        # here to set xlim and ylim
        if limx is not None:
            ax.set_xlim(limx[0],limx[1])
        if lim_y is not None:
            ax.set_ylim(lim_y[0],lim_y[1])
        
        if type(end)!=type(Figure()):
            fg.savefig(fn, bbox_inches='tight')
            print("File '{}' saved!".format("{}.svg".format(gtitle)))
        
    if not(build_in_console or export):
        print("WARNING: no action performed!")

