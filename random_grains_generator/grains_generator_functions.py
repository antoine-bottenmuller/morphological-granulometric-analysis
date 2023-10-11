# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 2023
@author: Antoine BOTTENMULLER
"""



## Libraries

import numpy as np
import cv2 as cv
from scipy import ndimage
from skimage.draw import polygon
from numba import jit

from utils import normalized, gradient_magnitude, disk





#%% Function to determine the actual circle best fitting with the corresponding structure



def get_lse_circle(cloud):
    """
    
    Parameters
    ----------
    cloud : Numpy array of size n*2
        List of all the 2D points in the cloud.

    Returns
    -------
    p_c : Tuple of size 2
        Central point of the circle.
    r : float64
        Radius of the circle.
    
    """
    n = cloud.shape[0]
    if n>2:
        gravity = np.mean(cloud, axis=0)
        cloudy = cloud-gravity #in the center-of-gravity coordinate system
        A = np.sum(cloudy[:,0]**2)
        B = np.sum(cloudy[:,1]**2)
        C = np.sum(cloudy[:,0]*cloudy[:,1])
        U = np.sum(cloudy[:,0]**3) + np.sum(cloudy[:,0]*cloudy[:,1]**2)
        V = np.sum(cloudy[:,1]**3) + np.sum(cloudy[:,1]*cloudy[:,0]**2)
        det_M = A*B-C**2
        if det_M!=0:
            x_p = (B*U-C*V)/det_M/2
            y_p = (A*V-C*U)/det_M/2
            r = np.sqrt( x_p**2 + y_p**2 + (A+B)/n )
            x = x_p + gravity[0]
            y = y_p + gravity[1]
            return (x,y) , r
        else:
            r = (A+B)/n
            return tuple(gravity) , r
    elif(n==2):
        gravity = np.mean(cloud, axis=0)
        vec_rad = (cloud[0] - gravity)**2
        r = np.sqrt(np.sum(vec_rad))
        return tuple(gravity) , r
    elif(n==1):
        gravity = cloud[0]
        return tuple(gravity) , 0
    else:
        print("Warning: No point in the cloud!")
        return (0,0) , 1



def get_true_corresponding_circle(struct):
    extended_length = len(struct) + 2
    extended_struct = np.zeros((extended_length, extended_length), dtype=struct.dtype)
    extended_struct[1:len(struct)+1,1:len(struct)+1] = struct
    struct_grad = gradient_magnitude(extended_struct)>0
    #show(struct_grad)
    x_center_window = struct.shape[0]/2+0.5
    y_center_window = struct.shape[1]/2+0.5
    true_center, true_radius = get_lse_circle(np.argwhere(struct_grad))
    diff_center = ( true_center[0]-x_center_window, true_center[1]-y_center_window )
    return true_radius, diff_center





#%% Function to create random disk or grain



def generate_random_grain(diameter, sigma=0.01, nb_int=20, data_type=bool):
    
    grad = np.random.normal(0, diameter*sigma, size=nb_int-1)
    
    x = np.arange(nb_int-1)
    xx, yy = np.meshgrid(x, x)
    triangle = 1*((xx+yy)>=nb_int-2)
    
    table = triangle * grad
    primitive = np.sum(table, axis=1)
    
    radi = np.ones([nb_int], dtype=np.float64) * (diameter-1)/2
    radi[1:] += primitive
    max_radius = int(np.ceil(np.max(radi)))
    
    angl = (np.arange(nb_int, dtype=np.float64)+0.5)/nb_int * np.pi # demi
    
    vect1 = np.array([np.cos(angl),np.sin(angl)], dtype=np.float64)
    vect1 *= radi
    
    vect2 = np.array([np.cos(angl),-np.sin(angl)], dtype=np.float64)
    vect2 *= radi
    
    center = np.array([max_radius, max_radius], dtype=int)
    
    coordinates = np.zeros([nb_int*2,2], dtype=np.float64)
    coordinates[:nb_int] = np.swapaxes(vect1,0,1)
    coordinates[nb_int:nb_int*2] = np.flip(np.swapaxes(vect2,0,1), axis=0)
    coordinates += center
    
    
    shape = (max_radius*2+1, max_radius*2+1)
    
    img = np.zeros(shape=shape)
    rr, cc = polygon(coordinates[:,0], coordinates[:,1], shape=shape)
    img[rr, cc] = 1
    
    
    angle = np.random.rand()*360
    
    
    rot_mat = cv.getRotationMatrix2D((int(center[0]),int(center[1])), angle, 1.0)
    rotated = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    
    # On recadre
    
    b_left = np.min(np.argwhere(np.sum(rotated,axis=0)>0))
    b_right = np.max(np.argwhere(np.sum(rotated,axis=0)>0))+1
    b_up = np.min(np.argwhere(np.sum(rotated,axis=1)>0))
    b_down = np.max(np.argwhere(np.sum(rotated,axis=1)>0))+1
    
    length = max(b_right-b_left,b_down-b_up)
    
    d_y = int((length - b_right + b_left)/2)
    d_x = int((length - b_down + b_up)/2)
    
    out = np.zeros([length,length])
    out[d_x:d_x+b_down-b_up, d_y:d_y+b_right-b_left] = rotated[b_up:b_down,b_left:b_right]
    
    
    series = np.zeros([nb_int], dtype = np.float64)
    series[0] = angle
    series[1:] = grad
    
    
    return out.astype(data_type), series



def get_grain(diameter, series, data_type=bool):
    
    nb_int = len(series)
    angle = series[0]
    grad = series[1:]
    
    x = np.arange(nb_int-1)
    xx, yy = np.meshgrid(x, x)
    triangle = 1*((xx+yy)>=nb_int-2)
    
    table = triangle * grad
    primitive = np.sum(table, axis=1)
    
    radi = np.ones([nb_int], dtype=np.float64) * (diameter-1)/2
    radi[1:] += primitive
    max_radius = int(np.ceil(np.max(radi)))
    
    angl = (np.arange(nb_int, dtype=np.float64)+0.5)/nb_int * np.pi # demi
    
    vect1 = np.array([np.cos(angl),np.sin(angl)], dtype=np.float64)
    vect1 *= radi
    
    vect2 = np.array([np.cos(angl),-np.sin(angl)], dtype=np.float64)
    vect2 *= radi
    
    center = np.array([max_radius, max_radius], dtype=int)
    
    coordinates = np.zeros([nb_int*2,2], dtype=np.float64)
    coordinates[:nb_int] = np.swapaxes(vect1,0,1)
    coordinates[nb_int:nb_int*2] = np.flip(np.swapaxes(vect2,0,1), axis=0)
    coordinates += center
    
    
    shape = (max_radius*2+1, max_radius*2+1)
    
    img = np.zeros(shape=shape)
    rr, cc = polygon(coordinates[:,0], coordinates[:,1], shape=shape)
    img[rr, cc] = 1
    
    
    rot_mat = cv.getRotationMatrix2D((int(center[0]),int(center[1])), angle, 1.0)
    rotated = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    
    # On recadre
    
    b_left = np.min(np.argwhere(np.sum(rotated,axis=0)>0))
    b_right = np.max(np.argwhere(np.sum(rotated,axis=0)>0))+1
    b_up = np.min(np.argwhere(np.sum(rotated,axis=1)>0))
    b_down = np.max(np.argwhere(np.sum(rotated,axis=1)>0))+1
    
    length = max(b_right-b_left,b_down-b_up)
    
    d_y = int((length - b_right + b_left)/2)
    d_x = int((length - b_down + b_up)/2)
    
    out = np.zeros([length,length])
    out[d_x:d_x+b_down-b_up, d_y:d_y+b_right-b_left] = rotated[b_up:b_down,b_left:b_right]
    
    return out.astype(data_type)





#%% Secondary functions for main function



def distance_disk(diameter, grad, power, light_dir, data_type=np.float64) :
    # defines a circular structuring element with its diameter
    sup_diameter = diameter*(1.2 + 0.2*np.sqrt(np.sum(light_dir**2))) #??? 0.5
    radius = (sup_diameter -1)/2
    x = np.arange(sup_diameter)-radius
    xx, yy = np.meshgrid(x, x)
    distance = xx**2 + yy**2
    powered_dis = distance** power
    #show(powered_dis,out=1)
    add_x = int( (sup_diameter-diameter)/2 * (1 - light_dir[0]) )
    add_y = int( (sup_diameter-diameter)/2 * (1 - light_dir[1]) )
    cropped_dis = powered_dis[add_x:add_x+diameter,add_y:add_y+diameter]
    #show(cropped_dis,out=1)
    adapted_dis = normalized(cropped_dis) * grad + (1-grad)
    return adapted_dis.astype(data_type)


def get_light_direction(light_direction):
    if(type(light_direction)==type(None)):
        ang = np.random.rand()*np.pi*2
        proj_up = np.cos( np.random.rand()*np.pi/2 )
        light_dir = np.array([np.cos(ang),np.sin(ang)], dtype=np.float64)*proj_up
        return light_dir
    else:
        return np.array(light_direction, dtype=np.float64)


def adapt_coord(infos_map, center, d_center, radius, struct):
    
    diameter = len(struct)
    
    inf_0 = int(np.round(center[0]))-int(diameter/2)
    sup_0 = int(np.round(center[0]))+int(np.ceil(diameter/2))
    inf_1 = int(np.round(center[1]))-int(diameter/2)
    sup_1 = int(np.round(center[1]))+int(np.ceil(diameter/2))
    
    if( inf_0>=0 and sup_0<=infos_map.shape[0] and inf_1>=0 and sup_1<=infos_map.shape[1] ):
        
        part = infos_map[inf_0:sup_0, inf_1:sup_1, 3] * struct
        max_part = np.max(part)
        
        if(max_part==0):
            return center
        
        else:
            max_part_coord = np.argwhere(part==max_part)[0]+np.array([inf_0,inf_1])
            ball_radius = infos_map[max_part_coord[0],max_part_coord[1],0]
            ball_center = infos_map[max_part_coord[0],max_part_coord[1],1:3]
            
            c = ( center[0] + d_center[0] , center[1] + d_center[1] )
            
            dis = np.sqrt(np.sum(( ball_center - c )**2)) + 0.0000000001
            sum_radius = ball_radius + radius + 3 #??? +3
            vec = ( c - ball_center )/dis*sum_radius
            
            new_center = ball_center + vec - d_center
            
            """
            uu = np.argwhere(part==max_part)[0]
            oo = infos_map[inf_0:sup_0,inf_1:sup_1,3].copy()
            oo[uu[0],uu[1]] = [np.max(oo),np.max(oo),np.max(oo),np.max(oo)]
            show(normalized(oo),out=0)
            im = infos_map[inf_0:sup_0,inf_1:sup_1,3].copy()
            im[uu[0],uu[1]] = np.max(im)*2
            show(normalized(im),out=0)
            """
            return new_center
    else:
        return center


def get_gray_value(i, nb_disks):
    gray_value = ( ( ((i+1)/nb_disks-0.5)*2 ) +1)/2
    return gray_value


def get_random_radius(normal_laws):
    if(len(normal_laws.shape)==2): # normal laws
        if(normal_laws.shape[1]==3): # sum of normal laws
            levels = normal_laws[:,2]
            levels_size = np.sum(levels)
            value_law = np.random.rand()*levels_size
            idx_law = 0
            while(np.sum(levels[:idx_law+1])<value_law):
                idx_law += 1
            mean, std = normal_laws[idx_law,:2]
            rad = np.random.normal(mean, std)
        else: # sum of normal laws with same occurence
            value_law = np.random.rand()
            idx_law = 0
            while( (idx_law+1)/normal_laws.shape[0] < value_law ):
                idx_law += 1
            mean, std = normal_laws[idx_law]
            rad = np.random.normal(mean, std)
    else: # log-normal law
        mean, std = normal_laws
        rad = np.random.lognormal(mean, std)
    return max( rad , 2 ) #??? min radius is 2


def get_center(i, nb_disks, infos_map, image_dims):
    height, length = image_dims
    if i>nb_disks*4/5:
        d_img = int(min(height,length)/96)
        dil = ndimage.binary_dilation(infos_map[:,:,3].copy()>0,structure=np.ones([3,3]))
        edt = ndimage.distance_transform_edt(1-dil)
        maxima = (ndimage.maximum_filter(edt,size=2*d_img)==edt)*(edt>1)
        maxima[:d_img+1,:] = False
        maxima[:,:d_img+1] = False
        maxima[maxima.shape[0]-d_img:,:] = False
        maxima[:,maxima.shape[1]-d_img:] = False
        m = np.argwhere(maxima)
        if len(m)>0:
            v = edt[m[:,0],m[:,1]]
            s = np.sort(v)
            rd_idx = len(s) * np.random.rand()**(1/5) #??? 1/5
            val = s[int(rd_idx)]
            coord = np.argwhere( (edt*maxima)==val )
            if len(coord)>0:
                return coord[0].astype(np.float64)
            else:
                return (np.random.rand(2)*[height,length]).astype(np.float64)
        else:
            return (np.random.rand(2)*[height,length]).astype(np.float64)
    else:
        return (np.random.rand(2)*[height,length]).astype(np.float64)


def get_textured_structure(struct, gray_level, light_dir):
    norm = np.sqrt(np.sum(light_dir**2))
    prop_light = (0.8 + np.sqrt(gray_level)*0.2)*0.9 * ( 0.8 + 0.2*norm )
    power_reflets = 1+2*norm
    gradient_texture = distance_disk(len(struct), prop_light, power_reflets, light_dir)
    texture = gradient_texture * gray_level**1.4 #??? 1.4
    return texture * struct


def fuzzied_and_noised_image(image, infos_map, nb_disks):
    
    img = normalized(image)
    
    full_disks = infos_map[:,:,3]>0
    #show(full_disks,out=0)
    borders = gradient_magnitude(ndimage.gaussian_filter(infos_map[:,:,3], 3 ))>0
    #show(borders,out=0)
    
    # adding noises
    img += ndimage.gaussian_filter( np.random.normal(0,0.56,size=img.shape) , 4.2 ) * img * ( ( 1*full_disks - 1*borders )>0 )
    img += ndimage.gaussian_filter( np.random.normal(0,0.18,size=img.shape) , 0.1 ) * img * ( ( 1*full_disks - 1*borders )>0 )
    
    # adding gaussian filter
    nb_layers = 4 #??? 4
    max_fluzz = 5 #??? 5
    #show(img,"0",out=1)
    for i in range(nb_layers-1):
        threshold = nb_disks/nb_layers*(i+1)
        protected = ndimage.binary_dilation(infos_map[:,:,3]>threshold, structure=disk(max_fluzz))
        #show(protected,str(i+1),out=1)
        out_prote = img*protected
        flou = ndimage.gaussian_filter(img, (nb_layers-1-i)/(nb_layers-1)*max_fluzz) * (1-1*protected)
        img = flou + out_prote
        #show(flou,str(i+1),out=1)
        #show(out_prote,str(i+1),out=1)
        #show(img,str(i+1),out=1)
    out = ndimage.gaussian_filter(img, 2.1) # global filter
    
    # adding global noise
    out += ndimage.gaussian_filter( np.abs( np.random.normal(0,0.06,size=out.shape) ) , 1.0 )
    
    return normalized(out)





#%% Functions to draw disks



def draw_grain_center(image,struct,center):
    diameter = len(struct)
    inf_0 = int(np.round(center[0]))-int(diameter/2)
    sup_0 = int(np.round(center[0]))+int(np.ceil(diameter/2))
    inf_1 = int(np.round(center[1]))-int(diameter/2)
    sup_1 = int(np.round(center[1]))+int(np.ceil(diameter/2))
    image[inf_0:sup_0,inf_1:sup_1] -= image[inf_0:sup_0,inf_1:sup_1]*(struct>0)
    image[inf_0:sup_0,inf_1:sup_1] += struct


@jit(nopython=True)
def draw_gray_pxl_2D(x,y,i,j,r_left,r_right,image,struct):
    if( struct[i-x+r_left,j-y+r_right] > 0 ):
        image[i,j] = struct[i-x+r_left,j-y+r_right]

@jit(nopython=True)
def draw_gray_pxl_3D(x,y,i,j,r_left,r_right,image,struct):
    if( struct[i-x+r_left,j-y+r_right,0] > 0 ):
        image[i,j] = struct[i-x+r_left,j-y+r_right]

def draw_grain_border(image,struct,center):
    x,y = np.array(center, dtype=np.int32)
    r_left = int(len(struct)/2)
    r_right = int(np.ceil(len(struct)/2))
    if(len(struct.shape)==2):
        for i in range(max(0,x-r_left),min(image.shape[0]-1,x+r_right)):
            for j in range(max(0,y-r_left),min(image.shape[1]-1,y+r_right)):
                draw_gray_pxl_2D(x,y,i,j,r_left,r_right,image,struct)
    else:
        for i in range(max(0,x-r_left),min(image.shape[0]-1,x+r_right)):
            for j in range(max(0,y-r_left),min(image.shape[1]-1,y+r_right)):
                draw_gray_pxl_3D(x,y,i,j,r_left,r_right,image,struct)


def draw_disk(image,struct,center):
    try:
        draw_grain_center(image,struct,center)
    except:
        try:
            draw_grain_border(image,struct,center)
        except:
            print("Warning: Can not draw disk!")



def draw_shadow_center(image,struct,center):
    diameter = len(struct)
    inf_0 = int(np.round(center[0]))-int(diameter/2)
    sup_0 = int(np.round(center[0]))+int(np.ceil(diameter/2))
    inf_1 = int(np.round(center[1]))-int(diameter/2)
    sup_1 = int(np.round(center[1]))+int(np.ceil(diameter/2))
    image[inf_0:sup_0,inf_1:sup_1] *= struct


@jit(nopython=True)
def draw_shadow_pxl(x,y,i,j,r_left,r_right,image,struct):
    image[i,j] *= struct[i-x+r_left,j-y+r_right]

def draw_shadow_border(image,struct,center):
    x,y = np.array(center, dtype=np.int32)
    r_left = int(len(struct)/2)
    r_right = int(np.ceil(len(struct)/2))
    for i in range(max(0,x-r_left),min(image.shape[0]-1,x+r_right)):
        for j in range(max(0,y-r_left),min(image.shape[1]-1,y+r_right)):
            draw_shadow_pxl(x,y,i,j,r_left,r_right,image,struct)



def draw_shadow(out,struct,center,light_dir):
    diam = len(struct)
    
    grad = ( 10 - np.sqrt(np.sum(light_dir**2)) )/10 * 0.8 #??? 4 , 0.6
    power = 2/(np.sqrt(np.sum(light_dir**2))+1) #??? 2/ , +1
    shadow = distance_disk(diam, grad, power, -light_dir)
    
    edges = ( 1*struct - 1*ndimage.binary_erosion(struct) )>0
    border = edges*shadow #np.ones([diam,diam]) + (shadow - 1) * edges
    s_max = np.max(border) #??? min ???
    s_min = np.min(shadow)
    shadow = (shadow-s_min)/(s_max-s_min)*(1-s_min)+ s_min
    
    shadow = shadow*struct#*(shadow<=1)
    shadow += 1*(shadow==0)
    
    ratio_resize = 1.3 + 0.3 * np.sqrt(np.sum(light_dir**2)) #??? 1.6
    shadow = cv.resize(shadow, 
                       dsize=(int(diam*ratio_resize),
                              int(diam*ratio_resize)),
                       interpolation=cv.INTER_CUBIC)
    
    #show(shadow/np.max(shadow),out=1) # on floute ensuite!
    
    pourcent = 0.14 #??? 0.14
    new_diam = len(shadow)
    out_diam = int(new_diam*(1+2*pourcent))
    d_diam = int(new_diam*pourcent)
    flou = np.ones([out_diam,out_diam], dtype=np.float64)
    flou[d_diam:d_diam+new_diam,d_diam:d_diam+new_diam] = shadow
    shadow = ndimage.convolve(flou, np.ones([d_diam+1,d_diam+1]))
    shadow/=np.max(shadow)
    
    #show(shadow,out=1)
    
    centre = center + light_dir*diam*0.28 #??? *0.28
    try:
        draw_shadow_center(out,shadow,centre)
    except:
        try:
            draw_shadow_border(out,shadow,centre)
        except:
            print("Warning: Can not draw disk!")





#%% Main function for random grains generation



def generate_random_grains(image_dims, nb_avg_grains, normal_laws, light_direction=None):
    """
    image_dims: tuple of size 2
    
    nb_circle: int

    normal_laws = [normal_law_1, normal_law_2, normal_law_3, ...]
    normal_law_i = [mean, std, level]
    
    light_direction: tuple of size 2
    if light_direction==None, then light_dir is randomly generated
    norm(light_direction) <= 1
    """
    
    nb_disks = int( nb_avg_grains*(1+(np.random.rand()-0.5)*2/5) )
    print("Number of grains: {}".format(nb_disks))
    
    std_sigma_grain, nb_corners_grain = 0.01, 20
    
    image = np.zeros(shape=image_dims, dtype=np.float64)
    L_RADIUS = np.zeros(nb_disks, dtype=np.float64)
    L_CENTER = np.zeros((nb_disks, 2), dtype=np.float64)
    SERIES = np.zeros((nb_disks, nb_corners_grain), dtype=np.float64)
    
    infos_map = np.zeros(shape=(image_dims[0],image_dims[1],4), dtype=np.float64)
    
    # light source direction (from source to balls)
    light_dir = get_light_direction(light_direction)
    print("Light direction: {}".format(light_dir))
    
    for i in range(nb_disks):
        
        # global gray level
        gray_level = get_gray_value(i, nb_disks)
        
        # radius of the grain
        rad = get_random_radius(normal_laws)
        
        # center of the grain
        x, y = get_center(i, nb_disks, infos_map, image_dims)
        
        # building struct and serie
        diameter = int(rad*2+1)
        struct, serie = generate_random_grain(diameter, std_sigma_grain, nb_corners_grain)
        
        # true corresponding circle centre and radius in the struct
        true_rad, delta_center = get_true_corresponding_circle(struct)
        
        # adaptating center coordinates
        x, y = adapt_coord(infos_map,(x,y),delta_center,true_rad,struct)
        
        # true appearing center coordinates (not for drawing! only as info!)
        true_x, true_y = x + delta_center[0], y + delta_center[1]
        
        # get texture with current configuration
        grain = get_textured_structure(struct, gray_level, light_dir)
        
        # drawing shadow
        draw_shadow(image,struct,(x,y),light_dir)
        
        # adding radius, center and i+1 to infos_map
        disc = np.array([struct*true_rad,struct*true_x,struct*true_y,struct*(i+1)])
        disc = np.swapaxes(disc,0,1)
        disc = np.swapaxes(disc,1,2)
        draw_disk(infos_map,disc,(x,y))
        
        # outputs : drawing grain and adding data to arrays
        draw_disk(image,grain,(x,y))
        L_RADIUS[i] = rad
        L_CENTER[i] = [x,y]
        SERIES[i] = serie
    
    image -= image*(image>1) # just in case there are values >1 (due to random)
    infos_map = np.concatenate((infos_map,np.expand_dims(image.copy(),2)),axis=2)
    
    image = fuzzied_and_noised_image(image, infos_map, nb_disks)
    
    return image, infos_map, L_RADIUS, L_CENTER, SERIES





#%% Function to create image with pre-given disks



def regenerate_grains(image_dims, l_radius, l_center, series=None, light_direction=None):
    """
    image_dims: tuple of size 2
    
    l_radius: NumPy array of size n

    l_center: NumPy array of size n * 2
    
    series: NumPy array of size n * m
    if series==None, then disks are generated as structures
    series[:,0] is angle
    series[:1:] is grad array
    
    light_direction: tuple of size 2
    if light_direction==None, then light_dir is randomly generated
    norm(light_direction) <= 1
    """
    
    nb_disks = len( l_radius )
    
    image = np.zeros(shape=image_dims, dtype=np.float64)
    infos_map = np.zeros(shape=(image_dims[0],image_dims[1],4), dtype=np.float64)
    
    # light source direction (from source to balls)
    light_dir = get_light_direction(light_direction)
    
    for i in range(nb_disks):
        
        # global gray level
        gray_level = get_gray_value(i, nb_disks)
        
        # radius and center of the grain
        rad = l_radius[i]
        x,y = l_center[i]
        
        # building struct and serie
        diameter = int(rad*2+1)
        if(type(series)==type(None)):
            struct = disk(diameter)
        else:
            struct = get_grain(diameter,series[i])
        
        # true corresponding circle centre and radius in the struct
        true_rad, (delta_x, delta_y) = get_true_corresponding_circle(struct)
        
        # true appearing center coordinates (not for drawing! only as info!)
        true_x, true_y = x + delta_x, y + delta_y
        
        # get texture with current configuration
        grain = get_textured_structure(struct, gray_level, light_dir)
        
        # adding shadow
        draw_shadow(image,struct,(x,y),light_dir)
        
        # adding radius, center and i+1 to infos_map
        disc = np.array([struct*true_rad,struct*true_x,struct*true_y,struct*(i+1)])
        disc = np.swapaxes(disc,0,1)
        disc = np.swapaxes(disc,1,2)
        draw_disk(infos_map,disc,(x,y))
        
        # outputs : drawing circle
        draw_disk(image,grain,(x,y))
    
    image -= image*(image>1) # just in case there are values >1 (due to random)
    infos_map = np.concatenate((infos_map,np.expand_dims(image.copy(),2)),axis=2)
    
    image = fuzzied_and_noised_image(image, infos_map, nb_disks)
    
    return image, infos_map





#%% Fonctions to get radius and center of visible circles only from infos_map


# Slow method but returns the same order of grains than the input (on indices)

def get_visible_circles_only_slow(infos_map, nb_min_pixels=3):
    int_indices_map = np.round(infos_map[:,:,3]).astype(np.uint32)
    max_idx = np.max(int_indices_map)
    idx_visible = []
    rad_visible = []
    cen_visible = []
    for i in range(1,max_idx+1):
        argues = np.argwhere(int_indices_map==i)
        if(len(argues)>nb_min_pixels):
            med = np.median(infos_map[argues[:,0],argues[:,1],:3], axis=0)
            idx_visible.append(i-1)
            rad_visible.append(med[0])
            cen_visible.append(med[1:])
    return np.array(idx_visible, dtype=np.uint32), np.array(rad_visible, dtype=np.float64), np.array(cen_visible, dtype=np.float64)


# Fast method but the output circles are sorted making the input order not preserved

def get_visible_circles_only_fast(infos_map, nb_min_pixels=10**2, nb_min_radius=3):
    radia, idx_radia, cnt_radia = np.unique(infos_map[:,:,0], return_index=True, return_counts=True)
    selected_data = ( cnt_radia > nb_min_pixels ) * ( radia > nb_min_radius )
    idx = idx_radia[selected_data]
    coor_radia = np.swapaxes(np.unravel_index(idx, shape=infos_map.shape[:2]),0,1)
    true_radia = radia[selected_data]
    true_centr = infos_map[coor_radia[:,0],coor_radia[:,1],1:3]
    return coor_radia, true_radia, true_centr


