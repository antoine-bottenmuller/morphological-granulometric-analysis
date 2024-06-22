#@title: Curvature Analysis Method (CAM)
#@description: Algorithm for 2D circular shape detection based on the analysis of edge curvature
#@author: Antoine BOTTENMULLER
#@created: Dec 2022
#@updated: Jun 2024

import cv2 as cv
import numpy as np
from typing import Tuple
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
from numba import jit

from utils import normalized, gradient_magnitude, resized, binary_ball


# %% ==============================================================================================================
# Color visualisation functions
# =================================================================================================================

# "_v" function from colorsys, but generalized to Numpy arrays
def __v(m1:np.ndarray|float, m2:np.ndarray|float, hue:np.ndarray|float) -> np.ndarray|float:
    hue = hue % 1.0
    one_sixth = np.bool_(hue < 1/6)
    one_halff = np.bool_(hue < 0.5)
    two_third = np.bool_(hue < 2/3)
    bool_1 =  one_sixth
    bool_2 = ~one_sixth * one_halff
    bool_3 = ~one_halff * two_third
    bool_4 = ~two_third
    float_1 = m1 + 6 * (m2-m1) * hue
    float_2 = m2
    float_3 = m1 + 6 * (m2-m1) * (2/3-hue)
    float_4 = m1
    return bool_1 * float_1 + bool_2 * float_2 + bool_3 * float_3 + bool_4 * float_4

# "hls_to_rgb" function from colorsys, but generalized to Numpy arrays
def _hls_to_rgb(h:np.ndarray|float, l:np.ndarray|float, s:np.ndarray|float) -> np.ndarray|tuple:
    halfL = np.bool_(l <= 0.5)
    dwL = l * (1 + s)
    upL = l + s - l * s
    m2 = halfL * dwL + ~halfL * upL
    m1 = 2 * l - m2
    c_red = __v(m1, m2, h+1/3)
    c_gre = __v(m1, m2, h)
    c_blu = __v(m1, m2, h-1/3)
    if type(c_gre) is np.ndarray:
        out = np.empty(shape=(3,)+c_gre.shape, dtype=c_gre.dtype)
        out[0] = c_red
        out[1] = c_gre
        out[2] = c_blu
        return out.transpose(tuple(np.arange(1,out.ndim))+(0,))
    return c_red, c_gre, c_blu

# Color image representing angles of best-fitting line vectors for each pixel on image support
def color_orientation_map(field2D:np.ndarray) -> np.ndarray:
    """
    Converts 2D vector field image into color image representing vector angles.\n
    Parameter:
    * field2D: 2D vector field image, with shape (m,n,2).\n
    Returns color image, Hue representing vector angles (modulo pi), with shape (m,n,3).
    """
    vecX = field2D[:,:,0]
    vecY = field2D[:,:,1]
    nullX = np.isclose(vecX, 0)
    angle = ~nullX * ( np.arctan( vecY / (vecX + nullX) ) / np.pi + 0.5 )
    return _hls_to_rgb(angle, 0.5, 1)


# %% ==============================================================================================================
# Fonctions to generate linear minimum MSE map
# =================================================================================================================

# Function to compute vector and its error which minimizes mean weighted distance to points in space
@jit(nopython=True)
def _min_L_E(area:np.ndarray, center:tuple, _xx:np.ndarray, _yy:np.ndarray) -> Tuple[tuple, float]:
    """
    Computes 2D vector attached to given center, which minimizes mean weighted squared distance (MSE) to points in area. 
    Returns tuple (vector, MSE).
    """
    xo = _xx[:area.shape[0],:area.shape[1]] - center[0]
    yo = _yy[:area.shape[0],:area.shape[1]] - center[1]
    area = area / area.sum()
    A = np.sum(area*xo**2)
    B = -2*np.sum(area*xo*yo)
    C = np.sum(area*yo**2)
    if np.isclose(B,0):
        if A > C: # a0 = 0
            return (1,0), C
        elif C > A: # a0 = +/- infini
            return (0,1), A
        return (0,0), C # Undefined!
    del_4 = (C-A)**2 + B**2
    a0 = (A - C - np.sqrt(del_4)) / B
    norm = np.sqrt(1+a0**2)
    L = (1/norm, a0/norm)
    E = (A*a0**2 + B*a0 + C)/(1+a0**2)
    return L, E

# Function to compute, and add in maps arrays, vector and its error which minimizes mean weighted distance to points in space
@jit(nopython=True)
def _update_maps(image:np.ndarray, vector_map:np.ndarray, merror_map:np.ndarray, structure:np.ndarray, full_error:float, i:int, j:int, _xx:np.ndarray, _yy:np.ndarray) -> None:
    """
    Computes and adds in 'vector_map' and 'merror_map' the 2D vector and its mean error, 
    which best fits points in area defined by given structure and centered on (i,j).
    """
    a_l, b_l = structure.shape

    # Define window borders
    a_min = max(i-int((a_l-1)/2),0)
    a_max = min(i+int(a_l/2)+1,image.shape[0])
    b_min = max(j-int((b_l-1)/2),0)
    b_max = min(j+int(b_l/2)+1,image.shape[1])
    
    # Select area defined by given structure and centered on (i,j) in 2D image
    area = image[a_min:a_max, b_min:b_max]
    area_s1, area_s2 = area.shape
    if area_s1 == a_l and area_s2 == b_l:
        area = area * structure
    
    # Check if normalization of gray values in selected area can be made (max!=min)
    area_m1, area_m2 = area.min(), area.max()
    if area_m1 != area_m2:
        
        # Normalise gray values in selected area
        area = (area - area_m1)/(area_m2 - area_m1) # Normalized area
        
        # Compute best-fitting vector and its corresponding MSE in selected area
        center = ((area_s1-1)/2, (area_s2-1)/2)
        d, e = _min_L_E(area, center, _xx, _yy)
        
    else: # Orientation is undefined, and dispersion is then maximal!
        d, e = (0,0), full_error
    
    # Add computed data in arrays
    vector_map[i,j] = d
    merror_map[i,j] = e

# Function to compute maps of vectors and errors which minimize mean weighted distance to line in associated area
def fit_lines(image:np.ndarray, width:int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes vector map and mean error map through the minimization of weighted distance to line in local space.\n
    Parameters:
    * image: 2D ndarray of the gradient magnitude image ;
    * width: diameter of the disk neighbourhood window.\n
    Returns tuple (vector_map, merror_map).
    """
    h,l = image.shape
    
    full_error = ((width-1)/2)**2 / 4 # error on flat structure, for a 2D disk: r**2 / 4
    
    vector_map = np.empty((h,l,2), dtype=np.float64) # initialization of 2D best-fitting vector map
    merror_map = np.empty((h,l)  , dtype=np.float64) # initialization of error map (floats)
    
    structure = binary_ball(width, ndim=image.ndim, dtype=image.dtype) # structure definition: a disk!

    _yy, _xx = np.meshgrid(*(np.arange(width),)*2) # 2D coordinate grids
    
    # Iteration on each pixel: the vector going through (i,j) and minimizing MSE is computed
    for i in range(h):
        for j in range(l):
            _update_maps(image, vector_map, merror_map, structure, full_error, i, j, _xx, _yy)
        
    return vector_map, merror_map


# %% ==============================================================================================================
# Fonctions to clean binary images on criteria
# =================================================================================================================

@jit(nopython=True)
def __add_label(out:np.ndarray, labels:np.ndarray, index:int, max_size:float) -> None:
    if np.sum(labels==index) > max_size:
        out[:] += labels==index

# Function to remove connected objects in binary image on a minimum size criteria
def clean_by_objectSize(binary_image:np.ndarray, max_size:float=10) -> np.ndarray:
    """
    Removes connected objects of size < max_size. Returns new cleaned image.
    """
    cleaned_image = np.zeros(shape=binary_image.shape, dtype=bool)
    labels, nb_labels = ndimage.label(binary_image, structure=np.ones((3,3)))
    for i in range(1, nb_labels+1):
        __add_label(cleaned_image, labels, i, max_size)
    return cleaned_image

def __remove_points(image:np.ndarray, int_width:int, ext_half_width:int) -> None:
    struct_white = binary_ball(int_width, ndim=image.ndim, dtype=np.uint8)
    struct_black = binary_ball(int_width+2*ext_half_width, ndim=image.ndim, dtype=np.uint8)
    struct_black[ext_half_width:int_width+ext_half_width, ext_half_width:int_width+ext_half_width] -= struct_white
    white = ndimage.convolve(image.astype(np.uint32), struct_white)  > 0
    black = ndimage.convolve(image.astype(np.uint32), struct_black) == 0
    check = ndimage.maximum_filter(white*black, footprint=struct_white)
    np.multiply(image, 1-check, out=image, casting='unsafe')

# Function to remove objects in binary image which do not have a minimum disk inside or are not in a maximum one
def clean_by_convolutionCheck(binary_image:np.ndarray, max_length:int, nb_inter:int, ext_half_width:int=2) -> np.ndarray:
    """
    Removes objects which do not respect a minimum size of disk inside and a maximum one outside. Returns new cleaned image.
    """
    cleaned_image = binary_image.copy()
    inter_length = int(np.ceil(max_length/nb_inter))
    start_length = inter_length + int(max_length%inter_length)
    for i in range(nb_inter):
        __remove_points(cleaned_image, start_length+i*inter_length, ext_half_width)
    return cleaned_image


# %% ==============================================================================================================
# Fonctions to compute binary image, from 'merror_map' (using K-means)
# =================================================================================================================

# Function to compute local difference map
def local_difference_map(image:np.ndarray, width:int) -> np.ndarray:
    """
    Computes normalized difference between image and local extrema in a window of size width.
    """
    min_map = ndimage.minimum_filter(image, size=width)
    max_map = ndimage.maximum_filter(image, size=width)
    diff = max_map - min_map
    null_diff = diff == 0
    distance_map = (image - min_map)/(diff + null_diff)
    return distance_map

# K-means algorithm for Ss ou M1 (K = 3)
def k_means(image:np.ndarray, k:int, i:int) -> np.ndarray:
    """
    K-means clustering algorithm from sklearn.\n
    Parameters:
    * image: 2D grayscale image ;
    * k: number of classes ;
    * i: index of target class.\n
    Returns binary image of target class.
    """
    if i<0:
        print("Warning: i<0, evaluation at i=0.")
        return k_means(image, k, 0)
    elif i>=k:
        print("Warning: i>=k, evaluation at i=k-1.")
        return k_means(image, k, k-1)
    data = np.expand_dims(image.flatten(), axis=1)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto').fit(data) # Kmeans object
    labels = kmeans.labels_
    labels.shape = image.shape # reshape labels
    centers = kmeans.cluster_centers_[:,0] # values of centers
    sorted_centers = np.sort(centers)
    center = sorted_centers[i]
    idx = np.argwhere(centers==center)[0,0]
    binary_image = labels==idx
    return binary_image


# %% ==============================================================================================================
# Fonctions to compute binary image, from 'vector_map' (using vector STD)
# =================================================================================================================

@jit(nopython=True)
def __set_std(std:np.ndarray, vector_map:np.ndarray, m:int, n:int, i:int, j:int, a:int, b:int) -> None:
    a_min = max(i-int((a-1)/2),0)
    a_max = min(i+int(a/2)+1,m)
    b_min = max(j-int((b-1)/2),0)
    b_max = min(j+int(b/2)+1,n)
    ref = vector_map[i,j]
    win = vector_map[a_min:a_max,b_min:b_max]
    s = np.sum( np.abs( win[:,:,0]*ref[1] - win[:,:,1]*ref[0] ) )
    std[i,j] = s

# Function to compute local STD in a window of given shape
def vec_std(vector_map:np.ndarray, w_shape:tuple=(2,2)) -> np.ndarray:
    """
    Computes local STD image in neighbourhood window of shape 'w_shape' from 2D vector image 'vector_map' of shape (m,n,2).
    """
    a,b = w_shape
    m,n = vector_map.shape[:2]
    std = np.empty(shape=(m,n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            __set_std(std, vector_map, m, n, i, j, a, b)
    return std


# %% ==============================================================================================================
# Fonctions to remove skeleton parts (corners, bends) from binary skeleton image
# =================================================================================================================

kernel_4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
kernel_8 = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)

# Function to associate a type label to each pixel of a skeleton image
def point_types(skeleton:np.ndarray, eight_neighbours:bool=True) -> np.ndarray:
    """
    Associates to each pixel in the skeleton image a label corresponding to its type:
    * ==0 <=> background point
    * ==1 <=> isolated point
    * ==2 <=> end-line point
    * ==3 <=> middle-line point
    * >=4 <=> intersection point\n
    Parameters:
    * skeleton: 2D binary image (bool) of edge skeleton ;
    * eight_neighbours: boolean for eight neighbour window (True) or four (False).\n
    Returns integer image of associated labels.
    """
    kernel = eight_neighbours * kernel_8 + (1-eight_neighbours) * kernel_4
    nb_neighbours = ndimage.convolve(skeleton.astype(np.uint8), kernel)
    return (nb_neighbours * skeleton).astype(np.uint8)

# Function to remove best corner-like pixels on each connected object in skeleton image
def remove_corner_pixels(skeleton:np.ndarray, blockSize:int) -> None:
    """
    Removes corner-like pixels from skeleton image.
    Parameters:
    * skeleton: 2D binary image (bool) of edge skeleton ;
    * blockSize: size of neighbourhood window for Harris corner detection.
    """
    # We use Harris corner detector from OpenCV
    corner_map = cv.cornerHarris(skeleton.astype(np.float32), blockSize, 3, 0.04)
    #show(normalized(corner_map),"corner map",1)
    labels, nb_labels = ndimage.label(skeleton, structure=np.ones(shape=(3,3)))
    end_pt = (point_types(skeleton)==2) * labels
    for i in range(1,nb_labels+1):
        if np.sum(end_pt==i)==0:
            obj = (labels==i) * corner_map
            coord = np.argwhere(obj==np.max(obj))[0]
            skeleton[coord[0]-1:coord[0]+2,coord[1]-1:coord[1]+2] = False

# Function to compute relative orientation between two continuous point paths (for GET_BENDS)
def get_relative_vector(l1:np.ndarray, l2:np.ndarray, prop:float=0.5) -> Tuple[float, float]:
    """
    Computes from two 2D point paths, l1 and l2, their relative orientation (d_angle, d_orien).
    """
    
    d1a = np.sum(l1[:int(len(l1)*prop)], axis=0).astype(np.float64)
    n1a = np.sqrt(d1a[0]**2+d1a[1]**2)
    d1a /= ( n1a + (n1a==0) * 1e-10 )
    
    d1b = np.sum(l1[int(len(l1)*prop):], axis=0).astype(np.float64)
    n1b = np.sqrt(d1b[0]**2+d1b[1]**2)
    d1b /= ( n1b + (n1b==0) * 1e-10 )
    
    v1x = np.sum(d1a*d1b)
    v1y = d1a[0]*d1b[1]-d1a[1]*d1b[0]
    v1 = np.array([v1x,v1y]) #v1
    
    d2a = np.sum(l2[:int(len(l2)*prop)], axis=0).astype(np.float64)
    n2a = np.sqrt(d2a[0]**2+d2a[1]**2)
    d2a /= ( n2a + (n2a==0) * 1e-10 )
    
    d2b = np.sum(l2[int(len(l2)*prop):], axis=0).astype(np.float64)
    n2b = np.sqrt(d2b[0]**2+d2b[1]**2)
    d2b /= ( n2b + (n2b==0) * 1e-10 )
    
    v2x = np.sum(d2a*d2b)
    v2y = d2a[0]*d2b[1]-d2a[1]*d2b[0]
    v2 = np.array([v2x,v2y]) #v2
    
    """
    v = ( vx , vy )
    = ( cos1*cos2+sin1*sin2 , cos1*sin2-cos2*sin1 )
    """
    """
    norm(v) = (cos1*cos2+sin1*sin2)^2+(cos1*sin2-cos2*sin1)^2 =
    
    = cos1^2 * cos2^2 + 2*cos1*cos2*sin1*sin2 + sin1^2 * sin2^2
    + cos1^2 * sin2^2 - 2*cos1*sin2*cos2*sin1 + cos2^2 * sin1^2
    
    = cos1^2 * cos2^2 + sin1^2 * sin2^2 + cos1^2 * sin2^2 + cos2^2 * sin1^2
    
    = cos1^2 * ( cos2^2 + sin2^2 ) + sin1^2 * ( sin2^2 + cos2^2 )
    
    = cos1^2 + sin1^2
    
    = 1 {OK!}
    """
    
    proj = max(min( np.sum(v1*v2) , 1), -1)
    d_angle = np.arccos(proj)
    
    proj_right_1 = d1b[0]*d1a[1]-d1b[1]*d1a[0]
    proj_right_2 = d2b[0]*d2a[1]-d2b[1]*d2a[0]
    d_orien = proj_right_2 * np.sign(proj_right_1)
    
    return d_angle, d_orien

# Function to get 'bends' in point paths through the analysis of path's curvature irregularities
def get_bends(skeleton:np.ndarray, nb_controllers:int=6) -> list:
    """
    Detects curvature irregularities ('bends') in connected point paths.\n
    Parameters:
    * skeleton: 2D binary image (bool) of edge skeleton ;
    * nb_controllers: size of moving control segment in point path.\n
    SET MANUALLY HYPERPARAMETERS INSIDE THE FUNCTION!!\n
    Returns position list of the detected curvature irregularities.
    """

    ###########################
    ##### HYPERPARAMETERS #####

    born_inf_coins = 0.7 #??? between 0 and 1 (included)
    born_sup_coins = 0.8 #??? between born_inf_coins and 2 (included)
    
    prop_reference = 0.5  #??? 0.5 for reference: is it good???
    
    d_angle_lim = np.pi/5 #??? maximum difference of angle: pi/5 is it good?
    d_orien_lim = np.cos(np.pi/6) #??? max error when wrong orientation: pi/6?
    
    ###########################
    ##### FUNCTION CORPUS #####

    L_COINS = [] # output: detected bend positions
    
    start_adding_current = min(int(nb_controllers*born_inf_coins), nb_controllers-1)
    stop_adding_current = min(int(nb_controllers*born_sup_coins)+1, nb_controllers)
    stop_adding_next = min(max(int(nb_controllers*(born_sup_coins-1)), 0), nb_controllers-1)
    
    S = skeleton.copy()
    B = point_types(S)==2
    
    AKA = np.argwhere(B) # S or B
    while len(AKA) > 0: # c is coordinate
        #print(len(AKA))
        
        k = AKA[0]
        B[k[0],k[1]] = False
        
        a_ = []
        
        a_ctr = []
        last_controle_coordinate = []
        
        first_complete = True
        encountered = False
        
        l_coins = []
        
        while True:
            S[k[0],k[1]] = False
            window = S[k[0]-1:k[0]+2,k[1]-1:k[1]+2]
            IKI = np.argwhere(window>0)
            if len(IKI) == 0:
                B[k[0],k[1]] = False
                break
            else:
                dp = IKI[0]-1
                c = k + dp
                
                #controle
                if len(a_ctr) < nb_controllers:
                    
                    a_ctr.append( dp )
                    last_controle_coordinate.append(c)
                    
                    if encountered: # here to add next pixels to l_coins
                        if len(a_ctr) < stop_adding_next:
                            l_coins.append(k)
                        else:
                            L_COINS.append(l_coins)
                            a_ctr = [] # controle
                            last_controle_coordinate = []
                            l_coins = []
                            c = k # stay on this pixel
                            encountered = False
                else:
                    if first_complete: # here only once per segment!! (when a_ctr is completed for the first time)
                        
                        a_ = a_ctr.copy()
                        
                        a_ctr.pop(0)
                        last_controle_coordinate.pop(0)
                        
                        a_ctr.append( dp )
                        last_controle_coordinate.append(c)
                        
                        first_complete = False
                        
                    else:
                        #checking well running
                        
                        d_angle, d_orien = get_relative_vector(a_,a_ctr,prop_reference)
                        centre_de_meme_direction = ( d_angle <= d_angle_lim ) & (d_orien >= -d_orien_lim )
                        
                        if centre_de_meme_direction: #continue
                            
                            a_ctr.pop(0)
                            last_controle_coordinate.pop(0)
                            
                            a_ctr.append( dp )
                            last_controle_coordinate.append(c)
                            
                        else: #stop
                            for h in range(start_adding_current,stop_adding_current): # add current defined pixels to l_coins
                                l_coins.append(last_controle_coordinate[h])
                            #reset data
                            a_ = []
                            a_ctr = [] # controle
                            last_controle_coordinate = []
                            encountered = True
                            first_complete = True
                
                k = c
        
        AKA = np.argwhere(B) # S or B
    
    return L_COINS

# Function to remove curvature irregularities ('bends') in skeleton image (= set of point paths)
def remove_bends(skeleton:np.ndarray, bends:list, length:int) -> None:
    """
    Removes bend-like pixels from skeleton image.\n
    Parameters:
    * skeleton: 2D binary image (bool) of edge skeleton ;
    * bends: list of connected bend point paths (use 'get_bends' func.) ;
    * length: diameter of the area to remove around each bend point.
    """
    for bend in bends:
        for i,j in bend:
            skeleton[i-length:i+length, j-length:j+length] = False


# %% ==============================================================================================================
# Fonctions to compute circles associated to skeleton point paths
# =================================================================================================================

# Function to determine the circle best fitting in a point cloud
def get_lse_circle(cloud:np.ndarray) -> Tuple[tuple, float]:
    """
    Determines center and radius of circle best fitting the given point cloud.

    Parameters
    ----------
    cloud : Numpy array of shape (n,2)
        List of all the 2D points in cloud.

    Returns
    -------
    p_c : Tuple of size 2
        Central point of computed circle.
    r : float64
        Radius of computed circle.
    """
    n = cloud.shape[0]
    if n > 2:
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
            return (x,y), r
        else:
            r = (A+B)/n
            return tuple(gravity), r
    elif n==2:
        gravity = np.mean(cloud, axis=0)
        vec_rad = (cloud[0] - gravity)**2
        r = np.sqrt(np.sum(vec_rad))
        return tuple(gravity), r
    elif n==1:
        gravity = cloud[0]
        return tuple(gravity), 0
    return (0,0), 0 # No point in the cloud!

# Function to compute corresponding circles for all segments
def get_circles(skeleton:np.ndarray) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Computes for each connected path in skeleton the circle best fitting its point cloud.\n
    Returns ordered list of associated point clouds, array of circle centers, and array of circle radii.
    """
    labels, nb_labels = ndimage.label(skeleton, structure=np.ones((3,3)))
    l_points = []
    l_center = np.empty(shape=(nb_labels,2))
    l_radius = np.empty(shape=(nb_labels))
    for i in range(nb_labels):
        cloud = np.argwhere(labels==i+1)
        center, radius = get_lse_circle(cloud)
        l_points.append(cloud)
        l_center[i] = center
        l_radius[i] = radius
    return l_points, l_center, l_radius

# Function to determine if two circles can be considered as the same
def are_assimilables(c1:tuple, r1:float, c2:tuple, r2:float, error:float=0.05) -> bool:
    """
    Determines if two circles can be considered as the same one given an error value.\n
    (c1, r1): center and radius of first circle ; (c2, r2): center and radius of the second one.
    """
    dr_max = error * min(r1,r2)
    assimilables = np.sqrt(np.sum((c1-c2)**2)) < dr_max and np.abs(r1-r2) < dr_max
    return assimilables

# Function to merge circles which can be considered as the same
def merge_close_circles(l_points:list, l_center:list|np.ndarray, l_radius:list|np.ndarray, error:float=0.05) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Merges circles which can be considered as the same one, given an error value (error).
    """
    L_POINTS = list(l_points)
    L_CENTER = list(l_center)
    L_RADIUS = list(l_radius)
    L_P = []
    L_C = []
    L_R = []
    i = 0
    while i < len(L_CENTER):
        l_same = []
        for j in range(len(L_CENTER)):
            if i!=j and are_assimilables(L_CENTER[j], L_RADIUS[j], L_CENTER[i], L_RADIUS[i], error):
                l_same.append(j)
        if len(l_same)==0:
            L_P.append(np.asarray(L_POINTS[i]))
            L_C.append(L_CENTER[i])
            L_R.append(L_RADIUS[i])
        else:
            l_p = L_POINTS[i]
            for k in range(len(l_same)):
                lpk = L_POINTS[l_same[k]]
                l_p = np.concatenate((l_p,lpk), axis=0)
            ctr, rad = get_lse_circle(l_p)
            L_P.append(l_p)
            L_C.append(ctr)
            L_R.append(rad)
            while len(l_same) != 0:
                u = np.argmax(l_same)
                v = l_same.pop(u)
                L_POINTS.pop(v)
                L_CENTER.pop(v)
                L_RADIUS.pop(v)
        i+=1
    return L_P, np.asarray(L_C), np.asarray(L_R)

# Function to merge circles whose segments can be considered as being on other circles
def merge_on_circles(l_points:list, l_center:list|np.ndarray, l_radius:list|np.ndarray, d_pxl:float=8) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Merges circles whose segments can be considered as being on other circles, given a pixel delta value (d_pxl).
    """
    L_POINTS = list(l_points)
    L_CENTER = list(l_center)
    L_RADIUS = list(l_radius)
    s = 0
    sort_idx = list(np.argsort(L_RADIUS)) # sorted by radius
    while s < len(sort_idx):
        i = sort_idx[s]
        is_confused = False
        lp = L_POINTS[i]
        cc = L_CENTER[i]
        mean_set = np.mean(lp,axis=0)
        for j in sort_idx:
            if j!=i:
                ra = L_RADIUS[j]
                ct = L_CENTER[j]
                dis = np.abs( np.sqrt(np.sum((lp-ct)**2,axis=1)) - ra )
                proj = np.sum( (mean_set-cc) * (mean_set-ct) )
                if np.max(dis) < d_pxl and proj > 0:
                    L_POINTS[j] = np.concatenate((L_POINTS[j],lp), axis=0)
                    L_CENTER[j], L_RADIUS[j] = get_lse_circle(L_POINTS[j])
                    is_confused = True
                    break
        if is_confused: # do remove
            sort_idx.pop(s)
        else: # do not remove
            s+=1
    L_P = [L_POINTS[a] for a in sort_idx]
    L_C = [L_CENTER[a] for a in sort_idx]
    L_R = [L_RADIUS[a] for a in sort_idx]
    return L_P, np.asarray(L_C), np.asarray(L_R)

# Function to remove circles whose segments are the far from forming a good circle (they have a too high MSE)
def remove_worst_circles(l_points:list, l_center:list|np.ndarray, l_radius:list|np.ndarray, max_error:float=2) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Removes circles whose associated segments have a too high circle-MSE, given a maximum error allowed (max_error).
    """
    L_POINTS = list(l_points)
    L_CENTER = list(l_center)
    L_RADIUS = list(l_radius)
    i = 0
    while i < len(L_POINTS):
        lp = L_POINTS[i]
        ct = L_CENTER[i]
        r2 = L_RADIUS[i]**2
        E = 0
        for k in range(len(lp)):
            p = lp[k]
            d_x = (p[0]-ct[0])**2
            d_y = (p[1]-ct[1])**2
            E += (d_x + d_y - r2)**2
        E = np.sqrt(E/len(lp))
        if E > r2 * max_error:
            L_POINTS.pop(i)
            L_CENTER.pop(i)
            L_RADIUS.pop(i)
        else:
            i+=1
    return L_POINTS, np.asarray(L_CENTER), np.asarray(L_RADIUS)

# Function to remove circles for which the inside (center) is not brighter than the outside
def remove_wrong_side(l_points:list, l_center:list|np.ndarray, l_radius:list|np.ndarray, image:np.ndarray, rad_scan:float=3) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Removes circles for which the inside (center) is not brighter than the outside in image, given a scanning radius (rad_scan).
    """
    L_POINTS = list(l_points)
    L_CENTER = list(l_center)
    L_RADIUS = list(l_radius)
    i=0
    rad_scan = int(np.ceil(rad_scan))
    while i < len(L_POINTS):
        ct = L_CENTER[i]
        lp = L_POINTS[i]
        int_value = 0
        ext_value = 0
        for k in range(len(lp)):
            pt = lp[k]
            vec_int = np.array([ct[0]-pt[0],ct[1]-pt[1]])
            nor_vec = np.sqrt(np.sum(vec_int**2))
            if nor_vec > 0:
                vec_int /= nor_vec
                for r in range(1,rad_scan+1):
                    vec = np.round(r*vec_int).astype(int)
                    pxr = image[pt[0]+vec[0], pt[1]+vec[1]]
                    pxm = image[pt[0]-vec[0], pt[1]-vec[1]]
                    int_value += pxr
                    ext_value += pxm
        #print("int: {}  |  ext: {}".format(int_value,ext_value))
        if ext_value > int_value:
            L_POINTS.pop(i)
            L_CENTER.pop(i)
            L_RADIUS.pop(i)
        else:
            i+=1
    return L_POINTS, np.asarray(L_CENTER), np.asarray(L_RADIUS)

# Function to remove circles not respecting some size and range properties
def filter_by_characteristics(l_points:list, l_center:list|np.ndarray, l_radius:list|np.ndarray, img_shape:tuple, min_perimeter_prop:float, min_radius:float, max_radius:float) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Removes circles which do not respect some properties:\n
    * a minimum proximity to the visible image support (circles whose center is too far from image area, defined by img_shape, are removed) ;
    * a minimum proportion between number of pixels on circle (in l_points) and the perimeter of the associated circle (min_perimeter_prop) ;
    * a minimum radius value (min_radius) ;
    * a maximum radius value (max_radius).
    """
    L_POINTS = list(l_points)
    L_CENTER = list(l_center)
    L_RADIUS = list(l_radius)
    k = 0
    while k < len(L_POINTS):
        if not(
            int(L_RADIUS[k]+0.5) > min_radius and 
            int(L_RADIUS[k]+0.5) < max_radius and 
            len(L_POINTS[k][:,0]) / (L_RADIUS[k]*2*np.pi) > min_perimeter_prop and 
            int(L_CENTER[k][0]) > -max_radius and 
            int(L_CENTER[k][0]) < img_shape[0] + max_radius and 
            int(L_CENTER[k][1]) > -max_radius and 
            int(L_CENTER[k][1]) < img_shape[1] + max_radius
        ):
            L_POINTS.pop(k)
            L_CENTER.pop(k)
            L_RADIUS.pop(k)
        else:
            k+=1
    return L_POINTS, np.asarray(L_CENTER), np.asarray(L_RADIUS)


# %% ==============================================================================================================
# MAIN -C.A.M.- FUNCTION
# =================================================================================================================

def CAM_circles_2D(
        image:np.ndarray, 
        resolution:float=0.5, 
        cleaning_power:float=1.0, 
        curvature_sensitivity:float=1.0, 
        circle_merging_power:float=1.0, 
        circle_accuracy_power:float=1.0, 
        min_radius_ratio:float=0.01, 
        max_radius_ratio:float=0.50
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Curvature Analysis Method (CAM): an algorithm for 2D circular shape detection based on the analysis of edge curvature.\n
    
    Parameters
    ----------
    * image: 2D grayscale image containing circular objects ;
    * resolution: image resize ratio (for downscaling: ratio < 1) ;
    * cleaning_power: power of binary cleaning in edge skeleton ;
    * curvature_sensitivity: power of curvature irregularity detection ;
    * circle_merging_power: power of the detection of similar circles ;
    * circle_accuracy_power: power of the selection of best circles ;
    * min_radius_ratio: ratio of the minimum circle radius allowed ;
    * max_radius_ratio: ratio of the maximum circle radius allowed.\n

    SET MANUALLY HYPERPARAMETERS INSIDE THE FUNCTION!!\n

    Returns
    -------
    (L_CENTER, L_RADIUS), the couple of ordered lists of centers and radii of detected circles.
    """

    ###########################
    ##### HYPERPARAMETERS #####

    img_resize_ratio = resolution
    sigma_preprocess = 1.0 * min(image.shape) / 1000
    line_window_size = 20. * min(image.shape) / 1000

    sigma_proba_maps = 2.0 * min(image.shape) / 1000
    sigma_delta_maps = 3.5 * min(image.shape) / 1000
    thresholding_idx = [[2,1],[2,1]]

    skelet_clean_pow = 0.5 * cleaning_power
    bend_control_pow = 1.2 / curvature_sensitivity

    similar_circle_merge_pow = 0.6 * circle_merging_power
    overlap_circle_merge_pow = 0.8 * circle_merging_power
    lighty_side_scanning_pow = 0.2 * circle_accuracy_power
    max_circle_error_allowed = 0.2 / circle_accuracy_power

    perim_size_prop = 0.08 * circle_accuracy_power
    min_radius_prop = min_radius_ratio
    max_radius_prop = max_radius_ratio
    
    ###########################
    ##### FUNCTION CORPUS #####

    # normalize image
    image = normalized(image, output_dtype=np.float64)

    # add small noize to image (std=0.01)
    noize = ndimage.gaussian_filter(np.abs(np.random.normal(0, 0.01, size=image.shape)), sigma_preprocess)
    image = image + noize

    # blur and resize image
    image = ndimage.gaussian_filter(image, sigma_preprocess)
    image = resized(image, img_resize_ratio)

    # compute gradient magnitude of image
    gradm = gradient_magnitude(image, 1)

    # compute line-fitting mean error map
    width = int(line_window_size * img_resize_ratio)
    _, merror_map = fit_lines(gradm, width)

    # compute and blur probability map of belonging to a line
    proba_map = 1 - normalized(merror_map)
    proba_map = ndimage.gaussian_filter(proba_map, sigma_proba_maps * img_resize_ratio)

    # compute and blur local difference map
    delta_map = local_difference_map(gradm, width)
    delta_map = ndimage.gaussian_filter(delta_map, sigma_delta_maps * img_resize_ratio)

    # compute and intersect binary images of proba map and delta map
    newData2 = k_means(proba_map, thresholding_idx[0][0], thresholding_idx[0][1])
    newData3 = k_means(delta_map, thresholding_idx[1][0], thresholding_idx[1][1])
    binary = newData2 * newData3

    # clean borders of binary image
    binary[:width//2+1,:] = False
    binary[:,:width//2+1] = False
    binary[binary.shape[0]-width//2-2:,:] = False
    binary[:,binary.shape[1]-width//2-2:] = False

    # compute naive skeleton from binary image
    skeleton = skeletonize(binary)

    # label point types and get intersection and middle points
    ptypes = point_types(skeleton)
    inter = ptypes >= 4
    middl = ptypes == 3
    
    # clean skeleton with point types criteria, object size criteria and convolution criteria
    skeleton = clean_by_objectSize(middl, width*skelet_clean_pow*2)
    skeleton = clean_by_convolutionCheck(skeleton, width*skelet_clean_pow, 5)
    skeleton = point_types(skeletonize(skeleton+inter)) >= 3
    skeleton = clean_by_objectSize(skeleton, int(width*skelet_clean_pow*2))
    skeleton = ndimage.binary_closing(skeleton, structure=binary_ball(int(width*skelet_clean_pow/2)))
    skeleton = skeletonize(skeleton)

    # keep only inner segments (connected paths)
    path_image = clean_by_objectSize((point_types(skeleton)==3)+(point_types(skeleton)==2), width*skelet_clean_pow)
    remove_corner_pixels(path_image, width)

    # detect and delete bends from path_image through curvature analysis
    bends = get_bends(path_image, int(width*bend_control_pow))
    remove_bends(path_image, bends, 2)
    
    # get all naive circles from prepared segments
    L_POINTS, L_CENTER, L_RADIUS = get_circles(path_image)
    
    # merge circles which are close to each other
    L_POINTS, L_CENTER, L_RADIUS = merge_close_circles(L_POINTS, L_CENTER, L_RADIUS, similar_circle_merge_pow)
    
    # merge circles which coincide with others
    L_POINTS, L_CENTER, L_RADIUS = merge_on_circles(L_POINTS, L_CENTER, L_RADIUS, width*overlap_circle_merge_pow)
    
    # keep only circles which fit well enough to their segment(s) (l_points)
    L_POINTS, L_CENTER, L_RADIUS = remove_worst_circles(L_POINTS, L_CENTER, L_RADIUS, max_circle_error_allowed)
    
    # keep only circles for which the inside is brighter than the outside, near their segment(s)
    L_POINTS, L_CENTER, L_RADIUS = remove_wrong_side(L_POINTS, L_CENTER, L_RADIUS, image, width*lighty_side_scanning_pow)
    
    # keep only the circles repspecting some size and range properties
    min_radius = min(image.shape) * min_radius_prop
    max_radius = min(image.shape) * max_radius_prop
    _, L_CENTER, L_RADIUS = filter_by_characteristics(L_POINTS, L_CENTER, L_RADIUS, image.shape, perim_size_prop, min_radius, max_radius)
    
    # resize circles to the original image's size
    L_CENTER = L_CENTER / img_resize_ratio
    L_RADIUS = L_RADIUS / img_resize_ratio
    
    return L_CENTER, L_RADIUS

