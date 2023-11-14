# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 2022
@author: Antoine BOTTENMULLER
"""



## Libraries

import numpy as np
import cv2 as cv
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
import colorsys
from numba import jit

from utils import normalized, gradient_magnitude, resized, disk





#%% ########## Best line fitting and energies scanning algorithm ########## %%#

# Fonctions de génération de la carte des MSE minimum linéaires

@jit(nopython=True)
def __min_L_E(zone, center, xx, yy):
    xo = xx[:zone.shape[0],:zone.shape[1]] - center[0]
    yo = yy[:zone.shape[0],:zone.shape[1]] - center[1]
    n = np.sum(zone)
    A = np.sum( xo**2 * zone )/n
    B = -2*np.sum( xo*yo * zone )/n
    C = np.sum( yo**2 * zone )/n
    if(B!=0):
        del_4 = (C-A)**2 + B**2 # Necessairement >=0
        a0 = (A-C - np.sqrt(del_4))/B
        norm = np.sqrt(1+a0**2)
        L = (1/norm,a0/norm)
        E = (A*a0**2 + B*a0 + C)/(1+a0**2)
        return L , E
    else: # Veritable solution!
        if(A>C): # a0 = 0
            return (1,0) , C
        elif(C>A): # a0 = +/- infini
            return (0,1) , A
        else: # a0 n'est pas defini
            return (0,0) , C # Orientation indefinie! Convention : a0 = 0


@jit(nopython=True)
def __updateC_d_e(image, largeur, dispersion_map, orientation_map, i, j, structure, distance_majorante, xx, yy):
    
    #Definition des bords de la fenetre
    a_min = max(i-int((largeur-1)/2),0)
    a_max = min(i+int(largeur/2)+1,image.shape[0])
    b_min = max(j-int((largeur-1)/2),0)
    b_max = min(j+int(largeur/2)+1,image.shape[1])
    
    #Selection de la fenetre sur l'image (pas de normalisation ici, car ca ne sert a rien, du moment qu'on fait un tri des valeurs par la suite!)
    carre = image[a_min:a_max, b_min:b_max]
    if(a_max-a_min==largeur and b_max-b_min==largeur):
        carre = carre*structure
    
    #Verification de possibilite de normalisation de la zone d'etude, i.e. qu'il existe des variations dans les valeurs de pixels de la zone, i.e. que le min est different du max
    if(np.max(carre) != np.min(carre)):
        
        #Normalisation de la zone d'etude
        zone = (carre-np.min(carre))/(np.max(carre)-np.min(carre))  #)**(1) # Float de 0 a 1
        
        #Calcul de la dispersion des points ("variance" ou "energie") correspondante a cette zone, par rapport a son centre
        centre = np.array([(zone.shape[0]-1)/2,(zone.shape[1]-1)/2])
        d , e = __min_L_E(zone,centre,xx,yy)
        
    else: # L'orientation est indefinie, et la dispersion est maximale
        d , e = (0,0) , distance_majorante
    
    #Ajout des donnees d'orientation et de dispersion aux deux listes respectives de sortie
    dispersion_map[i,j] = e
    orientation_map[i,j] = d


def fit_lines(image, largeur):
    if(type(image)==np.ndarray and type(largeur)==int):
        if(len(image.shape)==2 and largeur>=1):
            
            h,l = image.shape
            
            distance_majorante = int(np.ceil((largeur/2+1)**2 * np.max(image)))+1 # valeur majorante theoriquement inateignable
            
            dispersion_map = np.ones([h,l], dtype = np.float64) * distance_majorante # initialisation : array de sortie 1
            orientation_map = np.zeros([h,l,2], dtype = np.float64) # initialisation : array de sortie 2
            
            structure = disk(largeur, image.dtype) # definition de la structure : ici un disque!

            y_, x_ = np.arange(largeur), np.arange(largeur)
            yy, xx = np.meshgrid(x_, y_)
            
            #Iteration sur chaque pixel de l'image : on calcule l'orientation et la dispersion correspondantes dans la zone associee
            for i in range(h):#range(int((largeur-1)/2),h-int(largeur/2)-1):
                #print("Step : {}/{}".format(i+1,h)) # indication d'avancement sur les lignes
                for j in range(l):#range(int((largeur-1)/2),l-int(largeur/2)-1):
                    __updateC_d_e(image,largeur,dispersion_map,orientation_map,i,j,structure,distance_majorante,xx,yy)
            
            #On rapproche d'un certain pourcentage les pixels ayant atteint la 'distance_majorante' dans "dispersion_map" par rapport au maximum des dispersions calculees, avant la normalisation! Pour pas qu'il y ait un trop grand eloignement de ces pixels pour rien!
            if(np.max(dispersion_map)==distance_majorante):
                up = 1.02 # On augmente de 2% les valeurs des pixels ayant été à distance_majorante par rapport à la plus grande des valeurs calculees
                B = dispersion_map == distance_majorante
                K = dispersion_map - distance_majorante * B
                dispersion_map = K + up * np.max(K) * B
            
            #Normalisation lineaire de "dispersion_map"
            dispersion_map = 1-normalized(dispersion_map)
            
            return dispersion_map , orientation_map
        
        else:
            print("Error: Input variables are not correctly defined!")
            return None
    else:
        print("Error: Wrong types for input variables!")
        return None


# Image colorée des angles des meilleures lignes par pixel

def get_color_image_from_orientation_map(OMp):

    OMx = OMp[:,:,0] + ( OMp[:,:,0]<=0 ) * 0.000000000001
    OMy = OMp[:,:,1]
    ANG = ( np.arctan( OMy / OMx ) + np.pi/2 )/np.pi # ANG : from 0 to 1, modulo 1

    OM = np.zeros([ANG.shape[0],ANG.shape[1],3])
    for i in range(OM.shape[0]):
        for j in range(OM.shape[1]):
            OM[i,j] = colorsys.hls_to_rgb(ANG[i,j],0.5,1)

    return OM



#%% ###################### Binary cleaning functions ###################### %%#

@jit(nopython=True)
def __clean_label(out, labels, index, max_size):
    if np.sum(labels==index)>max_size:
        out += labels==index

def clean_by_objectSize(binary_image, max_size=10):
    cleaned_image = np.zeros(shape=binary_image.shape, dtype=bool)
    labels, nb_labels = ndimage.label(binary_image,structure=np.ones([3,3]))
    for i in range(1,nb_labels+1):
        __clean_label(cleaned_image,labels,i,max_size)
    return cleaned_image


def __delete_points(image, largeur_int, demi_largeur_ext):
    struct_white = disk(largeur_int, np.uint8)
    struct_black = disk(largeur_int+2*demi_largeur_ext, np.uint8)
    struct_black[demi_largeur_ext:largeur_int+demi_largeur_ext,demi_largeur_ext:largeur_int+demi_largeur_ext] -= struct_white
    white = ndimage.convolve(image.astype(np.uint32), struct_white)  > 0
    black = ndimage.convolve(image.astype(np.uint32), struct_black) == 0
    check = ndimage.maximum_filter(white*black, footprint=struct_white)
    new_image = image*(1 - check)
    return new_image

def clean_by_convolutionCheck(binary_image, max_length, nb_inter, demi_largeur_ext=2):
    cleaned_image = binary_image.copy()
    inter_length = int(max_length/nb_inter)
    start_length = inter_length + int(max_length%inter_length)
    for i in range(nb_inter):
        cleaned_image = __delete_points(cleaned_image,start_length+i*inter_length,demi_largeur_ext)
    return cleaned_image



#%% ########## Binarization of the image - With energy map ########## %%#

# Carte des distances

def distance_from_local_max(image, largeur):
    min_map = ndimage.minimum_filter(image,largeur)
    max_map = ndimage.maximum_filter(image,largeur)
    diff = max_map-min_map
    distances_map = (image-min_map)/(diff+(diff==0)*0.0000000001)
    return distances_map


# K means de Ss ou M1 (K = 3)

def k_means(image, k, i):
    if(i>=0 and i<k):
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
    elif(i<0):
        print("Warning: i<0, evaluation at i=0.")
        return k_means(image, k, 0)
    else:
        print("Warning: i>=k, evaluation at i=k-1.")
        return k_means(image, k, k-1)



#%% ########## Binarization of the image - With OMp map ########## %%#

@jit(nopython=True)
def set_std(std, h, l, i, j, a, b, color_map):
    a_min = max(i-int((a-1)/2),0)
    a_max = min(i+int(a/2)+1,h)
    b_min = max(j-int((b-1)/2),0)
    b_max = min(j+int(b/2)+1,l)
    ref = color_map[i,j]
    win = color_map[a_min:a_max,b_min:b_max]
    s = np.sum( np.abs( win[:,:,0]*ref[1] - win[:,:,1]*ref[0] ) )
    std[i,j] = s

def vec_std(color_map, size=(2,2)):
    std = np.zeros(shape=color_map.shape[:2], dtype=np.float64)
    a,b = size
    h,l = color_map.shape[:2]
    for i in range(h):
        for j in range(l):
            set_std(std, h, l, i, j, a, b, color_map)
    return std



#%% ################# from binary edges to clean skeleton #####################

######### D'abord, lier correctement les points d'intersection! ##################
################## ICI, retirer les endpoints du squelette (pour enlever les p'tits bouts nuls) ! ##################
####################### puis dillater et re-revenir ! #########################

kernel_4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
kernel_8 = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)

def points_types(skeleton_image, eight_neighbours=True):
    """
    ==0 <=> black point
    ==1 <=> isolated point
    ==2 <=> end-line point
    ==3 <=> middle-line point
    >=4 <=> intersection point
    """
    kernel = eight_neighbours * kernel_8 + (1-eight_neighbours) * kernel_4
    nb_neighbours = ndimage.convolve(skeleton_image.astype(np.uint8), kernel)
    return np.array( nb_neighbours * skeleton_image , dtype=np.uint8)



#%% ############# Remove best corner-like pixels on closed objects ##############

### Remove best coin-like pixel on each closed object

def remove_relevant_pixel_in_closed_sets(I, blockSize):
    corner_map = cv.cornerHarris(np.float32(I),blockSize,3,0.04)
    #show(normalized(corner_map),"corner map",1)
    labels, nb_labels = ndimage.label(I, structure=np.ones(shape=(3,3)))
    bouts = ( points_types(I)==2 ) * labels
    for i in range(1,nb_labels+1):
        if(np.sum(bouts==i)==0):
            obj = ( labels==i ) * corner_map
            coord = np.argwhere(obj==np.max(obj))[0]
            I[coord[0]-1:coord[0]+2,coord[1]-1:coord[1]+2] = False



#%% ############################ remove bends ################################

def get_relative_vector(l1, l2, prop=0.5):
    
    d1a = np.sum(l1[:int(len(l1)*prop)],axis=0).astype(np.float64)
    n1a = np.sqrt(d1a[0]**2+d1a[1]**2)
    d1a /= ( n1a + (n1a==0)*0.0000000001 )
    
    d1b = np.sum(l1[int(len(l1)*prop):],axis=0).astype(np.float64)
    n1b = np.sqrt(d1b[0]**2+d1b[1]**2)
    d1b /= ( n1b + (n1b==0)*0.0000000001 )
    
    v1x = np.sum(d1a*d1b)
    v1y = d1a[0]*d1b[1]-d1a[1]*d1b[0]
    v1 = np.array([v1x,v1y]) #v1
    
    d2a = np.sum(l2[:int(len(l2)*prop)],axis=0).astype(np.float64)
    n2a = np.sqrt(d2a[0]**2+d2a[1]**2)
    d2a /= ( n2a + (n2a==0)*0.0000000001 )
    
    d2b = np.sum(l2[int(len(l2)*prop):],axis=0).astype(np.float64)
    n2b = np.sqrt(d2b[0]**2+d2b[1]**2)
    d2b /= ( n2b + (n2b==0)*0.0000000001 )
    
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


def get_bends(skeleton, nb_controleurs = 6): # nb_controleurs: size of the control segment
    
    S = skeleton.copy()
    B = points_types(S)==2
    
    L_COINS = []
    
    born_inf_coins = 0.7 #??? between 0 and 1 (included)
    born_sup_coins = 0.8 #??? between born_inf_coins and 2 (included)
    
    start_adding_current = min(int(nb_controleurs*born_inf_coins), nb_controleurs-1)
    stop_adding_current = min(int(nb_controleurs*born_sup_coins)+1, nb_controleurs)
    stop_adding_next = min(max(int(nb_controleurs*(born_sup_coins-1)), 0), nb_controleurs-1)
    
    prop_reference = 0.5  #??? 0.5 for reference: is it good???
    
    d_angle_lim = np.pi/5 #??? maximum difference of angle: pi/5 is it good?
    d_orien_lim = np.cos(np.pi/6) #??? max error when wrong orientation: pi/6?
    
    AKA = np.argwhere(B) # S or B
    while(len(AKA)>0): # c is coordinate
        #print(len(AKA))
        
        k = AKA[0]
        B[k[0],k[1]] = False
        
        a_ = []
        
        a_ctr = []
        last_controle_coordinate = []
        
        first_complete = True
        encountered = False
        
        l_coins = []
        
        while(True):
            S[k[0],k[1]] = False
            window = S[k[0]-1:k[0]+2,k[1]-1:k[1]+2]
            IKI = np.argwhere(window>0)
            if(len(IKI)==0):
                B[k[0],k[1]] = False
                break
            else:
                dp = IKI[0]-1
                c = k + dp
                
                #controle
                if(len(a_ctr) < nb_controleurs):
                    
                    a_ctr.append( dp )
                    last_controle_coordinate.append(c)
                    
                    if(encountered): # here to add next pixels to l_coins
                        if(len(a_ctr) < stop_adding_next ):
                            l_coins.append(k)
                        else:
                            L_COINS.append(l_coins)
                            a_ctr = [] # controle
                            last_controle_coordinate = []
                            l_coins = []
                            c = k # stay on this pixel
                            encountered = False
                else:
                    if(first_complete): # here only once per segment!! (when a_ctr is completed for the first time)
                        
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
                        
                        if( centre_de_meme_direction ): #continue
                            
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


def delete_bends(image,coudes,length):
    img = image.copy()
    for coude in coudes:
        for c in coude:
            i,j = c
            img[i-length:i+length,j-length:j+length] = False
    return img



#%% Function to determine the circle best fitting in point cloud

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



#%% Estimation of corresponding circles for each segment

def get_circles(skeleton):
    labels, nb_labels = ndimage.label(skeleton,structure=np.ones((3,3)))
    l_points = []
    l_center = np.zeros(shape=(nb_labels,2))
    l_radius = np.zeros(shape=(nb_labels))
    for i in range(1,nb_labels+1):
        cloud = np.argwhere(labels==i)
        center, radius = get_lse_circle(cloud)
        l_points.append(cloud)
        l_center[i-1] = center
        l_radius[i-1] = radius
    return l_points, l_center , l_radius



#%% Assimilation of same circles

def are_assimilables(c1,r1,c2,r2,error=0.05):
    dr_max = error*min(r1,r2)
    assimilables = np.sqrt(np.sum((c1-c2)**2))<dr_max and np.abs(r1-r2)<dr_max
    return assimilables

def merge_close_circles(l_points, l_center , l_radius, error=0.05):
    L_POINTS = list(l_points)
    L_CENTER = list(l_center)
    L_RADIUS = list(l_radius)
    L_P = []
    L_C = []
    L_R = []
    i = 0
    while(i<len(L_CENTER)):
        l_same = []
        for j in range(len(L_CENTER)):
            if(i!=j and are_assimilables(L_CENTER[j],L_RADIUS[j],L_CENTER[i],L_RADIUS[i], error)):
                l_same.append(j)
        if len(l_same)==0:
            L_P.append(np.array(L_POINTS[i]))
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
            while(len(l_same)!=0):
                u = np.argmax(l_same)
                v = l_same.pop(u)
                L_POINTS.pop(v)
                L_CENTER.pop(v)
                L_RADIUS.pop(v)
        i+=1
    return L_P , np.array(L_C) , np.array(L_R)



#%% merge segments which are on other circles

def merge_on_circles(L_PTS, L_CTR, L_RAD, d_pxl=8):
    lpts = list(L_PTS)
    lctr = list(L_CTR)
    lrad = list(L_RAD)
    s=0
    sort_idx = list(np.argsort(lrad)) # sorted by radius
    while( s < len(sort_idx) ):
        i = sort_idx[s]
        is_confused = False
        lp = lpts[i]
        cc = lctr[i]
        mean_set = np.mean(lp,axis=0)
        for j in sort_idx:
            if(j!=i):
                ra = lrad[j]
                ct = lctr[j]
                dis = np.abs( np.sqrt(np.sum((lp-ct)**2,axis=1)) - ra )
                proj = np.sum( (mean_set-cc) * (mean_set-ct) )
                if( ( np.max(dis) < d_pxl ) and ( proj > 0 ) ):
                    lpts[j] = np.concatenate((lpts[j],lp), axis=0)
                    lctr[j], lrad[j] = get_lse_circle(lpts[j])
                    is_confused = True
                    break
        if( is_confused ): # do remove
            sort_idx.pop(s)
        else: # do not remove
            s+=1
    lp = [lpts[a] for a in sort_idx]
    lc = [lctr[a] for a in sort_idx]
    lr = [lrad[a] for a in sort_idx]
    return lp, np.array(lc), np.array(lr)



#%% remove segments for which MSE is too high

def keep_best_energies(L_PTS, L_CTR, L_RAD, max_error=2):
    lpts = list(L_PTS)
    lctr = list(L_CTR)
    lrad = list(L_RAD)
    i=0
    while( i < len(lpts) ):
        r2 = lrad[i]**2
        ct = lctr[i]
        lp = lpts[i]
        E = 0
        for k in range(len(lp)):
            p = lp[k]
            d_x = (p[0]-ct[0])**2
            d_y = (p[1]-ct[1])**2
            E += (d_x + d_y - r2)**2
        E = np.sqrt( E/len(lp) )
        if( E/r2 > max_error ):
            lpts.pop(i)
            lctr.pop(i)
            lrad.pop(i)
        else:
            i+=1
    return lpts, np.array(lctr), np.array(lrad)



#%% get only the ones for which the center is brighter

def keep_only_true_side(L_PTS, L_CTR, L_RAD, image, rad_scan=3):
    lpts = list(L_PTS)
    lctr = list(L_CTR)
    lrad = list(L_RAD)
    i=0
    while( i < len(lpts) ):
        ct = lctr[i]
        lp = lpts[i]
        int_value = 0
        ext_value = 0
        for k in range(len(lp)):
            pt = lp[k]
            vec_int = np.array([ct[0]-pt[0],ct[1]-pt[1]])
            nor_vec = np.sqrt(np.sum(vec_int**2))
            if(nor_vec>0):
                vec_int /= nor_vec
                for r in range(1,rad_scan+1):
                    vec = np.round(r*vec_int).astype(int)
                    pxr = image[pt[0]+vec[0], pt[1]+vec[1]]
                    pxm = image[pt[0]-vec[0], pt[1]-vec[1]]
                    int_value += pxr
                    ext_value += pxm
        #print("int: {}  |  ext: {}".format(int_value,ext_value))
        if( ext_value > int_value ):
            lpts.pop(i)
            lctr.pop(i)
            lrad.pop(i)
        else:
            i+=1
    return lpts, np.array(lctr), np.array(lrad)



#%% get only the ones respecting some size and range properties

def filter_by_characteristics(l_pts, l_ctr, l_rds, img_shape, disk_prop, min_radius, max_radius):
    L_POINTS = list(l_pts)
    L_CENTER = list(l_ctr)
    L_RADIUS = list(l_rds)
    k = 0
    while(k<len(L_POINTS)):
        if(not(
                int(L_RADIUS[k]+0.5) > min_radius and 
                int(L_RADIUS[k]+0.5) < max_radius and 
                (len(L_POINTS[k][:,0]) / (L_RADIUS[k]*2*np.pi)) > disk_prop and 
                int(L_CENTER[k][0]) > -max_radius and 
                int(L_CENTER[k][0]) < img_shape[0] + max_radius and 
                int(L_CENTER[k][1]) > -max_radius and 
                int(L_CENTER[k][1]) < img_shape[1] + max_radius
               )):
            L_POINTS.pop(k)
            L_CENTER.pop(k)
            L_RADIUS.pop(k)
        else:
            k+=1
    return L_POINTS, np.array(L_CENTER), np.array(L_RADIUS)



#%% Main functions

# Main CAM function

def CAM_circles_2D(
        image:np.ndarray, gaussian_sigma_pre_processing=1.0, processing_rescale_ratio=2.0, max_distinguishable_lines_length=20, 
        gaussian_sigma_energy=1.0, gaussian_sigma_distances=1.8, k_means_indices=[[2,1],[2,1]], skeleton_cleaning_weight=0.5, 
        bends_length_weight=1.2, merge_close_circles_coeff=0.6, merge_on_circles_weight=0.8, energy_max=0.2, true_side_scan_weight=0.2, 
        disks_pixels_proportion=0.08, min_radius_weight=0.2, max_radius_weight=0.3
        ):
    
    # prepare and resize image gradient magnitude E for analysis
    
    noize_I = normalized(image.astype(np.float64)) + ndimage.gaussian_filter( np.abs( np.random.normal(0,0.01,size=image.shape) ) , gaussian_sigma_pre_processing )
    gauss_I = ndimage.gaussian_filter(noize_I, gaussian_sigma_pre_processing)
    E = gradient_magnitude(resized(gauss_I, processing_rescale_ratio))
    
    
    # finding probability map of pixels to belong to a line (energy)
    
    largeur = int( max_distinguishable_lines_length / processing_rescale_ratio )
    energy , _ = fit_lines(E, largeur)
    ### OM = get_color_image_from_orientation_map(_)

    
    # get gaussian energy map from energy and compute gaussian distances map
    
    gaussian_energy = ndimage.gaussian_filter(energy, gaussian_sigma_energy)
    dis = distance_from_local_max(E, largeur)
    gaussian_dis = ndimage.gaussian_filter(dis, gaussian_sigma_distances)


    # intersec binary 2-means of gaussian energy map with binary 2-means of gaussian distances map

    newData2 = k_means(gaussian_energy, k_means_indices[0][0], k_means_indices[0][1])
    newData3 = k_means(gaussian_dis, k_means_indices[1][0], k_means_indices[1][1])
    binary = newData2 * newData3
    
    
    # clean borders of binary image and get a naive skeleton from it

    binary[:largeur//2+1,:] = False
    binary[:,:largeur//2+1] = False
    binary[binary.shape[0]-largeur//2-2:,:] = False
    binary[:,binary.shape[1]-largeur//2-2:] = False
    naive_skeleton = skeletonize(binary)
    
    
    # clean skeleton with points types checking, object size checking and convolution checking
    
    ptypes = points_types(naive_skeleton)
    
    intersections = ptypes >= 4
    middle_line_sets = ptypes == 3
    
    clean_nda = clean_by_objectSize(middle_line_sets, largeur*skeleton_cleaning_weight*2)
    clean_fusion = clean_by_convolutionCheck(clean_nda, largeur*skeleton_cleaning_weight, 5)
    clean_skeleton = points_types(skeletonize(clean_fusion+intersections))>=3
    clean_binary = clean_by_objectSize(clean_skeleton, int(largeur*skeleton_cleaning_weight*2))
    closed_skeleton = ndimage.binary_closing(clean_binary, structure = disk(int(largeur*skeleton_cleaning_weight/2)))
    skeleton = skeletonize(closed_skeleton)
    ### SK = skeleton
    
    
    # get clean inner segments from clean skeleton
    
    pure_middle_segments = clean_by_objectSize( (1*(points_types(skeleton)==3)+1*(points_types(skeleton)==2))>0 , largeur*skeleton_cleaning_weight )
    remove_relevant_pixel_in_closed_sets(pure_middle_segments, largeur)
    
    
    # detect and delete bends from inner segments with curvature analysis
    
    coudes = get_bends(pure_middle_segments, int(largeur*bends_length_weight))
    prepared = delete_bends(pure_middle_segments, coudes, 2)
    ### CB = np.concatenate( [np.expand_dims(pure_middle_segments,axis=2), np.expand_dims(prepared,axis=2), np.expand_dims(skeleton^pure_middle_segments^prepared,axis=2)], axis=2 )
    
    
    # get all naive circles from prepared segments
    
    l_points, l_center, l_radius = get_circles(prepared)
    
    
    # merge circles which are close to each other
    
    l_poi , l_cen , l_rad = merge_close_circles(l_points, l_center, l_radius, merge_close_circles_coeff)
    
    
    # merge circles which coincide with others
    
    l_pnt , l_cnt , l_rdu = merge_on_circles(l_poi, l_cen, l_rad, int(largeur*merge_on_circles_weight))
    
    
    # keep only circles which fit well enough to their segment(s) (l_points)
    
    l_pos, l_cer, l_rai = keep_best_energies(l_pnt, l_cnt, l_rdu, energy_max)
    
    
    # keep only circles for which the inside is brighter than the outside, near their segment(s)
    
    l_pts, l_ctr , l_rds = keep_only_true_side(l_pos, l_cer, l_rai, normalized(resized(gauss_I, processing_rescale_ratio)), int(largeur*true_side_scan_weight))
    

    # keep only the circles repspecting some size and range properties
    
    min_radius = np.ceil(largeur*min_radius_weight)
    max_radius = np.min(E.shape)*max_radius_weight
    _, L_CENTER, L_RADIUS = filter_by_characteristics(l_pts, l_ctr, l_rds, E.shape, disks_pixels_proportion, min_radius, max_radius)
    
    
    # resize circles to the original image's size
    
    l_centers = L_CENTER * processing_rescale_ratio
    l_radii   = L_RADIUS * processing_rescale_ratio
    
    
    return l_centers, l_radii


