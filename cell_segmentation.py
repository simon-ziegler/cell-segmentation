import sys
import os
import csv
import string
import cv2
import numpy as np

from skimage.feature import peak_local_max
from matplotlib import pyplot as plt

import random
import time

from datetime import datetime
from collections import defaultdict
import json


def read_csv(file_name,delimiter=';'):
    with open(file_name,'rb') as csvfile:
        in_csv=csv.reader(csvfile, delimiter=delimiter, quotechar='"')

        data=[]
        for row in in_csv:
            data.append(row)
        return data

def segment_image(img,img2,adapt_thres_winsize=91,adapt_thres_c=0,gaussian_filter_size=5,binary_thres_val=190,min_cell_area=20,intens_quantile=0.9,line_thickness=2,examine=False,same_color=False,low_quantile=0.1):
    """compute segmentation of image, result is a binary image of fore- and background

    Parameters
    ----------
    img : OpenCV image
        input image

    img2 : OpenCV image
        paints information into input image

    adapt_thres_winsize : uneven integer
        defines the window size for adaptive threshold

    adapt_thres_c : float
        value added in adaptive thresholding

    gaussian_filter_size : uneven integer
        size of gaussian filter for filtering the binary image

    binary_thres_val : integer
        threshold for gaussian filtered image (after adaptive thresholding) to create binary image

    min_cell_area : integer
        define minimal number of pixels for a segment to be a cell

    intense_quantile : float        
        quantile for maximal threshold of mean intensity to be a cell

    line_thickness : int
        thickness of lines drawed in output

    examine : boolean
        flag if intermediate results are saved

    same_color : boolean
        flag if all drawings are in the same color
    
    low_quantile : float        
        quantile for minimal threshold of mean intensity to be a cell


    Returns
    -------

    img2 : OpenCV image
        original image with markers

    markers_out : OpenCV image
        image with segments

    cont_in : list of OpenCV contours
        list cells

    cont_out : list of OpenCV contours
        list of segments which have to be segmented
        
    """

    thresh=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,adapt_thres_winsize,adapt_thres_c)


    if examine:
        cv2.imwrite('examine/img.png',img)
        cv2.imwrite('examine/adaptive.png',thresh)
    
    thresh=cv2.GaussianBlur(thresh,(gaussian_filter_size,gaussian_filter_size),8)

    if examine:
        cv2.imwrite('examine/blurred.png',thresh)

    ret, thresh=cv2.threshold(thresh,binary_thres_val,255,cv2.THRESH_BINARY)

    if examine:
        cv2.imwrite('examine/thresh.png',thresh)

    contours,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    intens_quant=np.quantile(img,intens_quantile)

    low_thres=np.quantile(img,low_quantile)

    cont_in=[]
    cont_out=[]
    cont_junk=[]
    cont_background=[]

    pt=cls+'/'+fl

    try:
        os.mkdir(pt)
    except:
        pass

    markers_out=np.zeros(img.shape)

    m_cnt=1
    for i in range(len(contours)):
        hull=cv2.convexHull(contours[i],False)

        perimeter=cv2.arcLength(contours[i],True)
        area=cv2.contourArea(contours[i])

        roundness=4.*np.pi*area/(perimeter*perimeter+0.0001)

        convexity=cv2.contourArea(hull)/(area+0.0001)



        x,y,w,h=cv2.boundingRect(contours[i])
        if x<=1 or y<=1 or x+w>=img.shape[1]-2 or y+h>=img.shape[0]-2:
            #switch off sort-out at the border, continue if sort-out is desired
            pass
            

        markers=np.zeros(img.shape)
        cv2.drawContours(markers,contours,i,(255),-1)

        markers_ext=markers[y:y+h,x:x+w]
        img_ext=img[y:y+h,x:x+w]

        pts=np.where(markers_ext!=255)
        img_ext[pts]=0      


        pix_vals_mean=np.mean(img_ext[np.where(markers_ext==255)])

        if pix_vals_mean<low_thres:
            cont_background.append(contours[i])
            continue


    
        sortOut=-1
        if convexity>1.05:
            sortOut=1
        else:
            sortOut=0
        if area<min_cell_area:
            sortOut=2

        if sortOut>0 and sortOut<2:
            cont_in.append(contours[i])
            cv2.drawContours(markers_out,contours,i,(m_cnt),-1)
            m_cnt+=1
        elif sortOut<2:
            if x<=1 or y<=1 or x+w>=img.shape[1]-2 or y+h>=img.shape[0]-2:
                continue
            cont_out.append(contours[i])
        else:
            x,y,w,h=cv2.boundingRect(contours[i])
            roi=img[y:y+h,x:x+w]
            cont_junk.append(contours[i])

            #bright cells sortout is deactivated, uncomment lines below and comment line above to activate
            #if quant_intens>intens_quant:
            #    cont_junk.append(contours[i])
            #else:
            #    cont_out.append(contours[i])
            


        if sortOut>0:
            continue

    if same_color:
        cv2.drawContours(img2,cont_in,-1,(0,0,255),line_thickness)
        cv2.drawContours(img2,cont_out,-1,(0,0,255),line_thickness)
        cv2.drawContours(img2,cont_junk,-1,(255,0,255),line_thickness)
        cv2.drawContours(img2,cont_background,-1,(255,255,0),line_thickness)
    else:
        cv2.drawContours(img2,cont_in,-1,(0,255,0),line_thickness)
        cv2.drawContours(img2,cont_out,-1,(0,0,255),line_thickness)
        cv2.drawContours(img2,cont_junk,-1,(255,0,255),line_thickness)
        cv2.drawContours(img2,cont_background,-1,(255,255,0),line_thickness)


    return img2, markers_out, cont_in, cont_out


def clustering(img,mask,adapt_thres_winsize,min_dist,binary_thres_val=190,gaussian_filter_size=21):
    """cluster segments in image to cells

    Parameters
    ----------
    img : OpenCV image
        input image

    mask : OpenCV binary image
        mask for the cell

    adapt_thres_winsize : integer
        threshold for adaptive thresholding

    min_dist : integer
        minimal pixel distance of local maxima in distance map, corresponds to distance between cells

    binary_thres_val : integer
        threshold for binarization of cluster

    gaussian_filter_size : uneven integer
        size of gaussian filter

    
    Returns
    -------

    seg_img : OpenCV image
        clustered image

    thresh : OpenCV image
        threholded image

    shed : OpenCV image
        watershed image

    dist_map_scaled : OpenCV image
        distance map for display

    contour_list : list of OpenCV contours
        list of contours of clustered cells
        
    """
    return seg_img, thresh, shed, (dist_map_scaled*255).astype(np.uint8), contour_list


    t_time=time.time()

    s_time=time.time()
    thresh=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,adapt_thres_winsize,0)

    
    thresh=cv2.GaussianBlur(thresh,(gaussian_filter_size,gaussian_filter_size),8)
    ret, thresh=cv2.threshold(thresh,binary_thres_val,255,cv2.THRESH_BINARY)
    

    s_time=time.time()
    dist_map=cv2.distanceTransform(thresh,cv2.DIST_L2,3)

    dist_map_scaled=dist_map.copy()
    cv2.normalize(dist_map,dist_map_scaled,0,1,cv2.NORM_MINMAX)

    s_time=time.time()

    peak_locs=peak_local_max(dist_map_scaled,min_distance=min_dist,exclude_border=0)

    img_c=thresh.copy()


    img_rgb=cv2.cvtColor(img_c,cv2.COLOR_GRAY2RGB)

    dist_map_int=255-(255*dist_map_scaled*thresh).astype(np.uint8)
    dist_map_col=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    dist_map_col=dist_map_col.astype(np.uint8)

    markers=255-thresh.copy()
    markers=markers.astype(np.int32)
    for i in range(peak_locs.shape[0]):
        cy=int(peak_locs[i][0])
        cx=int(peak_locs[i][1])

        cv2.circle(img_rgb,(cx,cy),5,(255,0,0))
        cv2.circle(thresh,(cx,cy),3,(127))

        cx=max(1,cx)
        cx=min(markers.shape[1]-2,cx)

        cy=max(1,cy)
        cy=min(markers.shape[0]-2,cy)
        markers[cy][cx]=i+1


    

    shed=cv2.watershed(dist_map_col,markers)

    segments=np.max(shed)
    segment_list=np.unique(shed)

    seg_img=np.zeros((img.shape[0],img.shape[1],3)).astype(np.uint8)

    random.seed(1)
    contour_list=[]
    for i in segment_list:
        if i>=255 or i<0:
            continue
        single=(shed==i)*1

        single_masked=single*mask
        if np.count_nonzero(single_masked)<np.count_nonzero(single)//2:
            continue
      
        contours,ret=cv2.findContours(single.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        col=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        col_seg=cv2.merge((single.astype(np.uint8)*col[0],single.astype(np.uint8)*col[1],single.astype(np.uint8)*col[2]))
    
        
        cv2.drawContours(seg_img,contours,0,col,1)
        contour_list.append(contours)

    return seg_img, thresh, shed, (dist_map_scaled*255).astype(np.uint8), contour_list


def cluster_img(img2,markers_out,seg_idx=1,adapt_thres_winsize=91,min_distance_frac=10,binary_thres_val=190,gaussian_filter_size=21):

    img_show=img2.copy()
    img_show=cv2.cvtColor(img_show,cv2.COLOR_RGB2GRAY)

    markers_int=markers_out.astype(int)
    segments=np.max(markers_int)

    img_draw=np.zeros((img2.shape[0],img2.shape[1],3)).astype(np.uint8)

    mask=(markers_int==seg_idx)*1
    cnt,ret=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_single=img_show*mask

    x,y,w,h=cv2.boundingRect(cnt[0])

    if min_distance_frac<1.0:
        min_distance=int(min_distance_frac*max(w,h))
    else:
        min_distance=int(min_distance_frac)

    cut_img_4bg=img_show[y:y+h,x:x+w]
    bg_val=int(np.quantile(cut_img_4bg,0.1))


    cut_img=img_show[y:y+h,x:x+w]
    
    cut_img_eq=cv2.equalizeHist(cut_img)
    seg_img,thresh,shed,dist_map_col,contour_list=clustering(cut_img_eq,(mask[y:y+h,x:x+w]).astype(np.uint8),adapt_thres_winsize,min_distance,binary_thres_val=binary_thres_val,gaussian_filter_size=gaussian_filter_size)
    
    contour_list_justsegment=[]
    for c in contour_list:
        at_border=False
        for j in range(len(c)):
            xb,yb,wb,hb=cv2.boundingRect(c[j])
            if xb+x<=1 or yb+y<=1 or xb+x+wb>=img.shape[1]-2 or yb+y+hb>=img.shape[0]-2:
                #pass
                at_border=True
        if not at_border:
            contour_list_justsegment.append(c)


    thresh_mask=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    thresh_mask[:,:,0]=mask[y:y+h,x:x+w]*255
    

    return cut_img_eq,seg_img,thresh_mask,contour_list_justsegment,[x,y,w,h]

def draw_contours(img,contour_list,ofs=(0,0),color=(0,0,0),thickness=1):
    random.seed(111)
    for contour in contour_list:
        if color==(0,0,0):
            col=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        else:
            col=color
        cv2.drawContours(img,contour,0,col,thickness,offset=ofs)
    return img

change=0

def nothing1(x):
    global change
    change=1

def nothing2(x):
    global change
    change=2

def nothing3(x):
    global change
    change=3

def nothing4(x):
    global change
    change=4
    

def select_cluster(event,x,y,flags,param):
    global selected_cluster_pos
    global select_cluster_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_cluster_pos=[x,y]
        select_cluster_changed=True
        print(selected_cluster_pos)

def get_chr_key(k):
    if k<0:
        return ''
    else:
        return chr(k)
try:
    in_dir=sys.argv[1]
except:
    in_dir='images'

if not os.path.exists(in_dir):
    print('directory',in_dir,'does not exist!')
    sys.exit(0)

config_dir='configs'
if not os.path.exists(config_dir):
    print('config directory',config_dir,'does not exist!')

cls='speck'

file_list=os.listdir(in_dir)

img_orig_load=[]
img2_orig_load=[]
f_names_load=[]

img_orig=[]
img2_orig=[]

img_orig_intens=[]
img2_orig_intens=[]

img_orig_aav=[]
img2_orig_aav=[]

cnt=0

if len(file_list)<1:
    print('directory',in_dir,'empty')
    sys.exit(0)

for fl in sorted(file_list):
    img_rgb=cv2.imread(in_dir+'/'+fl)
    img_type=fl.split('_')[-1]
    print(fl,img_type)

    img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
    img_orig_load.append(img_gray)
    img2_orig_load.append(img_rgb)
    f_names_load.append(fl)


    if 'Alexa' in img_type:
        img_orig_aav.append(img_gray)
        img2_orig_aav.append(img_rgb)

    if 'DAPI' in img_type:
        img_orig.append(img_gray)
        img2_orig.append(img_rgb)

    if 'DsRed' in img_type:
        img_orig_intens.append(img_gray)
        img2_orig_intens.append(img_rgb)

    if 'Cy5' in img_type:
        img_orig_intens.append(img_gray)
        img2_orig_intens.append(img_rgb)

    
    cnt+=1

if len(img_orig)<1 or len(img_orig_intens)<1 or len(img_orig_aav)<1:
    print('images do not exist!')
    sys.exit(0)

if not (len(img_orig)==len(img_orig_intens)==len(img_orig_aav)):
    print('number of images does not match!')
    sys.exit(0)

f_names_load_img=f_names_load[1::3]

scale_intens=255.0/float(np.max(img_orig_intens))
scale_aav=255.0/float(np.max(img_orig_aav))

for i in range(len(img_orig_aav)):
    print('i=',i)
    img_orig_intens[i]=(img_orig_intens[i]*scale_intens).astype(np.uint8)
    img2_orig_intens[i]=(img_orig_intens[i]*scale_intens).astype(np.uint8)

    img_orig_aav[i]=(img_orig_aav[i]*scale_intens).astype(np.uint8)
    img2_orig_aav[i]=(img_orig_aav[i]*scale_aav).astype(np.uint8)

cv2.namedWindow('localize',cv2.WINDOW_NORMAL)
cv2.namedWindow('cluster',cv2.WINDOW_NORMAL)
cv2.namedWindow('show_localize',cv2.WINDOW_NORMAL)

cv2.namedWindow('red',cv2.WINDOW_NORMAL)
cv2.namedWindow('green',cv2.WINDOW_NORMAL)

max_thick=3
cv2.createTrackbar('image','localize',0,len(img_orig)-1,nothing1)
cv2.createTrackbar('switch_orig','localize',0,1,nothing1)
cv2.createTrackbar('adapt_thres_win','localize',45,300,nothing1)
cv2.createTrackbar('adapt_thres_C','localize',10,20,nothing1)
cv2.createTrackbar('gaussian_win','localize',25,40,nothing1)
cv2.createTrackbar('binary_thres','localize',135,255,nothing1)
cv2.createTrackbar('min_cell_area','localize',20,500,nothing1)
#high intensity deactivated but available
cv2.createTrackbar('quantile','localize',95,1,nothing1)
cv2.createTrackbar('line_thickness','localize',0,max_thick,nothing1)

max_min_distance=30
cv2.createTrackbar('adapt_thres_win','cluster',45,300,nothing2)
cv2.createTrackbar('min_dist_relative','cluster',0,1,nothing2)
cv2.createTrackbar('min_distance','cluster',4,max_min_distance,nothing2)
cv2.createTrackbar('gaussian_win','cluster',3,7,nothing2)
cv2.createTrackbar('binary_thres','cluster',170,255,nothing2)

cv2.createTrackbar('adapt_thres_win','red',45,300,nothing3)
cv2.createTrackbar('gaussian_win','red',25,40,nothing3)
cv2.createTrackbar('binary_thres','red',135,255,nothing3)
cv2.createTrackbar('low_quantile','red',20,100,nothing3)

cv2.createTrackbar('adapt_thres_win','green',45,300,nothing4)
cv2.createTrackbar('gaussian_win','green',25,40,nothing4)
cv2.createTrackbar('binary_thres','green',135,255,nothing4)
cv2.createTrackbar('low_quantile','green',20,100,nothing3)

cv2.setMouseCallback('show_localize',select_cluster)
select_cluster(cv2.EVENT_LBUTTONDOWN,0,0,0,0)

selected_cluster=1
select_cluster_changed=True
selected_roi=[-10,-10,0,0]

empty_clust=np.zeros((10,10)).astype(np.uint8)
empty_seg=np.zeros((10,10,3)).astype(np.uint8)

clust_img=empty_clust.copy()
thresh_img=empty_seg.copy()

params=defaultdict(lambda: '')

load_config_keys=['1','2','3','4','5','6','7','8','9','0']
save_config_keys=['!','"','§','$','%','&','/','(',')','=']

selected_cluster_dict=defaultdict(lambda:0)
selected_roi_dict=defaultdict(lambda:[-10,-10,0,0])
selected_pos_dict=defaultdict(lambda:(-1,-1))

img_orig_intens_scaled=[]
img_orig_aav_scaled=[]
for img_idx in range(len(img_orig)):
    img_intens_scaled=(img_orig_intens[img_idx].copy()*scale_intens).astype(np.uint8)
    img_aav_scaled=(img_orig_aav[img_idx].copy()*scale_aav).astype(np.uint8)

    img_orig_intens_scaled.append(img_intens_scaled)
    img_orig_aav_scaled.append(img_aav_scaled)

change1=0
img_switch=cv2.getTrackbarPos('switch_orig','localize')
line_thickness=cv2.getTrackbarPos('line_thickness','localize')



try:
    fname='configs/config_6.json'
    f=open(fname,'rt')
    jstr=f.read()
    f.close()

    params=json.loads(jstr)


    cv2.setTrackbarPos('adapt_thres_win','localize',params['adapt_thres'])
    cv2.setTrackbarPos('adapt_thres_C','localize',params['adapt_c'])
    cv2.setTrackbarPos('gaussian_win','localize',params['gaussian_win'])
    cv2.setTrackbarPos('binary_thres','localize',params['binary_thres'])
    cv2.setTrackbarPos('min_cell_area','localize',params['min_cell_area'])
    cv2.setTrackbarPos('quantile','localize',params['quantile'])

    cv2.setTrackbarPos('adapt_thres_win','cluster',params['clust_adapt_thres_win'])
    cv2.setTrackbarPos('gaussian_win','cluster',params['clust_gaussian_win'])
    cv2.setTrackbarPos('binary_thres','cluster',params['clust_binary_thres'])
    cv2.setTrackbarPos('min_dist_relative','cluster',params['clust_dist_relative'])
    cv2.setTrackbarPos('min_distance','cluster',params['clust_min_distance'])

    cv2.setTrackbarPos('binary_thres','red',params['red_binary_thres'])
    cv2.setTrackbarPos('binary_thres','green',params['green_binary_thres'])

    cv2.setTrackbarPos('low_quantile','red',params['red_low_quantile'])
    cv2.setTrackbarPos('low_quantile','green',params['green_low_quantile'])

except:
    print('loading failed!')



while (1):

    old_idx=img_idx
    old_switch=img_switch
    old_thickness=line_thickness

    img_idx=cv2.getTrackbarPos('image','localize')
    img_switch=cv2.getTrackbarPos('switch_orig','localize')
    adapt_thres=cv2.getTrackbarPos('adapt_thres_win','localize')
    adapt_c=cv2.getTrackbarPos('adapt_thres_C','localize')
    gaussian_win=cv2.getTrackbarPos('gaussian_win','localize')
    binary_thres=cv2.getTrackbarPos('binary_thres','localize')
    min_cell_area=cv2.getTrackbarPos('min_cell_area','localize')
    quantile=cv2.getTrackbarPos('quantile','localize')
    line_thickness=cv2.getTrackbarPos('line_thickness','localize')

    cv2.setWindowTitle('show_localize',f_names_load_img[img_idx])

    clust_adapt_thres_win=cv2.getTrackbarPos('adapt_thres_win','cluster')
    clust_gaussian_win=cv2.getTrackbarPos('gaussian_win','cluster')
    clust_binary_thres=cv2.getTrackbarPos('binary_thres','cluster')
    clust_dist_relative=cv2.getTrackbarPos('min_dist_relative','cluster')
    clust_min_distance=cv2.getTrackbarPos('min_distance','cluster')

    red_adapt_thres=cv2.getTrackbarPos('adapt_thres_win','red')
    red_gaussian_win=cv2.getTrackbarPos('gaussian_win','red')
    red_binary_thres=cv2.getTrackbarPos('binary_thres','red')
    red_low_quantile=cv2.getTrackbarPos('low_quantile','red')

    green_adapt_thres=cv2.getTrackbarPos('adapt_thres_win','green')
    green_gaussian_win=cv2.getTrackbarPos('gaussian_win','green')
    green_binary_thres=cv2.getTrackbarPos('binary_thres','green')
    green_low_quantile=cv2.getTrackbarPos('low_quantile','green')

    if not old_idx==img_idx:
        change=0

    if not old_switch==img_switch:
        change=0

    if not old_thickness==line_thickness:
        change=0


    params['adapt_thres']=adapt_thres
    params['adapt_c']=adapt_c
    params['gaussian_win']=gaussian_win
    params['binary_thres']=binary_thres
    params['min_cell_area']=min_cell_area
    params['quantile']=quantile

    params['clust_adapt_thres_win']=clust_adapt_thres_win
    params['clust_gaussian_win']=clust_gaussian_win
    params['clust_binary_thres']=clust_binary_thres
    params['clust_dist_relative']=clust_dist_relative
    params['clust_min_distance']=clust_min_distance


    params['red_adapt_thres']=red_adapt_thres
    params['red_gaussian_win']=red_gaussian_win
    params['red_binary_thres']=red_binary_thres
    params['red_low_quantile']=red_low_quantile

    params['green_adapt_thres']=green_adapt_thres
    params['green_gaussian_win']=green_gaussian_win
    params['green_binary_thres']=green_binary_thres
    params['green_low_quantile']=green_low_quantile

    adapt_thres=adapt_thres*2+3
    adapt_c=adapt_c-10
    gaussian_win=gaussian_win*2+1
    quantile=quantile*0.01+0.0

    red_adapt_thres=red_adapt_thres*2+3
    red_gaussian_win=red_gaussian_win*2+1
    red_low_quantile=(red_low_quantile/100.0)*1.0


    green_adapt_thres=green_adapt_thres*2+3
    green_gaussian_win=green_gaussian_win*2+1
    green_low_quantile=(green_low_quantile/100.0)*1.0

    clust_adapt_thres_win=clust_adapt_thres_win*2+3
    clust_gaussian_win=clust_gaussian_win*2+1

    if line_thickness>=max_thick:
        line_thickness=-1
    else:
        line_thickness+=1

    if clust_dist_relative>0:
        clust_min_distance=float(clust_min_distance)/float(max_min_distance+1)
    
    img=img_orig[img_idx].copy()
    img2=img2_orig[img_idx].copy()
    img_rgb=img2_orig[img_idx].copy()

    img_red=img_orig_intens[img_idx].copy()
    img_green=img_orig_aav[img_idx].copy()

    img_red_rgb=cv2.cvtColor((img_red/np.max(img_red)*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    img_green_rgb=cv2.cvtColor((img_green/np.max(img_green)*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)

    if select_cluster_changed:
        try:
            selected_cluster_dict[img_idx]=int(markers_out[selected_cluster_pos[1]][selected_cluster_pos[0]])
            selected_pos_dict[img_idx]=[selected_cluster_pos[0],selected_cluster_pos[1]]
            print('SELECTED_CLUSTER_DICT=',selected_cluster_dict[img_idx])
            print('SELECTED_POS_DICT=',selected_pos_dict[img_idx])
        except:
            selected_cluster_dict[img_idx]=0
        select_cluster_changed=False

    if img_switch==0:
        if change==1 or change==0:
            res_img,markers_out,_,_=segment_image(img,img2,adapt_thres_winsize=adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=gaussian_win,binary_thres_val=binary_thres,min_cell_area=min_cell_area,intens_quantile=quantile,line_thickness=line_thickness)
        selected_cluster_dict[img_idx]=int(markers_out[selected_pos_dict[img_idx][1]][selected_pos_dict[img_idx][0]])        

        if change==3 or change==0:
            res_img_red,markers_out_red,_,_=segment_image(img_red,img_red_rgb,adapt_thres_winsize=red_adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=red_gaussian_win,binary_thres_val=red_binary_thres,min_cell_area=min_cell_area,intens_quantile=quantile,line_thickness=line_thickness,same_color=True,low_quantile=red_low_quantile)
        if change==4 or change==0:
            res_img_green,markers_out_green,_,_=segment_image(img_green,img_green_rgb,adapt_thres_winsize=green_adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=green_gaussian_win,binary_thres_val=green_binary_thres,min_cell_area=min_cell_area,intens_quantile=quantile,line_thickness=line_thickness,same_color=True,low_quantile=green_low_quantile)
         

        if selected_cluster_dict[img_idx]>0:
            clust_img,seg_img,thresh_img,contour_list,selected_roi_dict[img_idx]=cluster_img(img_rgb,markers_out,selected_cluster_dict[img_idx],adapt_thres_winsize=clust_adapt_thres_win,min_distance_frac=clust_min_distance,binary_thres_val=clust_binary_thres,gaussian_filter_size=clust_gaussian_win)
            clust_img_rgb=cv2.cvtColor(clust_img,cv2.COLOR_GRAY2RGB)
            draw_contours(clust_img_rgb,contour_list)
        else:
            clust_img_rgb=empty_seg.copy()
            seg_img=empty_seg.copy()
            thresh_img=empty_seg.copy()
            contour_list=[]
        seg_img_gray=seg_img[:,:,0]
        
        cv2.rectangle(res_img,(selected_roi_dict[img_idx][0],selected_roi_dict[img_idx][1]),(selected_roi_dict[img_idx][0]+selected_roi_dict[img_idx][2],selected_roi_dict[img_idx][1]+selected_roi_dict[img_idx][3]),(255,255),3)
    elif img_switch==2:
        #NOT USED AT THE MOMENT!
        res_img,markers_out,_,_=segment_image(img,img2,adapt_thres_winsize=adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=gaussian_win,binary_thres_val=binary_thres)        
        selected_cluster_dict[img_idx]=int(markers_out[selected_cluster_pos[1]][selected_cluster_pos[0]])

    else:
        res_img=img2_orig[img_idx]
        clust_img_rgb=cv2.cvtColor(clust_img,cv2.COLOR_GRAY2RGB)

        res_img_red=img_orig_intens[img_idx]
        res_img_green=img_orig_aav[img_idx]
                    

    small_img=cv2.resize(res_img,(res_img.shape[1]//1,res_img.shape[0]//1))

    cv2.imshow('show_localize',small_img)
    cv2.imshow('cluster',np.hstack((thresh_img,clust_img_rgb)))

    cv2.imshow('red',res_img_red)
    cv2.imshow('green',res_img_green)


    k=cv2.waitKey(1)

    k_chr=get_chr_key(k)

    if k_chr in save_config_keys:
        num=save_config_keys.index(k_chr)
        print('num=',num)
        jstr=json.dumps(params,indent=4)            
        f=open(config_dir+'/config_'+str(num)+'.json','wt')
        f.write(jstr+'\n')
        f.close()

    if k_chr in load_config_keys:
        num=load_config_keys.index(k_chr)

        fname=config_dir+'/config_'+str(num)+'.json'

        try:
            f=open(fname,'rt')
            jstr=f.read()
            f.close()

            params=json.loads(jstr)


            cv2.setTrackbarPos('adapt_thres_win','localize',params['adapt_thres'])
            cv2.setTrackbarPos('adapt_thres_C','localize',params['adapt_c'])
            cv2.setTrackbarPos('gaussian_win','localize',params['gaussian_win'])
            cv2.setTrackbarPos('binary_thres','localize',params['binary_thres'])
            cv2.setTrackbarPos('min_cell_area','localize',params['min_cell_area'])
            cv2.setTrackbarPos('quantile','localize',params['quantile'])

            cv2.setTrackbarPos('adapt_thres_win','cluster',params['clust_adapt_thres_win'])
            cv2.setTrackbarPos('gaussian_win','cluster',params['clust_gaussian_win'])
            cv2.setTrackbarPos('binary_thres','cluster',params['clust_binary_thres'])
            cv2.setTrackbarPos('min_dist_relative','cluster',params['clust_dist_relative'])
            cv2.setTrackbarPos('min_distance','cluster',params['clust_min_distance'])

            cv2.setTrackbarPos('binary_thres','red',params['red_binary_thres'])
            cv2.setTrackbarPos('gaussian_win','red',params['red_gaussian_win'])
            cv2.setTrackbarPos('adapt_thres_win','red',params['red_adapt_thres'])
            cv2.setTrackbarPos('low_quantile','red',params['red_low_quantile'])

            cv2.setTrackbarPos('binary_thres','green',params['green_binary_thres'])
            cv2.setTrackbarPos('gaussian_win','green',params['green_gaussian_win'])
            cv2.setTrackbarPos('adapt_thres_win','green',params['green_adapt_thres'])
            cv2.setTrackbarPos('low_quantile','green',params['green_low_quantile'])

        except:
            print('loading failed!')

    
    if k==27:
        #ESC is pessed, quit
        break
    if k==120 or k==115:
        #x is pressed, compute whole image
        #s is pressed, compute and save all images
        if k==120:
            img_idx_list=[img_idx]
            save_one=True
        else:
            img_idx_list=range(len(img_orig))
            now=datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")
            res_dir=dt_string+'_result'
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            params_withdir=params.copy()
            params_withdir['working_directory']=os.path.abspath(in_dir)

            jstr=json.dumps(params_withdir,indent=4)
            f=open(res_dir+'/config.json','wt')
            f.write(jstr+'\n')
            f.close()


                
            save_one=False

            

        for img_idx_s in img_idx_list:
            img=img_orig[img_idx_s].copy()
            img2=img2_orig[img_idx_s].copy()
            img_rgb=img2_orig[img_idx_s].copy()

        

            img_red=img_orig_intens[img_idx_s].copy()
            img_green=img_orig_aav[img_idx_s].copy()

            img_red_rgb=cv2.cvtColor(img_red,cv2.COLOR_GRAY2RGB)
            img_green_rgb=cv2.cvtColor(img_green,cv2.COLOR_GRAY2RGB)

            res_img_red,markers_out_red,cont_in_red,cont_out_red=segment_image(img_red,img_red_rgb,adapt_thres_winsize=red_adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=red_gaussian_win,binary_thres_val=red_binary_thres,min_cell_area=min_cell_area,intens_quantile=quantile,line_thickness=line_thickness,low_quantile=red_low_quantile)
            res_img_green,markers_out_green,cont_in_green,cont_out_green=segment_image(img_green,img_green_rgb,adapt_thres_winsize=adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=green_gaussian_win,binary_thres_val=green_binary_thres,min_cell_area=min_cell_area,intens_quantile=quantile,line_thickness=line_thickness,low_quantile=green_low_quantile)

            red_mask_img=np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
            cv2.drawContours(red_mask_img,cont_in_red,-1,(255),-1)
            cv2.drawContours(red_mask_img,cont_out_red,-1,(255),-1)

            green_mask_img=np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
            cv2.drawContours(green_mask_img,cont_in_green,-1,(255),-1)
            cv2.drawContours(green_mask_img,cont_out_green,-1,(255),-1)


            cell_mask_img=np.zeros((img.shape[0],img.shape[1])).astype(np.uint16)

            res_img,markers_out,cont_in,cont_out=segment_image(img,img2,adapt_thres_winsize=adapt_thres,adapt_thres_c=adapt_c,gaussian_filter_size=gaussian_win,binary_thres_val=binary_thres,min_cell_area=min_cell_area,intens_quantile=quantile,line_thickness=line_thickness)
            segments=(np.unique(markers_out)).astype(np.uint16)    
            print('cont_out:',len(cont_out))
            cv2.drawContours(img_rgb,cont_out,-1,(0,0,255),line_thickness)

            count_cont=1

            for i in range(len(cont_out)):
                cv2.drawContours(cell_mask_img,cont_out,i,(count_cont),-1)
                count_cont+=1

        

            print('segments:',segments)
            for sel_clust in segments:
                if sel_clust<1:
                    continue
                clust_imgx,seg_imgx,thresh_imgx,contour_listx,selected_roix=cluster_img(img_rgb,markers_out,sel_clust,adapt_thres_winsize=clust_adapt_thres_win,min_distance_frac=clust_min_distance,binary_thres_val=clust_binary_thres,gaussian_filter_size=clust_gaussian_win)
                draw_contours(img_rgb,contour_listx,ofs=(selected_roix[0],selected_roix[1]),thickness=line_thickness)

                for i in range(len(contour_listx)):
                    draw_contours(cell_mask_img,[contour_listx[i]],ofs=(selected_roix[0],selected_roix[1]),color=(count_cont),thickness=-1)
                    count_cont+=1

                    
            if not save_one:
                f_name_split=f_names_load_img[img_idx_s].split('_')

                res_name='_'.join(f_name_split[:-1]+['MASK.tif'])
                cv2.imwrite(res_dir+'/'+res_name,cell_mask_img)                

                res_name_red='_'.join(f_name_split[:-1]+['MASKred.tif'])
                cv2.imwrite(res_dir+'/'+res_name_red,red_mask_img)

                res_name_green='_'.join(f_name_split[:-1]+['MASKgreen.tif'])
                cv2.imwrite(res_dir+'/'+res_name_green,green_mask_img)

                print(res_name)  
                        
        if save_one:
            cv2.namedWindow('cell segments',cv2.WINDOW_NORMAL)
            cv2.imshow('cell segments',img_rgb)
                
            while cv2.waitKey(1)!=120:
                pass
            print('done')   

            cv2.destroyWindow('cell mask')
            cv2.destroyWindow('cell segments')
        
        
cv2.destroyAllWindows()
