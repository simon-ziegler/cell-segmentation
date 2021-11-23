import sys
import os
import csv
import string
import cv2
import numpy as np

from skimage.feature import peak_local_max, greycomatrix, greycoprops
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

import random
import time
import json
from collections import defaultdict
from datetime import datetime

import pickle

#global class_color
class_color=[(0,0,255),(255,0,0),(0,255,255)]

def compute_gaussian_list(img_intens_list,mask_list,num_gaussians=2,invert_mask=False,equalize=False,high_quantile=1.0):


    print('high_quantile:',high_quantile)
    
    X_flat=np.array([])
    
    for img_intens,mask in zip(img_intens_list,mask_list):
        print(img_intens.shape,mask.shape)

        mask_bin=(mask>0)*1
        if invert_mask:
            mask_bin=1-mask_bin

        if equalize:
            img_masked=cv2.equalizeHist(img_intens)*mask_bin
        else:
            img_masked=img_intens*mask_bin

        X=img_masked.ravel()
        X_notnull=X[X!=0]

        X_flat=np.concatenate((X_flat,X_notnull))

    X_flat[np.isnan(X_flat)]=1.

    q_val=np.quantile(X_flat,high_quantile)

    X_flat=X_flat[X_flat<q_val]
    


    gm=GaussianMixture(n_components=num_gaussians,random_state=0).fit(X_flat.reshape(-1,1))

    means=[]
    sigma=[]
    for i in range(num_gaussians):
        means.append(gm.means_[i][0])
        sigma.append(gm.covariances_[i][0][0])
    
    return means,sigma,X_flat

def classify_gaussians(img_intens,mask,means,sigma,img_show_rgb,mask_id_list=[]):
    
    num_seg=np.max(mask)+1


    print('num_seg:',num_seg)

    if len(mask_id_list)<1:
        mask_id_list=list(range(1,num_seg))

    mask_class_dict=defaultdict(lambda:-1)
    cell_means=defaultdict(lambda:-1)
    glcm_homo=defaultdict(lambda:-1)
    glcm_entropy=defaultdict(lambda:-1)
    glcm_energy=defaultdict(lambda:-1)

    for i in mask_id_list:
        mask_single=(mask==i)*1

        cont_single,ret=cv2.findContours(mask_single.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        img_masked_single=img_intens*mask_single
        flat_values=img_masked_single.ravel()
        flat_values_notnull=flat_values[flat_values>0]

        flat_values_notnull[np.isnan(flat_values_notnull)]=1.

        flat_mean=np.mean(flat_values_notnull)

        if np.isnan(flat_mean):
            mask_class_dict[i]=-1
            continue

        cell_means[i]=flat_mean

        pdens=[((m-flat_mean)**2)/s for m,s in zip(means,sigma)]
        c=pdens.index(min(pdens))

        #yellow: low expressure
        #blue: high expressure
        #red: background

        cv2.drawContours(img_show_rgb,cont_single,-1,class_color[c],2)
            
        mask_class_dict[i]=c

        #compute texture features
        angles=[0,0.5*np.pi,np.pi,1.5*np.pi]

        cont_single_largest=cont_single[0]
        for c in cont_single[1:]:
            if len(c)>len(cont_single_largest):
                cont_signle_largest=c


        x,y,w,h=cv2.boundingRect(cont_single_largest)
        cell_img=img_intens[y:y+h,x:x+w]



        glcm=greycomatrix(cell_img,distances=[1],angles=angles,levels=np.max(cell_img)+1,normed=True)
        glcm_nonzero=glcm[1:,1:,:,:]

        homogeneity=np.mean(greycoprops(glcm_nonzero,'homogeneity'))
        glcm_homo[i]=homogeneity

        entropy=0.0
        for k in range(glcm.shape[3]):
            entropy+=-np.sum((glcm_nonzero[:,:,0,k]+0.0000001)*np.log2(glcm_nonzero[:,:,0,k]+0.0000001))
        glcm_entropy[i]=np.exp(-0.2*entropy)

        energy=np.mean(greycoprops(glcm_nonzero,'energy'))
        glcm_energy[i]=np.exp(-1.0*energy)

    return mask_class_dict, cell_means, glcm_homo, glcm_entropy, glcm_energy

def draw_classes(img,contours,mask_class,thickness=2):

    img_cls=img.copy()
    for i in contours:
        cv2.drawContours(img_cls,[contours[i]],0,class_color[mask_class[i]],thickness)
    return img_cls

def load_images_result(res_dir):

    config_name=res_dir+'/config.json'
    
    try:
        f=open(config_name,'rt')
        jstr=f.read()
        f.close()
        
        params=json.loads(jstr,)
    except:
        print('can not load',config_name)
        return -1

    try:
        working_dir=params['working_directory']
    except:
        print('can not find working_directory in',config_name)
        return -1

    print('working_dir:',working_dir)

    in_dir=working_dir

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
            print('Alexa')

        if 'DAPI' in img_type:
            img_orig.append(img_gray)
            img2_orig.append(img_rgb)
            print('DAPI')

        #if 'DsRed' or 'CY5' in img_type:
        if 'DsRed' in img_type:
            img_orig_intens.append(img_gray)
            img2_orig_intens.append(img_rgb)
            print('DsRed')

        if 'Cy5' in img_type:
            print('Cy5')
            img_orig_intens.append(img_gray)
            img2_orig_intens.append(img_rgb)

        
        cnt+=1

    f_names_load_img=f_names_load[1::3]

    file_list=os.listdir(res_dir)
    mask_list=[]
    mask_red_list=[]
    mask_green_list=[]

    for fl in sorted(file_list):
        if not fl.endswith('.tif'):
            continue
        if not 'MASK' in fl:
            continue
        print('MASK_FILE:',fl)

        img_mask=cv2.imread(os.path.join(res_dir,fl),cv2.IMREAD_ANYDEPTH)

        if 'MASKred' in fl:
            mask_red_list.append(img_mask)
        elif 'MASKgreen' in fl:
            mask_green_list.append(img_mask)
        else:
            mask_list.append(img_mask)


    return img_orig,img_orig_intens,img_orig_aav,mask_list,mask_red_list,mask_green_list,f_names_load_img

def plot_gaussians(means,sigma,rawdata,plt,legends=None):

    if legends==None:
        legends=['']*len(means)
    elif len(legends)!=len(means):
        legends=['']*len(means)

    test_vals_4hist=np.asarray(range(1,int(np.max(rawdata)+1)))
    histo_comp=np.histogram(rawdata,bins=test_vals_4hist)[0]

    test_vals=test_vals_4hist[1:]

    histo=histo_comp

    max_pdf=0
    for mean,sigma,l in zip(means,sigma,legends):    
        pdf=stats.norm.pdf(test_vals,mean-1,np.sqrt(sigma))
        plt.plot(test_vals,pdf,label=l)
        max_pdf=max(max_pdf,np.max(pdf))

    histo_scaled=histo/rawdata.shape[0]

    plt.plot(test_vals,histo_scaled,label='raw_values')

    return plt    

def write_mask_id(img,mask,min_cell_size=10):
    img_id=img.copy()
    img_id_rgb=cv2.cvtColor(img_id.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    max_id=int(np.max(mask))

    centers=defaultdict(lambda:None)
    perimeters=defaultdict(lambda:None)
    areas=defaultdict(lambda:None)
    contour_list=defaultdict(lambda:np.zeros((2,1,2)))

    for i in range(1,max_id+1):
        mask_img=255*(mask==i)
        mask_area=np.count_nonzero(mask_img)
        if mask_area<min_cell_size:
            continue
        cont_single,ret=cv2.findContours(mask_img.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        color=(0,0,255)
        color_text=(255,255,255)

        cv2.drawContours(img_id_rgb,cont_single,0,color,2)

        best_cont=cont_single[0]
        for cont in cont_single[1:]:
            if len(cont)>len(best_cont):
                best_cont=cont

        M=cv2.moments(best_cont)
        cX=int(M['m10']/(M['m00']+0.00001))
        cY=int(M['m01']/(M['m00']+0.00001))
            
        if M['m00']==0:
            print('-------------------')
            print('cont_len:',len(best_cont))
            print('-------------------')

        contour_list[i]=best_cont
        centers[i]=(cX,cY)
        perimeters[i]=cv2.arcLength(best_cont,True)
        areas[i]=mask_area

        text=str(i)
        font=cv2.FONT_HERSHEY_SIMPLEX

        f_size=0.3
        f_stroke=1


        textsize=cv2.getTextSize(text,font,f_size,f_stroke)[0]

        textX=cX-textsize[0]//2
        textY=cY+textsize[1]//2

        cv2.putText(img_id_rgb,text,(textX,textY),font,f_size,color_text,1)
        
    
    return img_id_rgb,centers,perimeters,areas,contour_list

def str_r(a,dec=2):
    return str(round(a,dec))

def get_output_table(centers,perimeters,areas,red_cell_means,green_cell_means,mask_class_red_one,mask_class_green_one):    
    output_table=[]

    output_table.append(['ID','center','perimeter','cell area','red cell mean','green cell mean','class red','class green'])

    vals=list(centers)
    for i in vals:
        output_table.append([str(i),str(centers[i][0])+'-'+str(centers[i][1]),str_r(perimeters[i]),str_r(areas[i]),str_r(red_cell_means[i]),str_r(green_cell_means[i]),str(mask_class_red_one[i]),str(mask_class_green_one[i])])
        
    return output_table

def sortout_homo(mask_class_list,cell_homo,thres=0.1):
    mask_class_list_sortout=mask_class_list.copy()
    for i in mask_class_list_sortout:
        if cell_homo[i]<thres:
            mask_class_list_sortout[i]=0
    return mask_class_list_sortout

def classifiy_means_homo(cell_means,cell_homo,changed_by_hand,means,sigma,thres=0.1):
    mask_class_list=defaultdict(lambda:-1)

    bg_thres=sigma[0]<0.
        

    for i in cell_means:

        if changed_by_hand[i]>=0:
            mask_class_list[i]=changed_by_hand[i]
            continue
        
        if cell_homo[i]<thres:
            mask_class_list[i]=0
        else:
            if not bg_thres:
                pdens=[((m-cell_means[i])**2)/s for m,s in zip(means,sigma)]
                c=pdens.index(min(pdens))
                mask_class_list[i]=c
            else:
                pdens=[((m-cell_means[i])**2)/s for m,s in zip(means[1:],sigma[1:])]
                c=pdens.index(min(pdens))+1
                mask_class_list[i]=c
                if cell_means[i]<means[0]:
                    mask_class_list[i]=0
                
    return mask_class_list
        

def select_cell_red(event,x,y,flags,param):
    global selected_red_cell_pos
    global select_red_cell_pos_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_red_cell_pos=[x,y]
        print(selected_red_cell_pos)
        select_red_cell_pos_changed=True

def select_cell_green(event,x,y,flags,param):
    global selected_green_cell_pos
    global select_green_cell_pos_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_green_cell_pos=[x,y]
        print(selected_green_cell_pos)
        select_green_cell_pos_changed=True

def get_chr_key(k):
    if k<0:
        return ''
    else:
        return chr(k)

def nothing(x):
    pass

def cls_red(x):
    global cls_red_change
    cls_red_change=True

def cls_green(x):
    global cls_green_change
    cls_green_change=True

selected_red_cell_pos=0
selected_green_cell_pos=0

try:
    in_dir=sys.argv[1]
except:
    in_dir='result'

try:
    img_orig, img_orig_intens, img_orig_aav, mask_list, mask_red_list, mask_green_list, fnames=load_images_result(in_dir)
except:
    print('can not load data from directory',in_dir)
    sys.exit(0)

sc_fact=16.0
q_high=0.995

scale_all_single=False

sc_fact_list=[int(np.quantile(img_orig,q_high)),int(np.quantile(img_orig_intens,q_high)),int(np.quantile(img_orig_aav,q_high))]
sc_fact=np.min(sc_fact_list)


scale_orig=1.0/np.quantile(img_orig,q_high)*sc_fact
scale_intens=1.0/np.quantile(img_orig_intens,q_high)*sc_fact
scale_aav=1.0/np.quantile(img_orig_aav,q_high)*sc_fact

scale_orig_display=255.0/np.quantile(img_orig,q_high)
scale_intens_display=255.0/np.quantile(img_orig_intens,q_high)
scale_aav_display=255.0/np.quantile(img_orig_aav,q_high)

img_orig_display=[]
img_orig_intens_display=[]
img_orig_aav_display=[]

for i in range(len(img_orig)):
    if scale_all_single:
        img_orig_display.append(np.clip(img_orig[i].copy()*255.0/np.quantile(img_orig[i],q_high),0,255).astype(np.uint8))
        img_orig_intens_display.append(np.clip(img_orig_intens[i].copy()*255.0/np.quantile(img_orig_intens[i],q_high),0,255).astype(np.uint8))
        img_orig_aav_display.append(np.clip(img_orig_aav[i].copy()*255.0/np.quantile(img_orig_aav[i],q_high),0,255).astype(np.uint8))
    else:
        img_orig_display.append(np.clip(img_orig[i].copy()*scale_orig_display,0,255).astype(np.uint8))
        img_orig_intens_display.append(np.clip(img_orig_intens[i].copy()*scale_intens_display,0,255).astype(np.uint8))
        img_orig_aav_display.append(np.clip(img_orig_aav[i].copy()*scale_aav_display,0,255).astype(np.uint8))


img_orig_sc=[]
img_orig_intens_sc=[]
img_orig_aav_sc=[]

img_id_sc=[]

centers_list=[]
perimeters_list=[]
areas_list=[]
contour_list=[]

for i in range(len(img_orig)):
    if scale_all_single:
        img_orig_sc.append(np.clip(img_orig[i]*sc_fact/np.quantile(img_orig[i],q_high),0,sc_fact).astype(np.int))
        img_orig_intens_sc.append(np.clip(img_orig_intens[i]*sc_fact/np.quantile(img_orig_intens[i],q_high),0,sc_fact).astype(np.int))
        img_orig_aav_sc.append(np.clip(img_orig_aav[i]*sc_fact/np.quantile(img_orig_aav[i],q_high),0,sc_fact).astype(np.int))
    else:
        img_orig_sc.append(np.clip(img_orig[i].copy()*scale_orig,0,sc_fact_list[0]).astype(np.uint8))
        img_orig_intens_sc.append(np.clip(img_orig_intens[i].copy()*scale_intens,0,sc_fact_list[1]).astype(np.uint8))
        img_orig_aav_sc.append(np.clip(img_orig_aav[i].copy()*scale_aav,0,sc_fact_list[2]).astype(np.uint8))


    img_id,centers,perimeters,areas,contours=write_mask_id(img_orig_sc[i],mask_list[i])
    img_id_sc.append(img_id)
    centers_list.append(centers)
    perimeters_list.append(perimeters)
    areas_list.append(areas)
    contour_list.append(contours)

    cv2.imwrite('plots/'+'_'.join(fnames[i].split('_')[:-1])+'_red.jpg',img_orig_intens_sc[i])
    cv2.imwrite('plots/'+'_'.join(fnames[i].split('_')[:-1])+'_green.jpg',img_orig_aav_sc[i])



quantile=0.99

if True:

    means,sigma,rawdata_fg=compute_gaussian_list(img_orig_intens_sc[0:1],mask_red_list,num_gaussians=2,high_quantile=quantile)
    means_bg,sigma_bg,rawdata_bg=compute_gaussian_list(img_orig_intens_sc[0:1],mask_red_list,num_gaussians=1,invert_mask=True,high_quantile=quantile)
    print('red:',means,sigma)

    green_means,green_sigma,green_rawdata_fg=compute_gaussian_list(img_orig_aav_sc[0:1],mask_green_list,num_gaussians=2,high_quantile=quantile)
    green_means_bg,green_sigma_bg,green_rawdata_bg=compute_gaussian_list(img_orig_aav_sc[0:1],mask_green_list,num_gaussians=1,invert_mask=True,high_quantile=quantile)
    print('green:',green_means,green_sigma)


    pickle.dump([means,sigma,rawdata_fg],open('red_fg.bin','wb'))
    pickle.dump([means_bg,sigma_bg,rawdata_bg],open('red_bg.bin','wb'))

    pickle.dump([green_means,green_sigma,green_rawdata_fg],open('green_fg.bin','wb'))
    pickle.dump([green_means_bg,green_sigma_bg,green_rawdata_bg],open('green_bg.bin','wb'))

else:
    [means,sigma,rawdata_fg]=pickle.load(open('red_fg.bin','rb'))
    [means_bg,sigma_bg,rawdata_bg]=pickle.load(open('red_bg.bin','rb'))
    print('red:',means,sigma)

    [green_means,green_sigma,green_rawdata_fg]=pickle.load(open('green_fg.bin','rb'))
    [green_means_bg,green_sigma_bg,green_rawdata_bg]=pickle.load(open('green_bg.bin','rb'))


if means[0]>means[1]:
    means.reverse()
    sigma.reverse()

if green_means[0]>green_means[1]:
    green_means.reverse()
    green_sigma.reverse()


print('number of cells:',int(np.max(mask_list)))

ref_id=0

fig,axs=plt.subplots(2,1)
fig.tight_layout(pad=3.0)
_=plot_gaussians(means+means_bg,sigma+sigma_bg,np.concatenate((rawdata_fg,rawdata_bg)),axs[0],legends=['fg1','fg2','background'])
axs[0].set_title('red channel complete')
axs[0].legend()

_=plot_gaussians(means,sigma,rawdata_fg,axs[1],legends=['fg1','fg2'])
axs[1].set_title('red channel just foreground')
axs[1].legend()

fig.savefig('plots/tmpplot.png')
plot_img=cv2.imread('plots/tmpplot.png')

green_fig,axs2=plt.subplots(2,1)
green_fig.tight_layout(pad=3.0)
_=plot_gaussians(green_means+green_means_bg,green_sigma+green_sigma_bg,np.concatenate((green_rawdata_fg,green_rawdata_bg)),axs2[0],legends=['fg1_green','fg2_green','background_green'])
axs2[0].set_title('green channel complete')
axs2[0].legend()

_=plot_gaussians(green_means,green_sigma,green_rawdata_fg,axs2[1],legends=['fg1_green','fg2_green'])
axs2[1].set_title('green channel just foreground')
axs2[1].legend()

green_fig.savefig('plots/tmpplot_green.png')
plot_green_img=cv2.imread('plots/tmpplot_green.png')

pos_red=0
pos_green=0
total_red=0
total_green=0
pos_redgreen=0

img_classified_red=[]
img_classified_green=[]
red_means_list=[]
red_homo_list=[]
red_entropy_list=[]
red_energy_list=[]

green_means_list=[]
green_homo_list=[]
green_entropy_list=[]
green_energy_list=[]

red_class_list=[]
green_class_list=[]

red_changed_by_hand=[]
green_changed_by_hand=[]



for k in range(len(img_orig)):
    red_changed_by_hand.append(defaultdict(lambda:-1))
    green_changed_by_hand.append(defaultdict(lambda:-1))

for i in range(len(img_orig)):

    id_list=list(centers_list[i].keys())
    print('i=',i)
    
    img_show=img_orig_intens_display[i].copy().astype(np.uint8)
    img_show_rgb=cv2.cvtColor(img_show,cv2.COLOR_GRAY2RGB)
    mask_class_red,red_cell_means,red_cell_homo,red_cell_entropy,red_cell_energy=classify_gaussians(img_orig_intens_sc[i],mask_list[i],means_bg+means,sigma_bg+sigma,img_show_rgb,mask_id_list=id_list)
    img_classified_red.append(img_show_rgb)
    red_means_list.append(red_cell_means)
    red_class_list.append(mask_class_red)
    red_homo_list.append(red_cell_homo)
    red_entropy_list.append(red_cell_entropy)
    red_energy_list.append(red_cell_energy)


    img_show=img_orig_aav_display[i].copy().astype(np.uint8)
    img_show_rgb=cv2.cvtColor(img_show,cv2.COLOR_GRAY2RGB)
    mask_class_green,green_cell_means,green_cell_homo,green_cell_entropy,green_cell_energy=classify_gaussians(img_orig_aav_sc[i],mask_list[i],green_means_bg+green_means,green_sigma_bg+green_sigma,img_show_rgb,mask_id_list=id_list)
    img_classified_green.append(img_show_rgb)
    green_means_list.append(green_cell_means)
    green_class_list.append(mask_class_green)
    green_homo_list.append(green_cell_homo)
    green_entropy_list.append(green_cell_entropy)
    green_energy_list.append(green_cell_energy)

    mask_class_red_list=np.asarray([mask_class_red[c] for c in list(mask_class_red)])
    mask_class_green_list=np.asarray([mask_class_green[c] for c in list(mask_class_green)])


    pos_mask_class_red=1*(mask_class_red_list==2)
    pos_mask_class_green=1*(mask_class_green_list==2)

    pos_mask_class_redgreen=pos_mask_class_red*pos_mask_class_green

    pos_red=np.count_nonzero(mask_class_red_list[mask_class_red_list==2])
    total_red=np.count_nonzero(mask_class_red_list[mask_class_red_list>0])

    pos_green=np.count_nonzero(mask_class_green_list[mask_class_green_list==2])
    total_green=np.count_nonzero(mask_class_red_list[mask_class_green_list>0])

    pos_redgreen=np.count_nonzero(pos_mask_class_redgreen)


    fig_c,axs_c=plt.subplots(1,1)
    axs_c.bar(['red','green','both','total_red','total_green'],[pos_red,pos_green,pos_redgreen,total_red,total_green])

    fig_c.savefig('plots/tmpplot_classes'+str(i)+'.png')



fig_c,axs_c=plt.subplots(1,1)
axs_c.bar(['red','green','both','total_red','total_green'],[pos_red,pos_green,pos_redgreen,total_red,total_green])

fig_c.savefig('plots/tmpplot_classes.png')
plot_c=cv2.imread('plots/tmpplot_classes.png')

cv2.namedWindow('intens',cv2.WINDOW_NORMAL)
cv2.namedWindow('orig',cv2.WINDOW_NORMAL)
cv2.namedWindow('aav',cv2.WINDOW_NORMAL)
cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
cv2.namedWindow('plot_intens',cv2.WINDOW_NORMAL)
cv2.namedWindow('plot_aav',cv2.WINDOW_NORMAL)

cv2.namedWindow('select',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('image','select',0,len(img_orig)-1,nothing)

cv2.createTrackbar('orig_red','intens',0,2,nothing)
cv2.createTrackbar('class','intens',0,2,cls_red)
cv2.createTrackbar('texture_type','intens',0,2,nothing)
cv2.createTrackbar('texture_thres','intens',0,100,nothing)
cv2.createTrackbar('bg_quantile','intens',0,100,nothing)
cv2.createTrackbar('thres_vs_gaussian','intens',0,1,nothing)
cv2.setMouseCallback('intens',select_cell_red)
select_cell_red(cv2.EVENT_LBUTTONDOWN,0,0,0,0)
select_red_cell_pos_changed=False

cv2.createTrackbar('orig_green','aav',0,2,nothing)
cv2.createTrackbar('class','aav',0,2,cls_green)
cv2.createTrackbar('texture_type','aav',0,2,nothing)
cv2.createTrackbar('texture_thres','aav',0,100,nothing)
cv2.createTrackbar('bg_quantile','aav',0,100,nothing)
cv2.createTrackbar('thres_vs_gaussian','aav',0,1,nothing)
cv2.setMouseCallback('aav',select_cell_green)
select_cell_green(cv2.EVENT_LBUTTONDOWN,0,0,0,0)
select_green_cell_pos_changed=False

select_red_cell=defaultdict(lambda:0)
select_green_cell=defaultdict(lambda:0)

cls_red_change=False
cls_green_change=False

red_bg_quant=0.001
green_bg_quant=0.001
        

    

while True:

    red_bg_quant=cv2.getTrackbarPos('bg_quantile','intens')*0.01
    green_bg_quant=cv2.getTrackbarPos('bg_quantile','aav')*0.01

    red_bg_thres=np.quantile(img_orig_intens,red_bg_quant)
    green_bg_thres=np.quantile(img_orig_aav,green_bg_quant)

    red_thres=cv2.getTrackbarPos('thres_vs_gaussian','intens')
    green_thres=cv2.getTrackbarPos('thres_vs_gaussian','aav')


    ref_id=cv2.getTrackbarPos('image','select')

    red_homo_thres=cv2.getTrackbarPos('texture_thres','intens')*0.01
    red_texture_type=cv2.getTrackbarPos('texture_type','intens')

    if red_texture_type==0:
        red_texture_val=red_homo_list
    elif red_texture_type==1:
        red_texture_val=red_entropy_list
    else:
        red_texture_val=red_energy_list

    
    green_homo_thres=cv2.getTrackbarPos('texture_thres','aav')*0.01
    green_texture_type=cv2.getTrackbarPos('texture_type','aav')

    if green_texture_type==0:
        green_texture_val=green_homo_list
    elif green_texture_type==1:
        green_texture_val=green_entropy_list
    else:
        green_texture_val=green_energy_list




    red_class_sortout_list=[]
    green_class_sortout_list=[]
    for k in range(len(red_class_list)):
        if red_thres==0:
            red_class_sortout_single=classifiy_means_homo(red_means_list[k],red_texture_val[k],red_changed_by_hand[k],[red_bg_thres]+means,[-1]+sigma,red_homo_thres)
        else:
            red_class_sortout_single=classifiy_means_homo(red_means_list[k],red_texture_val[k],red_changed_by_hand[k],means_bg+means,sigma_bg+sigma,red_homo_thres)
            
        if green_thres==0:
            green_class_sortout_single=classifiy_means_homo(green_means_list[k],green_texture_val[k],green_changed_by_hand[k],[green_bg_thres]+green_means,[-1]+green_sigma,green_homo_thres)
        else:
            green_class_sortout_single=classifiy_means_homo(green_means_list[k],green_texture_val[k],green_changed_by_hand[k],green_means_bg+green_means,green_sigma_bg+green_sigma,green_homo_thres)

        red_class_sortout_list.append(red_class_sortout_single)
        green_class_sortout_list.append(green_class_sortout_single)
        
    if select_red_cell_pos_changed:
        print('selected_red_cell_pos:',selected_red_cell_pos)
        select_red_cell[ref_id]=mask_list[ref_id][selected_red_cell_pos[1]][selected_red_cell_pos[0]]
        print('ID:',select_red_cell[ref_id])
        select_red_cell_pos_changed=False

    if select_green_cell_pos_changed:
        print('selected_green_cell_pos:',selected_green_cell_pos)
        try:
            select_green_cell[ref_id]=mask_list[ref_id][selected_green_cell_pos[1]][selected_green_cell_pos[0]]
            print('ID:',select_green_cell[ref_id])
     
        except:
            select_green_cell[ref_id]=0
        select_green_cell_pos_changed=False

    if cls_red_change:
        if red_class_sortout_list[ref_id][select_red_cell[ref_id]]>=0:
            red_changed_by_hand[ref_id][select_red_cell[ref_id]]=cv2.getTrackbarPos('class','intens')
            cls_red_change=False

    if red_class_sortout_list[ref_id][select_red_cell[ref_id]]>=0:
        sel_cell=red_changed_by_hand[ref_id][select_red_cell[ref_id]]
        if sel_cell<0:
            sel_cell=red_class_sortout_list[ref_id][select_red_cell[ref_id]]

        cv2.setTrackbarPos('class','intens',sel_cell)
        cls_red_change=False

    if cls_green_change:
        if green_class_list[ref_id][select_green_cell[ref_id]]>=0:
            green_changed_by_hand[ref_id][select_green_cell[ref_id]]=cv2.getTrackbarPos('class','aav')
            cls_green_change=False

    if green_class_sortout_list[ref_id][select_green_cell[ref_id]]>=0:
        sel_cell=green_changed_by_hand[ref_id][select_green_cell[ref_id]]
        if sel_cell<0:
            sel_cell=green_class_sortout_list[ref_id][select_green_cell[ref_id]]

        cv2.setTrackbarPos('class','aav',sel_cell)
        cls_green_change=False

    plot_c=cv2.imread('plots/tmpplot_classes'+str(ref_id)+'.png')

    orig_red=cv2.getTrackbarPos('orig_red','intens')
    orig_green=cv2.getTrackbarPos('orig_green','aav')

    if orig_red==0:
        img_show_red=cv2.cvtColor(img_orig_intens_display[ref_id].copy().astype(np.uint8),cv2.COLOR_GRAY2RGB)
    elif orig_red==1:
        img_show_red=cv2.cvtColor(img_orig_intens_display[ref_id].copy().astype(np.uint8),cv2.COLOR_GRAY2RGB)
        img_show_red=draw_classes(img_show_red,contour_list[ref_id],red_class_sortout_list[ref_id])
    else:
        img_cp=img_orig_intens_sc[ref_id].copy()
        img_gray=255*(img_cp>red_bg_thres)
        img_show_red=cv2.cvtColor(img_gray.astype(np.uint8),cv2.COLOR_GRAY2RGB)
        img_show_red=draw_classes(img_show_red,contour_list[ref_id],red_class_sortout_list[ref_id])
        

    if not areas_list[ref_id][select_red_cell[ref_id]]==None:
        cv2.drawContours(img_show_red,[contour_list[ref_id][select_red_cell[ref_id]]],0,class_color[red_class_sortout_list[ref_id][select_red_cell[ref_id]]],4)
    cv2.imshow('intens',img_show_red)

    if orig_green==0:
        img_show_green=cv2.cvtColor(img_orig_aav_display[ref_id].copy().astype(np.uint8),cv2.COLOR_GRAY2RGB)
    elif orig_green==1:
        img_show_green=cv2.cvtColor(img_orig_aav_display[ref_id].copy().astype(np.uint8),cv2.COLOR_GRAY2RGB)
        img_show_green=draw_classes(img_show_green,contour_list[ref_id],green_class_sortout_list[ref_id])
    else:
        img_cp=img_orig_aav_sc[ref_id].copy()
        img_gray=255*(img_cp>green_bg_thres)
        img_show_green=cv2.cvtColor(img_gray.astype(np.uint8),cv2.COLOR_GRAY2RGB)
        img_show_green=draw_classes(img_show_green,contour_list[ref_id],green_class_sortout_list[ref_id])
        

    if not areas_list[ref_id][select_green_cell[ref_id]]==None:
        cv2.drawContours(img_show_green,[contour_list[ref_id][select_green_cell[ref_id]]],0,class_color[green_class_sortout_list[ref_id][select_green_cell[ref_id]]],4)
    cv2.imshow('aav',img_show_green)


    cv2.imshow('orig',img_orig_display[ref_id].astype(np.uint8))
    cv2.imshow('mask',1.0*(mask_list[ref_id].astype(np.float32)>0))

    cv2.imshow('plot_intens',plot_img)
    cv2.imshow('plot_aav',plot_green_img)
    
    k=cv2.waitKey(1)

    if k==27:
        break

    k_chr=get_chr_key(k)
    if k_chr=='s':
        now=datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        res_dir=dt_string+'_classified/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
            
        for i in range(len(fnames)):
            
            out_name=res_dir+'_'.join(fnames[i].split('_')[:-1])+'_info.csv'

            output_table=get_output_table(centers_list[i],perimeters_list[i],areas_list[i],red_means_list[i],green_means_list[i],red_class_sortout_list[i],green_class_sortout_list[i])

            f=open(out_name,'wt')

            f.write('gaussians;mean;variance\n\n')
            f.write('red low;'+str_r(means[0])+';'+str_r(sigma[0])+'\n')
            f.write('red high;'+str_r(means[1])+';'+str_r(sigma[1])+'\n')
            f.write('red background;'+str_r(means_bg[0])+';'+str_r(sigma_bg[0])+'\n')
            f.write('\n')
            f.write('green low;'+str_r(green_means[0])+';'+str_r(green_sigma[0])+'\n')
            f.write('green high;'+str_r(green_means[1])+';'+str_r(green_sigma[1])+'\n')
            f.write('green background;'+str_r(green_means_bg[0])+';'+str_r(green_sigma_bg[0])+'\n')
            f.write('\n')

            cls_idx=list(red_class_list[i])

            total=len(cls_idx)
            
            red_classes=np.asarray([red_class_sortout_list[i][j] for j in cls_idx])
            green_classes=np.asarray([green_class_sortout_list[i][j] for j in cls_idx])

            red_low_green_low=0
            red_high_green_high=0
            red_low_green_high=0
            red_high_green_low=0

            red_bg=0
            red_low=0
            red_high=0

            green_bg=0
            green_low=0
            green_high=0


            for j in range(len(red_classes)):
                if red_classes[j]==1 and green_classes[j]==1:
                    red_low_green_low+=1
                if red_classes[j]==2 and green_classes[j]==2:
                    red_high_green_high+=1
                if red_classes[j]==1 and green_classes[j]==2:
                    red_low_green_high+=1
                if red_classes[j]==2 and green_classes[j]==1:
                    red_high_green_low+=1

                if red_classes[j]==0:
                    red_bg+=1
                if red_classes[j]==1:
                    red_low+=1
                if red_classes[j]==2:
                    red_high+=1
            
                if green_classes[j]==0:
                    green_bg+=1
                if green_classes[j]==1:
                    green_low+=1
                if green_classes[j]==2:
                    green_high+=1
                

        
            f.write(';total;relative\n')
            f.write('red bg;'+str_r(red_bg)+';'+str_r(red_bg/total)+'\n')
            f.write('red low;'+str_r(red_low)+';'+str_r(red_low/total)+'\n')
            f.write('red high;'+str_r(red_high)+';'+str_r(red_high/total)+'\n')
            f.write('\n')
            f.write('green bg;'+str_r(green_bg)+';'+str_r(green_bg/total)+'\n')
            f.write('green low;'+str_r(green_low)+';'+str_r(green_low/total)+'\n')
            f.write('green high;'+str_r(green_high)+';'+str_r(green_high/total)+'\n')
            f.write('\n')
            f.write('red and green low;'+str_r(red_low_green_low)+';'+str_r(red_low_green_low/total)+'\n')
            f.write('red and green high;'+str_r(red_high_green_high)+';'+str_r(red_high_green_high/total)+'\n')
            f.write('red low and green high;'+str_r(red_low_green_high)+';'+str_r(red_low_green_high/total)+'\n')
            f.write('red high and green low;'+str_r(red_high_green_low)+';'+str_r(red_high_green_low/total)+'\n')
            f.write('\n')

            for output_row in output_table:
                f.write(';'.join(output_row)+'\n')
            f.close()

    
            cv2.imwrite(res_dir+'_'.join(fnames[i].split('_')[:-1])+'_ID.png',img_id_sc[i])        

            img_show_red=cv2.cvtColor(img_orig_intens_display[i].copy().astype(np.uint8),cv2.COLOR_GRAY2RGB)
            img_show_red=draw_classes(img_show_red,contour_list[i],red_class_sortout_list[i])
            cv2.imwrite(res_dir+'_'.join(fnames[i].split('_')[:-1])+'_class_red.jpg',img_show_red)

            img_show_green=cv2.cvtColor(img_orig_aav_display[i].copy().astype(np.uint8),cv2.COLOR_GRAY2RGB)
            img_show_green=draw_classes(img_show_green,contour_list[i],green_class_sortout_list[i])
            cv2.imwrite(res_dir+'_'.join(fnames[i].split('_')[:-1])+'_class_green.jpg',img_show_green)

            cv2.imwrite(res_dir+'_'.join(fnames[i].split('_')[:-1])+'_scaled_red.jpg',img_orig_intens_sc[i])
            cv2.imwrite(res_dir+'_'.join(fnames[i].split('_')[:-1])+'_scaled_green.jpg',img_orig_aav_sc[i])
            


cv2.destroyAllWindows()
