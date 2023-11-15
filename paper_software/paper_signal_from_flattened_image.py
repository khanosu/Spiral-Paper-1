#######################################################################
#              From Spiral Image to Discrete Signal
#
# STEP 1: Read Flattened Spiral Image file
# STEP 2: Not needed here
# STEP 3: Not needed here
# STEP 4: Crop to include pixels of a single stitched patient’s hand-drawing
# STEP 5: Derive the discrete signal and save it to Pandas data frames
#######################################################################
from PIL import Image as im
import matplotlib.pyplot as plt
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
import numpy as np
import math
import pandas as pd
from scipy import interpolate

#######################################################################
#                            STEP 1 
# Read Spiral Image file
#######################################################################

# input filename containing the image of the flattened patient's spiral
input_file_name = 'S3_flattened'
# this procedure should work on png/tif/bmp
input_file_type = 'png'
# number of samples in the final discrete signal
N_resample = 2000

# open log file
with open(f"{input_file_name}_log2", 'w') as log:
    pr = f"STEP 1 -- Read Spiral Image file"
    print('\n'+pr); log.write('\n'+pr+'\n')
        
    pr = f"input image filename: {input_file_name}.{input_file_type}" 
    print(pr); log.write(pr+'\n\n')

    number = 0
    coords = []
    def mouse_press(event):
        global coords
        global number
        if event.inaxes:
            coords.append((event.xdata, event.ydata))
            number = number + 1
            pr = f"{number}: {event.xdata}, {event.ydata}"
            print(pr); log.write(pr+'\n')

    # this procedure should work on png/tif/bmp
    # spiralImg = im.open(input_file_name + '.png')
    spiralImg = im.open(f"{input_file_name}.{input_file_type}")

    # np.array(spiralImg) converts image into a 3D np array. 
    # Dim 0 od data contains y pixels (number of pixels in the y dim of the image)
    # Dim 1 of data contains x pixels (number of pixels in the x dim of the image)
    # Dim 2 of data has either 3 dims: RGB or 4 dims: GRB and alpha (transparency)
    rdata1 = np.array(spiralImg)

    # pr = f"type(rdata1) = {type(rdata1) }" 
    # print(pr); log.write(pr+'\n')
    pr = f"y: rdata1.shape[0] = {rdata1.shape[0]}"
    print(pr); log.write(pr+'\n')
    pr = f"x: rdata1.shape[1] = {rdata1.shape[1]}"
    print(pr); log.write(pr+'\n')
    toprint = f"   rdata1.shape[2] = {rdata1.shape[2]}"
    print(pr); log.write(toprint+'\n\n')

    # For Dim 2 we only need 3 dims RGB. Discard alpha (transparency) data if present 
    rdata = rdata1[:, :, :3]

    # pr = f"type(rdata) = {type(rdata) }" 
    # print(pr); log.write(pr+'\n')
    pr = f"y: data.shape[0] = {rdata.shape[0]}"
    print(pr); log.write(pr+'\n')
    pr = f"x: data.shape[1] = {rdata.shape[1]}"
    print(pr); log.write(pr+'\n')
    toprint = f"   rdata.shape[2] = {rdata.shape[2]}"
    print(pr); log.write(toprint+'\n\n')
    
    pr = "--- Select the end-point of the longest flattened spiral in the image ---"                 
    print(pr); log.write(pr+'\n')
    plt.title('Multi-Zone Flattened Spiral Image', fontsize='22')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.imshow(np.flipud(rdata), origin='lower')
    number = 0
    plt.connect('button_press_event', mouse_press)
    plt.show()
    
    max_x_pixels = int(coords[-1][0])
    pr = f"max_x_pixels: {max_x_pixels}\n"                 
    print(pr); log.write(pr+'\n')

    pr = "--- Select two points on the graph to define a rectangle. Only data inside this rectangle will be kept ---"                 
    print(pr); log.write(pr+'\n')
    plt.title('Multi-Zone Flattened Spiral Image', fontsize='22')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.imshow(np.flipud(rdata), origin='lower')
    number = 0
    plt.connect('button_press_event', mouse_press)
    plt.show()

    #######################################################################
    #                            STEP 4 
    # Crop to include pixels of a single stitched patient’s hand-drawing  
    #######################################################################
    pr = f"STEP 4 -- Crop to include pixels of a single stitched patient’s hand-drawing"
    print('\n'+pr); log.write('\n'+pr+'\n')
    
    # select 2 points for the selection to choose data
    xselection_up_left = int(coords[-2][0])
    yselection_up_left = int(coords[-2][1])
    xselection_down_right= int(coords[-1][0])
    yselection_down_right = int(coords[-1][1])

    # keep track of cropped x boundaries
    current_max_x_pixels = xselection_down_right
    offset_x_pixels = xselection_up_left
    
    pr =f"current_max_x_pixels: {current_max_x_pixels}" 
    print(pr); log.write(pr+'\n')
    
    pr = f"offset_x_pixels: {offset_x_pixels}" 
    print(pr); log.write(pr+'\n')
    
    hselection = yselection_up_left - yselection_down_right
    wselection = xselection_down_right - xselection_up_left 
    sdata = np.full((hselection, wselection, 3), [255, 255, 255])

    for i in range(xselection_up_left, xselection_up_left + sdata.shape[1]): # x
        for j in range(yselection_down_right, yselection_down_right + sdata.shape[0]): # y
            ii = i - xselection_up_left
            jj = j -yselection_down_right
            sdata[(sdata.shape[0] - 1)-jj][ii] = rdata[(rdata.shape[0]-1)-j][i]

    pr = "--- Select two points on the graph to define a rectangle. Picture data inside this rectangle will be converted to a discrete signal. ... ---"
    print(pr); log.write('\n'+pr+'\n')

    plt.imshow(np.flipud(sdata), origin='lower')
    plt.title('Single-Zone Flattened Spiral Image', fontsize='22')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    number = 0 
    plt.connect('button_press_event', mouse_press) 
    plt.show() 

    #######################################################################
    #                            STEP 5 
    # Derive the discrete signal and save it to Pandas data frames
    #######################################################################
    pr = f"STEP 5 -- Derive the discrete signal and save it to Pandas data frames"
    print('\n'+pr); log.write('\n'+pr+'\n')
    
    xselection_up_left = int(coords[-2][0])  
    yselection_up_left = int(coords[-2][1])  
    xselection_down_right= int(coords[-1][0])  
    yselection_down_right = int(coords[-1][1]) 
    
    # keep track of cropped x boundaries
    current_max_x_pixels = xselection_down_right + offset_x_pixels 
    offset_x_pixels = offset_x_pixels + xselection_up_left

    pr =f"current_max_x_pixels: {current_max_x_pixels}" 
    print(pr); log.write(pr+'\n')
    pr = f"offset_x_pixels: {offset_x_pixels}" 
    print(pr); log.write(pr+'\n')

    cropping_ratio = ((current_max_x_pixels)**2 - (offset_x_pixels)**2)/(max_x_pixels)**2
    
    pr = f"cropping_ratio: {cropping_ratio}"
    print(pr); log.write('\n'+pr+'\n')

    # derive discrete signal disc_signal[i], i = 0, 1, 2, 3 ... from sdata
    disc_signal = []
    for i in range(xselection_up_left, xselection_down_right): 
        min = 300*3
        jmin = 0
        for j in range(yselection_down_right, yselection_up_left): 
            if (  sdata[(sdata.shape[0] - 1)-j][i][0]+sdata[(sdata.shape[0] - 1)-j][i][1]+sdata[(sdata.shape[0] - 1)-j][i][2]  < min):
                min = sdata[(sdata.shape[0] - 1)-j][i][0]+sdata[(sdata.shape[0] - 1)-j][i][1]+sdata[(sdata.shape[0] - 1)-j][i][2] 
                jmin = (sdata.shape[0] - 1)-j
        disc_signal.append(jmin)

    #---------------- correct the x-axis of the signal ----------
    # Please see eq (10) of the paper for this correction 
    # In terms of x-indices eq(10) is equivalent to the transformation i -> i*i
    disc_signal_x = []
    for i in range(current_max_x_pixels):
        # scaling the x axis nonuniformaly
        disc_signal_x.append(i*i)       
    disc_signal_x =  disc_signal_x[-len(disc_signal)-1:-1]

    #---------------- resample the corrected signal ----------
    # the non-uniform scaling has made the x samples nonuniform. We resample to make the
    # x samples uniform. The resampled signal has N_resample points, input ny the user at the start 
    # of this code
    xnew = np.linspace(disc_signal_x[0], disc_signal_x[-1], N_resample)
    flinear = interpolate.interp1d(disc_signal_x, disc_signal)
    ylinear = flinear(xnew)
    disc_signal = ylinear

    plt.plot( disc_signal)
    plt.title('Resampled Corrected Discrete Signal', fontsize='22')
    plt.xlabel('samle')
    plt.ylabel('signal')
    number = 0
    plt.connect('button_press_event', mouse_press)
    plt.show()

    # convert disc_signal to pandas data frame and save to disc
    dict = {input_file_name+'_signal': disc_signal}        
    df = pd.DataFrame(dict) 
    df.to_csv(input_file_name+'_disc_signal.csv', index=False) 
    pr = f"--- 'discrete signal' has been saved as {input_file_name+'_disc_signal.csv'} ---"
    print(pr); log.write('\n'+pr+'\n')
    
    log.close()
      
    #######################################################################
    #                            END 
    #######################################################################