#######################################################################
#              From Spiral Image to Discrete Signal
#
# STEP 1: Read Spiral Image file
# STEP 2: Determine the equation of the spiral by fitting to image data
# STEP 3: Use the spiral equation ot flatten the Spiral Image
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

# input filename containing the image of the spiral and the patient's spiral
input_file_name = 'S6'
# this procedure should work on png/tif/bmp
input_file_type = 'png'
# number of samples in the final discrete signal
N_resample = 1000

# open log file
with open(f"{input_file_name}_log", 'w') as log:
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
    data1 = np.array(spiralImg)
    # pr = f"type(data1) = {type(data1) }" 
    # print(pr); log.write(pr+'\n')
    pr = f"y: data1.shape[0] = {data1.shape[0]}"
    print(pr); log.write(pr+'\n')
    pr = f"x: data1.shape[1] = {data1.shape[1]}"
    print(pr); log.write(pr+'\n')
    toprint = f"   data1.shape[2] = {data1.shape[2]}"
    print(pr); log.write(toprint+'\n\n')
   
   # For Dim 2 we only need 3 dims RGB. Discard alpha (transparency) data if present 
    data = data1[:, :, :3]

    # pr = f"type(data1) = {type(data) }" 
    # print(pr); log.write(pr+'\n')
    pr = f"y: data.shape[0] = {data.shape[0]}"
    print(pr); log.write(pr+'\n')
    pr = f"x: data.shape[1] = {data.shape[1]}"
    print(pr); log.write(pr+'\n')
    toprint = f"   data.shape[2] = {data.shape[2]}"
    print(pr); log.write(toprint+'\n\n')

    #######################################################################
    #                            STEP 2 
    # Determine the equation of the spiral by fitting to image data
    #######################################################################

    pr = f"STEP 2 -- Determine the equation of the spiral by fitting to image data"
    print('\n'+pr); log.write('\n'+pr+'\n')
    
    pr = "--- Select the center by clicking on it ---"
    print(pr); log.write(pr+'\n')

    plt.title('Spiral Image', fontsize='22')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.imshow(np.flipud(data), origin='lower')
    number = 0
    plt.connect('button_press_event', mouse_press)
    plt.show()

    # select center
    xcenter = coords[-1][0]
    ycenter = coords[-1][1]

    pr = "--- Select two points (one in zone 1 and the other in zone 2) on the spiral by clicking on the points ---"
    print(pr); log.write('\n'+pr+'\n')

    plt.title('Spiral Image', fontsize='22')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.imshow(np.flipud(data), origin='lower')
    plt.connect('button_press_event', mouse_press)
    number = 0
    plt.axvline(x=xcenter, c="skyblue", label="x=0")
    plt.axhline(y=ycenter,  c="skyblue", label="y=0")
    plt.show()

    # select 2 points of the "scanned" spiral
    x11 = coords[-2][0] - xcenter
    y11 = coords[-2][1] - ycenter
    x22 = coords[-1][0] - xcenter
    y22 = coords[-1][1] - ycenter

    # determine parameters of the  spiral fitted to the two points
    r11 = math.sqrt(x11*x11 + y11*y11)
    r22 = math.sqrt(x22*x22 + y22*y22)
    # it is assumed that the user selected the first point in zone 1
    # and the second point in zone 2
    t11 = np.arctan2(y11,x11) # zone 1 
    t22 = np.arctan2(y22,x22) + 2*np.pi # zone 2 

    t_rr = (r22*t11 - r11*t22)/(r11-r22)
    t_rr = t_rr - 2*np.pi 
    bb = (r11-r22)/(t11-t22)

    pr = f"t_rr = {t_rr} radians --- {t_rr*180.0/np.pi} degrees"
    print(pr); log.write(pr+'\n')
    pr = f"bb = {bb}"
    print(pr); log.write(pr+'\n')

    pr = "--- Close this window and wait for the next window ... It will take about 40 sec. (in Python 3.11) ..."
    print(pr); log.write('\n'+pr+'\n')
    pr = "... for the next window to appear ---"   
    print(pr); log.write(pr+'\n')

    # plot fitted spiral along with the original scanned spiral
    plt.gca().set_aspect('equal')
    plt.title('Spiral Image and Fitted Spiral', fontsize='22')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')
    plt.imshow(np.flipud(data), origin='lower')
    plt.axvline(x=xcenter, c="black")
    plt.axhline(y=ycenter,  c="black")

    plt.hlines(y=ycenter, xmin=0, xmax=xcenter, linewidth=4, label="Branch-cut")
    # construct and plot the fitted spiral passing through the two points
    t = np.arange(-1*t_rr, 6*np.pi, 0.1 )
    xf=bb*(t+t_rr)*np.cos(t) + xcenter
    yf=bb*(t+t_rr)*np.sin(t) + ycenter

    plt.plot(xf,yf, linestyle = 'dashed', c="red", label="Fitted Spiral")

    x_rotation = [0+xcenter, 400*np.cos(t_rr)+xcenter]
    y_rotation = [0+ycenter, 400*np.sin(t_rr)+ycenter]

    plt.plot(x_rotation,y_rotation, linestyle = 'dashed', c="blue", label="Spiral Rotation")

    # plot the centter and two points throough which the fitted spiral passes
    plt.scatter([0+xcenter], [0+ycenter], marker = 'o', label="Center Selected Point")
    plt.scatter([x11+xcenter, x22+xcenter], [y11+ycenter, y22+ycenter], marker = 'o', label="Spiral Selected Points")
    plt.legend(loc='lower right')
    plt.show()

    #######################################################################
    #                            STEP 3 
    # Use the spiral equation ot flatten the Spiral Image
    #######################################################################
    pr = f"STEP 3 -- Use the spiral equation ot flatten the Spiral Image"
    print('\n'+pr); log.write('\n'+pr+'\n')

    pr = "... It will take about 1 min and 20 sec. for this window to close and the next window to appear ..."
    print(pr); log.write('\n'+pr+'\n')
    
    # transformation unravels the spiiral r = b*(t+t_r)
    # b and t_r (theta_r) are the parameters defining the spiral
    # input r and t (theta)
    # returns x and y coordinates in flattened space
    def trans(r, t, b, t_r):
        return [t + t_r, r - b*(t + t_r)]

    #   input r and t of an arbitraty point in spiral space, determines the zone
    #   this point belings to, and returns the factor appropriate for this zone
    #   w.r.t. a spiral defined by the parameters b and t_r (theta_r)
    def factor(r, t, b, t_r):
        factor = 0
        if( r > 0 and r <= b*(t + t_r)):
            # print("zone one")
            factor = -2*math.pi 
        elif (r > b*(t + t_r) and r <= b*(t + t_r) + 2*math.pi):
            # print("zone two")
            factor = 0*math.pi 
        elif (r > b*(t + t_r + 2*math.pi) and r <= b*(t + t_r) + 4*math.pi):
            # print("zone three")
            factor = 2*math.pi
        elif (r > b*(t + t_r + 4*math.pi) and r <= b*(t + t_r) + 6*math.pi):
            # print("zone four")
            factor = 4*math.pi
        elif (r > b*(t + t_r + 6*math.pi) and r <= b*(t + t_r) + 8*math.pi):
            # print("zone five")
            factor = 6*math.pi
        elif (r > b*(t + t_r + 8*math.pi) and r <= b*(t + t_r) + 10*math.pi):
            # print("zone six")
            factor = 8*math.pi
        return factor

    # transform (flatten) two points on the spiral
    
    # r and theta of the two points in spiral space
    r11 = math.sqrt(x11*x11 + y11*y11)
    t11 = math.atan2(y11, x11) + 2*np.pi 
    r22 = math.sqrt(x22*x22 + y22*y22)
    t22 = math.atan2(y22, x22) + 4*np.pi   

    # transform point 1 to flattened space
    p11 = trans(r11, t11, bb, t_rr)  
    px11 = p11[0]
    py11 = p11[1]

    # transform point 2 to flattened space
    p22 = trans(r22, t22, bb, t_rr) 
    px22 = p22[0]
    py22 = p22[1]

    rdata = np.full((2000, 3500, 3), [255, 255, 255])

    xshift = 700
    yshift = 750
    for i in range(data.shape[1]): # x
        for j in range(data.shape[0]): # y
            # data[i][j] is a 1d array with 3 or 4 elements
            # each element can be accessed as data[i][j][0] etc.
            x = i - xcenter 
            y = j - ycenter
            r = math.sqrt(x*x + y*y)
            t = math.atan2(y, x)
            t = t + factor(r, t, bb, t_rr) 
            d = trans(r, t, bb, t_rr)
            x = d[0]
            y = d[1]

            ii = int(x*4000/25) + xshift
            jj = int(y) + yshift
            if (((rdata.shape[0] - 1)-jj) >=0 and ((rdata.shape[0] - 1)-jj) < rdata.shape[0])  and (ii >=0 and ii < rdata.shape[1] )   :
                rdata[(rdata.shape[0] - 1)-jj][ii] = data[(data.shape[0]-1)-j][i]

    for i in range(data.shape[1]): # x
        for j in range(data.shape[0]): # y
            # data[i][j] is a 1d array with 3 or 4 elements
            # each element can be accessed as data[i][j][0] etc.
            x = i - xcenter 
            y = j - ycenter
            r = math.sqrt(x*x + y*y)
            t = math.atan2(y, x)
            t = t + factor(r, t, bb, t_rr) 
            d = trans(r, t, bb, t_rr)
            x = d[0]
            y = d[1]

            ii = int((x+2*np.pi)*4000/25) + xshift
            jj = int(y - bb*2*np.pi) + yshift
            if (((rdata.shape[0] - 1)-jj) >=0 and ((rdata.shape[0] - 1)-jj) < rdata.shape[0])  and (ii >=0 and ii < rdata.shape[1] )   :
                rdata[(rdata.shape[0] - 1)-jj][ii] = data[(data.shape[0]-1)-j][i]

    for i in range(data.shape[1]): # x
        for j in range(data.shape[0]): # y
            # data[i][j] is a 1d array with 3 or 4 elements
            # each element can be accessed as data[i][j][0] etc.
            x = i - xcenter 
            y = j - ycenter
            r = math.sqrt(x*x + y*y)
            t = math.atan2(y, x)
            t = t + factor(r, t, bb, t_rr) 
            d = trans(r, t, bb, t_rr)
            x = d[0]
            y = d[1]

            ii = int((x+4*np.pi)*4000/25) + xshift
            jj = int(y - bb*4*np.pi) + yshift
            if (((rdata.shape[0] - 1)-jj) >=0 and ((rdata.shape[0] - 1)-jj) < rdata.shape[0])  and (ii >=0 and ii < rdata.shape[1] )   :
                rdata[(rdata.shape[0] - 1)-jj][ii] = data[(data.shape[0]-1)-j][i]
 
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