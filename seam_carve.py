import cv2
import numpy as np
from numpy import linalg as la
import os
import sys
import math as m

def rotate_image(img, clock_rotate=0):#for flip image and unflip image
    r, c, ch = img.shape
    rotate = np.zeros((c, r, ch))#construct empty 3d array of rotated img
    if clock_rotate:
        flip = img 
        for channel in range(ch):
            for row in range(r):
                rotate[:, row, channel] = flip[row, :, channel]
    else:
        for channel in range(ch):
            for row in range(r):
                rotate[:, r-1-row, channel] = img[row, :, channel]
    return rotate

def min_helper(left,right,center):
    lrcMatrix = np.stack((left,right,center), axis = 0)
    return np.min(lrcMatrix,axis=0)

def print_helper(seam,i,orientation,cumulative_energy):#extra thicc helper function for print formmating
    print("\nPoints on seam %d:"%(i))
    if(orientation == 0):
        seam = np.fliplr(seam) 
    print("vertical" if orientation == 0 else "horizontal")
    print("%d, %d"%(seam[0,0], seam[0,1]))
    print("%d, %d"%(seam[len(seam)//2, 0], seam[len(seam)//2, 1]))
    print("%d, %d"%(seam[len(seam)-1, 0], seam[len(seam)-1, 1]) if len(seam)%2==0 \
            else "%d, %d"%(seam[len(seam) -2, 0], seam[0,1]))
    #gotta calculate the avg energy real quick
    avg_energy = np.min(cumulative_energy[cumulative_energy.shape[0]-1,:]) / len(seam)
    print("Energy of seam %d: %.2f"%(i,avg_energy)) 

def calc_energy(img):#function calculates image energy
    #calculate energy of image, first convert to greyscale then apply sobel
    gray_scale = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_scale,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray_scale,cv2.CV_64F,0,1,ksize=3)
    img_energy = np.abs(sobelx) + np.abs(sobely)
    return img_energy

def cumulative_energy(img_energy):
    row, col = img_energy.shape[:2]
    cumulative_energy = np.zeros((row, col))
    cumulative_energy[0,:] = img_energy[0,:]
    #set leftmost and right most col to 1mill
    cumulative_energy[0,0] = 1e6 
    cumulative_energy[0,-1] = 1e6
    for i in range(1, row):
        cumulative_energy[i-1,0] = 1e6 #set leftmost and right most col to 1mill
        cumulative_energy[i-1,-1] = 1e6
        #split into left right and center 
        left = cumulative_energy[i-1, :-2]
        right = cumulative_energy[i-1, 2:]
        center = cumulative_energy[i-1, 1:-1]
        mins = min_helper(left,right,center) #call min helper returns array of mins
        cumulative_energy[i,:] = img_energy[i,:] #set self energy value
        cumulative_energy[i,1:-1] = cumulative_energy[i,1:-1] + mins
        #set leftmost and right most col to 1mill
        cumulative_energy[i,0] = 1e6 
        cumulative_energy[i,-1] = 1e6
    return cumulative_energy

def seam_backtrace(cumlative_energy):
    row, col = cumlative_energy.shape[:2]
    seam = []
    prev_pos = 0
    for i in range(row-1, -1, -1):#need to iterate backwards to trace down path
        cur_row = cumlative_energy[i,:]#the row we are looking at currently
        if i == row -1: #if last row, then argmin is our starting point
            prev_pos = np.argmin(cur_row)
            seam.append([prev_pos, i])
        else:
            if(prev_pos - 1 < 0):
                left = 1e6
            else:
                left = cur_row[prev_pos - 1]
            if(prev_pos + 1 > col):
                right = 1e6
            else:
                right = cur_row[prev_pos + 1]
            middle = cur_row[prev_pos]
            prev_pos = prev_pos + np.argmin([left, middle, right]) - 1
            seam.append([prev_pos, i])
    #little clean up before we send the bois out
    seam = np.asarray(seam)#set as np array
    seam = seam[::-1]#reverse the array
    return seam

def del_seam(img, seam):
    after_img = np.zeros((img.shape[0], img.shape[1]-1, img.shape[2]))
    for x, y in seam:
        after_img[y, 0:x] = img[y, 0:x]
        after_img[y, x:img.shape[1] - 1] = img[y, x + 1:img.shape[1]]
    return after_img

def calc_seams(img, removal_count,orientation):#takes in image, and number of cols to remove
    ''' for each col removed we need to recaculate all values for the new image'''
    for remove in range(removal_count):
        img_energy = calc_energy(img) #calculate the energy across the image
        cumulative = cumulative_energy(img_energy) #calculate cumulative energy
        seam = seam_backtrace(cumulative)#backtrace to obtain the seam
        #some print logic for sample seam output, we want to print 0,1 and last seam
        if remove == 0 or remove == 1 or remove == removal_count-1:
            if remove == 0:
                seam_img = cv2.polylines(img, np.int32([seam]), False, (0, 0, 255))
                #writing and naming image
                name = os.path.splitext(IMG_NAME)[0]
                name_ext = os.path.splitext(IMG_NAME)[-1]
                seam_name = name + "_seam" + name_ext
                cv2.imwrite(seam_name,seam_img)
            print_helper(seam,remove,orientation,cumulative)#calls print helper for output formatting
        img = del_seam(img,seam)#delete seam
    return img

IMG_NAME = ""
if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    #handle cli arguments
    if len(sys.argv) != 2:
        print("Usage: python seam_carve.py <img>")
    img_in = sys.argv[1]
    IMG_NAME = sys.argv[1]
    img_pre = cv2.imread(img_in)
    img_row, img_col = img_pre.shape[:2] #row and cols of image
    """ Algorithm is designed for vertical seam carving until image is a square
        eg. col > row, in the case where horizontal seam carving is required
        eg. row > col, image is rotated 90 degree counter-clockwise for seam processing 
    """
    orientation  = 0 #flag for img orientation, default 0 = vertical, 1 = horizontal
    if img_col > img_row:
        img = img_pre
        remove = img_col - img_row
        final_img = calc_seams(img,remove,orientation)
    else:
        orientation = 1
        img = rotate_image(img_pre,1)#set clock_rotate to counter-clock; 1
        remove = img.shape[1] - img.shape[0]
        final_img = calc_seams(img,remove,orientation)
        final_img = rotate_image(final_img)#rotate image back to vertical
    #writing and naming image
    name = os.path.splitext(img_in)[0]
    name_ext = os.path.splitext(img_in)[-1]
    final_name = name + "_final" + name_ext
    cv2.imwrite(final_name,final_img)
