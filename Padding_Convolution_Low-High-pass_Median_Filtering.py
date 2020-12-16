     #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:04:39 2020

"""

import matplotlib.pyplot as plt
import matplotlib.image as mpltimg 
import numpy as np
from math import log10, sqrt




#method to add padding to the image

def padding(img,kernel_size,padding_type):
    
    ker_width=kernel_size//2
    
    xshape=img.shape[0]
    yshape=img.shape[1]
    
    xpad=xshape+2*ker_width
    ypad=yshape+2*ker_width
    
    image_Padded = np.zeros([xpad,ypad]) #filling the entire matrix with 0

    image_Padded[ker_width : -ker_width , ker_width : -ker_width] = img #filling the  non padded part with original image pixel values
    
	#zero padding done
    if padding_type == 0:
        return image_Padded

    #duplicate padding, works on the zero padded image    
    elif padding_type == 1:
        for i in range(ker_width,image_Padded.shape[0]-ker_width): 
            for j in range(ker_width,image_Padded.shape[1]-ker_width):
                
                if i == ker_width:
                    x=ker_width-1
                    while x > -1:
                        image_Padded[x,j]=img[i-ker_width,j-ker_width]
                        x -= 1
                
                elif i == image_Padded.shape[0] - ker_width:
                    x=image_Padded.shape[0]-ker_width+1
                    while x < image_Padded.shape[0]+1:
                        image_Padded[x,j]=img[i-ker_width,j-ker_width]
                        x += 1
                        
                elif j == ker_width:
                    x=ker_width-1
                    while x > -1:
                        image_Padded[i,x]=img[i-ker_width,j-ker_width]
                        x -= 1
                        
                elif j == image_Padded.shape[1] - ker_width:
                    x=image_Padded.shape[1]-ker_width+1
                    while x < image_Padded.shape[1]+1:
                        image_Padded[i,x]=img[i-ker_width,j-ker_width]
                        x += 1
        return image_Padded
    
	#duplicate padding, works on the zero padded image    
    else:
        for i in range(ker_width,image_Padded.shape[0]-ker_width):
            for j in range(ker_width,image_Padded.shape[1]-ker_width):
                
                if i == ker_width:
                    x=ker_width-1
                    while x > -1:
                        image_Padded[x,j]=255-img[i-ker_width,j-ker_width]
                        x -= 1
                
                elif i == image_Padded.shape[0] - ker_width:
                    x=image_Padded.shape[0]-ker_width+1
                    while x < image_Padded.shape[0]+1:
                        image_Padded[x,j]=255-img[i-ker_width,j-ker_width]
                        x += 1
                        
                elif j == ker_width:
                    x=ker_width-1
                    while x > -1:
                        image_Padded[i,x]=255-img[i-ker_width,j-ker_width]
                        x -= 1
                        
                elif j == image_Padded.shape[1] - ker_width:
                    x=image_Padded.shape[1]-ker_width+1
                    while x < image_Padded.shape[1]+1:
                        image_Padded[i,x]=255-img[i-ker_width,j-ker_width]
                        x += 1
        return image_Padded
		
		
#_______ 3 user defined functions which you have to code their functionality_________		

def convolution2D(img, kernel, padding_type):
    # write your code here
    # padding_type can take values 0, 1 or 2
        # 0 - zero padding
        # 1 - duplicate boundary pixels for padding
        # 2 - padding is done by mirroring the pixels
        
    # should handle kernel of any size but odd values only eg. 5x5, 7x7
    # image is a grayscale image
    
    imagePadded = padding(img, kernel.shape[0],padding_type)#padding the image
    
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    
    output = np.zeros_like(img,'uint8') #creating a matrix with 0s and size of image
    
    # Iterate through image
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum() #convolution
            
    return output
            
        
    


def medianFiltering(img, kernel_size, padding_type):
    # write your code here
    # padding_type can take values 0, 1 or 2
        # 0 - zero padding
        # 1 - duplicate boundary pixels for padding
        # 2 - padding is done by mirroring the pixels
        
    # should handle kernel of any size but odd values only eg. 5x5, 7x7
    # image is a grayscale image
    
    if kernel_size % 2 == 0: #handling even sized kernels
        kernel_size += 1
        
    image_padded=padding(img,kernel_size,padding_type) #padding the image

    blur_img=np.zeros(image_padded.shape,'uint8') #creating a matrix with 0s and size of padded image
    median_arr=np.zeros((kernel_size,kernel_size)) #creating the median matrix with size same as kernel
    height,width=image_padded.shape

    for i in range(height):
        for j in range(width):
            median_arr=image_padded[i:i+median_arr.shape[0], j:j+median_arr.shape[1]] 
            blur_img[i,j]=np.median(median_arr) #filling the pixel with median of the kernel sized matrix created in previous step
            
    new_img=np.zeros_like(img)
	
	#removing padding
    for i in range(kernel_size//2, blur_img.shape[0]-kernel_size//2):
        for j in range(kernel_size//2, blur_img.shape[1]-kernel_size//2):
            new_img[i-kernel_size//2,j-kernel_size//2] = blur_img[i,j]
            
    return new_img


# You can club the above two functions/write any other functions additionally if you wish

def computePSNR(image1, image2):
    psnr = 0
    
    mse = np.mean( (image1 - image2) ** 2)
    if (mse==0):
        psnr = 100
    max=255.0
    psnr = 20 * log10(max / sqrt(mse))  

    return psnr


# _____________________main program begins here___________________

def main():
    # reading a noisy image
    noisy_image = mpltimg.imread('images/noisy_image.jpg')

    original_image = mpltimg.imread('images/original.jpg')


# _____________________________________________________________________
# Average filter kernel
    kernel = 1/9 * np.array([[ 1, 1, 1],
                          [ 1, 1, 1],
                          [ 1, 1, 1]]) 
    



    low_pass_filtered_image = convolution2D(noisy_image, kernel, 1)

    
    avg_psnr = computePSNR(original_image,low_pass_filtered_image)


    med_filtered_image = medianFiltering(noisy_image, 4, 1);
    
    med_psnr = computePSNR(original_image,med_filtered_image)
    
    
    
# _____________________________________________________________________
# reading a blurry image
    blurry_image = mpltimg.imread('images/blurry_image.jpg')
    
    
    # Laplacian filter kernel
    kernel =        np.array([[ 1, 1, 1],
                              [ 1, -8, 1],
                              [ 1, 1, 1]]) 


    
    laplacian_filtered_image = convolution2D(blurry_image, kernel, 1)
	
	# perform the addition as in Eqn. 3.6.7 to obtain the sharpened image
    
    
    sharpened_image = blurry_image - laplacian_filtered_image
    
    
    
# _____________________________________________________________________
# Code to display the images
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    
    ax = axes.ravel()
    
    ax[0].imshow(noisy_image, cmap='gray')
    ax[0].set_title("Noisy image")
    ax[0].set_axis_off()
    
    ax[1].imshow(low_pass_filtered_image, cmap='gray')
    ax[1].set_title("Low Pass Filter Output")
    ax[1].set_axis_off()
    ax[1].text(x=40, y=230, s="PSNR = %1.2f db" %avg_psnr)
    
    ax[2].imshow(med_filtered_image, cmap='gray')
    ax[2].set_title("Median Filter Output")
    ax[2].set_axis_off()
    ax[2].text(x=40, y=230, s="PSNR = %1.2f db" %med_psnr)
    
    
    ax[3].imshow(blurry_image, cmap='gray')
    ax[3].set_title("Blurry Input Image")
    ax[3].set_axis_off()
    
    
    ax[4].imshow(laplacian_filtered_image, cmap='gray')
    ax[4].set_title("Laplacian Filter Output")
    ax[4].set_axis_off()
    
    
    ax[5].imshow(sharpened_image, cmap='gray')
    ax[5].set_title("Sharpened Image")
    ax[5].set_axis_off()
    
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

