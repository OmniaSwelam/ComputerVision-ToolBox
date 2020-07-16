from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
#%%RGB 2 Gray
def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
#%%Guassian Kernel
def gaussian_kernel( kernlen , std ):
    """Returns a 2D Gaussian kernel array."""
    kernlen= int(kernlen)
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d
#%%
def getCorners(img, kernlen, std, k, cornermeasure):
    if (len(img.shape)==3):
        img= rgb2gray(img)
    else:
        img= img
    #Smoothing using gaussian kernel    
    image_smooth= signal.convolve2d(img, gaussian_kernel(kernlen,std) ,'same')
    #Define gradient using Sobel
    sobel_h = np.array([[ -1 , 0 , 1 ] ,
                        [ -2 , 0 , 2 ] ,
                        [ -1 , 0 , 1 ]])
    sobel_v = sobel_h.transpose()
    image_Ix = signal.convolve2d( image_smooth , sobel_h ,'same') 
    image_Iy = signal.convolve2d( image_smooth , sobel_v ,'same')
    
    #Define $I_{xx}$, $I_{yy}$, and $I_{xy}$
    image_Ixx = np.multiply( image_Ix, image_Ix)
    image_Iyy = np.multiply( image_Iy, image_Iy)
    image_Ixy = np.multiply( image_Ix, image_Iy)

    #Compute Hessian matrix over a window
    #Now let's assume a uniform window of size 5x5,
    #in which we can apply a box filter over each of Ixx,Iyy,Ixy
    image_Ixx_hat = signal.convolve2d( image_Ixx ,  gaussian_kernel(21,1.0) ,'same') 
    image_Iyy_hat = signal.convolve2d( image_Iyy ,  gaussian_kernel(21,1.0) , 'same')
    image_Ixy_hat = signal.convolve2d( image_Ixy ,  gaussian_kernel(21,1.0)  ,'same')
    
    #compute corners
    #k is empirical constant, recommended k = 0.04-0.06
    #K = 0.05 
    image_detM = np.multiply(image_Ixx_hat,image_Iyy_hat) - np.multiply(image_Ixy_hat,image_Ixy_hat) 
    image_trM = image_Ixx_hat + image_Iyy_hat
    image_R = image_detM  - k * image_trM
    #ratio = 0.2 # Tunable value. to keep adaptivity per image.
    #image_corners = np.abs(image_R) >  np.quantile( np.abs(image_R),0.999)
    image_corners =np.abs(image_R) > cornermeasure * np.max(image_R)
    
    #plot corners
    plt.figure()
    #plt.axis('off')
    plt.imshow(img,zorder=1)
    corners_pos = np.argwhere(image_corners)
    plt.scatter(corners_pos[:,1],corners_pos[:,0],zorder=2, c = 'r',marker ='x')
    
    
    plt.set_cmap("gray")
    plt.ioff()
    plt.axis('off')
    plt.margins(0,0)
    plt.margins(0,0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig('cornerImg.jpg', bbox_inches = 'tight', pad_inches = 0)
    #plt.savefig('cornerImg.jpg')
    #plt.show()
    #cornerImage = plt.imread('corner image.jpg')
    #saveImage.saveImage(cornerImage,'corner image.jpg')
    #return cornerImage
    
    
'''    
#%%
#%%Smoothing using gaussian kernel
img_gr= rgb2gray(image)
image_smooth= signal.convolve2d(img_gr, gaussian_kernel(7,1.0) ,'same')
#plt.imshow(image_smooth)
#%%Define gradient using Sobel
sobel_h = np.array([[ -1 , 0 , 1 ] ,
                    [ -2 , 0 , 2 ] ,
                    [ -1 , 0 , 1 ]])
sobel_v = sobel_h.transpose()
image_Ix = signal.convolve2d( image_smooth , sobel_h ,'same') 
image_Iy = signal.convolve2d( image_smooth , sobel_v ,'same') 
#plt.imshow(image_Ix)
#plt.imshow(image_Iy)
#%%Define $I_{xx}$, $I_{yy}$, and $I_{xy}$
image_Ixx = np.multiply( image_Ix, image_Ix)
image_Iyy = np.multiply( image_Iy, image_Iy)
image_Ixy = np.multiply( image_Ix, image_Iy) 
#%%Compute Hessian matrix over a window
#Now let's assume a uniform window of size 5x5,
#in which we can apply a box filter over each of Ixx,Iyy,Ixy
image_Ixx_hat = signal.convolve2d( image_Ixx ,  gaussian_kernel(21,1.0) ,'same') 
image_Iyy_hat = signal.convolve2d( image_Iyy ,  gaussian_kernel(21,1.0) , 'same')
image_Ixy_hat = signal.convolve2d( image_Ixy ,  gaussian_kernel(21,1.0)  ,'same')
print('image_Ixy_hat is: '+ str(image_Ixy_hat.shape))
#%%compute corners
#k is empirical constant, recommended k = 0.04-0.06
K = 0.05 
image_detM = np.multiply(image_Ixx_hat,image_Iyy_hat) - np.multiply(image_Ixy_hat,image_Ixy_hat) 
image_trM = image_Ixx_hat + image_Iyy_hat
image_R = image_detM  - K * image_trM
ratio = 0.2 # Tunable value. to keep adaptivity per image.
#image_corners = np.abs(image_R) >  np.quantile( np.abs(image_R),0.999)
image_corners =np.abs(image_R) > 0.2 * np.max(image_R)
#%%
plt.imshow(image,zorder=1)
corners_pos = np.argwhere(image_corners)
plt.scatter(corners_pos[:,1],corners_pos[:,0],zorder=2, c = 'r',marker ='x')
plt.show()
'''
#%%load image
'''image = plt.imread('circles.png')
plt.imshow(image)
getCorners(image,11,1,0.05,.1)'''
