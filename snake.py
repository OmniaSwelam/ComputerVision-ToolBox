import numpy as np
import skimage.filter as filt
#import gaussianfilter
#%%
def create_A(a, b, N):
    """
    a: float
    alpha parameter

    b: float
    beta parameter

    N: int
    N is the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
    """
    #row = 1D vector (Nx1)
    #np.r_: creates an 1D array of these values with the size of (number of these values x1)
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N,N)) 
    for i in range(N): #0 : N-1 (loop over rows)
        A[i] = np.roll(row, i) #roll= circular shift
    return A #square matrix of num. of pts of snake
#%%
def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
#%%
def create_external_edge_force_gradients_from_img( img, sigma=30. ):
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.

    img: ndarray
        The image.
    """
    #Convert to gray scale
    if (len(img.shape)==3):
        img= rgb2gray(img)
    else:
        img= img
    
    # Gaussian smoothing.
    #smoothed = gaussianfilter.gaussianfilter( (img-img.min()) / (img.max()-img.min()),21, sigma )
    smoothed = filt.gaussian( (img-img.min()) / (img.max()-img.min()), sigma )
    
    # Gradient of the image in x and y directions.
    giy, gix = np.gradient( smoothed )
    # Gradient magnitude of the image.
    gmi = (gix**2 + giy**2)**(0.5)
    # Normalize. This is crucial (empirical observation).
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

    # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient( gmi )

    def fx(x, y):
        """
        Return external edge force in the x direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        #make numbers<0 =0 (no negative numbers if there)
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.
        #if x or y were out of the image, make them last pixel of image
        x[ x > img.shape[1]-1 ] = img.shape[1]-1 #rows (y)
        y[ y > img.shape[0]-1 ] = img.shape[0]-1 #cols (x)
        #ggmix was the 2nd gradient of the whole image
        #now I choose from it the 2nd gradient also but only for the chosen points on contour
        #(which are some points on the image)
        return ggmix[ (y.round().astype(int), x.round().astype(int)) ]

    def fy(x, y):
        """
        Return external edge force in the y direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmiy[ (y.round().astype(int), x.round().astype(int)) ]

    return fx, fy

def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    """
    x: ndarray
        intial x coordinates of the snake

    y: ndarray
        initial y coordinates of the snake
Note: length of x = length of y= number of points of snake
    a: float
        alpha parameter

    b: float
        beta parameter

    fx: callable
        partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.

    fy: callable
        see fx.

    gamma: float
        step size of the iteration
    
    n_iters: int
        number of times to iterate the snake

    return_all: bool
        if True, a list of (x,y) coords are returned corresponding to each iteration.
        if False, the (x,y) coords of the last iteration are returned.
    """
    A = create_A(a,b,x.shape[0]) 
    #B= inverse of (I- gamma*A) "NxN"
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append( (x_.copy(),y_.copy()) )

    if return_all:
        return snakes
    else:
        return (x,y)
