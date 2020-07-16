import matplotlib.pyplot as plt
#%%This function saves/writes an image to your pc (so that you can use it after that)
#with the ability to display or not display it. 

def saveImage(image, name):
    plt.figure()
    """if flag=="img":
        plt.imshow(image)
    elif flag== "hist":
        
        plt.hist(image.ravel(), 256, [0,256])"""
    plt.imshow(image)    
    # Turn interactive plotting off
    plt.set_cmap("gray")
    plt.ioff()
    plt.axis('off')
    plt.margins(0,0)
    plt.margins(0,0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(name, bbox_inches = 'tight', pad_inches = 0)
    #plt.show()
    plt.close()
#%%Test 
'''a= plt.imread("peppers.png")
saveImage(a,"a.jpg")'''
