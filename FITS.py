##Imports

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

##Reading a FITS image

hdulist=fits.open('E:/COURSES/Data driven astronomy-Coursera/fits1/0000.fits')
hdulist.info()


##To access particular elements

data=hdulist[0].data


##To get info 

print(data.shape)
print(data)


##To plot the image

plt.imshow(data,cmap=plt.cm.viridis)
plt.xlabel("x-pixels (RA)")
plt.ylabel("y-pixels (Dec)")
plt.colorbar()
plt.show()
plt.savefig("First_FITS.png")


##To find the position of max vaue
 
def load_fits(fitsfile):
    hdulist=fits.open(fitsfile)
    data=hdulist[0].data
    maxvalue=np.argmax(data)
    max_ind = np.unravel_index(maxvalue, data.shape)
    return max_ind


##To find the mean of the central value of several FITS images

def mean_fits(fitsfile):
    n = len(fitsfile)         
    if n > 0:
        hdulist=fits.open(fitsfile[0])
        required=hdulist[0].data
        for i in range(1,n):
            hdulist += fits.open(fitsfile[i])
            required +=hdulist[0].data
    data_mean = required/n
    return data_mean
    