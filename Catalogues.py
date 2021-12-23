import numpy as np
#Angular distance between two sources
# ra-Right accession
# dec-Declination

def angular_dist(ra1, dec1, ra2, dec2):
    r1 = np.radians(ra1)
    d1 = np.radians(dec1)
    r2 = np.radians(ra2)
    d2 = np.radians(dec2)
    a = np.sin(np.abs(d1-d2)/2)**2
    b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
    rad = 2*np.arcsin(np.sqrt(a + b))
    d = np.degrees(rad)
    return d
    

#Importing catalogues
    
#Bright Source Sample of AT20G Survey(Massardi+, 2008)
    
def import_bss():
    cat = np.loadtxt('bss.dat', usecols=range(1, 7))
    n=cat.tolist()
    halfout=[]
    for i in range(len(n)):
        ra = 15*(cat[i][0] + cat[i][1]/60 + cat[i][2]/(60*60))
        if cat[i][3]<0:
            d = (-1*(-cat[i][3] + cat[i][4]/60 + cat[i][5]/(60*60)))
        else:
            d = (cat[i][3] + cat[i][4]/60 + cat[i][5]/(60*60))
        o1 = [i+1]+[ra]+[d]
        o2 = tuple(o1)
        halfout.append(o2)
    out = np.asarray(halfout)
    return out


# SuperCOSMOS catalogue
    
def import_super():
    cat = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
    n=cat.tolist()
    halfout=[]
    for i in range(1,len(n)+1):
        o1 = [i]+list(cat[i-1])
        o2 = tuple(o1)
        halfout.append(o2)
    out = np.asarray(halfout)
    return out


#Finding closest position of a point
    
cat = import_bss()
def find_closest(cat,ra,d):
    dist_list=[]
    def angular_dist(ra1, dec1, ra2, dec2):
        r1 = np.radians(ra1)
        d1 = np.radians(dec1)
        r2 = np.radians(ra2)
        d2 = np.radians(dec2)
        a = np.sin(np.abs(d1-d2)/2)**2
        b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
        rad = 2*np.arcsin(np.sqrt(a + b))
        d = np.degrees(rad)
        return d
    for i in range(len(cat)):
        distance =  angular_dist(cat[i][1], cat[i][2], ra, d)
        dist_list.append(distance)
    closest = min(dist_list)
    index = dist_list.index(min(dist_list))+1
    return (index,closest)


#Crossmatching SuperCOSMOS and BSS

bss_cat = import_bss()
super_cat = import_super()    
def crossmatch(bss_cat, super_cat, max_dist):
    '''This one helps to confirm whether the object is the same object in both
    catalogues or more than one object that are really close. The accuracy 
    increases with decrease in the value of max_dist'''
    matches=[]
    no_matches=[]
    def angular_dist(ra1, dec1, ra2, dec2):
        r1 = np.radians(ra1)
        d1 = np.radians(dec1)
        r2 = np.radians(ra2)
        d2 = np.radians(dec2)
        a = np.sin(np.abs(d1-d2)/2)**2
        b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
        rad = 2*np.arcsin(np.sqrt(a + b))
        d = np.degrees(rad)
        return d
    for i in range(len(bss_cat)):
        for j in range(len(super_cat)):
            distance =  (angular_dist(bss_cat[i][1], bss_cat[i][2], super_cat[j][1], super_cat[j][2]))
            if distance < max_dist:
                element = (bss_cat[i][0], super_cat[j][0], distance)
                matches.append(element)
            else:
                element = bss_cat[i][0]
                no_matches.append(element)
    return (matches,no_matches)
    
                                                            
#Crossmatching function of astropy
#These are the test values from the above mentioned catalogues

ra1 = bss_cat[:,1]
dec1 = bss_cat[:,2] 
ra2 = super_cat[:,1]
dec2 = super_cat[:,2]

from astropy.coordinates import SkyCoord
from astropy import units as u
c = SkyCoord(ra = ra1*u.degree, dec=dec1*u.degree) 
catalogue = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
idx, d2d, d3d = c.match_to_catalog_sky(catalogue)
matches = catalogue[idx]
#Alternate
from astropy.coordinates import match_coordinates_sky
idx, d2d, d3d = match_coordinates_sky(c, catalogue)
idx, d2d, d3d = match_coordinates_sky(c.frame, catalogue.frame)

#To impose a separation constraint

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalogue)
sep_constraint = d2d < max_sep
c_matches = c[sep_constraint]
catalog_matches = catalogue[idx[sep_constraint]]

#A convenient way to create a frame centered on an arbitrary position on the sky suitable for computing positional offsets

from astropy.coordinates import SkyOffsetFrame, ICRS
center = ICRS(10*u.deg, 45*u.deg)
center.transform_to(SkyOffsetFrame(origin=center)) 
target = ICRS(11*u.deg, 46*u.deg)
target.transform_to(SkyOffsetFrame(origin=center))  
#Alternatively, the convenience method skyoffset_frame() lets you create a sky offset frame from an existing SkyCoord:
center = SkyCoord(10*u.deg, 45*u.deg)
aframe = center.skyoffset_frame()
target.transform_to(aframe) 
other = SkyCoord(9*u.deg, 44*u.deg, frame='fk5')
other.transform_to(aframe)  
#e.g. objects around M31 are sometimes shown in a coordinate frame aligned with standard ICRA RA/Dec, but on M31:
m31 = SkyCoord(10.6847083*u.deg, 41.26875*u.deg, frame='icrs')
ngc147 = SkyCoord(8.3005*u.deg, 48.5087389*u.deg, frame='icrs')
ngc147_inm31 = ngc147.transform_to(m31.skyoffset_frame())
xi, eta = ngc147_inm31.lon, ngc147_inm31.lat
