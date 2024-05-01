import hashlib
import pickle
from PIL import Image
import numpy as np
from pylibdmtx.pylibdmtx import decode #TODO Could make this optional
from alignment.calibrationsquare import CalibrationSquare
import matplotlib.pyplot as plt

class Photo():
    def __init__(self,camera=None,image=None,timeindex=None):
        """
        When using the codes to register the cameras, there are two options
         - either the codes stay stationary between multiple photos (in which case
                the time they were taken is irrelevant)
         - or the code(s) move between photos (in which case the time they were
                taken needs to be included in the uniqueness of the code.
        If the former is the case, leave timeindex as None.
        If the latter is the case, then you need to provide an index such that
        the index is the same for the photos from different cameras taken at (approximately)
        the same time.
        
        image = a numpy array.
        camera = Which camera took the photo (class Camera)
        """
        self.timeindex = timeindex
        self.camera = camera        
        self.image = image
        self.observations = []
            
    def decode(self,calsquares,timeout=None,decodes=None,max_count=1,usecache=True):
        """
         - calsquares = Pass the dictionary of current calibration squares (to potentially add to)
         - timeout = if we want to stop decoding early
         - max_count = default 1 (assumes 0 to 1 calibration squares are in the photo).
         - usecache = default True (only decodes if cache not found).
         
         for each decoded square,
         - if the square is not in the calsquares list: adds a CalibrationSquare to the calsquares list
         - calls the 'addPhoto' method for the calsquare (this adds the actual observation of that square).
        """
        im = self.image*10 #the library seems to work better if I brighten the image...
        im[im>255] = 255
        
        if decodes is None: #we haven't been given the codes...
            #try the cache...
            imghash = hashlib.md5(im.tobytes()).hexdigest()
            
            try:
                imgcache = pickle.load(open('decodecache.pkl','rb'))
            except FileNotFoundError:
                print("Failed to find cache file")
                imgcache = {}
            if (imghash in imgcache) and (usecache): #in cache...
                decodes = imgcache[imghash]
            else: #not in cache, need to decode...
                print("Not in cache, decoding...")

                #TODO Make decode import (and PIL import) optional, and allow user to supply the decode info
                img = Image.fromarray(im.astype(np.uint8))
                decodes = decode(img,max_count=max_count,timeout=timeout) #9.74s without max_count; 51ms with max_count=1
                imgcache[imghash] = decodes
            pickle.dump(imgcache,open('decodecache.pkl','wb'))
            
        for dec in decodes:
            calsqrid = str(dec.data)
            if self.timeindex is not None:
                calsqrid = calsqrid+str(self.timeindex)
            if calsqrid not in calsquares:
                calsquares[calsqrid] = CalibrationSquare(calsqrid)            
            calsquares[calsqrid].addPhoto(self,dec)      
                    
    def draw(self): #TODO Decide if we want a draw method at all
        """
        Draws the photo (with the associated observations).
        """
        img = 60*self.image/np.mean(self.image)
        img[img>200] = 200
        plt.imshow(img,cmap='gray',clim=[0,255])
        for obs in self.observations:
            obs.draw(plt.gca())