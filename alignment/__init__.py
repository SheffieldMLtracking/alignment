from PIL import Image
from pylibdmtx.pylibdmtx import encode, decode
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy
import regex as re
import hashlib
import pickle


class CalibrationSquareObservation():
    """
    An observation of a calibration square in a single photo, there might be several for each square, and
    several for each photo
    """
    def __init__(self,calsquare,photo,dec=None):
        """
            calsquare = the calibration square object that we have observed.
            photo = the photo object we have observed this calibration square in.    
            dec = the pylibdmtx decode tuple associated with the decoding (should contain cornersx, cornersy).
                     leave this as None if this hasn't really been observed, and is just being included to
                     be inferred later.
        """
        self.photo = photo
        self.calsquare = calsquare
        if dec is not None: #this is a real observation!
            self.cornersx = dec.cornersx
            self.cornersy = dec.cornersy
            #self.data = dec.data
            self.dist = None #don't know yet
            self.realobs = True
            self.compute_ellipse()
        else:
            self.realobs = False
        
    def get_ellipse_parameters(self,params):
        """
        Takes a list of parameters describing the ellipse: [x1,y1,x2,y2,d]
        x1,y1 and x2,y2 are the two foci.
        d is the distance (from focus1 to edge to focus2).
        returns middle, the two axes lengths and the angle to rotate (in radians)
        """
        #get the angle to rotate the ellipse
        angle = np.arctan((params[3]-params[1])/(params[2]-params[0]))
        #get the distance between the foci
        d = np.sqrt((params[3]-params[1])**2 + (params[2]-params[0])**2)
        #get the mid point
        mid = np.array([(params[0]+params[2])/2,(params[1]+params[3])/2])
        #the major axis is equal to the distance (between a focus, the edge and to the other focus)
        major = params[4]
        #the minor axis is equal to...
        if d>params[4]:
            minor = 0.0001
        else:
            minor = np.sqrt(params[4]**2 - d**2) #np.sqrt(params[4]**2/4 - d**2/4)*2    
        return mid,major,minor,angle


    def ellipse_cost(self,params):
        """
        Computes a cost function that penalises how far the corners are from the ellipse, and also the area
        of the ellipse (there is a analytic solution in a paper somewhere, but this does the job!)
        """
        dists = []
        for x,y in zip(self.cornersx,self.cornersy):
            dists.append(np.sqrt((x-params[0])**2 + (y-params[1])**2) + np.sqrt((x-params[2])**2 + (y-params[3])**2))
        dists = np.array(dists)
        err = np.sum((dists - params[4])**2)

        mid,major,minor,angle = self.get_ellipse_parameters(params)
        err+=(major*minor)/10000
        return err

    def compute_ellipse(self):
        """
        Computes the smallest ellipse that passes through the 4 corners of the code
        (stores it in this object)
        """
        points = np.c_[self.cornersx,self.cornersy] #the four corners in an array.
        mid = np.mean(np.c_[self.cornersx,self.cornersy],0) #a rough mid point
        #a starting value for the radius of the ellipse
        l = ((np.max(self.cornersx)-np.min(self.cornersx))+(np.max(self.cornersy)-np.min(self.cornersy)))/4
        #we'll start with the two foci on either side of the mid point, and the above approximate radius length
        params = np.r_[mid-2,mid+2,np.array([l])]
        results = scipy.optimize.minimize(self.ellipse_cost,params) #optimise & save in params
        self.ellipse_params = results.x
        #convert to mid,major,minor & angle
        self.mid, self.major, self.minor, self.angle = self.get_ellipse_parameters(self.ellipse_params)

        #check if the midpoint makes sense -- this is a useful heuristic for if the optmisier has failed
        check = (np.min(self.cornersx)<self.mid[0]<np.max(self.cornersx)) and \
                 np.min(self.cornersy)<self.mid[1]<np.max(self.cornersy)

        self.dist = None
        if not check: 
            print("Failed to compute distance...")
            print(self.mid)
            return #not sure what to do?

        
        hfov = self.photo.camera.hfov
        vfov = self.photo.camera.vfov
        photo = self.photo
        physicalwidth = self.calsquare.width*np.sqrt(2)
        
        theta = hfov * self.major/photo.camera.res[0]
        self.dist = physicalwidth/np.tan(theta)
        
        
        fromcam_pos = np.zeros(3)
        fromcam_pos[1] = self.dist * hfov * -(self.mid[0] - photo.camera.res[0]/2) / photo.camera.res[0]
        fromcam_pos[2] = self.dist * vfov * -(self.mid[1] - photo.camera.res[1]/2) / photo.camera.res[1]
        fromcam_pos[0] = self.dist

        self.spatial_position = fromcam_pos 
        
    def draw(self,ax):
        """
            Draw the observed calibration square result:
              - will draw the corners provided by the decoding
              - if we've got the ellipse around these corners computed successfully that is drawn too.
              - if the centre location of the calibration square is estimated in 3d, then a yellow 'x' is drawn.
              - if the 3d coordinates of the corners is know then these are drawn in blue (with '+'s).
            Parameters:
                ax = plot axes to draw on"""
        if self.realobs: 
            ax.plot(self.cornersx,self.photo.camera.res[1]-np.array(self.cornersy),'-w')
            #approximation approach....
            if self.dist: #if we've actually got a reliable ellipse...
                #convenience variable
                midflipped = np.array([self.mid[0],self.photo.camera.res[1]-self.mid[1]])        
                from matplotlib.patches import Ellipse
                ell = Ellipse(xy=midflipped,
                                width=self.major, height=self.minor,
                                angle=-np.rad2deg(self.angle))
                ax.add_artist(ell)
                ell.set_alpha(0.3)
                ell.set_facecolor([0.9,0.9,0])
                plt.text(midflipped[0],midflipped[1],"%0.2f, %0.2f, %0.2f" % tuple(self.spatial_position),fontsize=10,color='w')


        #exact corners etc...
        coord = self.photo.camera.get_pixel_loc(self.calsquare.loc)[0,:]
        plt.plot(coord[0],self.photo.camera.res[1]-coord[1],'xy') #rough location
        coords = self.photo.camera.get_pixel_loc(self.calsquare.get_corner_coords(),addstarttoend=True)
        plt.plot(coords[:,0],self.photo.camera.res[1]-coords[:,1],'+b-') #exact corners

    
class CalibrationSquare():
    """
    The calibration is done with QR codes. The 3d locations and orientations etc are defined by this class:
    note that each CalibrationSquare might be observed by multiple cameras in multiple photos.
    """
    def __init__(self, calsqrid=None, loc=None, orientation=None, width=None):
        """
        calsqrid = the id of the calibration square (this might either be just the code, if the squares don't move, or a code+time combination, if the squares can move!
        loc = location in 3d (initialised randomly by default)
        orientation = vector: yaw, pitch, roll (initialised randomly by default)
        width = one side of the code (in e.g. metres).
        """
        if loc is None:
            loc = np.random.rand(3)*10
        if orientation is None:
            orientation = np.random.rand(3)*np.pi*2-np.pi
        if width is None:
            width = 0.168 #16.8cm
        self.loc = loc
        self.orientation = orientation
        self.width = width
        #self.photo = photo
        self.calsqrid = calsqrid #not used!
        self.observations = []
        
    def get_corner_coords(self):
        """Get the coordinates of corners (returns a numpy array of 3d coordinates)"""
        w = self.width/2
        p = np.array([[w,w,0],[w,-w,0],[-w,w,0],[-w,-w,0]])
        r1 = R.from_euler('z', self.orientation[0], degrees=False) #yaw
        r2 = R.from_euler('Y', self.orientation[1], degrees=False) #pitch (intrinsic rotation around y axis)    
        r3 = R.from_euler('X', self.orientation[2], degrees=False) #roll (intrinsic rotation around x axis)    
        pvec = r3.apply(r2.apply(r1.apply(p)))
        return pvec + self.loc
    
    def addPhoto(self,photo,dec):
        """
        Add a photo associated with this calibration square.
            photo = photo object
            dec = the decode tuple (from pylibmtx)
        creates a new CalibrationSquareObservation object, adds to this object's list of observations and to the photo's list of observations.
        """
        obs = CalibrationSquareObservation(self,photo,dec)
        self.observations.append(obs)
        photo.observations.append(obs)
        

class Camera():
    """
    Each camera has a list of photos, a coordinate and orientation, and other parameters.
    """
    def __init__(self, loc=None, orientation=None, hfov = 0.846, vfov = None, res = (2048,1536)):
        """
        loc = location in 3d of camera (metres), default = random
        orientation = orientation (yaw,pitch,roll) of camera (radians), default random yaw, zero pitch or roll
        hfov = horizontal field of view (radians), (default = 48.5 degrees)
        vfov = vertical field of view (radians), (default = computed from res and hfov).
        res = tuple of resolution of camera (horizontal x vertical, default = (2048,1536))
        """
        if loc is None:
            loc = np.random.rand(3)*10 #default to random locaton
        if orientation is None:
            orientation = np.random.rand(3)*np.pi*2-np.pi #default to random orientation
            orientation[1:]=0
        self.loc = loc
        self.orientation = orientation
        
        self.res = res
        
        self.hfov = hfov
        if vfov is None:
            vfov = hfov * res[1]/res[0] #in radians, but e.g. 48.5 * (1436/2048) = 34 degrees.
    
        self.vfov = vfov
    
    def get_global_loc(self, spatial_position):
        """
        Given a location relative to camera, compute global coordiates in 3d.
        """
        
        
        r1 = R.from_euler('z', -self.orientation[0], degrees=False) #yaw
        r2 = R.from_euler('Y', -self.orientation[1], degrees=False) #pitch (intrinsic rotation around y axis)    
        r3 = R.from_euler('X', -self.orientation[2], degrees=False) #roll (intrinsic rotation around x axis)    
        pvec = r1.apply(r2.apply(r3.apply(spatial_position)))        
        pvec = np.array(pvec + self.loc) 
        return pvec
        
    def get_local_loc(self, spatial_position):
        """
        Given a global location, compute location in camera coordinates in 3d.
        """
        p = np.array(spatial_position - self.loc)
        
        r1 = R.from_euler('z', self.orientation[0], degrees=False) #yaw
        r2 = R.from_euler('Y', self.orientation[1], degrees=False) #pitch (intrinsic rotation around y axis)    
        r3 = R.from_euler('X', self.orientation[2], degrees=False) #roll (intrinsic rotation around x axis)    
        pvec = r3.apply(r2.apply(r1.apply(p)))
        return pvec
        
        
    def get_pixel_loc(self, coordinates,addstarttoend=False):
        """
        What 2d pixel location will 3d coordinates have for this camera.
        """
        pvec = self.get_local_loc(coordinates)
        
        
        if len(pvec.shape)==1:
            pvec = pvec[None,:]
        #We look down the x-axis...with x-axis being distance
        pvec[pvec[:,0]<0.001,0]=0.001 #nearly behind
        pixel_position = np.array([self.res[0]/2+self.res[0]*(-pvec[:,1]/pvec[:,0])/self.hfov,
                                   self.res[1]/2+self.res[1]*(-pvec[:,2]/pvec[:,0])/self.vfov]).T
        

        #assert np.all(np.array(self.old_get_pixel_loc(cam,markercoords))==res)
        if addstarttoend:
            return np.c_[pixel_position,pixel_position[:,0]]
        return pixel_position 
  
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
            
                
    def draw(self):
        """
        Draws the photo (with the associated observations).
        """
        img = 60*self.image/np.mean(self.image)
        img[img>200] = 200
        plt.imshow(img,cmap='gray',clim=[0,255])
        for obs in self.observations:
            obs.draw(plt.gca())

class NotInIntervalException(Exception):
    """
    If we try to find the timeindex of a photo that doesn't have one (lies outside all the time intervals)
    """
    pass
    

class Alignment():
    """
    To use this class, one needs to inherit from it, and extend depending on the type of images and calibration squares / timings you've collected.
    """
    def __init__(self,pathsToFiles=[]):
        raise NotImplementedError

    def setcameras(self,params):
        """
        Set the camera locations/orientations from the params list.
        """
        idx = 0
        for cam in self.cameras[1:]:
            cam.loc = params[idx:idx+3]
            idx+=3
            cam.orientation = params[idx:idx+3]
            idx+=3

    def costfn_firstpass(self,params):
        """Compute the cost function - this firstpass step only tries to match up the locations of the calibration squares (and locations & orientation of cameras)"""
        self.setcameras(params)
        err = 0
        for a,calsqr in self.calsquares.items():
            coords = []
            for obs in calsqr.observations:
                coords.append(obs.photo.camera.get_global_loc(obs.spatial_position))
            coords = np.array(coords)
            if len(coords)>1:
                err+=np.mean(np.var(coords,0))
        return err

    def firstpass(self,method='BFGS'):
        """
        Attempts to place the cameras and calibration squares in the right locations approxiamtely (doesn't orientate the calibration squares).
        """
        self.cameras[0].loc = np.zeros(3)
        self.cameras[0].orientation = np.zeros(3)

        params = np.random.randn(6*(len(self.cameras)-1))
        result = scipy.optimize.minimize(self.costfn_firstpass,params,method=method)
        self.firstpass_optimize_result = result
        self.setcameras(result.x)

        #for all the cameras that have each of the calibration squares in,
        #find the average location and set that as the cal. square's location
        for a,calsqr in self.calsquares.items():
            coords = []
            for obs in calsqr.observations:
                coords.append(obs.photo.camera.get_global_loc(obs.spatial_position))
            coords = np.array(coords)   
            calsqr.loc = np.mean(coords,0)

    def setcameras_and_calsqrs(self,params):
        """
        Set the camera & calibration square locations/orientations from the params list
        """
        idx = 0
        for cam in self.cameras[1:]:
            cam.loc = params[idx:idx+3]
            idx+=3
            cam.orientation = params[idx:idx+3]
            idx+=3
        for _, calsqr in self.calsquares.items():
            calsqr.loc = params[idx:idx+3]
            idx+=3
            calsqr.orientation = params[idx:idx+3]
            idx+=3

    def get_params_from_cameras_and_calsqrs(self):
        """
        Get the camera & calibration square locations/orientations from the list of objects.
        """   
        idx = 0
        params = []
        for cam in self.cameras[1:]:
            params.extend(list(cam.loc))
            params.extend(list(cam.orientation))  
        for a,calsqr in self.calsquares.items():
            params.extend(list(calsqr.loc))
            params.extend(list(calsqr.orientation))
        return params

    def costfn_secondpass(self,params):
        """
        Compute the cost function - this secondpass step tries to match up the locations AND orientaitons of the calibration squares (and cameras)
        """
        self.setcameras_and_calsqrs(params)

        #fix one camera!
        self.cameras[0].loc = np.zeros(3)
        self.cameras[0].orientation = np.zeros(3)
        #fix 2nd camera
        #cameras[1].loc = np.array([3.15,2.65,-1.2])
        #cameras[1].orientation = np.deg2rad(np.array([120,20,0]))

        sumerr = 0
        for photo in self.photos:
            for obs in photo.observations:
                pred_pix_loc = photo.camera.get_pixel_loc(obs.calsquare.get_corner_coords())
                #pred_pix_loc = cam.get_pixel_loc(obs.calsquare.get_corner_coords())
                act_pix_loc = np.array([obs.cornersx,obs.cornersy]).T  

                #if count%10==0: 
                #    print("cami",cami,"calindex",calindex,"camLoc",cam.loc,"calloc",calsqr.loc,"getcornercoords",calsqr.get_corner_coords(),"predPix",pred_pix_loc, "actPix",act_pix_loc)
                sumerr+=np.sum((pred_pix_loc-act_pix_loc)**2)



        if self.tempcount%1000==0: 
            #print("==============================")
            print(sumerr)
            #print(cameras[0].loc,cameras[1].loc,cameras[0].orientation,cameras[1].orientation)
            #print(calsqr.loc)
            #print("==============================")
        self.tempcount+=1
        return sumerr

    
    def secondpass(self,method='BFGS'):
        """
        This secondpass step tries to match up the locations AND orientaitons of the calibration squares (and cameras)
        """
        init_params = self.get_params_from_cameras_and_calsqrs()
        self.tempcount = 0
        result = scipy.optimize.minimize(self.costfn_secondpass,init_params,method=method)#, bounds=bounds)
        self.secondpass_optimize_result = result
        self.setcameras_and_calsqrs(result.x)
    
