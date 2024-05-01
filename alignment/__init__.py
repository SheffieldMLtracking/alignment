import numpy as np
import scipy

from alignment.camera import Camera
from alignment.calibrationsquare import CalibrationSquare
from alignment.calibrationsquareobservation import CalibrationSquareObservation
from alignment.photo import Photo


class Alignment():
    """
    Pass the photos, calibration squares and cameras.
    """
    def __init__(self, photos, calsquares, cameras):
        self.photos = photos
        self.calsquares = calsquares
        self.cameras = cameras 

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
            print(sumerr)
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
    
def compute_orientation(allimages,allintervals=None,timeout=2000,max_count=None):
    """
    allimages = either:...
                        - a list of list of images (each list is from a different camera)
                        - a list of images (each image is from a different camera)
    allintervals = either...
                        - a list of list of the indices of the time intervals (within each interval the
                        QR code didn't move). Leave as None to assume that the QR codes didn't move &
                        that we shouldn't take into account the time intervals.
                        - None


    max_count = number of tags to look for in each photo (defaults to 1 if allintervals!=None, as it 
                        is likely that there is only one in each photo. Otherwise defaults to 6.
    """
    photos = []
    calsquares = {}
    cameras = []

    if allintervals is None:
        if max_count is None: max_count = 6
        for image in allimages: 
            cam = Camera()
            cameras.append(cam)
            photo = Photo(cam,image)
            photo.decode(calsquares,timeout=2000,max_count=max_count)
            photos.append(photo)
    else:
        if max_count is None: max_count = 1
        for images,intervals in zip(allimages,allintervals): 
            cam = Camera()
            cameras.append(cam)
            for image,interval in zip(images,intervals):
                photo = Photo(cam,image,interval)
                photo.decode(calsquares,timeout=2000,max_count=max_count)
                photos.append(photo)
    al = Alignment(photos,calsquares,cameras)
    al.firstpass()
    al.secondpass()
    return al
