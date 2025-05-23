import numpy as np
import scipy
import matplotlib.pyplot as plt
from alignment.camera import Camera
from alignment.calibrationsquare import CalibrationSquare
from alignment.calibrationsquareobservation import CalibrationSquareObservation
from alignment.photo import Photo
from numpy.linalg import solve, norm


def getmidpoint(startA,startB,vectA,vectB):
    # for two skew lines, defined by start points and vectors,
    # find midpoint nearest to both
    #based on https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
    UC = np.cross(vectB, vectA);
    UC /= norm(UC)
    RHS = startB - startA
    LHS = np.array([vectA, -vectB, UC]).T
    res = solve(LHS, RHS)
    dist = res[2]
    midpoint = (((res[0]*vectA)+startA)+((res[1]*vectB)+startB))/2
    return midpoint,dist
    
# prints "[ 0. -0.  1.]"

n_steps = 0
def progressbar(x):
    global n_steps
    print("%04d" % n_steps,end="\r")
    n_steps+=1

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
                if obs.heldout: continue
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
        Compute the cost function - this secondpass step tries to match up the locations AND orientations of the calibration squares (and cameras)
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
                if obs.heldout: continue #this is going to be used for testing - so leave this observation out
                if not obs.realobs: continue #this isn't a real observation (we won't have access to obs.cornersx).
                pred_pix_loc = photo.camera.get_pixel_loc(obs.calsquare.get_corner_coords())
                act_pix_loc = np.array([obs.cornersx,obs.cornersy]).T  
                sumerr+=np.sum((pred_pix_loc-act_pix_loc)**2)
                
        #if self.tempcount%1000==0: 
        #    print(sumerr)
        self.tempcount+=1
        return sumerr

    
    def secondpass(self,method='BFGS'):
        """
        This secondpass step tries to match up the locations AND orientaitons of the calibration squares (and cameras)
        """
        init_params = self.get_params_from_cameras_and_calsqrs()
        self.tempcount = 0
        result = scipy.optimize.minimize(self.costfn_secondpass,init_params,method=method,callback=progressbar)#, bounds=bounds)
        self.secondpass_optimize_result = result
        self.setcameras_and_calsqrs(result.x)

    def reset_holdouts(self):
        """
        The cross validation process sets 'heldout' to true for subsets of the observations.
        Later, when we want to use all the data, this needs setting to False for all observations.        
        """
        for photo in self.photos:
            for obs in photo.observations:
                obs.heldout = False

    def compute_orientation(self):
        """Compute the 3d location, and orientation of the cameras (and calibration squares).
        
        The method ignores observations which have 'heldout' set to True.
        """
        print("Starting first pass...")
        self.firstpass()
        print("Starting second pass...")
        self.secondpass()
    
    def draw(self):
        maxintervals = max([photo.timeindex for photo in self.photos])
        for i in range(maxintervals):
            for j in range(len(self.cameras)):
                plt.subplot(maxintervals,len(self.cameras),i*len(self.cameras)+j+1)
                for photo in self.photos:
                    if photo.camera == self.cameras[j]:
                        if photo.timeindex == i:                
                            photo.draw()
                            plt.xlim([0,2048])
                            plt.ylim([1536,0]) 
                            plt.grid()
                            plt.axis('off')    
                            
    def run_cross_validation(self,Nfolds,Ntest=1):    
        """
        Leaves out observations [not photos] from the alignment algorithm, running cross-validation.
        It is really damaging to leave out several observations from one camera as we end up potentially with "unfairly" not enough to align,
        however running a full leave-one-out cross-validation is quite expensive. So here, one can set the Nfolds = number of x-validation runs
        to do, but the Ntest in each, is by default only one - so we don't get to test all the observations. If you set Ntest to None it will
        revert to testing the whole of fold, rather than just the first Ntest.       
                
        """
        summary_N_photos,summary_N_obs = self.compute_summary_table()
        maxintervals = max([photo.timeindex for photo in self.photos])
        xval_observations = []
        for timeindex in range(maxintervals):
            if np.sum(summary_N_obs[timeindex,:])<=2: continue #if there are 3+ photos at this time, we can use it for x-val
            for photo in self.photos:
                if (photo.timeindex != timeindex): continue
                for obs in photo.observations:   
                    xval_observations.append(obs)

        import random
        #random.shuffle(xval_observations)
        allxvalresults = []
        for cam in self.cameras:
            cam.xval_res = []

        #step = 1+len(xval_observations)//Nfolds
        #print("%d observations // %d Nfolds => %d step size" % (len(xval_observations),Nfolds,step))
        #if Ntest is None:
        #    Ntest = step
        ##for splits in range(0,len(xval_observations),step):
        for splits in range(0,Nfolds):
            print("Split %d/%d." % (1+splits,Nfolds))
            for obs in xval_observations:
                obs.heldout = False
            for obs in xval_observations[splits:splits+Ntest]: #just hold out one item
                obs.heldout = True
            self.compute_orientation() 

            #compute error
            for obs in xval_observations[splits:splits+Ntest]:
                predicted_coords = obs.photo.camera.get_pixel_loc(obs.calsquare.get_corner_coords())
                true_coords = np.c_[obs.cornersx,obs.cornersy]
                obs.predicted_coords = predicted_coords 
                maxerror = np.max(np.sqrt(np.sum((predicted_coords - true_coords)**2,1)))
                allxvalresults.append(maxerror)        
                obs.photo.camera.xval_res.append(maxerror)
    
    def print_crossvalidation_summary(self):
        print(" Camera     Errors")
        for cami,cam in enumerate(self.cameras):
            if hasattr(cam,'id'):
                print("%9s  " % cam.id[:9],end="")
            else:
                print("           ",end="")
            print("  ".join(["%0.1f" % xval for xval in cam.xval_res]))
                                          
    def compute_summary_table(self):
        maxintervals = max([photo.timeindex for photo in self.photos])
        summary_N_obs = np.zeros([maxintervals,len(self.cameras)])
        summary_N_photos = np.zeros([maxintervals,len(self.cameras)])
        
        for timeindex in range(maxintervals):
            for cami,camera in enumerate(self.cameras):
                N_photos = 0
                N_obs = 0
                for photo in self.photos:
                    #print("%d?=%d " % (photo.timeindex,timeindex),end="")
                    if (photo.timeindex == timeindex) and (photo.camera==camera):
                        N_photos+=1
                        N_obs+=len(photo.observations)>0
                        
                summary_N_photos[timeindex,cami] = N_photos
                summary_N_obs[timeindex,cami] = N_obs
        return summary_N_photos,summary_N_obs

    def summary(self):
        print("Number of cameras:      %d" % len(self.cameras))
        maxintervals = max([photo.timeindex for photo in self.photos])
        print("Number of time indices: %d" % maxintervals)
        summary_N_photos,summary_N_obs = self.compute_summary_table()
        
        if hasattr(self.cameras[0],'id'):
            print("       ",end="")
            for cam in self.cameras:
                print("%7s " % cam.id[:7],end="")
        print(" ")
        for timeindex in range(maxintervals):
            print(" %4d  " % timeindex,end="")
            for cami,camera in enumerate(self.cameras):
                N_obs = summary_N_obs[timeindex,cami]
                if N_obs == 0:
                    print("   .    ",end="")
                else:
                    print("   %d    " % (N_obs),end="")
                    
            print(" ")
        print("      ",end="")
        for cami,camera in enumerate(self.cameras):
            print("   %2d   " % np.sum(summary_N_obs[:,cami]),end="")
        print("")
        #print("       ",end="")
        #for g in greyscale:
        #    print(" g  " if g else " c  ",end="")

    def get3dpoint(self,cam0_coords,cam1_coords):
        """
        Given 2d photo locations for camera 0 and camera 1, reconstruct 3d location.
        
        - cam0_coords: An Nx2 array of pixel coordinates for camera 0
        - cam1_coords: An Nx2 array of pixel coordinates for camera 1
        returns a tuple (Nx3 array of locations, N array of distances between the 'intersecting' lines)
        """
        loc0 = self.cameras[0].loc
        loc1 = self.cameras[1].loc
        coords3d = []
        distances = []
        for coord0, coord1 in zip(cam0_coords,cam1_coords):
            vec0 = self.cameras[0].get_pixel_local_vector(coord0)
            vec1 = self.cameras[1].get_pixel_local_vector(coord1)
            pos, d = getmidpoint(loc0,loc1,vec0,vec1)
            coords3d.append(pos)
            distances.append(d)
        return np.array(coords3d), np.array(distances)        
         
                                         
def build_alignment_object(allimages,allintervals=None,timeout=2000,max_count=None,get_image_method=None,store_small=None,hfov = 0.846,usecache=True,calsquarewidth=None):
    """
    Returns an Alignment object (that contains the list of cameras and calibration squares),
    will have decoded codes from the photos, and be ready for running the alignment algorithm.
    

    allimages = either:...
                        - a list of list of images (each list is from a different camera)
                        - a list of images (each image is from a different camera)
    Each image is either:
                        - a 2d numpy array
                        - an identifier (e.g. a filename). You will need to set the get_image_method,
                          this takes the identifier and returns the numpy array.
    allintervals = either...
                        - a list of list of the indices of the time intervals (within each interval the
                        QR code didn't move). Leave as None to assume that the QR codes didn't move &
                        that we shouldn't take into account the time intervals.
                        - None

    The images instead be identifiers (e.g. filenames)


    max_count = number of tags to look for in each photo (defaults to 1 if allintervals!=None, as it 
                        is likely that there is only one in each photo. Otherwise defaults to 6.

    store_small = Whether to store the image as a small version
    
    hfov = horizontal field of view, in radians.
    
    usecache = whether to use the cache of decodings
    
    calsquarewidth = size of one side of the calibration square (defaults if None to 0.168 i.e. 16.8cm)
    """
    photos = []
    calsquares = {}
    cameras = []

    #we will store small version if get_image_method is set, as that implies
    #we can't fit the whole image set in memory.
    if store_small is None: store_small = get_image_method is not None

    if allintervals is None:
        if max_count is None: max_count = 6
        for image in allimages:
            image_reference = None
            if get_image_method is not None:
                image_reference = image
                image = get_image_method(image)
                if image is None: continue
            cam = Camera(hfov=hfov,res=image.shape[::-1])
            cameras.append(cam)
            photo = Photo(cam,image)
            photo.image_reference = image_reference #this allows us to link back to the file

            photo.decode(calsquares,timeout=timeout,max_count=max_count,store_small=store_small,usecache=usecache,calsquarewidth=calsquarewidth)

            photos.append(photo)
    else:
        if max_count is None: max_count = 1
        for images,intervals in zip(allimages,allintervals):  #a list of lists of images, each list is from a different camera
            firstimg = True
            for image,interval in zip(images,intervals):
                image_reference = None
                if get_image_method is not None:
                    image_reference = image
                    image = get_image_method(image)
                    if image is None: continue
                if firstimg: #create new camera object
                    cam = Camera(hfov=hfov,res=image.shape[::-1])
                    cameras.append(cam)
                    firstimg = False
                photo = Photo(cam,image,interval)

                photo.decode(calsquares,timeout=timeout,max_count=max_count,store_small=store_small,usecache=usecache)

                photo.image_reference = image_reference #this allows us to link back to the file

                photos.append(photo)
    return Alignment(photos,calsquares,cameras)
