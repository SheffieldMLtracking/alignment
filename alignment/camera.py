import numpy as np
from scipy.spatial.transform import Rotation as R

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
        self.photos = [] #a useful list linking back to the photos
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
