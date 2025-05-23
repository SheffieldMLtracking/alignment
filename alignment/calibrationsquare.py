import numpy as np
from alignment.calibrationsquareobservation import CalibrationSquareObservation
from scipy.spatial.transform import Rotation as R

class CalibrationSquare():
    """
    The calibration is done with QR codes. The 3d locations and orientations etc are defined by this class:
    note that each CalibrationSquare might be observed by multiple cameras in multiple photos.
    """
    def __init__(self, calsqrid=None, loc=None, orientation=None, width=0.168):
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
        
