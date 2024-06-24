from matplotlib import pyplot as plt
import numpy as np
import scipy


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
        self.heldout = False
        
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
            ax.plot(self.cornersx,self.photo.camera.res[1]-np.array(self.cornersy),'ow')
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
        
        if hasattr(self,'predicted_coords'):
            plt.plot(self.predicted_coords[:,0],self.photo.camera.res[1]-self.predicted_coords[:,1],'+y')
