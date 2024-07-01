import numpy as np
import pickle
from alignment.calibrationsquareobservation import CalibrationSquareObservation

class NotInIntervalException(Exception):
    """
    If we try to find the timeindex of a photo that doesn't have one (lies outside all the time intervals)
    """
    pass

def guesscamtypegetscore(fn):
    """
    Used to guess if camera is greyscale or colour. If the return value is less than about 0.02 it's probably greyscale.
    fn = filename.
    """
    photo = pickle.load(open(fn,'rb'))
    img = photo['img']
    if img is None: return np.NaN
    #e.g. 0.0001 = greyscale, 0.7 = colour
    score = np.abs(np.mean(img[0:-2:2,0:-2:2]/2+img[2::2,2::2]/2-img[1:-2:2,1:-2:2])/np.mean(img))
    return score
    
def guesscamtype(path,camid):
    """
    Pass the 
    Guesses camera type (returns a string either 'greyscale' or 'colour').
    Hopefully temporary.
   """
    score = np.nanmean([guesscamtypegetscore(fn) for fn in getimgfilelist(path,camid)[:50:5]])
    if score<0.02:
        return 'greyscale'
    else:
        return 'colour'

def getintervalstarts(times,interval_length):
    """
    Given a list of times (e.g. number of seconds from midnight), return the list
    of start times of each interval. For example: 233.14, 233.41, 236.16, 236.23...
    with an interval_length of one second, would return two interval start times, at 
    233.14 and 236.16. These can then be used to assign each time point to a particular
    time index.
    """
    assert len(times)>0, "Empty list of times."
    intervals = []
    intervalstart = 0
    for t in sorted(times):
        if t>intervalstart+interval_length:
            intervals.append(t)
            intervalstart=t
    intervals.append(intervals[-1]+interval_length)
    intervals = np.array(intervals)
    return intervals

def getinterval(t,intervals):
    """
    Returns which time interval index a given time is in.
    """
    try:
        interval = np.where(t>=intervals)[0][-1]
        if interval==len(intervals)-1:
            raise NotInIntervalException
    except IndexError as exc:
        raise NotInIntervalException from exc
    return interval
    
def addmissingobs_timeinterval(photos):
    """
    For the case in which one code moves between time intervals, we need to just add the
    code to those photos in which it wasn't seen.
    """
    for photo in photos:
        if len(photo.observations)==0:
            idx = photo.timeindex
            for same_idx_photo in photos:
                if same_idx_photo.timeindex!=idx: continue
                if len(same_idx_photo.observations)==0: continue
                obs = CalibrationSquareObservation(same_idx_photo.observations[0].calsquare,photo)
                photo.observations.append(obs)
                continue #only want to add one? TODO Need to handle if multiple codes are in use in moving code situation

def addmissingobs_stationarysquares(calsquares,photos):
    #we add synthetic observations here to see how well it does...
    for code,calsqr in calsquares.items():
        for photo in photos:
            if photo not in [obs.photo for obs in calsqr.observations]:
                obs = CalibrationSquareObservation(calsqr,photo)
                calsqr.observations.append(obs)
                photo.observations.append(obs)                
