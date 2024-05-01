import numpy as np
from alignment.calibrationsquareobservation import CalibrationSquareObservation

class NotInIntervalException(Exception):
    """
    If we try to find the timeindex of a photo that doesn't have one (lies outside all the time intervals)
    """
    pass

def getintervalstarts(times,interval_length):
    """
    Given a list of times (e.g. number of seconds from midnight), return the list
    of start times of each interval. For example: 233.14, 233.41, 236.16, 236.23...
    with an interval_length of one second, would return two interval start times, at 
    233.14 and 236.16. These can then be used to assign each time point to a particular
    time index.
    """
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

def addmissingobs_stationarysquares(calsquares,photos):
    #we add synthetic observations here to see how well it does...
    for code,calsqr in calsquares.items():
        for photo in photos:
            if photo not in [obs.photo for obs in calsqr.observations]:
                obs = CalibrationSquareObservation(calsqr,photo)
                calsqr.observations.append(obs)
                photo.observations.append(obs)                