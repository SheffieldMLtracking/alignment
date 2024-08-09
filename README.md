# alignment
- Uses a library to find the matrix codes in images from cameras at different angles
- Uses them to find out locations and orientations of cameras and the codes.

# Background to the Problem
Given multiple cameras pointing at a similar volume, how do we find their relative orientations and locations? In my problem the 4-7 cameras are typically in a field, mounted at arbitary angles, and potentially may not all be able to see the same volume.

# Other Related Tools
- OpenCV's triangulation
- https://strawlab.org/braid/
  
- Potential tools include:
whycon https://github.com/lrse/whycon - finding markers in video.

- AprilTag https://april.eecs.umich.edu/papers/details.php?name=olson2011tags good for localising the 6DOF of a 2d code from a single image [but I think this turns out to be somewhat inaccurate -- e.g. depth is often difficult, and when a long way away there is some ambiguity about orientation (e.g. could be tilted towards or away from the camera). It also doesn't appear to be for camera registration.

- Vicon's camera calibration -- feels quite proprietory. And seems to require that you use the particular hardware etc.
https://help.vicon.com/space/Nexus212/11248865/Calibrate+Vicon+cameras

- Various papers...
https://link.springer.com/chapter/10.1007/978-3-642-42057-3_101 -- uses depth camera.

- Off topic -- guessing depth from features 
https://arxiv.org/abs/2402.04883

- MATLAB https://www.sciencedirect.com/science/article/pii/S1474034623002203 -- but this looks close

- Various image-matching approaches are related ... https://arxiv.org/abs/2003.01587

- https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0653

- need to look at https://en.wikipedia.org/wiki/ARTag

- https://www.noldus.com/track3d
