This is a way to track my thesis progress work. Currently the most up to date files is orbit11. 
SatPCMCV2 is the most up to date nimble code.
These will simulate fake geostationary satellite data that has noise introduced to the velocity to help 
account for natural drift that occurs in the space. When the satellite reaches a max distance the satellite
will initiate station keeping to return it to the correct location over earth fixed at some longitude. Once
back on the correct orbit. The satellite will drop stationkeeping behavior. 

latV1.rmd infers the longitude location of geo orbit

We are now using the updated testLogUpdated.csv for satellites that include the ideal position and location.


Orbit14.py and j2TestLog.csv This is the initial orbit now with the effects of j2, solar radiation, and central gravity effects

The most up to date python sim is mainRun, orbits, and plotOrbits
