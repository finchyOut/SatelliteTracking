from satelliteCallFuncs import callBB, callPd, callLqr, createSim, multipleSimsAll
from satellitePlots import allPlots, plot3d, plot2d, plot_accel_x, plot_accel_y, plot_accel_z, plot_error, plot_paths_3d, plot_paths_xy
from MMD import dotKernel,  MMD, gaussianKernel, cosineKernel, laplacianKernel
import pandas as pd

#/opt/anaconda3/envs/cleanorbit/bin/python "/Users/emmafinch/Desktop/Thesis/Python Code/setPolicies/mainRun.py"
#bbDf = callBB(daysToProp= 1, time_Step = 30, longitudeDeg=45, maxDrift=36, noise_std=1e-8, maxA=1e-6, satId=1)
#pdDf = createSim(policy = 'PD',daysToProp= 1, time_Step = 30, longitudeDeg=45, maxDrift=36, noise_std=1e-8, maxA=1e-6, satId=1)
#allPlots(pdDf, 'pd')
#allPd, allBb, allLqr = multipleSimsAll(numReps = 3,daysToProp= 1, time_Step = 30, longitudeDeg=45, maxDrift=36, noise_std=1e-8, maxA=1e-6, satId=1 )

allPd, allBb, allLqr = multipleSimsAll(numReps = 3,daysToProp= 1, time_Step = 30, longitudeDeg=45, maxDrift=36, noise_std=1e-8, maxA=1e-6, satId=1 )


distanceDot = MMD(dotKernel, allPd, allBb)
distanceGaussianKernel = MMD(gaussianKernel, allPd, allBb, sigma =30000)
distanceCosKernel = MMD(cosineKernel, allPd, allBb, sigma =30000)
distanceLap = MMD(laplacianKernel, allPd, allBb, sigma =30000)

print("Dot kernel: ", distanceDot)
print("Gaussian kernel: ", distanceGaussianKernel)
print("Cosine kernel: ", distanceCosKernel)
print("Laplacian distance: ", distanceLap)