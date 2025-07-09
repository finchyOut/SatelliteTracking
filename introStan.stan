// saved as schools.stan
data {
  int<lower=0> J; // the number of time steps you want to investigate 
  int<lower=0> dt;
  array[J] real x; //the real satellite position in x
  array[J] real y; //the real satellite position in y
  array[J] real z; //the real satellite position in z
  array[J] real vx; //the real satellite velocity in x
  array[J] real vy; //the real satellite velocity in y
  array[J] real vz; //the real satellite velocity in z
  array[J] real ix; // ideal position in x
  array[J] real iy; // ideal position in y
  array[J] real iz; // ideal position in z
  array[J] real ivx; // ideal velocity in x
  array[J] real ivy; // ideal velocity in y
  array[J] real ivz; // ideal velocity in 
  //int<lower=0, upper=1> status[J]; // whether or not the satellite is station keeping
}
parameters {
  real<lower=50, upper = 1000> md;      // maximum distance allowed off orbit
  real<lower= -.006, upper = .005> kp;    // reacts to how far off the satellite is from desired orbit, position control
  real<lower=-.006, upper = .005> kd;     //dampens motion to avoid overshoot or oscillation, velocity control
  real<lower=1e-6> sigmaY; // standard Deviation in position to account for noise
  real<lower=1e-6> sigmaV; // standard Deviation in velocity to account for noise
}
// all of the equations that go into determining our prior and liklihood
transformed parameters {
  //define deterministic variables
  //the acceleration variables
  vector[J] ax;
  vector[J] ay;
  vector[J] az;
  
  //the difference between ideal and current position
  vector[J] dx;
  vector[J] dy;
  vector[J] dz;
  
  //the difference between ideal and current velocity
  vector[J] dvx;
  vector[J] dvy;
  vector[J] dvz;
  
  // the value of station keeping or not
  vector[J] driftDistance;
  real status[J];
  
  ax[1] = 0;
  ay[1] = 0;
  az[1] = 0;
  
  dx[1] = 0;
  dy[1] = 0;
  dz[1] = 0;
  
  dvx[1] = 0;
  dvy[1] = 0;
  dvz[1] = 0;
  
  //first get the simple changes in information
  for( j in 2:J-1){
    //get the difference in position
    dx[j] = x[j-1] - ix[j-1];
    dy[j] = y[j-1] - iy[j-1];
    dz[j] = z[j-1] - iz[j-1];
    if (is_nan(dx[j-1])){
      print("This is the time it failed ", j);
      print("x[j-1] is", x[j-1]);
      print("ix[j-1] is", ix[j-1]);
      print("dx[j-1] is", dx[j-1]);
    }
    
    //get the difference in velocity
    dvx[j] = vx[j-1] - ivx[j-1];
    dvy[j] = vy[j-1] - ivy[j-1];
    dvz[j] = vz[j-1] - ivz[j-1];
    
    
    //get the model for stationkeeping
    driftDistance[j] = sqrt(square(dx[j-1]) + square(dy[j-1]) + square(dz[j-1]));
    if (driftDistance[j] > md){
      status[j] = 1;
    }
    else{
      status[j] = 0;
    }
  }
  
  //Second get the deterministic values for the velocity
  for( j in 2:J-1){
    if (status[j-1] == 0){ // the satellite is non-stationkeeping
      ax[j] = 0;
      ay[j] = 0;
      az[j] = 0;
    }
    else{ //if the satellite is station keeping
      if(is_nan(dx[j-1])){
        print("dx[j-1 is nan: ", j);
      }
      if(is_nan(dvx[j-1])){
        print("dvx[j-1 is nan: ", j);
      }
      ax[j] = -kp *dx[j-1] - kd*dvx[j-1];
      ay[j] = -kp *dy[j-1] - kd*dvy[j-1];
      az[j] = -kp *dz[j-1] - kd*dvz[j-1];

    }
  }
}
model {
  //first model the Velocity
  for( j in 2:J-1){
    real mu = vx[j-1] + ax[j-1] * dt;
    if (is_nan(mu)) {
      print("mu is NaN at j = ", j);
      print("vx[j-1] = ", vx[j-1]);
      print("ax[j-1] = ", ax[j-1]);
      print("dt = ", dt);
    }
    vx[j] ~ normal(mu, sigmaV);
  }
  
  //next model the position
    for( j in 2:J-1){
      x[j] ~ normal(x[j-1] + ax[j-1]*dt, sigmaY);
      y[j] ~ normal(y[j-1] + ay[j-1]*dt, sigmaY);
      z[j] ~ normal(z[j-1] + az[j-1]*dt, sigmaY);
  }

}
