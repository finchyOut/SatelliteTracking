// saved as pdControllerStan.stan
functions{
//All of these functions are to model the reference trajectory
  //this gets the starting position for a satellite based on longitude in geo
  /*
  vector rot_z_deg(array[3] real v, real deg) {
    real c = cos(th);
    real s = sin(th);
    vector[3] out;
    out[1] =  c * v[1] - s * v[2];
    out[2] =  s * v[1] + c * v[2];
    out[3] =  v[3];
    return out;
  }
  //generate reference trajectory
  vector generateRefTrajectory(real timeSpan, real dt, real longitude){
    //convert to radians
    real th = longitude * pi() / 180; 
    real GEO_RADIUS = 6378.137e3 + 35786e3; //in meters
    vector[6] sNaut = [GEO_RADIUS, 0, 0, 0, GEO_SPEED, 0] ;
    vector[3] rNaut= [sNaut[1], sNaut[2], sNaut[3]];
    vector[3] vNaut= [sNaut[4], sNaut[5], sNaut[6]];
    //shift the start to be at the correct degree
    vector[6] finalS = [rot_z_deg(rNaut, th), rot_z_deg(vNaut, th)];
    
    //update dt to be half of itself
    dt = dt/2;
    real k = 0
    //create the log for the trajectory
    real size = timeSpan/dt;
    vector[size] log;
    //get the radius and omega
    real r = normMag(rNaut[1], rNaut[2], rNaut[3]);
    real w = sqrt(3.986004418e14 / (r*r*r));
    for t in timeSpan{
      real ang = w *t + th;
      real x = r * cos(ang);
      real y = r * sin(ang);
      real vx = -r * w*sin(ang);
      real vy = r*w*cos(ang);
      log[k] = [x, y, 0, vx, vy, 0];
      t = t + dt;
      k = k + 1;
    }
    return log;
    
  }
  
*/
//All of these functons are for the external forces
  //gets the magnitude of a distance
  real normMag(real x, real y, real z){
    //print("Inside the Magnitude ");
    real magnitude = sqrt(square(x) + square(y) +square(z));
    return magnitude;
  }
  
  //this has been updated for new version of SLV  
  //calculate the force attributed to central gravity acceleration
  vector centralGravity(real x, real y, real z){
    //print("Inside the central gravity ");
    real mag = normMag(x, y, z);
    //Earth's radius in kilometers (used to set orbital altitude)
    real MU =  3.986004418e14;    # m^3/s^2
    real totConst =  -MU / (mag * mag * mag);
    
    vector[3] a;
    
    //get the central gravity affact in each direction
    a[1] = x *totConst;
    a[2] = y *totConst;
    a[3] = z *totConst;
    
    return a;
  }
  
  //this has been checked to align with meters
  //calculate the j2 effect
  vector j2Affect(real x, real y, real z){
    //print("Inside the j2 affect ");
    //get known needed constants
    real J2 = 1.08263e-3;
    real EARTH_RADIUS = 6378.137e3;  # m
    real R = EARTH_RADIUS;
    real MU =  3.986004418e14;
    //get the magnitude of the actions being taken
    real r = normMag(x, y, z);
    // 1.5 * j2 * mu * R^2 / r^5
    real factor =  1.5 * J2 * MU * square(R)/ (square(r)*square(r)*r);
    // zx ratio
    real zx_ratio_sq = 5 * square(z) / square(r);
    
    vector[3] a;
    //get the central gravity affact in each direction
    a[1] = x * factor * (zx_ratio_sq -1);
    a[2] = y * factor * (zx_ratio_sq -1);
    a[3] = z * factor * (zx_ratio_sq -3);
    
    return a;
  }
  
  //this has been checked for new SLV
  //calculate the effects of solar radiation
 vector solarRadiation(real x, real y, real z,
                           real sunX, real sunY, real sunZ) {
    real P0   = 4.56e-6;        // N/m^2 at 1 AU
    real AU_m = 149597870.7e3;  // m
    real C_R  = 1.5;
    real A    = 20.0;           // m^2
    real m_kg = 1000.0;         // kg

    real dx = sunX - x ;
    real dy = sunY - y ;
    real dz = sunZ - z ;
    real d = normMag(dx, dy, dz);
    if (d <= 0) return rep_vector(0, 3);
    // a = (P0 * (AU/d)^2 * C_R * A / m) * unit_vector(Sun->sat)
    real a_mag = P0 * (AU_m*AU_m / d*d) * (C_R * A / m_kg);
    vector[3] a;
    a[1] = a_mag * (-dx / d);
    a[2] = a_mag * (-dy / d);
    a[3] = a_mag * (-dz / d);
    //print("solar radiation ");
    return a;
  }
  
  //this has been correctly compared to the slv file and updated
  //calculate the moon's positional effect on the sat
  vector moonAffect(real x, real y, real z, real moonx, real moony, real moonz){
    real moonMass = 4.9048695e12 ; //this is in meters
    //get difference in location
    real deltaX = moonx - x;
    real deltaY = moony - y;
    real deltaZ = moonz -z;
    real d = normMag(deltaX, deltaY, deltaZ);
    real rb = normMag(moonx, moony, moonz);
    
    real dThree = (d*d*d);
    real rbThree = (rb*rb*rb);
    
    vector[3] a;
    // a = moonMass *((dx/ d^3) - moonX / rb^3)
    //get x accel affect
    a[1] = moonMass * ((deltaX/dThree) - (moonx/ rbThree));
    //get y accel affect
    a[2] = moonMass * ((deltaY/ dThree) - (moony/ rbThree));
    //get z accel affect
    a[3] = moonMass * ((deltaZ/ dThree) - (moonz/ rbThree));
    //print("Inside the moon effect loop ");
    return a;
    
  }
  
    //this has been correctly updated for the SLV file
    vector sunAffect(real x, real y, real z, real sunx, real suny, real sunz){
    real sunMass = 1.32712440018e20;
    //get difference in location
    real deltaX = sunx - x;
    real deltaY = suny - y;
    real deltaZ = sunz -z;
    real d = normMag(deltaX, deltaY, deltaZ);
    real rb = normMag(sunx, suny, sunz);
    
    real dThree = (d*d*d);
    real rbThree = (rb*rb*rb);
    
    vector[3] a;
    
    //get x accel affect
    a[1] = sunMass * ((deltaX/ dThree) - (sunx/ rbThree));
    //get y accel affect
    a[2] = sunMass * ((deltaY/ dThree) - (suny/ rbThree));
    //get z accel affect
    a[3] = sunMass * ((deltaZ/ dThree) - (sunz/ rbThree));
    //print("Inside the sun affect");
    return a;
    
  }
  
  //combines all of the external forces listed above into one function
  vector totalForces(vector r, vector moonPos, vector sunPos){
    vector[3] cg = centralGravity( r[1], r[2],  r[3]);
    vector[3] j2 = j2Affect( r[1], r[2], r[3]);
    vector[3] solarRad = solarRadiation(r[1], r[2], r[3], sunPos[1], sunPos[2],  sunPos[3]);
    vector[3] moon = moonAffect(r[1], r[2], r[3], moonPos[1], moonPos[2], moonPos[3]);
    vector[3] sun = sunAffect(r[1], r[2], r[3], sunPos[1], sunPos[2],  sunPos[3]);
    
    vector[3] a;
    
    a[1] = cg[1] + j2[1] + solarRad[1] + moon[1] + sun[1];
    a[2] = cg[2] + j2[2] + solarRad[2] + moon[2] + sun[2];
    a[3] = cg[3] + j2[3] + solarRad[3] + moon[3] + sun[3];
    //print("Inside the total force");
    return a;
  }
  
// all of these functions are the different control policies
    vector pdControlPolicy(vector r, vector r_ref,
                            vector v, vector v_ref,
                            real kp, real kd, real md) {
    vector[3] dr = r - r_ref;   // position error
    vector[3] dv = v - v_ref;   // velocity error
    vector[3] a;

    real dr_norm = sqrt(dot_self(dr));  // ‖dr‖

    if (dr_norm > md) {
      a = -kp * dr - kd * dv;
    } else {
      a = rep_vector(0, 3);
    }
    //print("Inside the pd controller");
    return a;
  }
}

data {
  int<lower=0> J; // the number of time steps you want to investigate
  int<lower=0> Q;
  int<lower=0> dt;
  array[J] real x; //the real satellite position in x
  array[J] real y; //the real satellite position in y
  array[J] real z; //the real satellite position in z
  array[J] real vx; //the real satellite velocity in x
  array[J] real vy; //the real satellite velocity in y
  array[J] real vz; //the real satellite velocity in z
  array[Q] real ix; // ideal position in x
  array[Q] real iy; // ideal position in y
  array[Q] real iz; // ideal position in z
  array[Q] real ivx; // ideal velocity in x
  array[Q] real ivy; // ideal velocity in y
  array[Q] real ivz; // ideal velocity in 
  array[J] real moonX; // moon x position 
  array[J] real moonY; // moon y position 
  array[J] real moonZ; // moon z position 
  array[J] real sunX; // sun x position 
  array[J] real sunY; // sun y position 
  array[J] real sunZ; // sun z position 
  //int<lower=0, upper=1> status[J]; // whether or not the satellite is station keeping
}

parameters {
  real<lower=4000, upper = 7000> md;      // maximum distance allowed off orbit
  real<lower= -.006, upper = .005> kp;    // reacts to how far off the satellite is from desired orbit, position control
  real<lower=-.006, upper = .005> kd;     //dampens motion to avoid overshoot or oscillation, velocity control
  real<lower=1e-12> sigmaV; // standard Deviation in velocity to account for noise
  real<lower=45, upper = 46> longitude;
}

// all of the equations that go into determining our prior and liklihood
transformed parameters {
  vector[6] ref_trajectory;
  vector[6] ref_trajectory_half;
  vector[6] s_k;
  vector[3] r_k;
  vector[3] v_k;
  vector[3] r_ref_k;
  vector[3] v_ref_k;
  vector[3] r_half_step;
  vector[3] r_ref_k_half;
  vector[3] v_ref_k_half;
  vector[3] aExternal;
  vector[3] aControlHalfStep;
  vector[3] moonPos;
  vector[3] sunPos;
  array[J] vector[3] halfAccel; // Should be size J for logging

  // Initialize the first element of halfAccel as it's not set in the loop
  halfAccel[1] = rep_vector(0, 3);

  for (k in 2:(J-1)) {
    // --- Get current states and reference trajectories ---
    // Ideal trajectory at time step k
    ref_trajectory[1] = ix[2*k - 2];   // Stan arrays are 1-based, Python is 0-based
    ref_trajectory[2] = iy[2*k - 2];
    ref_trajectory[3] = iz[2*k - 2];
    ref_trajectory[4] = ivx[2*k - 2];
    ref_trajectory[5] = ivy[2*k - 2];
    ref_trajectory[6] = ivz[2*k - 2];

    // Ideal trajectory at half step k
    ref_trajectory_half[1] = ix[2*k - 1];
    ref_trajectory_half[2] = iy[2*k - 1];
    ref_trajectory_half[3] = iz[2*k - 1];
    ref_trajectory_half[4] = ivx[2*k - 1];
    ref_trajectory_half[5] = ivy[2*k - 1];
    ref_trajectory_half[6] = ivz[2*k - 1];

    // Current simulated state
    s_k[1] = x[k];
    s_k[2] = y[k];
    s_k[3] = z[k];
    s_k[4] = vx[k];
    s_k[5] = vy[k];
    s_k[6] = vz[k];

    // --- Unpack vectors ---
    r_k = s_k[1:3];
    v_k = s_k[4:6];
    r_ref_k = ref_trajectory[1:3];
    v_ref_k = ref_trajectory[4:6];

    // --- Leapfrog Verlet Step ---
    // 1. Half-step position update (using previous step's velocity)
    r_half_step = r_k + 0.5 * v_k * dt;

    // 2. Get reference state for the half step
    r_ref_k_half = ref_trajectory_half[1:3];
    v_ref_k_half = ref_trajectory_half[4:6];

    // 3. Get external forces at the half step
    moonPos = [moonX[k], moonY[k], moonZ[k]]'; // Transpose to make it a column vector
    sunPos  = [sunX[k], sunY[k], sunZ[k]]';
    aExternal = totalForces(r_half_step, moonPos, sunPos);

    // 4. Get control acceleration at the half step
    // NOTE: This uses the *full step* velocity v_k, matching your Python code
    aControlHalfStep = pdControlPolicy(r_half_step, r_ref_k_half, v_k, v_ref_k_half, kp, kd, md);

    // 5. Calculate total acceleration at the half-step and store it
    halfAccel[k] = aExternal + aControlHalfStep;

    // --- Optional: Print statements for debugging (use sparingly) ---
    // print("k=", k, " aExternal=", aExternal, " aControlHalfStep=", aControlHalfStep);
  }
}

model {
  //print("outside of model loop");
  //first model the Velocity
  for( j in 2:J-1){
    //print("inside of the model loop: ", j);
    real mux = vx[j-1] + halfAccel[j][1] * dt;
    real muy = vy[j-1] + halfAccel[j][2] * dt;
    real muz = vz[j-1] + halfAccel[j][3] * dt;
    if (is_nan(mux)) {
      print("mu is NaN at j = ", j);
      print("vx[j-1] = ", vx[j-1]);
      print("ax[j-1] = ", halfAccel[j-1]);
      print("dt = ", dt);
    }
    vx[j] ~ normal(mux, sigmaV);
    vy[j] ~ normal(muy, sigmaV);
    vz[j] ~ normal(muz, sigmaV);
  }
}
//for this to work need to load in the reference trajectory with the csv with the ideal x, vx etc
//you need to load in the moon and sun location also from the moonAndSunSim
//the actual position you need to use the pd sim
