// saved as pdController.stan
functions{

//All of these functons are for the external forces
  //gets the magnitude of a distance, just euclidian distance
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
    real MU =  3.986004418e14;    // m^3/s^2
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
    real EARTH_RADIUS = 6378.137e3;  // m
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
    real a_mag = P0 * square(AU_m / d) * (C_R * A / m_kg);
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
    //gets the affect of the sun's position on the moon
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
    vector[3] j2 = j2Affect( r[1], r[2], r[3]); //rep_vector(0, 3);
    vector[3] solarRad = solarRadiation(r[1], r[2], r[3], sunPos[1], sunPos[2],  sunPos[3]); //solarRad = rep_vector(0, 3);
    vector[3] moon = moonAffect(r[1], r[2], r[3], moonPos[1], moonPos[2], moonPos[3]); // rep_vector(0, 3);
    vector[3] sun = sunAffect(r[1], r[2], r[3], sunPos[1], sunPos[2],  sunPos[3]); //rep_vector(0, 3);
    
    vector[3] a;
    
    a[1] = cg[1] + j2[1] + solarRad[1] + moon[1] + sun[1];
    a[2] = cg[2] + j2[2] + solarRad[2] + moon[2] + sun[2];
    a[3] = cg[3] + j2[3] + solarRad[3] + moon[3] + sun[3];

    return a;
  }
  

// all of these functions are the different control policies
// so far I have only made the pd controller
    vector pdControlPolicy(vector r, vector r_ref,
                            vector v, vector v_ref,
                            real kp, real kd, real md) {
    vector[3] dr = r - r_ref;   // position error
    vector[3] dv = v - v_ref;   // velocity error
    vector[3] a;

    real dr_norm = sqrt(dot_self(dr));  // ‖dr‖
    
    a = -kp * dr - kd * dv;

 
    //print("Inside the pd controller");
    return a;
  }
}

data {
  int<lower=0> J; // the number of time steps you want to investigate
  int<lower=0> Q;// this is double j for the half and full steps
  real<lower=0> dt;// the change in the time steps
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
}

parameters {
  real<lower=1, upper = 4> logmd;      // maximum distance allowed off orbit
  real<lower= -9, upper = -5> logkp;    // reacts to how far off the satellite is from desired orbit, position control
  real<lower= -7, upper = -3> logkd;     //dampens motion to avoid overshoot or oscillation, velocity control
  real<lower=-9, upper =-3> logsigmaV; // standard Deviation in velocity to account for noise

}



// all of the equations that go into determining our prior and liklihood
transformed parameters {
  //Initialize variables
  real eps2 = 1e-8; //1e-8 got pretty quick convergence before
  real lambda =2;
  real kp = pow(10,logkp);
  real kd = pow(10,logkd);
  real sigmaV = pow(10,logsigmaV)*sqrt(dt);//; 1E-8
  real md = pow(10,logmd);
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
  array[J-1]vector[3] aExternal;
  array[J-1]vector[3] aControlHalfStep;
  vector[3] moonPos;
  vector[3] sunPos;
  array[J-1] real pi; 
  vector[3] dr;
  vector[3] dv;
  
  //loop through every time step in the data
  for (k in 1:(J-1)) {
    //print("k is: ", k);
    // --- Get current states and reference trajectories ---
    // Ideal trajectory at time step k
    ref_trajectory[1] = ix[2*k -1];   // this will get you the full step
    ref_trajectory[2] = iy[2*k -1]; // when k = 1, it returns 1
    ref_trajectory[3] = iz[2*k-1]; //when k =2, this returns 3, etc.
    ref_trajectory[4] = ivx[2*k-1];
    ref_trajectory[5] = ivy[2*k-1];
    ref_trajectory[6] = ivz[2*k-1];

    // Ideal trajectory at half step k
    ref_trajectory_half[1] = ix[2*k]; // this will get you the half time step
    ref_trajectory_half[2] = iy[2*k];// when k = 1, this returns 2,
    ref_trajectory_half[3] = iz[2*k];//when k =2, this returns 4, etc.
    ref_trajectory_half[4] = ivx[2*k];
    ref_trajectory_half[5] = ivy[2*k];
    ref_trajectory_half[6] = ivz[2*k];

    // Current simulated state, gets current position and velocity
    s_k[1] = x[k];
    s_k[2] = y[k];
    s_k[3] = z[k];
    s_k[4] = vx[k];
    s_k[5] = vy[k];
    s_k[6] = vz[k];

    // --- Unpack vectors ---
    //just puts these into a vector of size 3
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
    sunPos  = [sunX[k], sunY[k], sunZ[k]]'; //WNC: Should sun/moon be @ half step? Error will be negligle either way
    aExternal[k] = totalForces(r_half_step, moonPos, sunPos);

    // 4. Get control acceleration at the half step
    aControlHalfStep[k] = pdControlPolicy(r_half_step, r_ref_k_half, v_k, v_ref_k_half, kp, kd, md);

    
    dr = r_ref_k_half - r_half_step;   // position error
    dv = v_k- v_ref_k_half;   // velocity error

    real dr_norm = sqrt(dot_self(dr));  // ‖dr‖
    
    real raw_pi = inv_logit(lambda * (dr_norm - md));
    pi[k] = fmin(fmax(raw_pi, eps2), 1 - eps2);
  }
}


model {
  //Truncated normal priors over the range 
  logmd ~ normal(2,1);
  logkp ~ normal(-8.5,0.5);    // reacts to how far off the satellite is from desired orbit, position control
  logkd ~ normal(-7,0.5);
  
  //Priors over the initial speed... assumed they were pretty good
  vx[1] ~ normal(0, 1);
  vy[1] ~ normal(3074, 1);
  vz[1] ~ normal(0, 1);
  for( j in 2:(J-1)){
    
    //Turned the PD controller into a mixture model... 
    //Mean if not PD control applied
    real mux1 = vx[j-1] + aExternal[j-1][1] * dt;
    real muy1 = vy[j-1] + aExternal[j-1][2] * dt;
    real muz1 = vz[j-1] + aExternal[j-1][3] * dt;
    
    //Mean if PD Control applied
    real mux2 = mux1 + aControlHalfStep[j-1][1] * dt;
    real muy2 = muy1 + aControlHalfStep[j-1][2] * dt;
    real muz2 = muz1 + aControlHalfStep[j-1][3] * dt;
    
    // Use pi[j-1] is the probability of the control being on... below gives mixture densities for each velocity
    target += log_mix(pi[j-1],
      normal_lpdf(vx[j] | mux2, sigmaV),
      normal_lpdf(vx[j] | mux1, sigmaV)
    );

    target += log_mix(pi[j-1],
      normal_lpdf(vy[j] | muy2, sigmaV),
      normal_lpdf(vy[j] | muy1, sigmaV)
    );

    target += log_mix(pi[j-1],
      normal_lpdf(vz[j] | muz2, sigmaV),
      normal_lpdf(vz[j] | muz1, sigmaV)
    );
    

  }
}



generated quantities {
  //Below block is useful for debugging
  //array[J-1] vector[3] halfAccel_out;
  
  //for (j in 1:(J-1)) {
    //halfAccel_out[j] = halfAccel[j];
  //}
}

