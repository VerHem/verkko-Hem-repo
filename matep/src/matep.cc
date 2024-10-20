/* ------------------------------------------------------------------------------------------
 *
 * Copyright (C) 2023-present by Kuang. Zhang
 *
 * This library is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU Lesser General Public License as published by the Free Software Foundation; 
 * either version 2.1 of the License, or (at your option) any later version.
 *
 * Permission is hereby granted to use or copy this program under the
 * terms of the GNU LGPL, provided that the Copyright, this License 
 * and the Availability of the original version is retained on all copies.
 * User documentation of any code that uses this code or any modified
 * version of this code must cite the Copyright, this License, the
 * Availability note, and "Used by permission." 

 * Permission to modify the code and to distribute modified code is granted, 
 * provided the Copyright, this License, and the Availability note are retained,
 * and a notice that the code was modified is included.

 * The third party libraries which are used by this library are deal.II, Triinos and few others.
 * All components involved third party supports obey their Copyrights, Licence and permissions. 
 *  
 * ------------------------------------------------------------------------------------------
 *
 * author: Quang. Zhang (timohyva@github), 
 * Helsinki Institute of Physics, University of Helsinki;
 * 27. Kesäkuu. 2023.
 *
 */


#include <iostream>
#include <cstddef>
#include <cmath>
#include <vector>

#include "matep.h"


namespace FemGL_mpi
{
  
/* -------------------------------------------------------------------------
 * switch function for turning on/off SCC JWS2019
 * -------------------------------------------------------------------------
 */

void
Matep::with_SCC(const bool &key){
  scc_on = key;
}
  
//*********************************************************************
//***     member functions, interfaces of dimensional qualities     ***
//*********************************************************************

real_t
Matep::Tcp(real_t p){
  real_t Tc = lininterp(Tc_arr, p)*(1.0e-3);
  return Tc;
}

real_t
Matep::Tcp_mK(real_t p) {
  return lininterp(Tc_arr, p);
}


real_t
Matep::mEffp(real_t p){  
  real_t mEff = lininterp(Ms_arr, p)*m3;;
  return mEff;
}

real_t
Matep::vFp(real_t p){
  // unit m.s^-1
  real_t vF = lininterp(VF_arr, p);
  return vF;
}

real_t
Matep::xi0p(real_t p){
  real_t xi0 = lininterp(XI0_arr, p);
  return xi0;
}  

real_t
Matep::xi0GLp(real_t p){
  return xi0p(p) * std::sqrt((7.*zeta3)/20.);
}  

real_t
Matep::xiGLpT(real_t p, real_t T){
  return xi0GLp(p)/std::sqrt(1.-T/Tcp_mK(p));
}  
  
double
Matep::N0p(real_t p){
  /*
   * the maginitude of N0p is about 10^(50), it must be double type 
   */
  double N0 = (std::pow(mEffp(p),2)*vFp(p))/((2.0f*pi*pi)*(hbar*hbar*hbar));
  return N0;
}


//**********************************************************************
//***    member functions, interfaces of dimensionless coefficients  ***
//**********************************************************************

real_t
Matep::alpha_td(real_t t){ return 1.f*(t-1); }  


real_t
Matep::beta1_td(real_t p, real_t t){
  real_t beta1;
  
  if (scc_on == true)
    beta1 = c_betai*(-1.0f + (t)*lininterp(c1_arr, p));
  else if (scc_on == false)
    beta1 = c_betai*(-1.0f);    

  return beta1;
}  


real_t
Matep::beta2_td(real_t p, real_t t){
  real_t beta2;
  
  if (scc_on == true)
    beta2 = c_betai*(2.0f + (t)*lininterp(c2_arr, p));
  else if (scc_on == false)
    beta2 = c_betai*(2.0f);    

  return beta2;
}  


real_t
Matep::beta3_td(real_t p, real_t t){
  real_t beta3;
  
  if (scc_on == true)
    beta3 = c_betai*(2.0f + (t)*lininterp(c3_arr, p));
  else if (scc_on == false)
    beta3 = c_betai*(2.0f);    

  return beta3;
}  


real_t
Matep::beta4_td(real_t p, real_t t){
  real_t beta4;
  
  if (scc_on == true)
    beta4 = c_betai*(2.0f + (t)*lininterp(c4_arr, p));
  else if (scc_on == false)
    beta4 = c_betai*(2.0f);
  
  return beta4;
}


real_t
Matep::beta5_td(real_t p, real_t t){
  real_t beta5;
  
  if (scc_on == true)
    beta5 = c_betai*(-2.0f + (t)*lininterp(c5_arr, p));
  else if (scc_on == false)
    beta5 = c_betai*(-2.0f);    

  return beta5;
}  


//**********************************************************************
//***                 beta_A, beta_B and Gaps                        ***
//**********************************************************************

real_t
Matep::beta_A_td(real_t p, real_t t){
  return beta2_td(p, t) + beta4_td(p, t) + beta5_td(p, t);
}

real_t
Matep::beta_B_td(real_t p, real_t t){
  return beta1_td(p, t) + beta2_td(p, t) + (1.f/3.f)*(beta3_td(p, t) + beta4_td(p, t) + beta5_td(p, t));
}

// A-phase gap energy, in unit of Kb * Tc
real_t
Matep::gap_A_td(real_t p, real_t t){

  if (t <= 1.0)
    {
      real_t gap2 =-alpha_td(t)/(2.f*beta_A_td(p, t)); // (kb Tc)^2

      return std::sqrt(gap2);    
    }
  else //if (T > Tcp_mK(p))
    return 0.;

}

// B-phase gap energy, in unit of Kb * Tc
real_t
Matep::gap_B_td(real_t p, real_t t){

  if (t <= 1.0)
    {
      real_t gap2 =-alpha_td(t)/(2.f*beta_B_td(p, t)); // (kb Tc)^2

      return std::sqrt(gap2);
    }
  else //if (T > Tcp_mK(p))
    return 0.;
}

// A general gap function with message of equlibrium phase
real_t
Matep::gap_td(real_t p, real_t t){

  if (f_A_td(p, t) > f_B_td(p, t)){
    std::cout << " \nnow p, T are: " << p << ", " << t
              << ", equlibrum bulk phase is B phase. "
              << std::endl;
    return gap_A_td(p, t);
    
  } else if (f_A_td(p, t) < f_B_td(p, t)) { 
    
    std::cout << " \nnow p, T are: " << p << ", " << t
              << ", equlibrum bulk phase is A phase. "
              << std::endl;
    return gap_B_td(p, t);   

  } else {

    if (
	// (f_A_td(p, T) == f_B_td(p, T))
	// && (T < Tcp_mK(p))
	t < 1.0
       ){

       std::cout << " \nnow p, t are: " << p << ", " << t
                 << ", and A and B degenerate, return as -1. "
                 << std::endl;
       return -1.f;

    } else  //(
        	// (f_A_td(p, T) == f_B_td(p, T))
	        // && (T > Tcp_mK(p))
            // )
	  {

            std::cout << " \nnow p, t are: " << p << ", " << t
	            << ", system is in normal phase. "
	            << std::endl;
	    return 0.f;

          }

  }
}

// tAB_RWS 2019
real_t
Matep::tAB_RWS(real_t p){
  real_t t = 1.f/(3.f*lininterp(c1_arr, p)
           + lininterp(c3_arr, p)
           - 2.f*lininterp(c4_arr, p)
           - 2.f*lininterp(c5_arr, p));
  return t;
}


// A-Phase free energy density in unit of (1/3)(Kb Tc)^2 N(0)
real_t
Matep::f_A_td(real_t p, real_t t)
{
  if (t <= 1.0)
    {
     return (-1.f/4.f)*(std::pow(alpha_td(t),2.f))/beta_A_td(p, t);    
    }
  else //if (T > Tcp_mK(p))
    return 0.;
    
}

// B-Phase free energy density in unit of (1/3)(Kb Tc)^2 N(0)
real_t
Matep::f_B_td(real_t p, real_t t)
{
  if (t <= 1.0)
    {
     return (-1.f/4.f)*(std::pow(alpha_td(t),2.f))/beta_B_td(p, t);
    }
  else //if (T > Tcp_mK(p))
    return 0.;

}

} // FemGL_mpi namespace ends at here

