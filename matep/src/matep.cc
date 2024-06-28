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
  
//********************************************************************
//***          static physical constants of he3 members            ***
//********************************************************************
const real_t Matep::kb      = 1.380649*(1.0e-23)*J*1.0f;
const real_t Matep::u       = 1.66053906660f*(1.0e-27)*kg;
const real_t Matep::m3      = 3.016293f*u;
const real_t Matep::nm      = (1.0e-9)*m;
const real_t Matep::hbar    = 1.054571817f*(1.0e-34)*J*s;
const real_t Matep::zeta3   = 1.2020569031595942f;
const real_t Matep::c_betai = (7.0f*zeta3)/(80.0f*pi*pi); // 0.010657


// *******************************************************************
// >> data sheets of  strong coupling corrected material parameters <<
//'*******************************************************************

const real_t Matep::c1_arr[18] = {-0.0098, -0.0127, -0.0155, -0.0181, -0.0207, -0.0231, -0.0254, -0.0275, -0.0295, -0.0314, -0.0330, -0.0345, -0.0358, -0.0370, -0.0381, -0.0391, -0.0402, -0.0413};
const real_t Matep::c2_arr[18] = {-0.0419, -0.0490, -0.0562, -0.0636, -0.0711, -0.0786, -0.0861, -0.0936, -0.1011, -0.1086, -0.1160, -0.1233, -0.1306, -0.1378, -0.1448, -0.1517, -0.1583, -0.1645};
const real_t Matep::c3_arr[18] = {-0.0132, -0.0161, -0.0184, -0.0202, -0.0216, -0.0226, -0.0233, -0.0239, -0.0243, -0.0247, -0.0249, -0.0252, -0.0255, -0.0258, -0.0262, -0.0265, -0.0267, -0.0268};
const real_t Matep::c4_arr[18] = {-0.0047, -0.0276, -0.0514, -0.0760, -0.1010, -0.1260, -0.1508, -0.1751, -0.1985, -0.2208, -0.2419, -0.2614, -0.2795, -0.2961, -0.3114, -0.3255, -0.3388, -0.3518};
const real_t Matep::c5_arr[18] = {-0.0899, -0.1277, -0.1602, -0.1880, -0.2119, -0.2324, -0.2503, -0.2660, -0.2801, -0.2930, -0.3051, -0.3167, -0.3280, -0.3392, -0.3502, -0.3611, -0.3717, -0.3815};

const real_t Matep::Tc_arr[18] = {0.929, 1.181, 1.388, 1.560, 1.705, 1.828, 1.934, 2.026, 2.106, 2.177, 2.239, 2.293, 2.339, 2.378, 2.411, 2.438, 2.463, 2.486}; // mK
const real_t Matep::Ms_arr[18] = {2.80, 3.05, 3.27, 3.48, 3.68, 3.86, 4.03, 4.20, 4.37, 4.53, 4.70, 4.86, 5.02, 5.18, 5.34, 5.50, 5.66, 5.82}; // in unit of helium-3 atom
const real_t Matep::VF_arr[18] = {59.03, 55.41, 52.36, 49.77, 47.56, 45.66, 44.00, 42.51, 41.17, 39.92, 38.74, 37.61, 36.53, 35.50, 34.53, 33.63, 32.85, 32.23}; // fermi velosity, m.s^-1
const real_t Matep::XI0_arr[18] = {77.21, 57.04, 45.85, 38.77, 33.91, 30.37, 27.66, 25.51, 23.76, 22.29, 21.03, 19.94, 18.99, 18.15, 17.41, 16.77, 16.22, 15.76};


/*********************************************************************
 ***     member functions, interfaces of dimensional qualities     ***
 *********************************************************************
 */

// Matep::Matep(const bool &key)
//   : scc_on(key)
// {}  


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
  real_t xi0 = lininterp(XI0_arr, p)*nm;
  return xi0;
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

//**********************************************************************
//***              public member: Levi-Civita symbol                 ***
//**********************************************************************

real_t
Matep::epsilon(int al, int be, int ga)
{
  if (
      (al == 0 && be == 1 && ga == 2)   // 123
      ||(al == 1 && be == 2 && ga == 0) // 231
      ||(al == 2 && be == 0 && ga == 1) // 312
     )
    {return 1.0;}
  else if
     (
      (al == 2 && be == 1 && ga == 0)   // 321
      ||(al == 0 && be == 2 && ga == 1) // 132
      ||(al == 1 && be == 0 && ga == 2) // 213
     )
    {return -1.0;}
  else if ((al == be) || (al == ga) || (be == ga))
    {return 0.0;}
  else
    {return 0.0;}
}  

//**********************************************************************
//***       private method :  linear intepolation function           ***
//**********************************************************************

real_t
Matep::lininterp(const real_t *cX_arr, real_t p){
  float pk, fp;
  size_t k;

  if ((p >= 0.0) && (p < 2.0)) { pk = 0.0f; k = 0; }

  if ((p >= 2.0) && (p < 4.0)) { pk = 2.0f; k = 1; }
 
  if ((p >= 4.0) && (p < 6.0)) { pk = 4.0f; k = 2; }

  if ((p >= 6.0) && (p < 8.0)) { pk = 6.0f; k = 3; }

  if ((p >= 8.0) && (p < 10.0)) { pk = 8.0f; k = 4; }

  if ((p >= 10.0) && (p < 12.0)) { pk = 10.0f; k = 5; }

  if ((p >= 12.0) && (p < 14.0)) { pk = 12.0f; k = 6; }

  if ((p >= 14.0) && (p < 16.0)) { pk = 14.0f; k = 7; }

  if ((p >= 16.0) && (p < 18.0)) { pk = 16.0f; k = 8; }

  if ((p >= 18.0) && (p < 20.0)) { pk = 18.0f; k = 9; }

  if ((p >= 20.0) && (p < 22.0)) { pk = 20.0f; k = 10; }

  if ((p >= 22.0) && (p < 24.0)) { pk = 22.0f; k = 11; }

  if ((p >= 24.0) && (p < 26.0)) { pk = 24.0f; k = 12; }

  if ((p >= 26.0) && (p < 28.0)) { pk = 26.0f; k = 13; }

  if ((p >= 28.0) && (p < 30.0)) { pk = 28.0f; k = 14; }

  if ((p >= 30.0) && (p < 32.0)) { pk = 30.0f; k = 15; }

  if ((p >= 32.0) && (p < 34.0)) { pk = 32.0f; k = 16; }

  fp = ((cX_arr[k+1]-cX_arr[k])/2.0)*(p-pk)+cX_arr[k];
  return fp; 
}


} // FemGL_mpi namespace ends at here

