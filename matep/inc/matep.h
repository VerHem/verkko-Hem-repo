/*
 * 
 * This is the *.hpp header file of the Strong Coupling Correction Object (SCCO) 
 * or Material Parameters Object (Matep).
 * 
 * Member functions are declared at here, and then defined at Matep.cpp.
 *
 * author: Quang. Zhang (timohyva@github)
 *
 */ 

#ifndef MATEP_H
#define MATEP_H

#include <iostream>
#include <cstddef>
#include <cmath>
#include <vector>

namespace FemGL_mpi
{
 using real_t = double;

 class Matep {
 public:
        Matep() = default;                                                  // default constructor

  // ************************************************************************** //
  // >>>>>>>>>>>        interfaces of dimensional qualities        <<<<<<<<<<<< //
  // ************************************************************************** //
	
        real_t Tcp(real_t p);                  // in unit of Kelvin
        real_t Tcp_mK(real_t p);               // in unit of mK
  
        real_t mEffp(real_t p);                // quisiparticle effective mass 
        real_t vFp(real_t p);                  // Fermi velocity
        real_t xi0p(real_t p);                 // zero Temperature coherent length
        double N0p(real_t p);                  // deisty of state on Fermi surface
  

  // ************************************************************************* //
  // >>>>  interfaces of dimensionless coeficients; SC-correction parts: <<<<< //
  // ************************************************************************* //
  
        real_t alpha_td(real_t t);
        real_t beta1_td(real_t p, real_t t);
        real_t beta2_td(real_t p, real_t t);
        real_t beta3_td(real_t p, real_t t);
        real_t beta4_td(real_t p, real_t t);
        real_t beta5_td(real_t p, real_t t);

  // >>>>>>>>>    interfaces for beta_A, beta_B, gaps and tAB_RWS    <<<<<<<<< //
  
        real_t beta_A_td(real_t p, real_t t);
        real_t beta_B_td(real_t p, real_t t);
        real_t gap_A_td(real_t p, real_t t);
        real_t gap_B_td(real_t p, real_t t);
        real_t gap_td(real_t p, real_t t);      // gap for given p,T, and showing message

        real_t tAB_RWS(real_t p);               

  // >>>>>>> interfaces for f_{A}, f_{B}, in unit of (1/3)(Kb Tc)^2 N(0) <<<<< //
  
        real_t f_A_td(real_t p, real_t t);
        real_t f_B_td(real_t p, real_t t);
        

  // >>>>>>>>>>>>>>> menmber funcition Levi_Civita symbol <<<<<<<<<<<<<<<<<<<< //

        real_t epsilon(int al, /*alpha*/
         	       int be, /*beta*/
		       int ga /*gamma*/);
  
  private:
        // SI unit
        static constexpr real_t Kelvin =1.0f, J = 1.0f, s = 1.0f, m = 1.0f, kg = 1.0f
                         	,pi = 3.14159265358979323846264338328f, p_pcp = 21.22f;
        
        // physical constants for he3
        static const real_t u, m3, nm, hbar, kb
                            ,zeta3, c_betai;
        
        // SC-data sheets arries, All associate to SCCO class
        static const real_t c1_arr[18]; 
        static const real_t c2_arr[18];
        static const real_t c3_arr[18];
        static const real_t c4_arr[18];
        static const real_t c5_arr[18];

        // ***************************************************
        static const real_t Tc_arr[18];
        static const real_t Ms_arr[18];
        static const real_t VF_arr[18];
        static const real_t XI0_arr[18];

        // linear interpolation function:
        real_t lininterp(const real_t *cX_arr, real_t p);

 }; // matep class declaration ends at here

} // FemGL_mpi namespace ends at here  

#endif
