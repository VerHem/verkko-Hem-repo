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


#include "simplem.h"
#include <iostream>

namespace FemGL_mpi
{

template <typename numbertype>
void SimpleMatrix<numbertype>::mTmult(SimpleMatrix &C, const SimpleMatrix &B, const bool adding) const
{
  // a00*b00 + a01*b01 + a02*b02
  const double c00 = (this->mData[(0u *nCol) + 0u]) * (B.mData[(0u *nCol) + 0u])    
                     + (this->mData[(0u *nCol) + 1u]) * (B.mData[(0u *nCol) + 1u] ) 
                     + (this->mData[(0u *nCol) + 2u]) * (B.mData[(0u *nCol) + 2u]); 

  // a00*b10 + a01*b11 + a02*b12
  const double c01 = (this->mData[(0u *nCol) + 0u]) * (B.mData[(1u *nCol) + 0u])    
                     + (this->mData[(0u *nCol) + 1u]) * (B.mData[(1u *nCol) + 1u] ) 
                     + (this->mData[(0u *nCol) + 2u]) * (B.mData[(1u *nCol) + 2u]); 

  // a00*b20 + a01*b21 + a02*b22
  const double c02 = (this->mData[(0u *nCol) + 0u]) * (B.mData[(2u *nCol) + 0u])    
                     + (this->mData[(0u *nCol) + 1u]) * (B.mData[(2u *nCol) + 1u] ) 
                     + (this->mData[(0u *nCol) + 2u]) * (B.mData[(2u *nCol) + 2u]); 

  // a10*b00 + a11*b01 + a12*b02
  const double c10 = (this->mData[(1u *nCol) + 0u]) * (B.mData[(0u *nCol) + 0u])    
                     + (this->mData[(1u *nCol) + 1u]) * (B.mData[(0u *nCol) + 1u]) 
                     + (this->mData[(1u *nCol) + 2u]) * (B.mData[(0u *nCol) + 2u]); 

  // a10*b10 + a11*b11 + a12*b12
  const double c11 = (this->mData[(1u *nCol) + 0u]) * (B.mData[(1u *nCol) + 0u])    
                     + (this->mData[(1u *nCol) + 1u]) * (B.mData[(1u *nCol) + 1u]) 
                     + (this->mData[(1u *nCol) + 2u]) * (B.mData[(1u *nCol) + 2u]); 

  // a10*b20 + a11*b21 + a12*b22
  const double c12 = (this->mData[(1u *nCol) + 0u]) * (B.mData[(2u *nCol) + 0u])    
                     + (this->mData[(1u *nCol) + 1u]) * (B.mData[(2u *nCol) + 1u]) 
                     + (this->mData[(1u *nCol) + 2u]) * (B.mData[(2u *nCol) + 2u]); 

  // a20*b00 + a21*b01 + a22*b02
  const double c20 = (this->mData[(2u *nCol) + 0u]) * (B.mData[(0u *nCol) + 0u])    
                     + (this->mData[(2u *nCol) + 1u]) * (B.mData[(0u *nCol) + 1u]) 
                     + (this->mData[(2u *nCol) + 2u]) * (B.mData[(0u *nCol) + 2u]); 

  // a20*b10 + a21*b11 + a22*b12
  const double c21 = (this->mData[(2u *nCol) + 0u]) * (B.mData[(1u *nCol) + 0u])    
                     + (this->mData[(2u *nCol) + 1u]) * (B.mData[(1u *nCol) + 1u]) 
                     + (this->mData[(2u *nCol) + 2u]) * (B.mData[(1u *nCol) + 2u]); 

  // a20*b20 + a21*b21 + a22*b22
  const double c22 = (this->mData[(2u *nCol) + 0u]) * (B.mData[(2u *nCol) + 0u])    
                     + (this->mData[(2u *nCol) + 1u]) * (B.mData[(2u *nCol) + 1u]) 
                     + (this->mData[(2u *nCol) + 2u]) * (B.mData[(2u *nCol) + 2u]); 

  // add cxx to Cxx if adding is true
  if ( adding == true )
    {
      C.set(0u, 0u, (C.mData[0] + c00)); C.set(0u, 1u, (C.mData[1] + c01)); C.set(0u, 2u, (C.mData[2] + c02));
      C.set(1u, 0u, (C.mData[3] + c10)); C.set(1u, 1u, (C.mData[4] + c11)); C.set(1u, 2u, (C.mData[5] + c12));
      C.set(2u, 0u, (C.mData[6] + c20)); C.set(2u, 1u, (C.mData[7] + c21)); C.set(2u, 2u, (C.mData[8] + c22));
    }
  else
    {
     C.set(0u, 0u, c00); C.set(0u, 1u, c01); C.set(0u, 2u, c02);
     C.set(1u, 0u, c10); C.set(1u, 1u, c11); C.set(1u, 2u, c12);
     C.set(2u, 0u, c20); C.set(2u, 1u, c21); C.set(2u, 2u, c22);
    }
} // func mTmult block ends here


template class SimpleMatrix<long double>;
template class SimpleMatrix<double>;
template class SimpleMatrix<float>;
template class SimpleMatrix<int>;

} // namespace FemGL_mpi block ends here  
