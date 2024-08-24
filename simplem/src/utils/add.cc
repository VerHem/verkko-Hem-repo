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


#include "SimpleMatrix.h"
#include <iostream>

namespace FemGL_mpi
{

template <typename numbertype>
void SimpleMatrix<numbertype>::add(const numbertype a, const SimpleMatrix &A)
{

  for (unsigned int m = 0; m < 9u; ++m)
    this->mData[m] += a * A.mData[m];  
}

template <typename numbertype>
void SimpleMatrix<numbertype>::add(const numbertype a, const SimpleMatrix &A, const numbertype b, const SimpleMatrix &B)
{

  for (unsigned int m = 0; m < 9u; ++m)
    this->mData[m] += (a * A.mData[m] + b * B.mData[m]);  
}



template class SimpleMatrix<long double>;
template class SimpleMatrix<double>;
template class SimpleMatrix<float>;
template class SimpleMatrix<int>;

} // namespace FemGl_mpi block ends here       
