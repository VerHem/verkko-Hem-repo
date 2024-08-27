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
SimpleMatrix<numbertype>::SimpleMatrix(unsigned int M, unsigned int N, const numbertype number)
  : nRow(M), nCol(N)
{
  for (unsigned int m = 0; m < 9u; ++m)
     mData[m] = 0.0;

  // generate diagnolized Matrix if bumber != 0
  if (number != 0.0)
    { mData[0] = number; mData[4] = number; mData[8] = number; }  
}

template <typename numbertype>
SimpleMatrix<numbertype> &SimpleMatrix<numbertype>::operator= (const numbertype &number)
{
  for (unsigned int m = 0; m < 9u; ++m)
     mData[m] = number;
  
  return *this; 
}

template <typename numbertype>
numbertype SimpleMatrix<numbertype>::operator() (const unsigned int i, const unsigned int j) const
{
  return (this->mData[(i * nCol) + j]); 
}
  
template <typename numbertype>
void SimpleMatrix<numbertype>::set(unsigned int i, unsigned int j, const numbertype &melement)
{
  mData[(i * nCol) + j] = melement;
}

template <typename numbertype>
numbertype SimpleMatrix<numbertype>::trace()
{
  return ((this->mData[0]) + (this->mData[4]) + (this->mData[8]));
}  

template <typename numbertype>
void SimpleMatrix<numbertype>::print(unsigned int i, unsigned int j) const
{
  std::cout << mData[(i * nCol) + j] << "\n";  
}

template <typename numbertype>
void SimpleMatrix<numbertype>::print() const
{
  for (unsigned int i = 0; i<nRow; ++i)
     for (unsigned int j = 0; j<nCol; ++j)
       {
	 ( j == (nCol - 1u) ) ? (std::cout << mData[(i * nCol) + j] << "\n")
	                      : (std::cout << mData[(i * nCol) + j] << " ");
       }
  std::cout << std::endl;
}

template class SimpleMatrix<long double>;
template class SimpleMatrix<double>;
template class SimpleMatrix<float>;
template class SimpleMatrix<int>;


} // nammespace FemGL_mpi ends here  
