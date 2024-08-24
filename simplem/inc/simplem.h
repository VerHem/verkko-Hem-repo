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

#ifndef SIMPLEM_H
#define SIMPLEM_H

namespace FemGL_mpi
{
 template <typename numbertype>
 class SimpleMatrix
 {
  public:
   // constructor
   SimpleMatrix(unsigned int M, unsigned int N);

   // overloading assignment oprator
   SimpleMatrix<numbertype> &operator= (const numbertype &number);

   // set element value on (i,j)
   void set(unsigned int i, unsigned int j, const numbertype &melement);

   // matrix trace
   numbertype trace();

   // print element (i,j)
   void print(unsigned int i, unsigned int j) const;

   // print *this matrix
   void print() const;

   /***************************************/
   /* API for adding and matrix multiplay */   
   /***************************************/

   // *this += a * A
   void add(const numbertype, const SimpleMatrix &);

   // *this += a * A + b * B
   void add(const numbertype, const SimpleMatrix &, const numbertype, const SimpleMatrix &);

   // C = A * B
   void mmult(SimpleMatrix &, const SimpleMatrix &, const bool adding = false) const;

   // C = AT * B
   void Tmmult(SimpleMatrix &, const SimpleMatrix &, const bool adding = false) const;

   // C = A * BT
   void mTmult(SimpleMatrix &, const SimpleMatrix &, const bool adding = false) const;

   // C = AT * BT
   void TmTmult(SimpleMatrix &, const SimpleMatrix &, const bool adding = false) const;      
  
  private:
   // row number and column number
   unsigned int nRow, nCol;

   // 1D array, hodling data of Matrix element, A(i,j) is mData[i * 3 + j]
   numbertype mData[9];

 };

}

#endif
