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


#ifndef BINB_H
#define BINB_H

#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <cmath>
#include <fstream>
#include <iostream>

//#include "matep.h"

namespace FemGL_mpi
{
  using namespace dealii;

  /* ------------------------------------------------------------------------------------------
   * class template BinA inhereted from Function<dim>.
   * set the reference value_list to B-in-A configuration for full-step newton iteration.
   * ------------------------------------------------------------------------------------------
   */
  
  template <int dim>
  class BinB : public Function<dim>
  {
  public:
    BinB(const double &r)
      : Function<dim>(18) // tell base Function<dim> class I want a 2-components vector-valued function
      , radius(r)
    {}

    double radius;         // in unit of \xi_0    
    
    virtual void vector_value(const Point<dim> &point /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 18, ExcDimensionMismatch(values.size(), 18));

      if ( (point(0)*point(0) + point(1)*point(1)) <= (radius*radius) )     //inside sphere, B-phase
	{
          values[0] = 0.859734; values[9] = 0.859734;
          values[1] = -0.285668; values[10] = -0.285668;
          values[2] = -1.76655; values[11] = -1.76655;
	  
          values[3] = 1.58845; values[12] = 1.58845;
          values[4] = -1.25931; values[13] = -1.25931;
          values[5] = 0.763255;	values[14] = 0.763255;	  

          values[6] = 1.26923; values[15] = 1.26923;
          values[7] = 1.77689; values[16] = 1.77689;
          values[8] = 0.245094; values[17] = 0.245094;	  	  	  	  
	}
      else if ( (point(0)*point(0) + point(1)*point(1)) > (radius*radius) ) //outside sphere, B-phase
	{
          values[0] = 0.668541; values[9] = 0.668541;
          values[1] = 0.632763; values[10] = 0.632763;
          values[2] = 1.7257; values[11] = 1.7257;
	  
          values[3] = 1.63496; values[12] = 1.63496;
          values[4] = -1.50349; values[13] = -1.50349;
          values[5] = -0.0439564; values[14] = -0.0439564;	  

          values[6] = 1.3531; values[15] = 1.3531;
          values[7] = 1.50482; values[16] = 1.50482;
          values[8] = -0.799212; values[17] = -0.799212;	  	  	  	  
          
     	}
    } // vector_value() function ends here

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        BinB<dim>::vector_value(points[p], value_list[p]);
    }
  };

} // namespace FemGL_mpi

#endif
