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


#ifndef A_PHASE_MONOPOLE_CONFIG_3_INNER_WITH_FLUCTUATION_H
#define A_PHASE_MONOPOLE_CONFIG_3_INNER_WITH_FLUCTUATION_H

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
  class APhaseMonopleconfig : public Function<dim>
  {
  public:
    APhaseMonopleconfig(const double hx, const double hy, const double hz, const double gap_para)
      : Function<dim>(18) // tell base Function<dim> class I want a 2-components vector-valued function
      , half_x_length(hx)
      , half_y_length(hy)
      , half_z_length(hz)
      , gap(gap_para)	
    {}

    double half_x_length, half_y_length, half_z_length, gap;   // in unit of \xi^GL_0    
    
    virtual void vector_value(const Point<dim> &point /*p*/,
                              Vector<double> &values) const override
    {
      const double diff_rolrence = 3e-1;
      
      Assert(values.size() == 18, ExcDimensionMismatch(values.size(), 18));

      /*****************************************************/
      /* configuraions on boundary with x as normal vector */
      /*****************************************************/      
      if ( std::fabs(point(0) - (-half_x_length)) < (diff_rolrence * half_x_length) ) 
	{
	  /* d = x, m = z, n = y on x-direction boundary*/
          values[0] = 0.;  /*u11*/  values[9] = 0.;   /*v11*/
          values[1] = 0.;  /*u12*/  values[10] = gap/std::sqrt(2.);  /*v12*/
          values[2] = gap/std::sqrt(2.);  /*u13*/  values[11] = 0.;  /*v13*/
	  
          values[3] = 0.;  /*u21*/ values[12] = 0.; /*v21*/
          values[4] = 0.;  /*u22*/ values[13] = 0.; /*v22*/
          values[5] = 0.;  /*u23*/ values[14] = 0.; /*v23*/  

          values[6] = 0.;   /*u31*/ values[15] = 0.;  /*v31*/
          values[7] = 0.;   /*u32*/ values[16] = 0.;  /*v32*/
          values[8] = 0.;   /*u33*/ values[17] = 0.;  /*v33*/	  	  	  	  
	}
      
      if ( std::fabs(point(0) - (half_x_length)) < (diff_rolrence * half_x_length) ) 
	{
	  /* d = x, m = z, n = -y on x-direction boundary*/	  
          values[0] = 0.;  /*u11*/  values[9] = 0.;   /*v11*/
          values[1] = 0.;  /*u12*/  values[10] = (gap/std::sqrt(2.)); /*v12*/
          values[2] = -(gap/std::sqrt(2.));  /*u13*/  values[11] = 0.;  /*v13*/
	  
          values[3] = 0.;  /*u21*/ values[12] = 0.; /*v21*/
          values[4] = 0.;  /*u22*/ values[13] = 0.; /*v22*/
          values[5] = 0.;  /*u23*/ values[14] = 0.; /*v23*/  

          values[6] = 0.;   /*u31*/ values[15] = 0.;  /*v31*/
          values[7] = 0.;   /*u32*/ values[16] = 0.;  /*v32*/
          values[8] = 0.;   /*u33*/ values[17] = 0.;  /*v33*/	  	  	  	  

     	}

      /*****************************************************/
      /* configuraions on boundary with y as normal vector */
      /*****************************************************/            
      if ( std::fabs(point(1) - (-half_y_length)) < (diff_rolrence * half_y_length) ) 
	{
	  /* d = x, m = z, n = -x on y-direction boundary*/
          values[0] = gap/std::sqrt(2.);  /*u11*/  values[9] = 0.;   /*v11*/
          values[1] = 0.;  /*u12*/  values[10] = 0.;  /*v12*/
          values[2] = 0.;  /*u13*/  values[11] = (gap/std::sqrt(2.));  /*v13*/
	  
          values[3] = 0.;  /*u21*/ values[12] = 0.; /*v21*/
          values[4] = 0.;  /*u22*/ values[13] = 0.; /*v22*/
          values[5] = 0.;  /*u23*/ values[14] = 0.; /*v23*/  

          values[6] = 0.;   /*u31*/ values[15] = 0.;  /*v31*/
          values[7] = 0.;   /*u32*/ values[16] = 0.;  /*v32*/
          values[8] = 0.;   /*u33*/ values[17] = 0.;  /*v33*/	  	  	  	  
	}
      
      if ( std::fabs(point(1) - (half_y_length)) < (diff_rolrence * half_y_length) ) 
	{
	  /* d = x, m = z, n = x on y-direction boundary*/	  
          values[0] = gap/std::sqrt(2.);  /*u11*/  values[9] = 0.;   /*v11*/
          values[1] = 0.;  /*u12*/  values[10] = 0.; /*v12*/
          values[2] = 0.;  /*u13*/  values[11] = -gap/std::sqrt(2.);  /*v13*/
	  
          values[3] = 0.;  /*u21*/ values[12] = 0.; /*v21*/
          values[4] = 0.;  /*u22*/ values[13] = 0.; /*v22*/
          values[5] = 0.;  /*u23*/ values[14] = 0.; /*v23*/  

          values[6] = 0.;   /*u31*/ values[15] = 0.;  /*v31*/
          values[7] = 0.;   /*u32*/ values[16] = 0.;  /*v32*/
          values[8] = 0.;   /*u33*/ values[17] = 0.;  /*v33*/	  	  	  	  

     	}

      /*****************************************************/
      /* configuraions on boundary with x as normal vector */
      /*****************************************************/      
      if ( std::fabs(point(2) - (-half_z_length)) < (diff_rolrence * half_z_length) ) 
	{
	  /* d = x, m = -x, n = y on z-direction boundary*/
          values[0] = gap/std::sqrt(2.);  /*u11*/  values[9] = 0.;   /*v11*/
          values[1] = 0.;  /*u12*/  values[10] = -(gap/std::sqrt(2.));  /*v12*/
          values[2] = 0.;  /*u13*/  values[11] = 0.;  /*v13*/
	  
          values[3] = 0.;  /*u21*/ values[12] = 0.; /*v21*/
          values[4] = 0.;  /*u22*/ values[13] = 0.; /*v22*/
          values[5] = 0.;  /*u23*/ values[14] = 0.; /*v23*/  

          values[6] = 0.;   /*u31*/ values[15] = 0.;  /*v31*/
          values[7] = 0.;   /*u32*/ values[16] = 0.;  /*v32*/
          values[8] = 0.;   /*u33*/ values[17] = 0.;  /*v33*/	  	  	  	  
	}

      if ( std::fabs(point(2) - (half_z_length)) < (diff_rolrence * half_z_length) ) 
	{
	  /* d = x, m = x, n = y on z-direction boundary*/	  
          values[0] = gap/std::sqrt(2.);  /*u11*/  values[9] = 0.;   /*v11*/
          values[1] = 0.;  /*u12*/  values[10] = gap/std::sqrt(2.); /*v12*/
          values[2] = 0.;  /*u13*/  values[11] = 0.;  /*v13*/
	  
          values[3] = 0.;  /*u21*/ values[12] = 0.; /*v21*/
          values[4] = 0.;  /*u22*/ values[13] = 0.; /*v22*/
          values[5] = 0.;  /*u23*/ values[14] = 0.; /*v23*/  

          values[6] = 0.;   /*u31*/ values[15] = 0.;  /*v31*/
          values[7] = 0.;   /*u32*/ values[16] = 0.;  /*v32*/
          values[8] = 0.;   /*u33*/ values[17] = 0.;  /*v33*/	  	  	  	  

     	}

            

      /*********************************************************/
      /* d=1, m=x, n=y for point doesn't locate on any bundary */
      /*  Random number s generated for every element as noise */
      /*********************************************************/
      std::random_device rd{};
      std::mt19937       gen{rd()};
      std::normal_distribution<double> gaussian_dis{0.0, 0.25};
      if (
           !( std::fabs(point(0) - (-half_x_length)) < (diff_rolrence * half_x_length) )
	   &&
	   !( std::fabs(point(0) - (half_x_length)) < (diff_rolrence * half_x_length) )
	   &&
	   !( std::fabs(point(1) - (-half_y_length)) < (diff_rolrence * half_y_length) )
	   &&
	   !( std::fabs(point(1) - (half_y_length)) < (diff_rolrence * half_y_length) )
	   &&
	   !( std::fabs(point(2) - (-half_z_length)) < (diff_rolrence * half_z_length) )
	   &&
	   !( std::fabs(point(2) - (half_z_length)) < (diff_rolrence * half_z_length) )
         )
	{
          values[0] = (gap/std::sqrt(2.)) + gaussian_dis(gen);  /*u11*/  values[9] = gaussian_dis(gen);   /*v11*/
          values[1] = gaussian_dis(gen);  /*u12*/  values[10] = (gap/std::sqrt(2.)) + gaussian_dis(gen); /*v12*/
          values[2] = gaussian_dis(gen);  /*u13*/  values[11] = gaussian_dis(gen);  /*v13*/
 
          values[3] = gaussian_dis(gen);  /*u21*/ values[12] = gaussian_dis(gen); /*v21*/
          values[4] = gaussian_dis(gen);  /*u22*/ values[13] = gaussian_dis(gen); /*v22*/
          values[5] = gaussian_dis(gen);  /*u23*/ values[14] = gaussian_dis(gen); /*v23*/  

          values[6] = gaussian_dis(gen);   /*u31*/ values[15] = gaussian_dis(gen);  /*v31*/
          values[7] = gaussian_dis(gen);   /*u32*/ values[16] = gaussian_dis(gen);  /*v32*/
          values[8] = gaussian_dis(gen);   /*u33*/ values[17] = gaussian_dis(gen);  /*v33*/	  	  	  	  
	}      
            
    } // vector_value() function ends here

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        APhaseMonopleconfig<dim>::vector_value(points[p], value_list[p]);
    }
  };

} // namespace FemGL_mpi

#endif
