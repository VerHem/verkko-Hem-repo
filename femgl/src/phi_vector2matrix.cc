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


#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
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

#include "femgl.h"
#include "dirichlet.h"
#include "confreader.h"
#include "matep.h"
#include "BinA.h"

namespace FemGL_mpi
{
  using namespace dealii;

/* --------------------------------------------------------------------------------
   * old_solution_matrix_generator
   * phi_matrix_generator
   * grad_phi_matrix_container_generator
   * 
   * Homo-Robin BC face phi_matrix generator
   * --------------------------------------------------------------------------------
   */
  template <int dim>
  void FemGL<dim>::phi_matrix_generator(const FEValues<dim> &fe_values,
 				        const unsigned int  x, const unsigned int q,
					FullMatrix<double>  &phi_u_at_x_q,
					FullMatrix<double>  &phi_v_at_x_q)
  {
    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	phi_u_at_x_q.set(comp_index/3u, comp_index%3u, fe_values[components_u[comp_index]].value(x, q));
	phi_v_at_x_q.set(comp_index/3u, comp_index%3u, fe_values[components_v[comp_index]].value(x, q));
      }
    /*matrix_u_at_x_q.set(0,0,fe_values[u11_component].value(x, q));
      matrix_v_at_x_q.set(2,2,fe_values[v33_component].value(x, q));*/
  }

  template <int dim>
  void FemGL<dim>::grad_phi_matrix_container_generator(const FEValues<dim> &fe_values,
							     const unsigned int x, const unsigned int q,
							     std::vector<FullMatrix<double>> &container_grad_phi_u_x_q,
							     std::vector<FullMatrix<double>> &container_grad_phi_v_x_q)
  {
    /* auto grad_u11_x_q = fe_values[u11_component].gradient(x, q);
       auto grad_v33_x_q = fe_values[v33_component].gradient(x, q);*/
    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	auto gradient_uxx_at_q = fe_values[components_u[comp_index]].gradient(x, q);
	auto gradient_vxx_at_q = fe_values[components_v[comp_index]].gradient(x, q);

	for (unsigned int k = 0; k < dim; ++k)
	  {
	    container_grad_phi_u_x_q[k].set(comp_index/3u, comp_index%3u, gradient_uxx_at_q[k]);
	    container_grad_phi_v_x_q[k].set(comp_index/3u, comp_index%3u, gradient_vxx_at_q[k]);
	  }
      }

  }

  template <int dim>
  void FemGL<dim>::phi_matrix_face_generator(const FEFaceValues<dim> &fe_face_values,
 				             const unsigned int  x, const unsigned int q,
					     FullMatrix<double>  &phi_u_face_at_x_q,
					     FullMatrix<double>  &phi_v_face_at_x_q,
					     types::boundary_id b_id )
  {
    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
        if (
	     /* the 1st column commponents for normal vector x*/
	     ((comp_index == 0) || (comp_index == 3) || (comp_index == 6))
	     && (b_id == 2)
	   )
	  {
           phi_u_face_at_x_q.set(comp_index/3u, comp_index%3u, 0.);
	   phi_v_face_at_x_q.set(comp_index/3u, comp_index%3u, 0.);
	  }	
        else if (
	         /* the 2nd column commponents for normal vector y*/
	         ((comp_index == 1) || (comp_index == 4) || (comp_index == 7))
	         && (b_id == 3)
	        )
	  {
           phi_u_face_at_x_q.set(comp_index/3u, comp_index%3u, 0.);
	   phi_v_face_at_x_q.set(comp_index/3u, comp_index%3u, 0.);
	  }	
        else if (
	         /* the 3rd column commponents for normal vector z*/
	         ((comp_index == 2) || (comp_index == 5) || (comp_index == 8))
	         && (b_id == 4)
	        )
	  {
           phi_u_face_at_x_q.set(comp_index/3u, comp_index%3u, 0.);
	   phi_v_face_at_x_q.set(comp_index/3u, comp_index%3u, 0.);
	  }
	else
	  {
           phi_u_face_at_x_q.set(comp_index/3u, comp_index%3u, fe_face_values[components_u[comp_index]].value(x, q));
	   phi_v_face_at_x_q.set(comp_index/3u, comp_index%3u, fe_face_values[components_v[comp_index]].value(x, q));
	  }
      }
  }
 

  template class FemGL<3>;  
} //namespace FemGL_mpi ends here  
