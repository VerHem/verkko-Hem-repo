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

  template <int dim>
  void FemGL<dim>::newton_iteration()
  {
    TimerOutput::Scope t(computing_timer, "newton_iteration");

    { // newton iteraion block with line search, all local objects will be released to save momory leak

     /* ------------------------------------- */
     /*  loading line search step Paramters   */
     /* ------------------------------------- */      
     conf.enter_subsection("control parameters");
     const double line_search_step = conf.get_double("primary step length of dampped newton iteration");
     const bool dampped_newton     = conf.get_bool("Using dampped Newton iteration");          
     conf.leave_subsection();
     /* ------------------------------------- */
     /*  line search step Paramters ends here */
     /* ------------------------------------- */      

      
     LA::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
     LA::MPI::Vector distributed_newton_update(locally_owned_dofs, mpi_communicator);          
     double previous_residual = system_rhs.l2_norm();

     if (dampped_newton == false)
       {// full-newton iteration block
         distributed_newton_update = locally_relevant_newton_solution;
         distributed_solution      = local_solution;

	 // damped iteration:
	 distributed_solution.add(1.0, distributed_newton_update);

	 // AffineConstraint::distribute call
         constraints_solution.distribute(distributed_solution);	

	 // assign un-ghosted solution to ghosted solution
	 // Don't be confused by the name with "dampped", there is NO dampped
	 locally_relevant_damped_vector = distributed_solution;

	 compute_residual(/*locally_relevant_damped_vector*/);
	 double current_residual = residual_vector.l2_norm();

	 pcout << " we are in full newton-iteration "
	       << ", residual is: "          << current_residual
	       << ", previous_residual is: " << previous_residual
	       << std::endl;
	 
	 if (current_residual < previous_residual)
	   {
	     pcout << " ohh! current_residual < previous_residual, we get better solution ! "
	           << std::endl;
           }
	 else
	   {
             pcout << " Humm ! current_residual >= previous_residual, maybe you need a better guess ! This is full-newton"
	           << std::endl;
	   }

       } // full-newton iteration block ends here
     else
       {// line search loop, dampped newton iteration

        for (unsigned int i = 0; i < 100; ++i)
         {
 	  const double alpha = std::pow(line_search_step, static_cast<double>(i));
          distributed_newton_update = locally_relevant_newton_solution;
          distributed_solution      = local_solution;

	  // damped iteration:
	  distributed_solution.add(alpha, distributed_newton_update);

	  // AffineConstraint::distribute call
          constraints_solution.distribute(distributed_solution);	

	  // assign un-ghosted solution to ghosted solution
	  locally_relevant_damped_vector = distributed_solution;

          //local_solution = distributed_solution;	 

	  compute_residual(/*locally_relevant_damped_vector*/);
	  double current_residual = residual_vector.l2_norm();

	  pcout << " step length alpha is: "  << alpha
	        << ", residual is: "          << current_residual
	        << ", previous_residual is: " << previous_residual
	        << std::endl;
	  if (current_residual < previous_residual)
	    {
	     pcout << " ohh! current_residual < previous_residual, we get better solution ! "
	           << std::endl;
  	     break;
            }
	  else
	    {
             pcout << " haa! current_residual >= previous_residual, more line search ! "
	           << std::endl;
	    }
	  
          } // for loop ends at here

        } // line search block 
    
     local_solution = distributed_solution;
     
    } // Newton iteraion block ends at here, both full-newton and dammped newton are possible
      // all local objects will be released to save momory leak

  } //femgl memeber newton_iteration() block ends at here

  template class FemGL<3>;  
} // namespace FemGL_mpi

