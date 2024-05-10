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
  void FemGL<dim>::refine_grid(std::string &refinement_strategy)
  {
    TimerOutput::Scope t(computing_timer, "refine");

    if (refinement_strategy == "global")
      {
	{
	 // solution transfer block
         parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_transfer(dof_handler);
         solution_transfer.prepare_for_coarsening_and_refinement(local_solution);

         triangulation.refine_global(1);
	 setup_system();
         pcout << "setup_system() call is done ! this is global refinment"
	       << std::endl;	 
         TrilinosWrappers::MPI::Vector distributed_solution_tmp(locally_owned_dofs,
							        mpi_communicator);	 
  
         solution_transfer.interpolate(distributed_solution_tmp);
         constraints_solution.distribute(distributed_solution_tmp);
         local_solution = distributed_solution_tmp;
	}
      }
    else if (refinement_strategy == "adaptive")
      {
        {
         /*---------------------------------------*/ 
         /* loading refinements control paramters */
         /*---------------------------------------*/
         conf.enter_subsection("control parameters");
         const double refine_ratio  = conf.get_double("adaptive refinment ratio");
         const double coarsen_ratio = conf.get_double("adaptive coarsen ratio");	 
	 conf.leave_subsection();
         /*---------------------------------------*/
	 /*    paramters loading ends at here     */
         /*---------------------------------------*/
	  
         Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
         KellyErrorEstimator<dim>::estimate(dof_handler,
				            QGauss<dim - 1>(fe.degree + 1),
					    std::map<types::boundary_id, const Function<dim> *>(),
					    local_solution,
					    estimated_error_per_cell);

         parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
					  				        estimated_error_per_cell,
										refine_ratio, /*0.3*/
										coarsen_ratio /*0.0*/);
        }

        triangulation.prepare_coarsening_and_refinement();

        {
         parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_transfer(dof_handler);
         solution_transfer.prepare_for_coarsening_and_refinement(local_solution);
         triangulation.execute_coarsening_and_refinement();

         // after execute refining, you must do it at here otherwise local_solution will not match tmp Vector
         setup_system();
         pcout << "setup_system() call is done !" << std::endl;
      
         TrilinosWrappers::MPI::Vector distributed_solution_tmp(locally_owned_dofs,
							        mpi_communicator);
         solution_transfer.interpolate(distributed_solution_tmp);
         constraints_solution.distribute(distributed_solution_tmp);
         local_solution = distributed_solution_tmp;
        }
        //compute_nonlinear_residual(solution);
        pcout << "adaptive_refine_grid() call is done !" << std::endl;
      } // adaptive block ends here
    
  }

  template class FemGL<3>;  
} // namespace FemGL_mpi

