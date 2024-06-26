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
  void FemGL<dim>::run()
  {
    pcout << "Running using Trilinos." << std::endl;

    /*---------------------------------------*/
    /* loading refinements control paramters */
    /*---------------------------------------*/    
    conf.enter_subsection("control parameters"); 
    const unsigned int n_cycles                = conf.get_integer("Number of refinements");
    const unsigned int n_iteration             = conf.get_integer("Number of interations");
    const double       Cycle0_refine_threshold = conf.get_double("Cycle 0 refinement threshold");
    const double       Cycle0_solve_tol        = conf.get_double("Cycle 0 linear solver tol");
    const double       Cycle1_refine_threshold = conf.get_double("Cycle 1 refinement threshold");
    const bool         Cycle1_do_global_refine = conf.get_bool("Cycle 1 do global refinement");
    const double       Cycle1_solve_tol        = conf.get_double("Cycle 1 linear solver tol");    
    const double       Cycle2_refine_threshold = conf.get_double("Cycle 2 refinement threshold");
    const bool         Cycle2_do_global_refine = conf.get_bool("Cycle 2 do global refinement");
    const double       Cycle2_solve_tol        = conf.get_double("Cycle 2 linear solver tol");    
    const double       Cycle3_refine_threshold = conf.get_double("Cycle 3 refinement threshold");
    const bool         Cycle3_do_global_refine = conf.get_bool("Cycle 3 do global refinement");
    const double       Cycle3_solve_tol        = conf.get_double("Cycle 3 linear solver tol");    
    const bool         Cycle4_do_global_refine = conf.get_bool("Cycle 4 do global refinement");
    const double       Cycle4_solve_tol        = conf.get_double("Cycle 4 linear solver tol");    
    const double       converge_acc            = conf.get_double("converge accuracy");
    conf.leave_subsection();
    /*---------------------------------------*/
    /*    paramters loading ends at here     */
    /*---------------------------------------*/

    /* arameters lists for refine stratagy */
    std::string ref_str;

    std::vector<double> cycleX_refine_threshold = {Cycle0_refine_threshold,
						   Cycle1_refine_threshold,
						   Cycle2_refine_threshold,
						   Cycle3_refine_threshold};

    std::vector<bool>   cycleX_refine_strategy = {Cycle1_do_global_refine,
					          Cycle2_do_global_refine,
					          Cycle3_do_global_refine,
					          Cycle4_do_global_refine};

    std::vector<double> cycleX_solve_tol{Cycle0_solve_tol,
                                         Cycle1_solve_tol,
                                         Cycle2_solve_tol,
                                         Cycle3_solve_tol,
                                         Cycle4_solve_tol};
        
    for (cycle = 0; cycle <= n_cycles; ++cycle)
      {
        pcout << "\n"
	      << "Refinement Cycle is " << cycle
	      << "\n"
	      << "------------------------------------------------------" << "\n"
	      << "------------------------------------------------------" << "\n"
	      << std::endl;

	//std::cout << " this rank has active cells : " << triangulation.n_active_cells() << std::endl;	

	if (cycle == 0)
	  {
  	   pcout << " cycle 0 will be globally refined in make_grid() call "
                 << "\n"
		 << std::endl;

	   make_grid();
           setup_system();
           {
             std::string dirc = "./setup_config/";
             output_results(dirc);
           }
	   
	  }
	else if (cycle > 0)
	  {
	   pcout << " 0th rank has active cells : " << triangulation.n_active_cells()
	         << " cycleX_refine_strategy[cycle-1] is " << cycleX_refine_strategy[cycle-1]
	         << "\n"
	         << std::endl;

	   if (cycleX_refine_strategy[cycle-1] == false)
            ref_str = "adaptive";
           else
            ref_str = "global";

	   refine_grid(ref_str);		   	   
	  }
	
        // if (cycle == 0)
	//   {
	//    make_grid();
        //    setup_system();	
	//   }
        // else
	//   refine_grid(ref_str);	

	//std::cout << " this rank has active cells : " << triangulation.n_active_cells() << std::endl;
	
	double residual_last_iter = 0.0; // residual_vector.norm() in last time interation
        for (iteration_loop = 0; iteration_loop <= n_iteration; ++iteration_loop)
	  {
	     pcout << "Refinement cycle : " << cycle << ", "
                   << "iteration_loop: " << iteration_loop	       
		   << std::endl;

             assemble_system();
	     pcout << " assembly is done !" << std::endl;
             solve(cycleX_solve_tol[cycle]);
	     pcout << " AMG preconditioned solving is done ! With solver_tol " << cycleX_solve_tol[cycle] << std::endl;	     
	     newton_iteration();
	     pcout << " newton iteration is done !" << std::endl;	     	     

             if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 12800)
              {
               std::string dirc = "./refine-cycle_" + std::to_string(cycle) + "/";
	       
               TimerOutput::Scope t(computing_timer, "output");
	       output_results(dirc);
              }

             computing_timer.print_summary();
             computing_timer.reset();

             pcout << std::endl;

	     const double residual_l2_norm = residual_vector.l2_norm();
	     if (/* Cycle x stuck-refine condtion */
		 (std::fabs(residual_l2_norm - residual_last_iter) < cycleX_refine_threshold[cycle])
		 && (residual_l2_norm > converge_acc)
		 && (cycle < n_cycles)
		)
	       break;
	     else if (/* this part is for final converge checking */
                      (residual_l2_norm <= converge_acc)
                     )
	       break;
	     else
	       residual_last_iter = residual_l2_norm; /* if iteration isn't stuck or grid has been 
                                                         maximum refined, this statement will run.
                                                         This push ieteration either to stuck & refine,
                                                         or in to final converge.
                                                       */

	  } // iteration loop ends at here
	
	// Stop Cycle loop if required acccuracy is achieved
	if (residual_vector.l2_norm() <= converge_acc)
	  break;

      } // Cycle loop end here
     computing_timer.print_summary();	
  }

  template class FemGL<3>;
  //template class FemGL<2>;      

} // namespace FemGL_mpi

