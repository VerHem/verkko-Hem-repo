/*
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

namespace FemGL_mpi
{
  using namespace dealii;


  template <int dim>
  void FemGL<dim>::run()
  {
    pcout << "Running using Trilinos." << std::endl;

    std::string ref_str = "adaptive";

    /*---------------------------------------*/
    /* loading refinements control paramters */
    /*---------------------------------------*/    
    conf.enter_subsection("control parameters"); 
    const unsigned int n_cycles                = conf.get_integer("Number of adaptive refinements");
    const unsigned int n_iteration             = conf.get_integer("Number of interations");
    const double       Cycle0_refine_threshold = conf.get_double("threshold of Cycle 0 refinement");    
    const double       refine_threshold        = conf.get_double("threshold of refinement");
    const double       converge_acc            = conf.get_double("converge accuracy");    
    conf.leave_subsection();
    /*---------------------------------------*/
    /*    paramters loading ends at here     */
    /*---------------------------------------*/    

    
    for (cycle = 0; cycle <= n_cycles; ++cycle)
      {
        pcout << "\n"
	      << "Refinement Cycle is " << cycle
	      << std::endl;

        if (cycle == 0)
	  {
	   make_grid();
           setup_system();	
	  }
        else
	  refine_grid(ref_str);

	double residual_last_iter = 0.0; // residual_vector.norm() in last time interation
        for (unsigned int iteration_loop = 0; iteration_loop <= n_iteration; ++iteration_loop)
	  {
	     pcout << "Refinement cycle : " << cycle << ", "
                   << "iteration_loop: " << iteration_loop	       
		   << std::endl;

             assemble_system();
	     pcout << "assembly is done !" << std::endl;
             solve();
	     pcout << " AMG preconditioned solving is done !" << std::endl;	     
	     newton_iteration();
	     pcout << " newton iteration is done !" << std::endl;	     	     

             if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 12800)
              {
               TimerOutput::Scope t(computing_timer, "output");
	       output_results(iteration_loop);
              }

             computing_timer.print_summary();
             computing_timer.reset();

             pcout << std::endl;

	     const double residual_l2_norm = residual_vector.l2_norm();
	     if (/* Cycle 0 stuck-refine condtion */
		 (std::fabs(residual_l2_norm - residual_last_iter) < Cycle0_refine_threshold)
		 && (residual_l2_norm > converge_acc)
		 && (cycle == 0)
		)
	       break;
	     else if (/* Cycle < n_cycles stuck-refine condtion */
		      (std::fabs(residual_l2_norm - residual_last_iter) < refine_threshold)
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

