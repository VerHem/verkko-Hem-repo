/*
 *
 */


#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

// The following chunk out code is identical to step-40 and allows
// switching between PETSc and Trilinos:

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
 

namespace FemGL_mpi
{
  using namespace dealii;


  template <int dim>
  void FemGL<dim>::run()
  {
    pcout << "Running using Trilinos." << std::endl;

    std::string ref_str = "adaptive";
    const unsigned int n_cycles    = 3;
    const unsigned int n_iteration = 50;    
    for (cycle = 0; cycle <= n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
	  {
	   make_grid();
           setup_system();	
	  }
        else
	  refine_grid(ref_str);
	
        for (unsigned int iteration_loop = 0; iteration_loop <= n_iteration; ++iteration_loop)
	  {
	     pcout << "cycle : " << cycle << ", "
                   << "iteration_loop: " << iteration_loop	       
		   << std::endl;

             assemble_system();
	     pcout << "assembly is done !" << std::endl;
             solve();
	     pcout << " AMG solve is done !" << std::endl;	     
	     newton_iteration();
	     pcout << " newton iteration is done !" << std::endl;	     	     

             if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 1280)
              {
               TimerOutput::Scope t(computing_timer, "output");
	       output_results(iteration_loop);
              }

             computing_timer.print_summary();
             computing_timer.reset();

             pcout << std::endl;

	     if ((system_rhs.l2_norm() < 1e-2) && (cycle == 0))
	       break;
	     else if ((system_rhs.l2_norm() < 5e-3) && (cycle == 1))
	       break;
	     else if ((system_rhs.l2_norm() < 5e-4) && (cycle == 2))
	       break;
	     else if ((system_rhs.l2_norm() < 1e-6) && (cycle == 3))
	       break;
	     /*else if ((system_rhs.l2_norm() < 5e-6) && (cycle == 4))
	       break;	     
	     else if ((system_rhs.l2_norm() < 5e-6) && (cycle == 5))
	       break;
	     else if ((system_rhs.l2_norm() < 5e-6) && (cycle == 6))
	       break;
	     else if ((system_rhs.l2_norm() < 5e-6) && (cycle == 7))
	       break;
	     else if ((system_rhs.l2_norm() < 5e-7) && (cycle == 8))
	       break;	     	     	     
	     */
	  }

      }
     computing_timer.print_summary();	
  }

  template class FemGL<3>;
  //template class FemGL<2>;      

} // namespace FemGL_mpi

