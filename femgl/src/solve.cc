/*
 *
 */


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

#include "femgl.h"
#include "dirichlet.h"
 

namespace FemGL_mpi
{
  using namespace dealii;
  
  template <int dim>
  void FemGL<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    { //linear solver block, will be released after it finishes to save memory leak
     LA::MPI::PreconditionAMG preconditioner;
     LA::MPI::Vector distributed_newton_update(locally_owned_dofs, mpi_communicator);
     {
      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");

      pcout << " start to build AMG preconditioner." << std::endl;      
      std::vector<std::vector<bool>> constant_modes;
      DoFTools::extract_constant_modes(dof_handler,
				       ComponentMask(),
				       constant_modes);

      TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
      additional_data.constant_modes        = constant_modes;
      additional_data.elliptic              = true;  /* elliptic actully faster than non-elliptic the sence to acchive same accuracy. 
                                                         In a 64cores run, ellipic used 30mins to achive 0.04, while non-eeliptic used
                                                         60mins to achive 0.09 */ 
      additional_data.n_cycles              = 4;
      additional_data.w_cycle               = true;
      additional_data.output_details        = false;
      additional_data.smoother_sweeps       = 2;
      additional_data.aggregation_threshold = 1e-2;

      preconditioner.initialize(system_matrix, additional_data);
      pcout << " AMG preconditioner is built up." << std::endl;      
     }

     {
      // With that, we can finally set up the linear solver and solve the system:
      pcout << " system_rhs.l2_norm() is " << system_rhs.l2_norm() << std::endl;
      SolverControl solver_control(10*system_matrix.m(),
                                   7e-1 * system_rhs.l2_norm());

      //SolverMinRes<LA::MPI::Vector> solver(solver_control);
      SolverFGMRES<LA::MPI::Vector> solver(solver_control);
      //SolverGMRES<LA::MPI::Vector> solver(solver_control, gmres_adddata);

      // what this .set_zero() is doing ?
      // AffineContraint::set_zero() set the values of all constrained DoFs in a vector to zero. 
      // constraints_newton_update.set_zero(distributed_newton_update);

      solver.solve(system_matrix,
                   distributed_newton_update,
                   system_rhs,
                   preconditioner);

      pcout << "   Solved in " << solver_control.last_step() << " iterations."
            << std::endl;

     }

    constraints_newton_update.distribute(distributed_newton_update);

    locally_relevant_newton_solution = distributed_newton_update;

    } // solver block ends at here to save memory leak

  }

  template class FemGL<3>;  
} // namespace FemGL_mpi

