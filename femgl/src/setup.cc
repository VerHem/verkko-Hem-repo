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
  void FemGL<dim>::setup_system()
  {
    {
      TimerOutput::Scope t(computing_timer, "setup");
      dof_handler.distribute_dofs(fe);

      pcout << "   Number of degrees of freedom: "
 	    << dof_handler.n_dofs()
	    << std::endl;

      locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);    
    }

    {
      constraints_newton_update.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_newton_update);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               DirichletBCs_newton_update<dim>(),
                                               constraints_newton_update);
                                               //fe.component_mask(velocities));
      constraints_newton_update.close();
    }

    {
      constraints_solution.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints_solution);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               constraints_solution);
                                               //fe.component_mask(velocities));
      constraints_solution.close();
    }

    {
      system_matrix.clear();

      TrilinosWrappers::SparsityPattern sp(locally_owned_dofs, mpi_communicator);

      DoFTools::make_sparsity_pattern(dof_handler, sp, constraints_newton_update, false,
				      Utilities::MPI::this_mpi_process(mpi_communicator));
      // exchange local dsp entries between processes      
      sp.compress();
      system_matrix.reinit(sp);
    }

    // initialize local_solution Vector by gaussian random numbers in cycle 0
    {
     if (cycle == 0)
       {
        /*  set up initial local_solution Vector */
        /*---------------------------------------*/
        std::random_device rd{};         // rd will be used to obtain a seed for the random number engine
        std::mt19937       gen{rd()};    // Standard mersenne_twister_engine seeded with rd()
        std::normal_distribution<double> gaussian_distr{2.0, 0.2}; // gaussian distribution, 1st arg is mean. 2nd arg is STD

        local_solution.reinit(locally_relevant_dofs,
    			     mpi_communicator,
    			     false);
        LA::MPI::Vector distrubuted_tmp_solution(locally_owned_dofs,
                                                mpi_communicator);

        for (auto it = distrubuted_tmp_solution.begin(); it != distrubuted_tmp_solution.end(); ++it)
         {
	  *it = gaussian_distr(gen);
         }

        // AffineConstriant::distribute call
        constraints_solution.distribute(distrubuted_tmp_solution);

        local_solution = distrubuted_tmp_solution;    
        /*---------------------------------------*/
       }
     else
       local_solution.reinit(locally_relevant_dofs, mpi_communicator/*, false*/);
    }

    {
      locally_relevant_newton_solution.reinit(locally_relevant_dofs, mpi_communicator);
      locally_relevant_damped_vector.reinit(locally_relevant_dofs, mpi_communicator);
      system_rhs.reinit(locally_owned_dofs, mpi_communicator);
      residual_vector.reinit(locally_owned_dofs, mpi_communicator);    
    }
  }

  template class FemGL<3>;  
} // namespace FemGL_mpi

