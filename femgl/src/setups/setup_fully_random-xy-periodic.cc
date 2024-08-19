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
 

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  void FemGL<dim>::setup_system()
  {
    /* Initialize the component mask object by a bool vector
     * This is necessary for setting up AffineConstraints objects
     * in the interpolate_boundary_values() call.
     * According to AdGR BCs, 0, 3, 6, 9, 12, 15 components are zero-Dirichlet when normal vector is x
     * According to AdGR BCs, 1, 4, 7, 10, 13, 16 components are zero-Dirichlet when normal vector is y
     * According to AdGR BCs, 2, 5, 8, 11, 14, 17 components are zero-Dirichlet when normal vector is z
     */
    ComponentMask comp_mask_x(Dirichlet_x_marking_list);
    ComponentMask comp_mask_y(Dirichlet_y_marking_list);
    ComponentMask comp_mask_z(Dirichlet_z_marking_list);

    /*---------------------------------------*/
    /* identify matched faces pairs          */
    /* for periodicity along x, y direction  */
    /*---------------------------------------*/
    
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>> matched_pairs_x, matched_pairs_y;

    GridTools::collect_periodic_faces(dof_handler,
                                      5 /*b_id1 */, 6 /*b_id2*/,
                                      0, /*spatial direction of periodicity */
                                      matched_pairs_x);       

    GridTools::collect_periodic_faces(dof_handler,
                                      7 /*b_id1 */, 8 /*b_id2*/,
                                      1, /*spatial direction of periodicity */
                                      matched_pairs_y);

    /* cook first_vector_component up for make_periodicity_constraints */
    std::vector<unsigned int> first_vector_components;
    first_vector_components.push_back(0);      
    
    /*---------------------------------------*/
    /* inditify matched face ends here       */
    /*---------------------------------------*/
    
    
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
      constraints_newton_update.clear();
      constraints_newton_update.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_newton_update);

      VectorTools::interpolate_boundary_values(dof_handler,
                                               2, // 0 for all Dirichlet, 2 for diffuse, which 6 components are Dirichlet along x 
                                               DirichletBCs_newton_update<dim>(),
                                               constraints_newton_update,
					       comp_mask_x);

      
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3, // 0 for all Dirichlet, 2 for diffuse, which 6 components are Dirichlet along y
                                               DirichletBCs_newton_update<dim>(),
                                               constraints_newton_update,
					       comp_mask_y);

      
      VectorTools::interpolate_boundary_values(dof_handler,
                                               4, // 0 for all Dirichlet, 2 for diffuse, which 6 components are Dirichlet along z
                                               DirichletBCs_newton_update<dim>(),
                                               constraints_newton_update,
					       comp_mask_z);
                                               //fe.component_mask(velocities));

      /*---------------------------------------*/
      /* identify matched faces pairs          */
      /* add perodicuty info into triangulation*/
      /*---------------------------------------*/
      
      DoFTools::make_periodicity_constraints<dim, dim>(matched_pairs_x,
                                                       constraints_newton_update
                                                       //{}  default, all components,
						       //first_vector_components first_vector_components of whole vector is 0
						       );

      DoFTools::make_periodicity_constraints<dim, dim>(matched_pairs_y,
                                                       constraints_newton_update
                                                       //{} default, all components,
						       //first_vector_components first_vector_components of whole vector is 0
						       );
   
      /*---------------------------------------*/
      /* periodicity info handling ends here   */
      /*---------------------------------------*/
      
	      
      constraints_newton_update.close();
    }

    {
      constraints_solution.clear();
      constraints_solution.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints_solution);

      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,  // 0 for all Dirichlet, 2 for diffuse, which 6 components are Dirichlet along x
                                               BoundaryValues<dim>(),
                                               constraints_solution,
					       comp_mask_x);
      
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,  // 0 for all Dirichlet, 2 for diffuse, which 6 components are Dirichlet along y
                                               BoundaryValues<dim>(),
                                               constraints_solution,
					       comp_mask_y);
      
      VectorTools::interpolate_boundary_values(dof_handler,
                                               4,  // 0 for all Dirichlet, 2 for diffuse, which 6 components are Dirichlet along z
                                               BoundaryValues<dim>(),
                                               constraints_solution,
					       comp_mask_z);
                                               //fe.component_mask(velocities));
      /*---------------------------------------*/
      /* identify matched faces pairs          */
      /* add perodicuty info into triangulation*/
      /*---------------------------------------*/
      
      DoFTools::make_periodicity_constraints<dim, dim>(matched_pairs_x,
                                                       constraints_solution
                                                       //{} default, all components,
						       //first_vector_components first_vector_components of whole vector is 0
						       );

      DoFTools::make_periodicity_constraints<dim, dim>(matched_pairs_y,
                                                       constraints_solution
                                                       //{} default, all components,
						       //first_vector_components first_vector_components of whole vector is 0
						       );
   
      /*---------------------------------------*/
      /* periodicity info handling ends here   */
      /*---------------------------------------*/

      
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

        /*---------------------------------------*/ 
        /* loading refinements control paramters */
        /*---------------------------------------*/
        conf.enter_subsection("physical parameters");
        const double gaussian_mean           = conf.get_double("gaussian random mean value");
        const double gaussian_std            = conf.get_double("gaussian random STD");
        conf.leave_subsection();
        /*---------------------------------------*/
        /*    paramters loading ends at here     */
        /*---------------------------------------*/

	 
        /*  set up initial local_solution Vector */
        /*---------------------------------------*/
        std::random_device rd{};         // rd will be used to obtain a seed for the random number engine
        std::mt19937       gen{rd()};    // Standard mersenne_twister_engine seeded with rd()
        std::normal_distribution<double> gaussian_distr{gaussian_mean, gaussian_std}; // gaussian distribution, 1st arg is mean. 2nd arg is STD
        //std::normal_distribution<double> gaussian_distr2{0.0, 0.1}; // gaussian distribution, 1st arg is mean. 2nd arg is STD	

        local_solution.reinit(locally_relevant_dofs,
    			      mpi_communicator,
    			      false);
        LA::MPI::Vector distrubuted_tmp_solution(locally_owned_dofs,
                                                 mpi_communicator);
	 
        for (auto it = distrubuted_tmp_solution.begin(); it != distrubuted_tmp_solution.end(); ++it)
         {
	  //*it = 0.0;
	  //*it = gaussian_distr2(gen);
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

