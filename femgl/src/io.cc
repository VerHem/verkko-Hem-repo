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
  void FemGL<dim>::output_results(const std::string &dirc) const
  {

    std::vector<std::string> newton_update_components_names;
    newton_update_components_names.emplace_back("du_11");
    newton_update_components_names.emplace_back("du_12");
    newton_update_components_names.emplace_back("du_13");
    newton_update_components_names.emplace_back("du_21");
    newton_update_components_names.emplace_back("du_22");
    newton_update_components_names.emplace_back("du_23");
    newton_update_components_names.emplace_back("du_31");
    newton_update_components_names.emplace_back("du_32");
    newton_update_components_names.emplace_back("du_33");
    newton_update_components_names.emplace_back("dv_11");
    newton_update_components_names.emplace_back("dv_12");
    newton_update_components_names.emplace_back("dv_13");
    newton_update_components_names.emplace_back("dv_21");
    newton_update_components_names.emplace_back("dv_22");
    newton_update_components_names.emplace_back("dv_23");
    newton_update_components_names.emplace_back("dv_31");
    newton_update_components_names.emplace_back("dv_32");
    newton_update_components_names.emplace_back("dv_33");

    std::vector<std::string> solution_components_names;
    solution_components_names.emplace_back("u_11");
    solution_components_names.emplace_back("u_12");
    solution_components_names.emplace_back("u_13");
    solution_components_names.emplace_back("u_21");
    solution_components_names.emplace_back("u_22");
    solution_components_names.emplace_back("u_23");
    solution_components_names.emplace_back("u_31");
    solution_components_names.emplace_back("u_32");
    solution_components_names.emplace_back("u_33");
    solution_components_names.emplace_back("v_11");
    solution_components_names.emplace_back("v_12");
    solution_components_names.emplace_back("v_13");
    solution_components_names.emplace_back("v_21");
    solution_components_names.emplace_back("v_22");
    solution_components_names.emplace_back("v_23");
    solution_components_names.emplace_back("v_31");
    solution_components_names.emplace_back("v_32");
    solution_components_names.emplace_back("v_33");
    
    { // DataOut block starts here, release memory
     DataOut<dim> data_out;
     data_out.attach_dof_handler(dof_handler);

     data_out.add_data_vector(locally_relevant_newton_solution, newton_update_components_names,
			     DataOut<dim>::type_dof_data);
     data_out.add_data_vector(local_solution, solution_components_names,
			     DataOut<dim>::type_dof_data);

     Vector<float> subdomain(triangulation.n_active_cells());
     for (unsigned int i = 0; i < subdomain.size(); ++i)
       subdomain(i) = triangulation.locally_owned_subdomain();
     data_out.add_data_vector(subdomain, "subdomain");

     data_out.build_patches();

     data_out.write_vtu_with_pvtu_record(
      dirc, "solution", iteration_loop, mpi_communicator, 2);
    } // DataOut block ends here, release memory

  }

  template class FemGL<3>;  
} // namespace FemGL_mpi

