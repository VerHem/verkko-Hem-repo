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

#include "confreader.h"

namespace FemGL_mpi
{
  using namespace dealii;

  /* ------------------------------------------------
   * configuration file reader for reading parameters 
   * from configuration file
   * ------------------------------------------------
   */

  // ParameterReader constructor
  confreader::confreader(ParameterHandler &configuration)
    : prm(configuration)
  {}

  // read_parameters function
  void confreader::read_parameters(const std::string &configuration_file)
  {
    declare_parameters();

    prm.parse_input(configuration_file);
  }
  
  // phrase declarations()
  void confreader::declare_parameters()
  {
    prm.enter_subsection("physical parameters");
    {
      prm.declare_entry("t_reduced", "0.0", Patterns::Double(0), "reduced temperature");

      prm.declare_entry("b1_diffuse", "1.0e10", Patterns::Double(0), "u component diffuse parameter");

      prm.declare_entry("b2_diffuse", "1.0e10", Patterns::Double(0), "v component diffuse parameter");      
    }
    prm.leave_subsection();

    prm.enter_subsection("switches");
    {
      prm.declare_entry("flip_normal", "false", Patterns::Bool(), "flipping Robin face normal to inforward");

    }
    prm.leave_subsection();
  }
  
  /* ----------------------------------------------
   * ConfigurationReader defination ends at here
   * ----------------------------------------------
   */
  
} // namespace FemGL_mpi

