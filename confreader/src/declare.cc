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

  // phrase declarations()
  void confreader::declare_parameters()
  {
    /*****************************************************
     * declrations of phyiscal parameters
     *****************************************************/
    prm.enter_subsection("physical parameters");
    {
      prm.declare_entry("t_reduced", "0.0", Patterns::Double(0), "reduced temperature");

      prm.declare_entry("AdGR diffuse length", "1.0e10", Patterns::Double(0), "AdGR diffuse parameter");
    }
    prm.leave_subsection();

    /*****************************************************
     * declrations of control parameters of iteration,
     * linear solver, AMG precondtioner and refinements
     *****************************************************/
    prm.enter_subsection("control parameters");
    {
      /* grid control paramters */
      prm.declare_entry("Number of adaptive refinements", "5", Patterns::Integer(0), "default value is 5, this is needed in run()");

      prm.declare_entry("Number of interations", "40", Patterns::Integer(0), "number of iterations run in one Cycle");

      prm.declare_entry("threshold of Cycle 0 refinement", "1.0e0", Patterns::Double(0), "Refine grid when the difference of residual.norm() smaller than this value in Cycle 0");            

      prm.declare_entry("threshold of refinement", "1.0e-3", Patterns::Double(0), "Refine grid when the difference of residual.norm() smaller than this value");      

      prm.declare_entry("converge accuracy", "5.0e-6", Patterns::Double(0), "terminate current Cycle if residual.norm() smaller than this value");      
      
      
      prm.declare_entry("adaptive refinment ratio", "0.3", Patterns::Double(0), "refiment parameter for refine_and_coarsen_fixed_number() call");

      prm.declare_entry("adaptive coarsen ratio", "0.0", Patterns::Double(0), "coarsen parameter for refine_and_coarsen_fixed_number() call");

      prm.declare_entry("Number of initial global refinments", "4", Patterns::Integer(0), "default value is 4, which is used in makegrid() call");

      
      /* iteration and linear solver control parameters */
      prm.declare_entry("Number of n-cycle in AdditionalData", "4", Patterns::Integer(0), "AMG precondtioner control parameter in solve() call");

      prm.declare_entry("tolrence of linear SolverControl", "7.0e-1", Patterns::Double(0), "default value is 7.0e-1 * system_rhs.norm()");

      prm.declare_entry("primary step length of dampped newton iteration", "0.83", Patterns::Double(0), "0.83 seems work well, not drop too fast");
                   
    }
    prm.leave_subsection();

    
    // prm.enter_subsection("switches");
    // {
    //   prm.declare_entry("flip_normal", "false", Patterns::Bool(), "flipping Robin face normal to inforward");

    // }
    // prm.leave_subsection();
  }
  
  /* ----------------------------------------------
   * ConfigurationReader defination ends at here
   * ----------------------------------------------
   */
  
} // namespace FemGL_mpi

