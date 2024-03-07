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
  double FemGL<dim>::mat_lhs_K1(std::vector<FullMatrix<double>> &grad_phi_u_i_q,
				std::vector<FullMatrix<double>> &grad_phi_v_i_q,
				std::vector<FullMatrix<double>> &grad_phi_u_j_q,
				std::vector<FullMatrix<double>> &grad_phi_v_j_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak
    /* --------------------------------------------------------------------------------
     * matrics for free energy terms: gradient terms, alpha-term, beta-term,
     * they are products of phi tensors or u/v tensors.
     * --------------------------------------------------------------------------------
     */
    FullMatrix<double>              K1grad_matrics_sum_i_j_q(3,3);
    
    /*---------------------------------------------------------------------------------------------*/
    /* grad_phi^u_i_q, grad_phi^v_j_q matrices have beeen cooked up in other functions             */
    /*---------------------------------------------------------------------------------------------*/

    /* --------------------------------------------------
     * conduct matrices multiplacations
     * --------------------------------------------------
     */
    K1grad_matrics_sum_i_j_q = 0.0; // initialize 

    for (unsigned int k = 0; k < dim; ++k)
      {
	/* FullMatrix::mTmult does C+=A.BT if the adding boolen is true
         */
        grad_phi_u_i_q[k].mTmult(K1grad_matrics_sum_i_j_q, grad_phi_u_j_q[k], true); 
	grad_phi_v_i_q[k].mTmult(K1grad_matrics_sum_i_j_q, grad_phi_v_j_q[k], true);
      }
    
    return  K1grad_matrics_sum_i_j_q.trace();
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

