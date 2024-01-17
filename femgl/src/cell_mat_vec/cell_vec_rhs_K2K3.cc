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
  double FemGL<dim>::vec_rhs_K2K3(std::vector<FullMatrix<double>> &grad_phi_u_i_q,
				  std::vector<FullMatrix<double>> &grad_phi_v_i_q,
				  std::vector<FullMatrix<double>> &grad_old_u_q,
				  std::vector<FullMatrix<double>> &grad_old_v_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak
    /* --------------------------------------------------------------------------------
     * matrics for free energy terms: gradient terms, alpha-term, beta-term,
     * they are products of phi tensors or u/v tensors.
     * --------------------------------------------------------------------------------
     */

    /* declear a FullMatrix<> to use extract_submatrix_from() funtion to hold part_x_phi_u/v_x */
    FullMatrix<double> mat_partial_x_phi_x(3,3),
                       mat_partial_x_s_x(3,3);

    /* container of all grad_phi_u/v_i_q and grad_old_u/v_q */
    std::vector<std::vector<FullMatrix<double>>> container_grad{grad_phi_u_i_q, grad_phi_v_i_q, grad_old_u_q, grad_old_v_q};
    
    /*deal.ii::Vector<> doesn't need to be initialized in the way,which FullMatrix<> does i.e., '=0.0'.*/ 
    /*dealii::Vector<> v(3) will do the job, in which a 3-components Vector is defines as 0-vector.    */ 
    
    Vector<double>  partial_x_phi_u_x_i(3);  //Vector of part_x_phi_u^mu_x_i, with i is DoF index, x is spatial-prbital index and they contract
    Vector<double>  partial_x_phi_v_x_i(3);  //Vector of part_x_phi_u^mu_x_i, with i is DoF index, x is spatial-prbital index and they contract

    Vector<double>  partial_x_old_u_x(3);  //Vector of part_x_u^mu_x, with mu is spin-index, x is spatial-prbital index and they contract
    Vector<double>  partial_x_old_v_x(3);  //Vector of part_x_u^mu_x, with mu is spin-index, x is spatial-prbital index and they contract

    /* container of all partial_x_phi_u/v_i_q and grad_old_u/v_q */
    std::vector<Vector<double>> container_px_x{partial_x_phi_u_x_i, partial_x_phi_v_x_i, partial_x_old_u_x, partial_x_old_v_x};
    
    /*---------------------------------------------------------------------------------------------*/
    /* grad_phi^u_i_q, grad_phi^v_j_q matrices have beeen cooked up in other functions             */
    /*---------------------------------------------------------------------------------------------*/

    /* --------------------------------------------------
     * conduct matrices multiplacations
     * --------------------------------------------------
     */

    /* >>> construct part_x_phi_u^mu_x_i, part_x_phi_v^mu_x_i vectors <<< */
    /* >>>   construct part_x_pld_u^mu_x, part_x_old_v^mu_x vectors <<<   */    

    std::vector<unsigned int> row_index_set{0,1,2};
    for (unsigned int w = 0; w < 2; ++w)
      {
        mat_partial_x_phi_x = 0.0;
	mat_partial_x_s_x   = 0.0;

	/* the following loop extracts partial_x_phi^u/v_x into xth column of mat_p_x_phi_x   */
	/* z is the index of FullMatrix elements in grad_xxx, it corresponds to spatial index */
	for (unsigned int z = 0; z < 3; ++z)
	  {
	    std::vector<unsigned int> column_index_set{z};
  	    mat_partial_x_phi_x.extract_submatrix_from(container_grad[w][z],
				                       row_index_set,
				                       column_index_set);

  	    mat_partial_x_s_x.extract_submatrix_from(container_grad[w+2][z],
				                       row_index_set,
				                       column_index_set);
	    
	  }

	/* FullMatrix<>::add_col() A(1...n,i) += s*A(1...n,j) + t*A(1...n,k). Multiple addition of columns of this. */
	/* This operation adds up all p_x_phi_x in to the 0th col of mat_p_x_phi_x matrix.                          */
	mat_partial_x_phi_x.add_col(0 /*i=0th col*/,
				    1 /*s*/, 1 /* j=1st col */,
				    1 /*t*/, 2 /* k=2nd col*/);

	mat_partial_x_s_x.add_col(0 /*i=0th col*/,
				    1 /*s*/, 1 /* j=1st col */,
				    1 /*t*/, 2 /* k=2nd col*/);
	
	/* FullMatrix::begin() returns iterator starting at the first entry of row r. */
	container_px_x[w][0] = *mat_partial_x_phi_x.begin(0);
	container_px_x[w][1] = *mat_partial_x_phi_x.begin(1);
	container_px_x[w][2] = *mat_partial_x_phi_x.begin(2);

	container_px_x[w+2][0] = *mat_partial_x_s_x.begin(0);
	container_px_x[w+2][1] = *mat_partial_x_s_x.begin(1);
	container_px_x[w+2][2] = *mat_partial_x_s_x.begin(2);	
	
      }

        
    /* p_x_phi_u^mu_x dot p_x_old_u^mu_x + p_x_phi_v^mu_x dot p_x_old_v^mu_x */
    return  (container_px_x[0] * container_px_x[2]) + (container_px_x[1] * container_px_x[3]);
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

