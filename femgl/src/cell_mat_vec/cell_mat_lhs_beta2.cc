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

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  double FemGL<dim>::mat_lhs_beta2(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
				   FullMatrix<double> &phi_u_i_q,FullMatrix<double> &phi_u_j_q,
		                   FullMatrix<double> &phi_v_i_q,FullMatrix<double> &phi_v_j_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak

    FullMatrix<double>    phi_u_phi_ut(3,3),phi_v_phi_vt(3,3),
                          phi_phit_matrics_i_j_q(3,3);

    FullMatrix<double>    old_u_old_ut(3,3), old_v_old_vt(3,3);

    FullMatrix<double>              old_u_phi_ut_i_q(3,3),
                                    old_u_phi_ut_j_q(3,3),
                                    old_v_phi_vt_i_q(3,3),
                                    old_v_phi_vt_j_q(3,3);
    
    /*-------------------------------------------------------------*/
    /* phi^u, phi^v matrices have been cooked up in other fuctions */
    /* old_u old_v matrices have been cooked up in other functions */
    /*-------------------------------------------------------------*/

    /* -------------------------------------------------------------
     * conduct matrices multiplacations: initlize matrices
     * -------------------------------------------------------------
     */
    phi_u_phi_ut        = 0.0;
    phi_v_phi_vt        = 0.0;
    phi_phit_matrics_i_j_q   = 0.0;

    old_u_old_ut   = 0.0;
    old_v_old_vt   = 0.0;

    old_u_phi_ut_i_q    = 0.0;
    old_u_phi_ut_j_q    = 0.0;
    old_v_phi_vt_i_q    = 0.0;
    old_v_phi_vt_j_q    = 0.0;

    /* -------------------------------------------------------------
     * conduct matrices multiplacations: matrices multiplication
     * -------------------------------------------------------------
     */
   
    old_solution_u.mTmult(old_u_old_ut, old_solution_u);
    old_solution_v.mTmult(old_v_old_vt, old_solution_v);

    phi_u_i_q.mTmult(phi_u_phi_ut, phi_u_j_q);
    phi_v_i_q.mTmult(phi_v_phi_vt, phi_v_j_q);
    
    phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);

    old_solution_u.mTmult(old_u_phi_ut_i_q, phi_u_i_q);
    old_solution_u.mTmult(old_u_phi_ut_j_q, phi_u_j_q);
    old_solution_v.mTmult(old_v_phi_vt_i_q, phi_v_i_q);
    old_solution_v.mTmult(old_v_phi_vt_j_q, phi_v_j_q);

    
    return  ((old_u_old_ut.trace() + old_v_old_vt.trace()) * phi_phit_matrics_i_j_q.trace()
	     + 2.0 * ((old_u_phi_ut_i_q.trace() + old_v_phi_vt_i_q.trace())
		      * (old_u_phi_ut_j_q.trace() + old_v_phi_vt_j_q.trace())));
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

