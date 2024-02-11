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
  double FemGL<dim>::mat_lhs_beta4(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
				   FullMatrix<double> &phi_u_i_q,FullMatrix<double> &phi_u_j_q,
		                   FullMatrix<double> &phi_v_i_q,FullMatrix<double> &phi_v_j_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak

                          /* default DoF indices pattern of multiplication 
                           * between phis is phi_phit_matrics_i_j_q(3,3);
                           */
    FullMatrix<double>    phi_u_phi_ut(3,3), phi_v_phi_vt(3,3),
                          phi_u_phi_vt(3,3), phi_v_phi_ut(3,3),
                          phi_ut_phi_u(3,3), phi_vt_phi_v(3,3),
                          phi_ut_phi_v(3,3), phi_vt_phi_u(3,3);

    FullMatrix<double>    u0_u0t(3,3), v0_v0t(3,3),
                          u0_v0t(3,3), v0_u0t(3,3),
                          v0t_v0(3,3), /*old_vt_old_u(3,3),*/
                          v0t_u0(3,3);

    FullMatrix<double>              phi_u_i_q_u0t(3,3), /*o*/
                                    phi_u_i_q_v0t(3,3), /*^*/
                                    phi_v_i_q_v0t(3,3),
                                    phi_v_i_q_u0t(3,3),
                                    /* ************** */
                                    phi_u_j_q_u0t(3,3),
                                    phi_u_j_q_v0t(3,3),      
                                    phi_v_j_q_u0t(3,3), /**/
                                    phi_v_j_q_v0t(3,3),
                                    /* ************** */

















      
                                    old_vt_phi_u_i_q(3,3),
                                    old_vt_phi_v_i_q(3,3);      


                                    /* ************** */
    FullMatrix<double>              phi_u_j_q_old_ut(3,3),
                                    phi_v_j_q_old_ut(3,3),
                                    /* ************** */      
                                    phi_ut_j_q_old_v(3,3),
                                    phi_vt_j_q_old_v(3,3);

    // Matrices for saving trace polynomils NO. I, II, II and IV
    // see note for understanding details
    FullMatrix<double>              poly_I, poly_II,
                                    poly_III, poly_IV;

          
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
    phi_u_phi_vt        = 0.0;
    phi_v_phi_ut        = 0.0;
    //phi_phit_matrics_i_j_q   = 0.0;
    phi_ut_phi_u        = 0.0;
    phi_vt_phi_v        = 0.0;
    phi_ut_phi_v        = 0.0;
    phi_vt_phi_u        = 0.0;        

    old_u_old_ut   = 0.0;
    old_v_old_vt   = 0.0;
    old_u_old_vt   = 0.0;
    old_vt_old_v   = 0.0;
    old_vt_old_u   = 0.0;    

    old_u_phi_ut_i_q    = 0.0;
    old_u_phi_ut_j_q    = 0.0;
    old_v_phi_vt_i_q    = 0.0;
    old_v_phi_vt_j_q    = 0.0;
    /* *********************/
    old_u_phi_vt_i_q    = 0.0;
    old_u_phi_vt_j_q    = 0.0;      
    old_v_phi_ut_i_q    = 0.0;
    old_v_phi_ut_j_q    = 0.0;
    /* *********************/
    old_vt_phi_u_i_q    = 0.0;
    old_vt_phi_v_i_q    = 0.0;    
    /* *********************/
    phi_ut_j_q_old_v    = 0.0;
    phi_vt_j_q_old_v    = 0.0;
    /* ************** */
    phi_u_j_q_old_ut    = 0.0;
    phi_v_j_q_old_ut    = 0.0;
      
    /* -------------------------------------------------------------
     * conduct matrices multiplacations: matrices multiplication
     * -------------------------------------------------------------
     */
    // C = A*B^T
    phi_u_i_q.mTmult(phi_u_phi_ut, phi_u_j_q);
    phi_v_i_q.mTmult(phi_v_phi_vt, phi_v_j_q);
    phi_u_i_q.mTmult(phi_u_phi_vt, phi_v_j_q);
    phi_v_i_q.mTmult(phi_v_phi_ut, phi_u_j_q);

    // C = A^T*B
    phi_u_i_q.Tmmult(phi_ut_phi_u, phi_u_j_q);
    phi_v_i_q.Tmmult(phi_vt_phi_v, phi_v_j_q);
    phi_u_i_q.Tmmult(phi_ut_phi_v, phi_v_j_q);
    phi_v_i_q.Tmmult(phi_vt_phi_u, phi_u_j_q);            

   
    old_solution_u.mTmult(old_u_old_ut, old_solution_u);
    old_solution_v.mTmult(old_v_old_vt, old_solution_v);
    old_solution_u.mTmult(old_u_old_vt, old_solution_v);
    old_solution_v.mTmult(old_v_old_ut, old_solution_u);    

    old_solution_v.Tmmult(old_vt_old_v, old_solution_v);
    old_solution_v.Tmmult(old_vt_old_u, old_solution_u);        
    
    //phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);

    old_solution_u.mTmult(old_u_phi_ut_i_q, phi_u_i_q);
    old_solution_u.mTmult(old_u_phi_ut_j_q, phi_u_j_q);
    old_solution_v.mTmult(old_v_phi_vt_i_q, phi_v_i_q);
    old_solution_v.mTmult(old_v_phi_vt_j_q, phi_v_j_q);
    /* ********************* */
    old_solution_u.mTmult(old_u_phi_vt_i_q, phi_v_i_q);
    old_solution_u.mTmult(old_u_phi_vt_j_q, phi_v_j_q);
    old_solution_v.mTmult(old_v_phi_ut_i_q, phi_u_i_q);
    old_solution_v.mTmult(old_v_phi_ut_j_q, phi_u_j_q);

    /* ********************* */
    old_solution_v.Tmmult(old_vt_phi_u_i_q, phi_u_i_q);
    old_solution_v.Tmmult(old_vt_phi_v_i_q, phi_v_i_q);


    /* ********************* */
    phi_u_j_q.Tmmult(phi_u_j_q_old_ut, old_solution_u);
    phi_v_j_q.Tmmult(phi_v_j_q_old_ut, old_solution_u);                

    /* ********************* */
    phi_u_j_q.Tmmult(phi_ut_j_q_old_v, old_solution_v);
    phi_v_j_q.Tmmult(phi_vt_j_q_old_v, old_solution_v);

    /* --------------------------------------------- */
    /*         build poly_I, II, III and IV          */
    /* --------------------------------------------- */

    poly_I = 0.0;

    old_u_phi_ut_i_q.mmult(poly_I, old_u_phi_ut_j_q, true);
    old_v_phi_ut_i_q.mmult(poly_I, old_v_phi_ut_j_q, true);
    old_v_phi_ut_i_q.mmult(poly_I, old_u_phi_vt_j_q, true);

    // -old_u_phi_ut_i_q    
    FullMatrix<double> n_old_u_phi_ut_i_q(IdentityMatrix(3));
    n_old_u_phi_ut_i_q.copy_from(old_u_phi_ut_i_q);
    n_old_u_phi_ut_i_q *= -1.0; 
    n_old_u_phi_ut_i_q.mmult(poly_I, old_u_phi_ut_j_q, true);    

    /* ------------------------- */
    
    poly_II = 0.0;

    // -old_v_phi_vt_i_q
    FullMatrix<double> n_old_v_phi_vt_i_q(IdentityMatrix(3));
    n_old_v_phi_vt_i_q.copy_from(old_v_phi_vt_i_q);
    n_old_v_phi_vt_i_q *= -1.0; 
    n_old_v_phi_vt_i_q.mmult(poly_II, old_u_phi_ut_j_q, true);
    
    old_u_phi_vt_i_q.mmult(poly_II, old_v_phi_ut_j_q, true);
    old_u_phi_vt_i_q.mmult(poly_II, old_u_phi_vt_j_q, true);
    old_v_phi_vt_i_q.mmult(poly_II, old_v_phi_vt_j_q, true);

    /* ------------------------- */

    poly_III = 0.0;

    old_u_phi_ut_i_q.mmult(poly_III, phi_u_j_q_old_ut, true);
    old_u_old_vt.mmult(poly_III, phi_u_phi_vt, true);
    old_u_old_vt.mmult(poly_III, phi_v_phi_ut, true);    

    n_old_v_phi_vt_i_q.mmult(poly_III, phi_u_j_q_old_ut, true);

    old_u_old_ut.mmult(poly_III, phi_u_phi_ut, true);

    // -old_u_old_ut
    FullMatrix<double> n_old_u_old_ut(IdentityMatrix(3));
    n_old_u_old_ut.copy_from(old_u_old_ut);
    n_old_u_old_ut *= -1.0; 
    n_old_u_old_ut.mmult(poly_III, phi_v_phi_vt, true);

    old_v_phi_ut_i_q.mmult(poly_III, phi_v_j_q_old_ut, true);
    old_u_phi_vt_i_q.mmult(poly_III, phi_v_j_q_old_ut, true);
    
    /* ------------------------- */

    poly_IV = 0.0;

    old_vt_old_v.mmult(poly_IV, phi_ut_phi_u, true);
    old_v_old_ut.mmult(poly_IV, phi_u_phi_vt, true);
    old_v_old_ut.mmult(poly_IV, phi_v_phi_ut, true);
    old_vt_old_u.mmult(poly_IV, phi_vt_phi_u, true);

    // -old_vt_phi_u_i_q
    FullMatrix<double> n_old_vt_phi_u_i_q(IdentityMatrix(3));
    n_old_vt_phi_u_i_q.copy_from(old_vt_phi_u_i_q);
    n_old_vt_phi_u_i_q *= -1.0; 
    n_old_vt_phi_u_i_q.mmult(poly_IV, phi_ut_j_q_old_v, true);

    old_vt_phi_v_i_q.mmult(poly_IV, phi_vt_j_q_old_v, true);
    
    // -old_vt_old_u
    FullMatrix<double> n_old_vt_old_u(IdentityMatrix(3));
    n_old_vt_old_u.copy_from(old_vt_old_u);
    n_old_vt_old_u *= -1.0; 
    n_old_vt_old_u.mmult(poly_IV, phi_ut_phi_v, true);

    old_vt_old_v.mmult(poly_IV, phi_vt_phi_v, true);







    
    
    return  (poly_I.trace() + poly_II.trace() + poly_III.trace() + poly_IV.trace());
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

