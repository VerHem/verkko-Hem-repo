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
#include "matep.h"
#include "BinA.h"

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  double FemGL<dim>::mat_lhs_beta3(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
				   FullMatrix<double> &phi_u_i_q,FullMatrix<double> &phi_u_j_q,
		                   FullMatrix<double> &phi_v_i_q,FullMatrix<double> &phi_v_j_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak

                          /* default DoF indices pattern of multiplication 
                           * between phis is phi_phit_matrics_i_j_q(3,3);
                           */
    FullMatrix<double>    phi_u_phi_ut(3,3) /* *** */, phi_v_phi_vt(3,3), /*ooo*/
                          phi_u_phi_vt(3,3) /*~~~*/, phi_v_phi_ut(3,3), /*->->->*/
                          phi_ut_phi_u(3,3) /*888*/, phi_vt_phi_v(3,3), /*aaa*/ // aaa is a bug
                          phi_ut_phi_v(3,3) /*\\\*/, phi_vt_phi_u(3,3); /*///*/

    FullMatrix<double>    u0_u0t(3,3) /*(..)*/, //v0_v0t(3,3), /*WTF is this*/ 
                          u0_v0t(3,3) /*^.*/, v0_u0t(3,3) /*LTLT*/,
                          v0t_v0(3,3) /*sss*/, 
                          v0t_u0(3,3); /*o+*/

    FullMatrix<double>              u0_phi_ut_i_q(3,3), /*o*/
                                    u0_phi_ut_j_q(3,3), /*^*/
                                    v0_phi_vt_i_q(3,3), /*o.*/
                                    v0_phi_vt_j_q(3,3), /* * */
                                    /* ************** */
                                    u0_phi_vt_i_q(3,3), /*s.*/
                                    u0_phi_vt_j_q(3,3), /*h*/     
                                    v0_phi_ut_i_q(3,3), /*s*/
                                    v0_phi_ut_j_q(3,3), /*^v*/
                                    /* ************** */
                                    v0t_phi_u_i_q(3,3), /*xxx*/
                                    v0t_phi_v_i_q(3,3); /*---*/     


                                    /* ************** */
    FullMatrix<double>              phi_u_j_q_u0t(3,3), /*ox*/
                                    phi_v_j_q_u0t(3,3), /*^^^*/
                                    /* ************** */      
                                    phi_ut_j_q_v0(3,3), /*+++*/
                                    phi_vt_j_q_v0(3,3); /*--- ---*/

    FullMatrix<double>              mm1(3,3), mm2(3,3),
                                    mm3(3,3), mm4(3,3),
                                    mm5(3,3), mm6(3,3); // mmx matrix is for handling "-" minus matrix
                                    //ms1(3,3), ms2(3,3);
    
    // Matrices for saving trace polynomils NO. I, II, II and IV
    // see note for understanding details
    FullMatrix<double>              poly_I(3,3), poly_II(3,3),
                                    poly_III(3,3), poly_IV(3,3);

          
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

    u0_u0t   = 0.0;
    u0_v0t   = 0.0;
    v0_u0t   = 0.0;
    v0t_v0   = 0.0;
    v0t_u0   = 0.0;    

    u0_phi_ut_i_q    = 0.0;
    u0_phi_ut_j_q    = 0.0;
    v0_phi_vt_i_q    = 0.0;
    v0_phi_vt_j_q    = 0.0;
    /* *********************/
    u0_phi_vt_i_q    = 0.0;
    u0_phi_vt_j_q    = 0.0;      
    v0_phi_ut_i_q    = 0.0;
    v0_phi_ut_j_q    = 0.0;
    /* *********************/
    v0t_phi_u_i_q    = 0.0;
    v0t_phi_v_i_q    = 0.0;    
    /* *********************/

    phi_u_j_q_u0t    = 0.0;
    phi_v_j_q_u0t    = 0.0;
    /* ************** */
    phi_ut_j_q_v0    = 0.0;
    phi_vt_j_q_v0    = 0.0;
    
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

   
    old_solution_u.mTmult(u0_u0t, old_solution_u);
    old_solution_u.mTmult(u0_v0t, old_solution_v);
    old_solution_v.mTmult(v0_u0t, old_solution_u);    
    old_solution_v.Tmmult(v0t_v0, old_solution_v);
    old_solution_v.Tmmult(v0t_u0, old_solution_u);        
    
    //phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);

    old_solution_u.mTmult(u0_phi_ut_i_q, phi_u_i_q);
    old_solution_u.mTmult(u0_phi_ut_j_q, phi_u_j_q);
    old_solution_v.mTmult(v0_phi_vt_i_q, phi_v_i_q);
    old_solution_v.mTmult(v0_phi_vt_j_q, phi_v_j_q);
    /* ********************* */
    old_solution_u.mTmult(u0_phi_vt_i_q, phi_v_i_q);
    old_solution_u.mTmult(u0_phi_vt_j_q, phi_v_j_q);
    old_solution_v.mTmult(v0_phi_ut_i_q, phi_u_i_q);
    old_solution_v.mTmult(v0_phi_ut_j_q, phi_u_j_q);

    /* ********************* */
    old_solution_v.Tmmult(v0t_phi_u_i_q, phi_u_i_q);
    old_solution_v.Tmmult(v0t_phi_v_i_q, phi_v_i_q);


    /* ********************* */
    phi_u_j_q.mTmult(phi_u_j_q_u0t, old_solution_u);
    phi_v_j_q.mTmult(phi_v_j_q_u0t, old_solution_u);                

    /* ********************* */
    phi_u_j_q.Tmmult(phi_ut_j_q_v0, old_solution_v);
    phi_v_j_q.Tmmult(phi_vt_j_q_v0, old_solution_v);

    /* ----------------------------------------------*/
    /*           construct ms_x matrics              */
    /*   this method is used in beta4, beta5 funcs   */
    /* ----------------------------------------------*/
    
    /* --------------------------------------------- */
    /*         build poly_I, II, III and IV          */
    /* --------------------------------------------- */

    poly_I = 0.0;
    mm1    = 0.0; // o.* term 

    u0_phi_ut_i_q.mmult(poly_I, u0_phi_ut_j_q, true);
    v0_phi_ut_i_q.mmult(poly_I, v0_phi_ut_j_q, true);
    v0_phi_ut_i_q.mmult(poly_I, u0_phi_vt_j_q, true);

    u0_phi_ut_i_q.mmult(mm1, v0_phi_vt_j_q);
    poly_I.add(-1.0, mm1);    

    /* ------------------------- */
    
    poly_II = 0.0;
    mm2     = 0.0; // o. . ^ term

    v0_phi_vt_i_q.mmult(mm2, u0_phi_ut_j_q);
    poly_II.add(-1.0, mm2);
    
    u0_phi_vt_i_q.mmult(poly_II, v0_phi_ut_j_q, true);
    u0_phi_vt_i_q.mmult(poly_II, u0_phi_vt_j_q, true);
    v0_phi_vt_i_q.mmult(poly_II, v0_phi_vt_j_q, true);

    /* ------------------------- */

    poly_III = 0.0;
    mm3      = 0.0;
    mm4      = 0.0;

    u0_phi_ut_i_q.mmult(poly_III, phi_u_j_q_u0t, true);
    u0_v0t.mmult(poly_III, phi_u_phi_vt, true);
    u0_v0t.mmult(poly_III, phi_v_phi_ut, true);    

    v0_phi_vt_i_q.mmult(mm3, phi_u_j_q_u0t, true);
    poly_III.add(-1.0, mm3);

    u0_u0t.mmult(poly_III, phi_u_phi_ut, true);

    u0_u0t.mmult(mm4, phi_v_phi_vt);
    poly_III.add(-1.0, mm4);
    
    v0_phi_ut_i_q.mmult(poly_III, phi_v_j_q_u0t, true);
    u0_phi_vt_i_q.mmult(poly_III, phi_v_j_q_u0t, true);
    
    /* ------------------------- */

    poly_IV = 0.0;
    mm5     = 0.0;
    mm6     = 0.0;    

    v0t_v0.mmult(poly_IV, phi_ut_phi_u, true);
    v0_u0t.mmult(poly_IV, phi_u_phi_vt, true);
    v0_u0t.mmult(poly_IV, phi_v_phi_ut, true);
    v0t_u0.mmult(poly_IV, phi_vt_phi_u, true);

    v0t_phi_u_i_q.mmult(mm5, phi_ut_j_q_v0);
    poly_IV.add(-1.0, mm5);

    v0t_phi_v_i_q.mmult(poly_IV, phi_vt_j_q_v0, true);

    v0t_u0.mmult(mm6, phi_ut_phi_v);
    poly_IV.add(-1.0, mm6);

    v0t_v0.mmult(poly_IV, phi_vt_phi_v, true);







    
    
    return  (poly_I.trace() + poly_II.trace() + poly_III.trace() + poly_IV.trace());
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

