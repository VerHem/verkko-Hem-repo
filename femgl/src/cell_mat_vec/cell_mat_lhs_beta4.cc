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
                          phi_u_phi_vt(3,3), phi_v_phi_ut(3,3);

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
                                    u0_phi_ut_j_q(3,3),
                                    u0_phi_vt_j_q(3,3),
                                    v0_phi_ut_j_q(3,3),
                                    v0_phi_vt_j_q(3,3);      



    FullMatrix<double>              ms1(3,3), ms2(3,3),
                                    ms3(3,3), ms4(3,3),
                                    ms5(3,3), ms6(3,3);
    
    // Matrices for saving trace polynomils NO. I, II, II and IV
    // see note for understanding details
    FullMatrix<double>              poly_I, poly_II,
                                    poly_III;

          
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
    
    u0_u0t   = 0.0;
    u0_v0t   = 0.0;
    v0_v0t   = 0.0;    
    v0_u0t   = 0.0;


    phi_u_i_q_u0t = 0.0;
    phi_u_i_q_v0t = 0.0;
    phi_v_i_q_v0t = 0.0;
    phi_v_i_q_u0t = 0.0;

    phi_u_j_q_u0t = 0.0;
    phi_u_j_q_v0t = 0.0;
    phi_v_j_q_u0t = 0.0;
    phi_v_j_q_v0t = 0.0;

    u0_phi_ut_j_q = 0.0;
    u0_phi_vt_j_q = 0.0;
    v0_phi_ut_j_q = 0.0;
    v0_phi_vt_j_q = 0.0;     
    
    
    /* -------------------------------------------------------------
     * conduct matrices multiplacations: matrices multiplication
     * -------------------------------------------------------------
     */
    // C = A*B^T
    phi_u_i_q.mTmult(phi_u_phi_ut, phi_u_j_q);
    phi_v_i_q.mTmult(phi_v_phi_vt, phi_v_j_q);
    phi_u_i_q.mTmult(phi_u_phi_vt, phi_v_j_q);
    phi_v_i_q.mTmult(phi_v_phi_ut, phi_u_j_q);
    
    /* ********************* */
    old_solution_u.mTmult(u0_u0t, old_solution_u);
    old_solution_u.mTmult(u0_v0t, old_solution_v);
    old_solution_v.mTmult(v0_v0t, old_solution_v);
    old_solution_v.mTmult(v0_u0t, old_solution_u);        
        
    //phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);


    /* ********************* */
    phi_u_i_q.mTmult(phi_u_i_q_u0t, old_solution_u);
    phi_u_i_q.mTmult(phi_u_i_q_v0t, old_solution_v);
    phi_v_i_q.mTmult(phi_v_i_q_v0t, old_solution_v);
    phi_v_i_q.mTmult(phi_v_i_q_u0t, old_solution_u);    

    phi_u_j_q.mTmult(phi_u_j_q_u0t, old_solution_u);
    phi_u_j_q.mTmult(phi_u_j_q_v0t, old_solution_v);
    phi_v_j_q.mTmult(phi_v_j_q_u0t, old_solution_u);
    phi_v_j_q.mTmult(phi_v_j_q_v0t, old_solution_v);

    old_solution_u.mTmult(u0_phi_ut_j_q, phi_u_j_q);
    old_solution_u.mTmult(u0_phi_vt_j_q, phi_v_j_q);
    old_solution_v.mTmult(v0_phi_ut_j_q, phi_u_j_q);
    old_solution_v.mTmult(v0_phi_vt_j_q, phi_v_j_q);                
    
    
    /* --------------------------------------------- */
    /*     build ms1, ms2, ms3, ms4, ms5, ms6        */
    /* --------------------------------------------- */

    ms1 = 0.0;
    ms1.add(1.,phi_u_i_q_u0t);
    ms1.add(1.,phi_v_i_q_v0t);    

    ms2 = 0.0;
    ms2.add(1.,phi_u_j_q_u0t);
    ms2.add(1.,phi_v_j_q_v0t);    

    ms3 = 0.0;
    ms3.add(1.,phi_u_i_q_v0t);
    ms3.add(-1.,phi_v_i_q_u0t);    

    ms4 = 0.0;
    ms4.add(1.,phi_v_j_q_u0t);
    ms4.add(-1.,phi_u_j_q_v0t);    

    ms5 = 0.0;
    ms5.add(1.,phi_v_phi_ut);
    ms5.add(-1.,phi_u_phi_vt);    

    ms6 = 0.0;
    ms6.add(1.,phi_u_phi_ut);
    ms6.add(1.,phi_v_phi_vt);    
    

    
    /* --------------------------------------------- */
    /*              build poly_I, II, III            */
    /* --------------------------------------------- */

    poly_I = 0.0;

    ms1.mmult(poly_I, ms2, true);
    ms3.mmult(poly_I, ms4, true);    

    /* ------------------------- */

    poly_II = 0.0;

    ms1.mmult(poly_II, u0_phi_ut_j_q, true);
    ms5.mmult(poly_II, u0_v0t, true);
    ms6.mmult(poly_II, u0_u0t, true);

    FullMatrix<double> n_ms3 = ms3;
    n_ms3 *= -1.0;
    n_ms3.mmult(poly_II, u0_phi_vt_j_q, true);            

    /* ------------------------- */

    poly_III = 0.0;

    ms3.mmult(poly_III, v0_phi_ut_j_q, true);

    FullMatrix<double> n_ms5 = ms5;
    n_ms5 *= -1.0;
    n_ms5.mmult(poly_III, v0_u0t, true);        

    ms6.mmult(poly_III, v0_v0t, true);
    ms1.mmult(poly_III, v0_phi_vt_j_q, true);        



    
    return  (poly_I.trace() + poly_II.trace() + poly_III.trace());
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

