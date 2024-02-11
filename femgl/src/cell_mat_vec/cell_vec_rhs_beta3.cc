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
  double FemGL<dim>::vec_rhs_beta3(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
				   FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak

    FullMatrix<double>    old_u_old_vt(3,3),
                          old_ut_old_u(3,3), old_ut_old_v(3,3),
                          old_vt_old_v(3,3);

    FullMatrix<double>             old_ut_phi_u_i_q(3,3),
                                   old_ut_phi_v_i_q(3,3),
                                   old_vt_phi_u_i_q(3,3),
                                   old_vt_phi_v_i_q(3,3),      
                                    /* ************** */
                                   old_v_phi_ut_i_q(3,3),
                                    /* ************** */
                                   phi_ut_i_q_old_v(3.3),
                                   phi_vt_i_q_old_u(3.3);

    FullMatrix<double>   poly_I = 0.0;
          
    /*-------------------------------------------------------------*/
    /* phi^u, phi^v matrices have been cooked up in other fuctions */
    /* old_u old_v matrices have been cooked up in other functions */
    /*-------------------------------------------------------------*/

    /* -------------------------------------------------------------
     * conduct matrices multiplacations: initlize matrices
     * -------------------------------------------------------------
     */
    
    old_u_old_vt   = 0.0;
    /* *********************/
    old_ut_old_u    = 0.0;
    old_ut_old_v    = 0.0;
    old_vt_old_v    = 0.0;    
    /* *********************/

    
    old_ut_phi_u_i_q    = 0.0;
    old_ut_phi_v_i_q    = 0.0;
    old_vt_phi_u_i_q    = 0.0;
    old_vt_phi_v_i_q    = 0.0;    
    /* *********************/    
    old_v_phi_ut_i_q    = 0.0;
    /* *********************/        
    phi_ut_i_q_old_v    = 0.0;
    phi_vt_i_q_old_u    = 0.0;    
    
    /* -------------------------------------------------------------
     * conduct matrices multiplacations: matrices multiplication
     * -------------------------------------------------------------
     */
   
    old_solution_u.mTmult(old_u_old_vt, old_solution_v);
    /* *********************/            
    old_solution_u.Tmmult(old_ut_old_u, old_solution_u);
    old_solution_u.Tmmult(old_ut_old_v, old_solution_v);
    old_solution_v.Tmmult(old_vt_old_v, old_solution_v);            
    
    //phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);

    old_solution_u.Tmmult(old_ut_phi_u_i_q, phi_u_i_q);
    old_solution_u.Tmmult(old_ut_phi_v_i_q, phi_v_i_q);    
    old_solution_v.Tmmult(old_vt_phi_u_i_q, phi_u_i_q);
    old_solution_v.Tmmult(old_vt_phi_v_i_q, phi_v_i_q);    
    /* ********************* */
    old_solution_v.mTmult(old_v_phi_ut_i_q, phi_u_i_q);
    /* ********************* */    
    phi_u_i_q.Tmmult(phi_ut_i_q_old_v, old_solution_v);
    phi_v_i_q.Tmmult(phi_vt_i_q_old_u, old_solution_u);


    /* ********  construct the matrices summation ******** */

    old_ut_old_u.mmult(poly_I, old_ut_phi_u_i_q, true);
    old_u_old_vt.mmult(poly_I, old_v_phi_ut_i_q, true);
    old_ut_old_v.mmult(poly_I, phi_ut_i_q_old_v, true);

    // -old_ut_old_v
    /*FullMatrix<double> n_old_ut_old_v{old_ut_old_v};*/
    FullMatrix<double> n_old_ut_old_v = old_ut_old_v;    
    n_old_ut_old_v *= -1.0;
    n_old_ut_old_v.mmult(poly_I, old_vt_phi_u_i_q, true);
    n_old_ut_old_v.mmult(poly_I, phi_vt_i_q_old_u, true);

    old_ut_old_v.mmult(poly_I, old_ut_phi_v_i_q, true);
    old_ut_old_u.mmult(poly_I, old_vt_phi_v_i_q, true);
    old_vt_old_v.mmult(poly_I, old_vt_phi_v_i_q, true);        

    /* **  matrices summation construction ends here ***** */
    
    return  poly_I.trace();
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

