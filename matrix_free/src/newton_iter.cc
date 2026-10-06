#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

// #include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/tools.h>

//#include <deal.II/multigrid/mg_coarse.h>
//#include <deal.II/multigrid/mg_matrix.h>
//#include <deal.II/multigrid/mg_smoother.h>
//#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
//#include <deal.II/multigrid/mg_transfer_matrix_free.h>
//#include <deal.II/multigrid/multigrid.h>
//#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include "roctracer/roctx.h"

#include "LinearGLProblem.h"
// #include "LinearGLOperator.h"
// #include "LocalLinearGLOperator.h"
#include "initial_UV_ConfFunction/bgSolution_U.h"
#include "initial_UV_ConfFunction/bgSolution_V.h"
#include "LinearGL_RightHandSide/LinearGLRHSCellOperator.h"
// #include "LaplaceDiagonalCellOperatorQuad.h"
// #include "preconditioner/BlockDiagonalJacobiPreconditioner.h"

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::newton_iter(const Number lambda)
  {
    roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter()");

    roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() rhs_0_residual");
    const Number current_iter_InitResidual = rhs.l2_norm();
    pcout << "rhs_0_residual = " << rhs_0_residual
          << std::endl;
    roctxRangePop();

    // initialize bg_newton_iter vector
    mf_data_ptr->initialize_dof_vector(bg_newton_iter);

    pcout << "lambda = " << lambda << std::endl;
    
    for (unsigned int i = 0; i < 100; ++i)
      {
	const Number alpha = std::pow(lambda, static_cast<Number>(i));

        pcout << "iteration i = " << i << ", alpha = " << alpha << std::endl;	  
	

	roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() copy bgSol into bg_newton_iter");
	// do I need a renit call for bg_newton_iter before copy?
	bg_newton_iter = bg_solution;
	roctxRangePop();	

	roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() bg_N_iter +=al * liSol");
	bg_newton_iter.add(alpha, linear_solution);
	roctxRangePop();

	roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() mf_data_ptr constrain bg_N_iter");	
        mf_data_ptr->set_constrained_values(Number(0.0), bg_newton_iter.block(0), 0);
        mf_data_ptr->set_constrained_values(Number(0.0), bg_newton_iter.block(1), 1);
	roctxRangePop();	
	
        {
	  roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() rhs_residual");
          /* ------------------------------------------------------------
           * using updted device vector bg_solution.
           * ------------------------------------------------------------
           */	  
          // mf_data_ptr->initialize_dof_vector(rhs);
          rhs = 0.0; // do I need this? yes!

          const Number K1    = 0.42072;
          const Number alpha = -0.4;
          const Number beta2 = 0.1;
       
          LinearGLRHSCellOperator<dim, fe_degree, Number>
	    rhs_operator(K1, alpha, beta2);

          /*  bg_solution.block(0) = u^0 ,bg_solution.block(1) = v^0 */
          roctxRangePush("ROCTX-RANGE:newtin_iter *mf_data_ptr::cell_loop() ");
          mf_data_ptr->cell_loop(rhs_operator, bg_newton_iter /*src*/, rhs /*dst*/);
          roctxRangePop();

          /*
           * Newton updates satisfies homogeneous Dirichlet conditions.
           * Therefore constrained RHS entries must be zero.
           */
          mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(0), 0);
          mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(1), 1);

	  roctxRangePop();
      
        } // residual calculation block ends here

	const Number linearSearch_trail_residual = rhs.l2_norm();

        pcout << "iteration i = " << i << ", alpha = " << alpha
              << ", linearSearch_trail_residual is "
	      << linearSearch_trail_residual
	      << std::endl;	  
		
        if (linearSearch_trail_residual < current_iter_InitResidual)
	  {
	    bg_solution = bg_newton_iter;
	    break;
	  }	  
	
      } // newton interation loop
         
    roctxRangePop();
 } // LinearGLProblem<...>::newton_iter() ends here

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
