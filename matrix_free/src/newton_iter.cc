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
#include "bgSolution_U.h"
#include "bgSolution_V.h"
#include "LinearGLRHSCellOperator.h"
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
    const Number rhs_0_residual = rhs.l2_norm();
    roctxRangePop();
    
    for (unsigned int i = 0; i < 100; ++i)
      {
	const Number alpha = std::pow(lambda, static_cast<Number>(i));

	roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() bgSol+=al * liSol");
	bg_solution.add(alpha, linear_solution);
	roctxRangePop();

        {
	  roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() rhs_residual");
          /* ------------------------------------------------------------
           * using updted device vector bg_solution.
           * ------------------------------------------------------------
           */	  
          // mf_data_ptr->initialize_dof_vector(rhs);
          // rhs = 0.0; // do I need this?

          const Number K1    = 0.42072;
          const Number alpha = -0.4;
          const Number beta2 = 0.1;
       
          LinearGLRHSCellOperator<dim, fe_degree, Number>
	    rhs_operator(K1, alpha, beta2);

          /*  bg_solution.block(0) = u^0 ,bg_solution.block(1) = v^0 */
          roctxRangePush("ROCTX-RANGE:newtin_iter *mf_data_ptr::cell_loop() ");
          mf_data_ptr->cell_loop(rhs_operator, bg_solution /*src*/, rhs /*dst*/);
          roctxRangePop();

          /*
           * Newton updates satisfies homogeneous Dirichlet conditions.
           * Therefore constrained RHS entries must be zero.
           */
          mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(0), 0);
          mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(1), 1);

	  roctxRangePop();
      
        } // residual calculation block ends here
	
        if (rhs.l2_norm() < rhs_0_residual) break;
	
      } // newton interation loop
         
    roctxRangePop();
 } // LinearGLProblem<...>::newton_iter() ends here

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
