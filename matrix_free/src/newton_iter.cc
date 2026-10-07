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

    // roctxRangePush("ROCTX-RANGE:LinearGLProblem::newton_iter() rhs_0_residual");
    // roctxRangePop();
    
    for (unsigned int iter = 0; iter < 100; ++iter)
      {
	solve();
	line_search(lambda);

	pcout << iter << "th newton iteration with current_iter_InitResidual = "
	      << current_iter_InitResidual
	      << "\n ---------------------------------"
	      << std::endl;
	// pcout << "---------------------------------" << std::endl;

	
	if (current_iter_InitResidual <= convTol_newton_iter)
	  {
	    pcout << " current_iter_InitResidual <= convTol_newton_iter, newton iter ends "
		  << std::endl;
            break;
	  }
      } // newton interation loop
         
    roctxRangePop();
 } // LinearGLProblem<...>::newton_iter() ends here

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
