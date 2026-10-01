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
#include <deal.II/lac/solver_gmres.h>

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
#include "LinearGLOperator.h"
// #include "LocalLinearGLOperator.h"
// #include "bgSolution_U.h"
// #include "bgSolution_V.h"
// #include "LinearGLRHSCellOperator.h"
// #include "LaplaceDiagonalCellOperatorQuad.h"
// #include "preconditioner/BlockDiagonalJacobiPreconditioner.h"

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::solve()
  {
    roctxRangePush("ROCTX-RANGE:LinearGLProblem::solve()");
    // LinearGLOperator<dim, fe_degree, Number> LinearGL_Operator(mf_data);
    LinearGLOperator<dim, fe_degree, Number>
      LinearGL_Operator(mf_data_ptr,
                        DoFHandler_U, DoFHandler_V,
                        constraints_U, constraints_V,
                        bg_solution /*background_U_V_sol*/);

    mf_data_ptr->initialize_dof_vector(linear_solution);

    {
      dealii::Timer t(tria.get_mpi_communicator());
      LinearGL_Operator.vmult(linear_solution, rhs);
      const double time          = t.wall_time();
      const double dofs_per_second = static_cast<double>(linear_solution.size()) / time;
      pcout << "LinearGL Operator: " << time << " s, DoFs/s: " << dofs_per_second
            << std::endl;
      linear_solution = 0.0;
    }

    /* -------------------------------------
     * define solver and solver control
     * -------------------------------------
     */
    SolverControl solver_control(1000, 1e-8 * rhs.l2_norm());

    SolverGMRES<BlockVectorType> solver(
      solver_control,
      typename SolverGMRES<BlockVectorType>::AdditionalData(50, true));

    /* -----------------------------------------------
     *  preconditioner construction blocks start here
     *  cheap first preconditioner.
     * -----------------------------------------------
     */
    roctxRangePush("ROCTX-RANGE:LinearGLProblem preconditioner");
    DiagonalMatrix<VectorType> inverse_diagonal_U;
    DiagonalMatrix<VectorType> inverse_diagonal_V;

    VectorType &diagonal_U_vec = inverse_diagonal_U.get_vector();
    VectorType &diagonal_V_vec = inverse_diagonal_V.get_vector();

    mf_data_ptr->initialize_dof_vector(diagonal_U_vec);
    mf_data_ptr->initialize_dof_vector(diagonal_V_vec);
    
    LaplaceDiagonalCellOperatorQuad<dim, fe_degree, Number> laplace_diagonal_operator;

    /* U block */
    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, n_components, Number>
      (*mf_data_ptr, diagonal_U_vec, laplace_diagonal_operator,
        EvaluationFlags::gradients,
        EvaluationFlags::gradients, 0);

    /* V block */
    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, n_components, Number>
      (*mf_data_ptr, diagonal_V_vec, laplace_diagonal_operator,
        EvaluationFlags::gradients,
        EvaluationFlags::gradients, 1);

    roctxRangePush("ROCTX-RANGE:inv diagonal_U_vec");
    /* Invert diagonal. */
    for (auto &x : diagonal_U_vec)
      x = (std::abs(x) > 1e-12) ? 1./x : 1.;
    roctxRangePop();    
    
    roctxRangePush("ROCTX-RANGE:inv diagonal_V_vec");
    for (auto &x : diagonal_V_vec)
      x = (std::abs(x) > 1e-12) ? 1./x : 1.;
    roctxRangePop();
    
    BlockDiagonalJacobiPreconditioner<dim, fe_degree, Number>
      preconditioner(inverse_diagonal_U, inverse_diagonal_V);

    roctxRangePop();
    /* -----------------------------------------------
     *  preconditioner construction blocks ends here
     * -----------------------------------------------
     */
    
    dealii::Timer t(tria.get_mpi_communicator());
    roctxRangePush("ROCTX-RANGE:LinearGLProblem::solver.solve()");
    solver.solve(LinearGL_Operator, linear_solution, rhs, preconditioner);
    roctxRangePop();
    t.stop();

    pcout << "Solver converged in " << solver_control.last_step()
          << " iterations in " << t.wall_time() << " seconds" << std::endl;

    roctxRangePop();
  } // LinearGLProblem<...>::solve() ends here

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
