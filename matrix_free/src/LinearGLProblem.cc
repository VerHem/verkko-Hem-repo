#include <deal.II/base/conditional_ostream.h>
// #include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

// #include <deal.II/grid/grid_generator.h>

// #include <deal.II/lac/affine_constraints.h>
// #include <deal.II/lac/diagonal_matrix.h>
// #include <deal.II/lac/la_parallel_block_vector.h>
// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_gmres.h>

// #include <deal.II/matrix_free/operators.h>
// #include <deal.II/matrix_free/portable_fe_evaluation.h>
// #include <deal.II/matrix_free/portable_matrix_free.h>
// #include <deal.II/matrix_free/tools.h>

//#include <deal.II/multigrid/mg_coarse.h>
//#include <deal.II/multigrid/mg_matrix.h>
//#include <deal.II/multigrid/mg_smoother.h>
//#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
//#include <deal.II/multigrid/mg_transfer_matrix_free.h>
//#include <deal.II/multigrid/multigrid.h>
//#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

// #include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/vector_tools_integrate_difference.h>

#include "roctracer/roctx.h"

// #include "LinearGLOperator.h"
// #include "LocalLinearGLOperator.h"
// #include "bgSolution_U.h"
// #include "bgSolution_V.h"
// #include "LinearGLRHSCellOperator.h"
// #include "LaplaceDiagonalCellOperatorQuad.h"
// #include "preconditioner/BlockDiagonalJacobiPreconditioner.h"
#include "LinearGLProblem.h"

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  LinearGLProblem<dim, fe_degree, Number>::LinearGLProblem()
    : tria(MPI_COMM_WORLD)
    , mapping(fe_degree)
    , fe_U(FE_Q<dim>(fe_degree), 9)
    , fe_V(FE_Q<dim>(fe_degree), 9)
    , DoFHandler_U(tria)
    , DoFHandler_V(tria)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
   roctxRangePush("ROCTX-RANGE:LinearGLProblem Constructor");
   roctxRangePop();
  }

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
