#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

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

#include "LinearGLOperator.h"
// #include "LocalLinearGLOperator.h"
#include "initial_UV_ConfFunction/bgSolution_U.h"
#include "initial_UV_ConfFunction/bgSolution_V.h"
#include "LinearGL_RightHandSide/LinearGLRHSCellOperator.h"
#include "LinearGL_Preconditioner/LaplaceDiagonalCellOperatorQuad.h"
#include "LinearGL_Preconditioner/BlockDiagonalJacobiPreconditioner.h"

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  class LinearGLProblem
  {
  public:

    static constexpr unsigned int n_components = 9;

    LinearGLProblem();

    void run();

    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using BlockVectorType =
      LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Default>;
    using DiagonalMatrixType = DiagonalMatrix<VectorType>;
    
  private:
    void setup_dofs();

    void solve();
    void newton_iter(const Number lambda);
    void line_search(const Number lambda);    
    
    void postprocess();

    parallel::distributed::Triangulation<dim> tria;

    MappingQ<dim> mapping;

    FESystem<dim> fe_U;
    FESystem<dim> fe_V;
    
    DoFHandler<dim> DoFHandler_U;
    DoFHandler<dim> DoFHandler_V;

    AffineConstraints<Number> constraints_U;
    AffineConstraints<Number> constraints_V;

    std::shared_ptr<Portable::MatrixFree<dim, Number>> mf_data_ptr;
    BlockVectorType                                    linear_solution;
    BlockVectorType                                    rhs;
    BlockVectorType                                    bg_solution;
    BlockVectorType                                    bg_newton_iter;
    ConditionalOStream                                 pcout;

    Number current_iter_InitResidual{0.0};
    const Number convTol_newton_iter{1e-4};
  }; // LinearGLProblem declearation ends here


} // namespace VerHem ends here
