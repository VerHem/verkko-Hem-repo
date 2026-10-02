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
#include "bgSolution_U.h"
#include "bgSolution_V.h"
#include "LinearGLRHSCellOperator.h"
#include "preconditioner/LaplaceDiagonalCellOperatorQuad.h"
#include "preconditioner/BlockDiagonalJacobiPreconditioner.h"

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

    /* -------------------------------------------
     *        preconditioner class
     * -------------------------------------------*/
    // class BlockDiagonalJacobiPreconditioner
    // {
    //  public:
    //   BlockDiagonalJacobiPreconditioner(const DiagonalMatrix<VectorType> &inverse_diagonal_U,
    //                                     const DiagonalMatrix<VectorType> &inverse_diagonal_V)
    //     : inverse_diagonal_U(inverse_diagonal_U)
    //     , inverse_diagonal_V(inverse_diagonal_V)
    //   {}
    //   //
    //   void vmult(BlockVectorType &dst,
    //              const BlockVectorType &src) const
    //   {
    // 	roctxRangePush("ROCTX-RANGE:BlockDiagonalJacobiPreconditioner::vmult()");
    //     inverse_diagonal_U.vmult(dst.block(0), src.block(0));
    //     inverse_diagonal_V.vmult(dst.block(1), src.block(1));
    // 	roctxRangePop();
    //   }
      
    //   void Tvmult(BlockVectorType &dst,
    //               const BlockVectorType &src) const
    //   {
    // 	roctxRangePush("ROCTX-RANGE:BlockDiagonalJacobiPreconditioner::Tvmult()");
    // 	vmult(dst, src);
    // 	roctxRangePop();
    //   }
    //   private:
    //     const DiagonalMatrix<VectorType> &inverse_diagonal_U;
    //     const DiagonalMatrix<VectorType> &inverse_diagonal_V;
    // }; // preconditioner ends here
    /* -------------------------------------------
     *     preconditioner class ends here
     * -------------------------------------------*/
    
  private:
    void setup_dofs();

    void solve();
    void newton_iter(const Number lambda);
    
    void postprocess();

    parallel::distributed::Triangulation<dim> tria;

    MappingQ<dim> mapping;

    // FESystem<dim> fe_u;
    // FE_Q<dim>     fe_p;

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
    ConditionalOStream                                 pcout;
  }; // LinearGLProblem declearation ends here


} // namespace VerHem ends here

// template class VerHem::LinearGLProblem<3, 1, float>;
// template class VerHem::LinearGLProblem<3, 1, double>;
