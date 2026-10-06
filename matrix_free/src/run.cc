#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

// #include <deal.II/lac/affine_constraints.h>
// #include <deal.II/lac/diagonal_matrix.h>
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
// #include "bgSolution_U.h"
// #include "bgSolution_V.h"
// #include "LinearGLRHSCellOperator.h"
// #include "LaplaceDiagonalCellOperatorQuad.h"
// #include "preconditioner/BlockDiagonalJacobiPreconditioner.h"

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::run()
  {
    roctxRangePush("ROCTX-RANGE:LinearGLProblem::run()");
    pcout << std::setprecision(10);
    pcout << "Running on " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
          << " MPI ranks (with " << MultithreadInfo::n_threads()
          << " threads each) in ";
    if constexpr (running_in_debug_mode())
      pcout << "DEBUG mode";
    else
      pcout << "RELEASE mode";

    pcout << "\nKokkos execution space: "
          << Kokkos::DefaultExecutionSpace::name();
    pcout << '\n'
          << "dim: " << dim << '\n'
          << "Element: Q" << fe_degree << "-Q" << fe_degree << std::endl;

    unsigned int n_refinements = 1;

    for (unsigned int i = 0; i < n_refinements; ++i)
      {
        if (i == 0)
          {
	    roctxRangePush("ROCTX-RANGE:Starting hyper_cube GridGenerator");
            GridGenerator::hyper_cube(tria, -20, 20);
	    roctxRangePop();

	    roctxRangePush("ROCTX-RANGE:Starting tria.refine_global(2)");
            // tria.refine_global(10);
	    tria.refine_global(7);
	    // tria.refine_global(6);
	    roctxRangePop();
          }
        else
          { tria.refine_global(1); }

        setup_dofs();

        pcout << "\nrefinement: " << i
              << ", n_dofs: " << DoFHandler_U.n_dofs() + DoFHandler_V.n_dofs()
              << " = " << DoFHandler_U.n_dofs() << " + " << DoFHandler_V.n_dofs()
              << std::endl;

        solve();

	newton_iter(0.5);
	
        postprocess();
      }
    
    roctxRangePop();
  } // LinearGLProblem::run() ends here
} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
