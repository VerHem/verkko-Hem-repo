// #include <deal.II/base/conditional_ostream.h>
// #include <deal.II/base/timer.h>

// #include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// #include <deal.II/fe/fe_q.h>
// #include <deal.II/fe/fe_system.h>

// #include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
// #include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/operators.h>
// #include <deal.II/matrix_free/portable_fe_evaluation.h>
// #include <deal.II/matrix_free/portable_matrix_free.h>
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
  void LinearGLProblem<dim, fe_degree, Number>::postprocess()
  {
    roctxRangePush("ROCTX-RANGE:LinearGLProblem::postprocess()");
    LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Host> linear_solution_host;
    mf_data_ptr->initialize_dof_vector(linear_solution_host);

    linear_solution_host.block(0).import_elements(linear_solution.block(0),
                                           VectorOperation::insert);
    linear_solution_host.block(1).import_elements(linear_solution.block(1),
                                           VectorOperation::insert);

    constraints_U.distribute(linear_solution_host.block(0));
    constraints_V.distribute(linear_solution_host.block(1));
    linear_solution_host.update_ghost_values();
    // const double mean_pressure = VectorTools::compute_mean_value(
    //   dof_p, QGauss<dim>(degree_p + 2), solution_host.block(1), 0);
    // solution_host.block(1).add(-mean_pressure);

    // const QGauss<dim> quadrature_formula(degree_u + 1);

    // Vector<double> cellwise_errors_ul2(tria.n_active_cells());
    // Vector<double> cellwise_errors_pl2(tria.n_active_cells());

    // VectorTools::integrate_difference(dof_u,
    //                                   solution_host.block(0),
    //                                   xxxx<dim, Number>(),
    //                                   cellwise_errors_ul2,
    //                                   quadrature_formula,
    //                                   VectorTools::L2_norm);
    // VectorTools::integrate_difference(dof_p,
    //                                   solution_host.block(1),
    //                                   yyyy<dim, Number>(),
    //                                   cellwise_errors_pl2,
    //                                   quadrature_formula,
    //                                   VectorTools::L2_norm);

    // const double u_l2 = VectorTools::compute_global_error(tria,
    //                                                       cellwise_errors_ul2,
    //                                                       VectorTools::L2_norm);
    // const double p_l2 = VectorTools::compute_global_error(tria,
    //                                                       cellwise_errors_pl2,
    //                                                       VectorTools::L2_norm);

    // pcout << "velocity error: " << u_l2 << " pressure error: " << p_l2
    //       << std::endl;

    roctxRangePop();
  } // LinearGLProblem<...>::postprocess() ends here

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
