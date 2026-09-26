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

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include "LinearGLOperator.h"
// #include "LocalLinearGLOperator.h"
#include "bgSolution_U.h"
#include "bgSolution_V.h"
#include "LinearGLRHSCellOperator.h"
#include "LaplaceDiagonalCellOperatorQuad.h"

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
    class BlockDiagonalJacobiPreconditioner
    {
     public:
      BlockDiagonalJacobiPreconditioner(const DiagonalMatrix<VectorType> &inverse_diagonal_U,
                                        const DiagonalMatrix<VectorType> &inverse_diagonal_V)
        : inverse_diagonal_U(inverse_diagonal_U)
        , inverse_diagonal_V(inverse_diagonal_V)
      {}
      //
      void vmult(BlockVectorType &dst,
                 const BlockVectorType &src) const
      {
        inverse_diagonal_U.vmult(dst.block(0), src.block(0));
        inverse_diagonal_V.vmult(dst.block(1), src.block(1));
      }
      
      void Tvmult(BlockVectorType &dst,
                  const BlockVectorType &src) const
      { vmult(dst, src); }
      private:
        const DiagonalMatrix<VectorType> &inverse_diagonal_U;
        const DiagonalMatrix<VectorType> &inverse_diagonal_V;
    }; // preconditioner ends here
    /* -------------------------------------------
     *     preconditioner class ends here
     * -------------------------------------------*/
    
  private:
    void setup_dofs();

    void solve();

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
  };


  template <int dim, int fe_degree, typename Number>
  LinearGLProblem<dim, fe_degree, Number>::LinearGLProblem()
    : tria(MPI_COMM_WORLD)
    , mapping(fe_degree)
    , fe_U(FE_Q<dim>(fe_degree), 9)
    , fe_V(FE_Q<dim>(fe_degree), 9)
    , DoFHandler_U(tria)
    , DoFHandler_V(tria)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::setup_dofs()
  {
    DoFHandler_U.distribute_dofs(fe_U);
    DoFHandler_V.distribute_dofs(fe_V);

    /* U dofs */
    const IndexSet &owned_set_U   = DoFHandler_U.locally_owned_dofs();
    const IndexSet relevant_set_U = DoFTools::extract_locally_relevant_dofs(DoFHandler_U);
    constraints_U.reinit(owned_set_U, relevant_set_U);
    
    DoFTools::make_hanging_node_constraints(DoFHandler_U, constraints_U);
    VectorTools::interpolate_boundary_values(
      DoFHandler_U, 0, Functions::ZeroFunction<dim, Number>(dim), constraints_U);
    constraints_U.close();

    /* V dofs */
    const IndexSet &owned_set_V   = DoFHandler_V.locally_owned_dofs();
    const IndexSet relevant_set_V = DoFTools::extract_locally_relevant_dofs(DoFHandler_V);
    constraints_V.reinit(owned_set_V, relevant_set_V);
    
    DoFTools::make_hanging_node_constraints(DoFHandler_V, constraints_V);
    VectorTools::interpolate_boundary_values(
      DoFHandler_V, 0, Functions::ZeroFunction<dim, Number>(dim), constraints_V);
    constraints_V.close();

    /* ------------------------------------------------
     * container of DoFHandlers of U and V
     * ------------------------------------------------
     */
    std::vector<const DoFHandler<dim> *> DoFHandlers_U_V
      = {&DoFHandler_U, &DoFHandler_V};

    /* ------------------------------------------------
     * container of AffineConstraints of U and V
     * ------------------------------------------------
     */     
    std::vector<const AffineConstraints<Number> *> constraints_U_V
      = {&constraints_U, &constraints_V};

    /* ------------------------------------------------
     * allocate ssmart pointer for Portable::MatrixFree<..>
     * ------------------------------------------------
     */
    mf_data_ptr = std::make_shared<Portable::MatrixFree<dim, Number>>();

    const QGauss<1> quad(fe_degree + 1);
    // const QGauss<1> quad(degree_p + 2);
    typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values
      | update_gradients | update_JxW_values | update_quadrature_points;    
    // additional_data.mapping_update_flags = update_values | update_gradients;
    /*------------------------------------------------------------
     * using similar multi-DoFHandler pattern as step-104 for block structure
     * DoFHandler 0 -> U, DoFHandler 1 -> V, Here two Dofhandlers are provided
     * when Portable::Matrixfree is initialized.
     * ------------------------------------------------------------
     */    
    mf_data_ptr->reinit(mapping,
                        DoFHandlers_U_V, constraints_U_V,
                        quad, additional_data);

    /* ------------------------------------------------------------
     * create the background_solution on the host and move to device:
     * ------------------------------------------------------------
     */ 
    LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Host> bgSolution_host;
    mf_data_ptr->initialize_dof_vector(bgSolution_host);

    VectorTools::interpolate(mapping, DoFHandler_U,
                               bgSolution_U<dim, Number>(),
                               bgSolution_host.block(0));
    
    VectorTools::interpolate(mapping, DoFHandler_V,
                               bgSolution_V<dim, Number>(),
                               bgSolution_host.block(1));

    // prepare moving host vector to device vector.
    mf_data_ptr->initialize_dof_vector(bg_solution);
    bg_solution.block(0).import_elements(bgSolution_host.block(0), VectorOperation::insert);
    bg_solution.block(1).import_elements(bgSolution_host.block(1), VectorOperation::insert);
    /* ------------------------------------------------------------
     * background_solution on the host and  device is done
     * ------------------------------------------------------------
     */ 
         
    {
      /* ------------------------------------------------------------
       * using the device vector bg_solution.
       * ------------------------------------------------------------
       */
       mf_data_ptr->initialize_dof_vector(rhs);
       rhs = Number(0.0); // do I need this?

       const Number K1    = /* your K1 */;
       const Number alpha = /* your alpha */;
       const Number beta2 = /* your beta2 */;
       
       LinearGLRHSCellOperator<dim, fe_degree, Number> rhs_operator(K1, alpha, beta2);

      /*  bg_solution.block(0) = u^0 ,bg_solution.block(1) = v^0 */
       mf_data_ptr->cell_loop(rhs_operator, bg_solution /*src*/, rhs /*dst*/);

      /*
       * Newton updates satisfies homogeneous Dirichlet conditions.
       * Therefore constrained RHS entries must be zero.
       */
       mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(0), 0);
       mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(1), 1);
      
    } // rhs setting block ends here
    
  } // LinearGLProblem<...>::setup_dofs() ends here

  // In the solve() function we set up the preconditioner and
  // run the GMRES solver.
  // For this, we construct the multigrid
  // hierarchy for the GMG v-cycle with a Chebyshev iteration around the
  // point-Jacobi scheme, i.e., the inverse of the diagonal of $A$, to
  // approximate the action of $A^{-1}$.

  // We approximate the Schur Complement with a Chebyshev iteration
  // applied to the pressure mass matrix (without multigrid).
  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::solve()
  {
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
    DiagonalMatrix<VectorType> inverse_diagonal_U;
    DiagonalMatrix<VectorType> inverse_diagonal_V;

    VectorType &diagonal_U_vec = inverse_diagonal_U.get_vector();
    VectorType &diagonal_V_vec = inverse_diagonal_V.get_vector();

    mf_data_ptr->initialize_dof_vector(diagonal_U_vec);
    mf_data_ptr->initialize_dof_vector(diagonal_V_vec);
    
    LaplaceDiagonalOperation<dim, fe_degree, Number> laplace_diagonal_operator;

    /* U block */
    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, n_components, Number,
      MemorySpace::Default>(*mf_data_ptr, diagonal_U_vec, laplace_diagonal_operator,
        EvaluationFlags::gradients,
        EvaluationFlags::gradients, 0);

    /* V block */
    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, n_components, Number,
      MemorySpace::Default>(*mf_data_ptr, diagonal_V_vec, laplace_diagonal_operator,
        EvaluationFlags::gradients,
        EvaluationFlags::gradients, 1);
    
    /* Invert diagonal. */
    for (auto &x : diagonal_U_vec)
      x = (std::abs(x) > 1e-12) ? 1./x : 1.;

    for (auto &x : diagonal_V_vec)
      x = (std::abs(x) > 1e-12) ? 1./x : 1.;
    
    BlockDiagonalJacobiPreconditioner preconditioner(inverse_diagonal_U, inverse_diagonal_V);

    /* -----------------------------------------------
     *  preconditioner construction blocks ends here
     * -----------------------------------------------
     */
    
    dealii::Timer t(tria.get_mpi_communicator());
    solver.solve(LinearGL_Operator, linear_solution, rhs, preconditioner);
    t.stop();

    pcout << "Solver converged in " << solver_control.last_step()
          << " iterations in " << t.wall_time() << " seconds" << std::endl;

  } // LinearGLProblem<...>::solve() ends here


  // The postprocess() function moves the solution to host memory
  // and integrates the difference to the manufactured solution to
  // compute errors.
  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::postprocess()
  {
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
    //                                   VelocitySolution<dim, Number>(),
    //                                   cellwise_errors_ul2,
    //                                   quadrature_formula,
    //                                   VectorTools::L2_norm);
    // VectorTools::integrate_difference(dof_p,
    //                                   solution_host.block(1),
    //                                   PressureSolution<dim, Number>(),
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
    
  } // LinearGLProblem<...>::postprocess() ends here


  // The run() function prints some statistics and
  // then performs refinement loop.
  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::run()
  {
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

    unsigned int n_refinements = 10;

    for (unsigned int i = 0; i < n_refinements; ++i)
      {
        if (i == 0)
          {
            GridGenerator::hyper_cube(tria);
            tria.refine_global(2);
          }
        else
          {
            tria.refine_global(1);
          }
        setup_dofs();

        pcout << "\nrefinement: " << i
              << ", n_dofs: " << DoFHandler_U.n_dofs() + DoFHandler_V.n_dofs()
              << " = " << DoFHandler_U.n_dofs() << " + " << DoFHandler_V.n_dofs()
              << std::endl;

        solve();
        postprocess();
      }
  }
} // namespace VerHem ends here

// The only interesting bits here are the template arguments that
// specify dimension and polynomial degree to be used.
int main(int argc, char **argv)
{
  using namespace VerHem;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const unsigned int                   dim      = 3;
  const unsigned int                   FE_degree = 1;
  StokesProblem<dim, FE_degree, float> LinearGL_problem;
  LinearGL_problem.run();
}
