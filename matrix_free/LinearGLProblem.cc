#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include ""

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  class LinearGLProblem
  {
  public:
    // static constexpr unsigned int degree_u = degree_p + 1;

    LinearGLProblem();

    void run();

    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using BlockVectorType =
      LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Default>;

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
    ConditionalOStream                                 pcout;
  };


  template <int dim, int fe_degree, typename Number>
  LinearGLProblem<dim, fe_degree, Number>::LinearGLProblem()
    : tria(MPI_COMM_WORLD)
    , mapping(1)
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

    /* container of DoFHandlers of U and V */
    std::vector<const DoFHandler<dim> *> DoFHandlers_U_V
      = {&DoFHandler_U, &DoFHandler_V};

    /* container of AffineConstraints of U and V */    
    std::vector<const AffineConstraints<Number> *> constraints_U_V
      = {&constraints_U, &constraints_V};

    // allocate ssmart pointer for Portable::MatrixFree<..>
    mf_data_ptr = std::make_shared<Portable::MatrixFree<dim, Number>>();

    const QGauss<1> quad(fe_degree + 1);
    // const QGauss<1> quad(degree_p + 2);
    typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;
    mf_data_ptr->reinit(mapping, DoFHandlers_U_V, constraints_U_V, quad, additional_data);

    {
      // create the right hand side on the host and move to device:
      LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Host> rhs_host;
      mf_data_ptr->initialize_dof_vector(rhs_host);

      VectorTools::create_right_hand_side(mapping,
                                          DoFHandler_U,
                                          QGauss<dim>(fe_degree + 1),
                                          // VelocityRightHandSide<dim, Number>(),
                                          RU_RightHandSide<dim, Number>(),
                                          rhs_host.block(0),
                                          constraints_U);

      VectorTools::create_right_hand_side(mapping,
                                          DoFHandler_V,
                                          QGauss<dim>(fe_degree + 1),
                                          // VelocityRightHandSide<dim, Number>(),
                                          RV_RightHandSide<dim, Number>(),
                                          rhs_host.block(1),
                                          constraints_V);
      
      mf_data_ptr->initialize_dof_vector(rhs);
      rhs.block(0).import_elements(rhs_host.block(0), VectorOperation::insert);
      rhs.block(1).import_elements(rhs_host.block(1), VectorOperation::insert);
    } // rhs.host block ends here

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
      LinearGL_Operator(DoFHandler_U, DoFHandler_V,
                        constraints_U, constraints_V,
                        background_U_V_sol);

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
     * -----------------------------------------------
     */
    // using LevelMatrixType = PortableMFVelocityOperator<dim, degree_u, degree_p, Number>;
    using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
    using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                               VectorType,
                                               SmootherPreconditionerType>;
    using MGTransferType = MGTransferMatrixFree<dim, Number, MemorySpace::Default>;

    const auto coarse_grid_triangulations =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);

    const unsigned int max_level = coarse_grid_triangulations.size() - 1;
    // Do not go down to level 0, because this will lead to slower runtime as
    // the problem becomes very small:
    const unsigned int min_level = std::min(3U, max_level - 1);

    // mg_dof_handlers
    MGLevelObject<DoFHandler<dim>> mg_dof_handlers(min_level, max_level);
    // mg_constraints
    MGLevelObject<AffineConstraints<Number>> mg_constraints(min_level, max_level);
    // mg_matrices
    MGLevelObject<LevelMatrixType>           mg_matrices(min_level, max_level);
    // mg_transfers
    MGLevelObject<Portable::MGTwoLevelTransfer<dim, VectorType>> mg_transfers(min_level, max_level);

    // container of smart pointer os MatrixFree<..>
    std::vector<std::shared_ptr<Portable::MatrixFree<dim, Number>>> mf_data_levels;

    // Prepare the operators and data structures
    // on all levels of the multigrid hierarchy
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        auto &dof_handler = mg_dof_handlers[level];
        auto &constraint  = mg_constraints[level];

        dof_handler.reinit(*coarse_grid_triangulations[level]);
        dof_handler.distribute_dofs(fe_U);

        constraint.reinit(dof_handler.locally_owned_dofs(),
                          DoFTools::extract_locally_relevant_dofs(dof_handler));

        DoFTools::make_zero_boundary_constraints(dof_handler, constraint);
        constraint.close();

        typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
        additional_data.mapping_update_flags = update_JxW_values | update_gradients;

        if (level == max_level)
          // On the finest level we can reuse the MatrixFree object from the
          // LinearGL operator. This way we can solve significantly larger
          // problems before we run out of device memory.
          mf_data_levels.emplace_back(mf_data);
        else
          {
            const QGauss<1> quad(fe_degree + 2);
            mf_data_levels.emplace_back(std::make_shared<Portable::MatrixFree<dim, Number>>());

            mf_data_levels.back()->reinit(mapping, dof_handler, constraint, quad, additional_data);
          }

        mg_matrices[level].reinit(mf_data_levels.back());
      }

    mg::Matrix<VectorType> mg_matrix(mg_matrices);

    // transfer operator
    for (unsigned int level = min_level; level < max_level; ++level)
      mg_transfers[level + 1].reinit_geometric_transfer(
        mg_dof_handlers[level + 1],
        mg_dof_handlers[level],
        mg_constraints[level + 1],
        mg_constraints[level]);

    MGTransferType mg_transfer(mg_transfers, [&](const auto l, auto &vec) {
      mg_matrices[l].initialize_dof_vector(vec);
    });

    // smoother
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);

    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>(*mg_matrices[level].get_matrix_diagonal_inverse());
        smoother_data[level].constraints.copy_from(mg_constraints[level]);

        if (level == min_level)
          {
            // Use the Chebyshev iteration as an (approximate) solver on the
            // coarsest level. In this mode @p smoothing_range is a relative
            // target tolerance and must be strictly less than one; the number
            // of iterations is then chosen automatically by setting
            // @p degree to numbers::invalid_unsigned_int. We also use more
            // CG iterations for the eigenvalue estimate because when
            // @p min_level > 0, the coarse problem can still be reasonably
            // large and badly conditioned.
            smoother_data[level].smoothing_range = 1e-3;
            smoother_data[level].degree = numbers::invalid_unsigned_int;
            smoother_data[level].eig_cg_n_iterations = 40;
          }
        else
          {
            // These values are chosen by experimentation for the problem at
            // hand. We chose the smoothing range first. A good value will allow
            // the smoother to effectively separate large and small scale
            // oscillations in the residual and as such improve the convergence
            // of the Chebyshev iteration and the multigrid method. Finally, the
            // degree is chosen to minimize total runtime (a larger value
            // increases the cost but improves the outer number of GMRES
            // iterations).
            smoother_data[level].smoothing_range     = 5;
            smoother_data[level].degree              = 4;
            smoother_data[level].eig_cg_n_iterations = 20;
          }
      }

    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
    mg_smoother.initialize(mg_matrices, smoother_data);

    // Estimate and print the eigenvalue spectrum of the velocity block on each
    // level. This spectrum is later used by the Chebyshev iteration.
    pcout << "GMG velocity block smoothers:" << std::endl;
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        VectorType vec;
        mg_matrices[level].initialize_dof_vector(vec);
        auto eigenvalue_info = mg_smoother.smoothers[level].estimate_eigenvalues(vec);
        pcout << "    level: " << level << " n_dofs: " << vec.size()
              << ", eigenvalue spectrum: [ "
              << eigenvalue_info.min_eigenvalue_estimate << ", "
              << eigenvalue_info.max_eigenvalue_estimate << " ]" << std::endl;
      }

    // coarse-grid solver
    MGCoarseGridApplySmoother<VectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    // put everything together
    Multigrid<VectorType> mg(mg_matrix,
                             mg_coarse,
                             mg_transfer,
                             mg_smoother,
                             mg_smoother,
                             min_level,
                             max_level);


    dealii::Timer timer_smoother;
    dealii::Timer timer_transfer;
    dealii::Timer timer_coarse;
    dealii::Timer timer_residual;
    {
      timer_smoother.reset();
      timer_transfer.reset();
      timer_coarse.reset();
      timer_residual.reset();

      auto make_timer_lambda = [&](dealii::Timer &timer) {
        return [&](const bool before, const unsigned int /*level*/) {
          if (before)
            timer.start();
          else
            timer.stop();
        };
      };
      mg.connect_pre_smoother_step(make_timer_lambda(timer_smoother));
      mg.connect_post_smoother_step(make_timer_lambda(timer_smoother));
      mg.connect_residual_step(make_timer_lambda(timer_residual));
      mg.connect_restriction(make_timer_lambda(timer_transfer));
      mg.connect_prolongation(make_timer_lambda(timer_transfer));
      mg.connect_coarse_solve(make_timer_lambda(timer_coarse));
    }

    using APreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;
    APreconditionerType preconditioner_A(DoFHandler_U, mg, mg_transfer);

    // PortableMFMassOperator<dim, degree_u, degree_p, Number> mass_operator(mf_data);
    // mass_operator.compute_diagonal();

    // using SPreconditionerType = PreconditionChebyshev<
    //   PortableMFMassOperator<dim, degree_u, degree_p, Number>,
    //   VectorType>;

    // SPreconditionerType preconditioner_schur;
    // {
    //   typename SPreconditionerType::AdditionalData additional_data;
    //   additional_data.smoothing_range     = 15.;
    //   additional_data.degree              = 3;
    //   additional_data.eig_cg_n_iterations = 10;
    //   additional_data.constraints.copy_from(constraints_p);
    //   additional_data.preconditioner =
    //     mass_operator.get_matrix_diagonal_inverse();

    //   preconditioner_schur.initialize(mass_operator, additional_data);
    // }

    // using BTOperatorType =
    //   PortableMFBTOperator<dim, degree_u, degree_p, Number>;
    // BTOperatorType BT_operator(mf_data);

    BlockSchurPreconditioner<APreconditionerType,
                             SPreconditionerType,
                             BTOperatorType,
                             BlockVectorType>
      preconditioner(preconditioner_A, preconditioner_schur, BT_operator);
    /* -----------------------------------------------
     *  preconditioner construction blocks ends here
     * -----------------------------------------------
     */
    
    dealii::Timer t(tria.get_mpi_communicator());
    solver.solve(LinearGL_Operator, linear_solution, rhs, preconditioner);
    t.stop();

    pcout << "Solver converged in " << solver_control.last_step()
          << " iterations in " << t.wall_time() << " seconds" << std::endl;

    pcout << "Velocity block GMG timings:"
          << "\n    smoother: " << timer_smoother.wall_time()
          << " s\n    transfer: " << timer_transfer.wall_time()
          << " s\n    coarse  : " << timer_coarse.wall_time()
          << " s\n    residual: " << timer_residual.wall_time() << " s"
          << std::endl;

  } // LinearGLProblem<...>::solve() ends here


  // The postprocess() function moves the solution to host memory
  // and integrates the difference to the manufactured solution to
  // compute errors.
  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::postprocess()
  {
    LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Host> solution_host;
    mf_data->initialize_dof_vector(solution_host);

    solution_host.block(0).import_elements(linear_solution.block(0),
                                           VectorOperation::insert);
    solution_host.block(1).import_elements(linear_solution.block(1),
                                           VectorOperation::insert);

    constraints_U.distribute(solution_host.block(0));
    constraints_V.distribute(solution_host.block(1));
    solution_host.update_ghost_values();
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


  // The run() function prints some statistics and then performs a familiar
  // refinement loop.
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
