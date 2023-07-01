/* ------------------------------------------------------------------------------------------
 * 
 * This is source code of finite element solver of complex valued scalar GL equation 
 * i.e., s-wave SC/SF GL equation.
 * It is developed on the top of deal.II 9.3.3 finite element C++ library. 
 * 
 * License of this code is GNU Lesser General Public License, which been 
 * published by Free Software Fundation, either version 2.1 and later version.
 * You are free to use, modify and redistribute this program.
 *
 * ------------------------------------------------------------------------------------------
 *
 * author: Quang. Zhang (timohyva@github), 
 * QUEST-DMC project, University of Sussex;
 * Helsinki Institute of Physics, University of Helsinki;
 * 27. Kesäkuu. 2023.
 *
 */
#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

// The following chunk out code is identical to step-40 and allows
// switching between PETSc and Trilinos:

#include <deal.II/lac/generic_linear_algebra.h>

/* #define FORCE_USE_OF_TRILINOS */

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace complexGL_mpi
{
  using namespace dealii;
  /************************************************************/
  /* >>>>>>>>>>>>>>>    preconditioner class   <<<<<<<<<<<<<< */
  
  template <class PreconditionerA, class PreconditionerS>
  class BlockDiagonalPreconditioner : public Subscriptor
  {
    public:
      BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                  const PreconditionerS &preconditioner_S);

      void vmult(LA::MPI::BlockVector &      dst,
                 const LA::MPI::BlockVector &src) const;

    private:
      const PreconditionerA &preconditioner_A;
      const PreconditionerS &preconditioner_S;
  };

  template <class PreconditionerA, class PreconditionerS>
  BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
  BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                              const PreconditionerS &preconditioner_S)
    : preconditioner_A(preconditioner_A)
    , preconditioner_S(preconditioner_S)
  {}


  template <class PreconditionerA, class PreconditionerS>
  void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
  LA::MPI::BlockVector &      dst,
  const LA::MPI::BlockVector &src) const
  {
      preconditioner_A.vmult(dst.block(0), src.block(0));
      preconditioner_S.vmult(dst.block(1), src.block(1));
  }

  /* >>>>>>>> preconditioner class defination ends <<<<<<<<<< */
  /************************************************************/

  /* ------------------------------------------------------------------------------------------
   *
   * class template BoundaryValues inhereted from Function<dim>.
   * set the reference value_list as Zero Dirichilet BC.
   *
   * ------------------------------------------------------------------------------------------
   */
  
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(2) // tell base Function<dim> class I want a 2-components vector-valued function
    {}

    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));

      values(0) = 0;
      values(1) = 0;
    }

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        BoundaryValues<dim>::vector_value(points[p], value_list[p]);
    }
  };

  
  /* ------------------------------------------------------------------------------------------
   *
   * Zero Dirichlet BCs of newton-update
   *
   * ------------------------------------------------------------------------------------------
   */
  template <int dim>
  class DirichletBCs_newton_update : public Function<dim>
  {
  public:
    DirichletBCs_newton_update()
      : Function<dim>(2) // tell base Function<dim> class I want a 2-components vector-valued function
    {}

    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));

      values(0) = 0;
      values(1) = 0;
    }

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        DirichletBCs_newton_update<dim>::vector_value(points[p], value_list[p]);
    }
  };



  
  /* ------------------------------------------------------------------------------------------
   *
   * The following functions are members of ComplexValuedScalarGLSolver till the end of ScalarGL
   * namespace.
   *
   * ------------------------------------------------------------------------------------------
   */
    
  template <int dim>
  class complexGL
  {
  public:
    complexGL(unsigned int Q_degree);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    unsigned int  degree;
    //unsigned int velocity_degree;
    MPI_Comm     mpi_communicator;

    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    std::vector<IndexSet> owned_partitioning;
    std::vector<IndexSet> relevant_partitioning;

    AffineConstraints<double> constraints;

    LA::MPI::BlockSparseMatrix system_matrix;
    //LA::MPI::BlockSparseMatrix preconditioner_matrix;
    LA::MPI::BlockVector       locally_relevant_newton_solution;
    LA::MPI::BlockVector       local_solution;    // this is final solution Vector
    LA::MPI::BlockVector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };


  template <int dim>
  complexGL<dim>::complexGL(unsigned int Q_degree)
    : degree(Q_degree)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(Q_degree), 2)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  // The Kovasnay flow is defined on the domain [-0.5, 1.5]^2, which we
  // create by passing the min and max values to GridGenerator::hyper_cube.
  template <int dim>
  void complexGL<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, -0.5, 1.5);
    triangulation.refine_global(3);
  }

  template <int dim>
  void complexGL<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> sub_blocks_vector(2, 0);
    sub_blocks_vector[1] = 1;
    DoFRenumbering::component_wise(dof_handler, sub_blocks_vector);

    const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler,
												  sub_blocks_vector);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_v = dofs_per_block[1];

    pcout << "   Number of degrees of freedom: "
	  << dof_handler.n_dofs()
	  << " (" << n_u << '+' << n_p << ')'
	  << std::endl;

    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
    owned_partitioning[1] = dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_v);

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_v);

    {
      constraints.reinit(locally_relevant_dofs);

      //FEValuesExtractors::Vector u_component(0);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               DirichletBCs_newton_update<dim>(),
                                               constraints);
                                               //fe.component_mask(velocities));
      constraints.close();
    }

    {
      system_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(2, 2);
      for (unsigned int c = 0; c < 2; ++c)
        for (unsigned int d = 0; d < 2; ++d)
          if (c == d)
            coupling[c][d] = DoFTools::always;
          else if ((c == 0 && d == 1)
		   || (c == 1 && d == 0))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none; // how about Ribin BC's contribution ?

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp, constraints, false);

      // exchange local dsp entries between processes
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    /*  set up initial local_solution Vector */
    /*---------------------------------------*/
    std::random_device rd{};         // rd will be used to obtain a seed for the random number engine
    std::mt19937       gen{rd()};    // Standard mersenne_twister_engine seeded with rd()

    std::normal_distribution<double> gaussian_distr{3.0, 2.0}; // gaussian distribution with mean 10. STD 6.0

    local_solution.reinit(owned_partitioning,
			  relevant_partitioning,
			  mpi_communicator);
    
    for (auto it = local_solution.begin(); it != local_solution.end(); ++it)
      *it = gaussian_distr(gen);
    /*---------------------------------------*/
    
    locally_relevant_newton_solution.reinit(owned_partitioning,
                                            relevant_partitioning,
                                            mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }



  template <int dim>
  void complexGL<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    system_rhs            = 0;

    const QGauss<dim> quadrature_formula(velocity_degree + 1);

    FEValues<dim> fe_values(fe,
			    quadrature_formula,
			    update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    /* --------------------------------------------------*/
    //std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    //std::vector<double>         div_phi_u(dofs_per_cell);
    //std::vector<double>         phi_p(dofs_per_cell);
    /* --------------------------------------------------*/

    /*--------------------------------------------------*/
    /* >>>>>>>>>>  old solution for r.h.s   <<<<<<<<<<< */
    // vector to holding u^n, grad u^n on cell:
    std::vector<Tensor<1, dim>> old_solution_gradients_u(n_q_points);
    std::vector<double>         old_solution_u(n_q_points);

    // vector to holding v^n, grad v^n on cell:
    std::vector<Tensor<1, dim>> old_solution_gradients_v(n_q_points);
    std::vector<double>         old_solution_v(n_q_points);
    /* --------------------------------------------------*/    

    /*std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector     velocities(0);
    const FEValuesExtractors::Scalar     pressure(dim);*/

    // FEValuesExtractors, works for FEValues, FEFaceValues, FESubFaceValues
    const FEValuesExtractors::Scalar u_component(0);
    const FEValuesExtractors::Scalar v_component(1);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
	{
	  cell_matrix  = 0;
	  cell_rhs     = 0;

	  fe_values.reinit(cell);
	  //right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
	  
	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {
	      /*--------------------------------------------------*/
	      /*for (unsigned int k = 0; k < dofs_per_cell; ++k)
		{
		  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
		  div_phi_u[k]  = fe_values[velocities].divergence(k, q);
		  phi_p[k]      = fe_values[pressure].value(k, q);
		  }*/
              /*--------------------------------------------------*/

	      /* FeValuesViews[Extractor] returns on-cell values,
	       * gradients of unkown functions corresponding to
	       * given indexed component through Extractor.*/
	      fe_values[u_component].get_function_gradients(local_solution, old_solution_gradients_u);
	      fe_values[u_component].get_function_values(local_solution, old_solution_u);
	      fe_values[v_component].get_function_gradients(local_solution, old_solution_gradients_v);
	      fe_values[v_component].get_function_values(local_solution, old_solution_v);

	      /* --------------------------------------------------*/
              // alpha + bete (3*u^(n)^2 + v^{n}^2), u-coeff, system_matrix
	      const double bulkTerm_u_coeff_sysMatr =
		alpha_0 * (t-1.0)
		+ (beta * (3.0 * old_solution_u[q] * old_solution_u[q] + old_solution_v[q] * old_solution_v[q]));

	      // alpha + bete (3*v^(n)^2 + u^{n}^2), v-coeff, system_matrix
	      const double bulkTerm_v_coeff_sysMatr =
		alpha_0 * (t-1.0)
		+ (beta * (3.0 * old_solution_v[q] * old_solution_v[q] + old_solution_u[q] * old_solution_u[q]));

	      const double bulkTerm_uv_coeff_sysMatr = (2.0 * beta * old_solution_v[q] * old_solution_u[q]);

	      // alpha + bete (u^(n)^2 + v^(n)^2), u/v-coef, rhs
	      const double bulkTerm_coeff_rhs =
		alpha_0 * (t-1.0)
		+ beta * (old_solution_u[q] * old_solution_u[q] + old_solution_v[q] * old_solution_v[q]);
	      /* --------------------------------------------------*/
	      
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
		  for (unsigned int j = 0; j < dofs_per_cell; ++j)
		    {
		      cell_matrix(i, j) +=
			/*(viscosity *
			 scalar_product(grad_phi_u[i], grad_phi_u[j]) -
			 div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
			 fe_values.JxW(q);*/
			(((fe_values[u_component].gradient(i, q)       // ((\partial_q \phi^u_i
			   * fe_values[u_component].gradient(j, q)     //   \partial_q \phi^u_j
			   * 0.5)                                      //   * 1/2)
			  +                                 //  +
			  (bulkTerm_u_coeff_sysMatr                //  ((\alpha + \beta (3 u^(n)^2 + v^(n)^2)
			   * fe_values[u_component].value(i, q)    //    * \phi_i
			   * fe_values[u_component].value(j, q))  //    * \phi_j))
			  +
			  (bulkTerm_uv_coeff_sysMatr
			   * fe_values[v_component].value(i, q)
			   * fe_values[u_component].value(j, q))
			  +
			  (fe_values[v_component].gradient(i, q)       // ((\partial_q \phi^v_i
			   * fe_values[v_component].gradient(j, q)     //   \partial_q \phi^v_j
			   * 0.5)                                      //   * 1/2)
			  +
			  (bulkTerm_v_coeff_sysMatr                //  ((\alpha + \beta (3 u^(n)^2 + v^(n)^2)
			   * fe_values[v_component].value(i, q)    //    * \phi_i
			   * fe_values[v_component].value(j, q))  //    * \phi_j))
			  +
			  (bulkTerm_uv_coeff_sysMatr
			   * fe_values[u_component].value(i, q)
			   * fe_values[v_component].value(j, q)))
			 * fe_values.JxW(q));                    // * dx			  
		    }

		  const unsigned int component_i =
		      fe.system_to_component_index(i).first;
		  cell_rhs(i) += fe_values.shape_value(i, q) *
			      rhs_values[q](component_i) * fe_values.JxW(q);
		}
	    }


	  cell->get_dof_indices(local_dof_indices);
	  constraints.distribute_local_to_global(cell_matrix,
						 cell_rhs,
						 local_dof_indices,
						 system_matrix,
						 system_rhs);

       }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim>
  void complexGL<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::PreconditionAMG B_inv;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;

      B_inv.initialize(system_matrix.block(?????????????????????, ???????????????????????), data);
    }

    LA::MPI::PreconditionAMG BT_inv;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;

      prec_S.initialize(preconditioner_matrix.block(1, 0), data);
    }

    // The InverseMatrix is used to solve for the mass matrix:
    // using mp_inverse_t = LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
    //                                                  LA::MPI::PreconditionAMG>;
    // const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1), prec_S);

    // This constructs the block preconditioner based on the preconditioners
    // for the individual blocks defined above.
    const BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
                                      /*mp_inverse_t*/
                                      LA::MPI::PreconditionAMG> preconditioner(B_inv, /*mp_inverse*/BT_inv);

    // With that, we can finally set up the linear solver and solve the system:
    SolverControl solver_control(system_matrix.m(),
                                 1e-10 * system_rhs.l2_norm());

    SolverMinRes<LA::MPI::BlockVector> solver(solver_control);

    LA::MPI::BlockVector distributed_solution(owned_partitioning, mpi_communicator);

    // what this .set_zero() is doing ?
    // AffineContraint::set_zero() set the values of all constrained DoFs in a vector to zero. 
    constraints.set_zero(distributed_solution);

    solver.solve(system_matrix,
                 distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(distributed_solution);

  locally_relevant_newton_solution = distributed_solution;

  }


  template <int dim>
  void complexGL<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    triangulation.refine_global();
  }



  template <int dim>
  void complexGL<dim>::output_results(const unsigned int cycle) const
  {

    /*std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);*/
    std::vector<std::string> newton_update_components_names;
    newton_update_components_names.emplace_back("Re_du");
    newton_update_components_names.emplace_back("Im_dv");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    /*data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);*/
    data_out.add_data_vector(locally_relevant_newton_solution, newton_update_components_names);    


    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);
  }



  template <int dim>
  void complexGL<dim>::run()
  {
    pcout << "Running using Trilinos." << std::endl;

    const unsigned int n_cycles = 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          make_grid();
        else
          refine_grid();

        setup_system();

        assemble_system();
        solve();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 128)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(cycle);
          }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace complexGL_mpi



int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace complexGL_mpi;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      complexGLm<2> GLsolver(2);
      GLSolver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
