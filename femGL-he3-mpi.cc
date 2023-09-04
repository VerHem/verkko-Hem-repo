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
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace FemGL_mpi
{
  using namespace dealii;

  /* ------------------------------------------------------------------------------------------
   * class template BoundaryValues inhereted from Function<dim>.
   * set the reference value_list as Zero Dirichilet BC.
   * ------------------------------------------------------------------------------------------
   */
  
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(18) // tell base Function<dim> class I want a 2-components vector-valued function
    {}

    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 18, ExcDimensionMismatch(values.size(), 18));
      for (auto &value_of_index : values)                                                                              
        value_of_index = 0.0;    
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
   * Zero Dirichlet BCs of newton-update
   * ------------------------------------------------------------------------------------------
   */
  template <int dim>
  class DirichletBCs_newton_update : public Function<dim>
  {
  public:
    DirichletBCs_newton_update()
      : Function<dim>(18) // tell base Function<dim> class I want a 2-components vector-valued function
    {}

    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 18, ExcDimensionMismatch(values.size(), 18));
      for (auto &value_of_index : values)                                                                              
        value_of_index = 0.0;    
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
   * The following functions are members of ComplexGL-mpi till the end of complexGL-mpi
   * namespace.
   *
   * ------------------------------------------------------------------------------------------
   */
    
  template <int dim>
  class FemGL
  {
  public:
    FemGL(unsigned int Q_degree);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void compute_residual(/*const LA::MPI::Vector &*/);
    void solve();
    void newton_iteration();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    // matrices constraction function for last step u, v tensors
    void   vector_matrix_generator(const FEValues<dim>  &fe_values,
				   const char &vector_flag,
				   const unsigned int   q, const unsigned int   n_q_point,
				   FullMatrix<double>   &u_matrix_at_q,
				   FullMatrix<double>   &v_matrix_at_q);

    // matrices constraction function for last step u, v tensors
    void   grad_vector_matrix_generator(const FEValues<dim>  &fe_values,
					const char &vector_flag,
					const unsigned int   q, const unsigned int   n_q_point,
					std::vector<FullMatrix<double>> &grad_u_at_q,
					std::vector<FullMatrix<double>> &grad_v_at_q);

    // matrices construction function for shape function phi_u, phi_v at given local dof x and Gaussian q
    void   phi_matrix_generator(const FEValues<dim> &fe_values,
				const unsigned int  x, const unsigned int q,
				FullMatrix<double>  &phi_u_at_x_q,
				FullMatrix<double>  &phi_v_at_x_q);

    // matrices construction function for gradient of shape function, grad_k_phi_u/v at local dof x and Gaussian q
    void   grad_phi_matrix_container_generator(const FEValues<dim> &fe_values,
					       const unsigned int x, const unsigned int q,
					       std::vector<FullMatrix<double>> &container_grad_phi_u_x_q,
					       std::vector<FullMatrix<double>> &container_grad_phi_v_x_q); 

    unsigned int       degree;    
    MPI_Comm           mpi_communicator;
    
    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    /* container for FEValuesExtractors::scalar
     * FEValuesExtractors, works for FEValues, FEFaceValues, FESubFaceValues
     * for he3 GL eqn, 18 copies of scalar extractors are needed, they are
     * defined by the matrix components they represents for.*/

    std::vector<FEValuesExtractors::Scalar> components_u;
    std::vector<FEValuesExtractors::Scalar> components_v;


    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints_newton_update;
    AffineConstraints<double> constraints_solution;    

    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector       locally_relevant_newton_solution;
    LA::MPI::Vector       local_solution;    // this is final solution Vector
    LA::MPI::Vector       locally_relevant_damped_vector;
    LA::MPI::Vector       system_rhs;
    LA::MPI::Vector       residual_vector;

    const double K1      = 0.5;
    const double alpha_0 = 2.0;
    const double beta    = 0.5;

    double reduced_t;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
        
  };


  template <int dim>
  FemGL<dim>::FemGL(unsigned int Q_degree)
    : degree(Q_degree)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(Q_degree), 18)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , components_u(9, FEValuesExtractors::Scalar())
    , components_v(9, FEValuesExtractors::Scalar())      
    , reduced_t(0.0)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {
    /* -------------------------------------------------- 
     * Initiate FEValuesExtractors::Scalar container
     * --------------------------------------------------
     */
    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	FEValuesExtractors::Scalar extractor_u(comp_index), extractor_v(comp_index + 9);
	components_u[comp_index] = extractor_u; components_v[comp_index] = extractor_v;
      }
    /*--------------------------------------------------*/
  }


  template <int dim>
  void FemGL<dim>::make_grid()
  {
    if (dim==2)
      {
        const double half_length = 10.0, inner_radius = 2.0;
        GridGenerator::hyper_cube_with_cylindrical_hole(triangulation,
						    inner_radius, half_length);
        triangulation.refine_global(3);
        // Dirichlet pillars centers
        const Point<dim> p1(0., 0.);

        for (const auto &cell : triangulation.cell_iterators())
        for (const auto &face : cell->face_iterators())
 	  {
            const auto center = face->center();
	    if (
	        (std::fabs(center(0) - (-half_length)) < 1e-12)
	        ||
	        (std::fabs(center(0) - (half_length)) < 1e-12)
	        ||
	        (std::fabs(center(1) - (-half_length)) < 1e-12)
	        ||
	        (std::fabs(center(1) - (half_length)) < 1e-12)
               )
	       face->set_boundary_id(1);

	    if ((std::fabs(center.distance(p1) - inner_radius) <=0.15))
	      face->set_boundary_id(0);
	  }

        triangulation.refine_global(1);
      }
    else if (dim==3)
      {
	const double half_length = 10.0, inner_radius = 2.0, z_extension =10.0;
	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation,
							inner_radius, half_length, z_extension);

	triangulation.refine_global(3);
	// Dirichlet pillars centers
	const Point<dim> p1(0., 0., 0.);

	for (const auto &cell : triangulation.cell_iterators())
	  for (const auto &face : cell->face_iterators())
	    {
	      const auto center = face->center();
	      if (
		  (std::fabs(center(0) - (-half_length)) < 1e-12)
		                   ||
		  (std::fabs(center(0) - (half_length)) < 1e-12)
		                   ||
		  (std::fabs(center(1) - (-half_length)) < 1e-12)
		                   ||
		  (std::fabs(center(1) - (half_length)) < 1e-12)
		                   ||
		  (std::fabs(center(2) - 0.0) < 1e-12)
		                   ||
		  (std::fabs(center(2) - (z_extension)) < 1e-12)
		  )
		face->set_boundary_id(1);

	      if (std::fabs(std::sqrt(center(0) * center(0)
				      + center(1) * center(1))
			    - inner_radius) <=0.15)
		face->set_boundary_id(0);
	    }

	triangulation.refine_global(3);
      }
  }

  template <int dim>
  void FemGL<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs(fe);

    pcout << "   Number of degrees of freedom: "
	  << dof_handler.n_dofs()
	  << std::endl;

    locally_owned_dofs     = dof_handler.locally_owned_dofs();
    //locally_relevant_dofs  = DoFTools::extract_locally_relevant_dofs(dof_handler);
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);    

    {
      constraints_newton_update.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_newton_update);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               DirichletBCs_newton_update<dim>(),
                                               constraints_newton_update);
                                               //fe.component_mask(velocities));
      constraints_newton_update.close();
    }

    {
      constraints_solution.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints_solution);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               constraints_solution);
                                               //fe.component_mask(velocities));
      constraints_solution.close();
    }

    {
      system_matrix.clear();

      TrilinosWrappers::SparsityPattern sp(locally_owned_dofs, mpi_communicator);

      DoFTools::make_sparsity_pattern(dof_handler, sp, constraints_newton_update, false,
				      Utilities::MPI::this_mpi_process(mpi_communicator));
      // exchange local dsp entries between processes      
      sp.compress();
      system_matrix.reinit(sp);
    }

    /*  set up initial local_solution Vector */
    /*---------------------------------------*/
    std::random_device rd{};         // rd will be used to obtain a seed for the random number engine
    std::mt19937       gen{rd()};    // Standard mersenne_twister_engine seeded with rd()

    std::normal_distribution<double> gaussian_distr{3.0, 1.0}; // gaussian distribution with mean 10. STD 6.0

    local_solution.reinit(locally_relevant_dofs,
    			  mpi_communicator,
    			  false);
    LA::MPI::Vector distrubuted_tmp_solution(locally_owned_dofs,
                                             mpi_communicator);
    //distrubuted_tmp_solution = 3.0;
    for (auto it = distrubuted_tmp_solution.begin(); it != distrubuted_tmp_solution.end(); ++it)
      {
	*it = gaussian_distr(gen);
      }

    // AffineConstriant::distribute call
    constraints_solution.distribute(distrubuted_tmp_solution);

    local_solution = distrubuted_tmp_solution;    
    /*---------------------------------------*/
    
    locally_relevant_newton_solution.reinit(locally_relevant_dofs,
                                            mpi_communicator);
    locally_relevant_damped_vector.reinit(locally_relevant_dofs,
				          mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    residual_vector.reinit(locally_owned_dofs, mpi_communicator);    
  }



  template <int dim>
  void FemGL<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    system_rhs            = 0;

    const QGauss<dim> quadrature_formula(degree + 1);

    FEValues<dim> fe_values(fe,
			    quadrature_formula,
			    update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    /* --------------------------------------------------------------------------------
     * matrices objects for old_solution_u, old_solution_v tensors,
     * matrices objects for old_solution_gradients_u, old_solution_gradients_v.
     * --------------------------------------------------------------------------------
     */

    FullMatrix<double> old_solution_u(3,3);
    FullMatrix<double> old_solution_v(3,3);

    // containers of old_solution_gradients_u(3,3), old_solution_gradient_v(3,3);
    const FullMatrix<double>        identity (IdentityMatrix(3));
    std::vector<FullMatrix<double>> grad_old_u_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_old_v_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]

    /* --------------------------------------------------------------------------------
     * matrices objects for shape functions tensors phi^u, phi^v.    
     * These matrices depend on local Dof and Gaussian quadrature point
     * --------------------------------------------------------------------------------
     */

    FullMatrix<double> phi_u_i_q(3,3);
    FullMatrix<double> phi_u_j_q(3,3);
    FullMatrix<double> phi_v_i_q(3,3);
    FullMatrix<double> phi_v_j_q(3,3);

    //types::global_dof_index spacedim = (dim == 2)? 2 : 3;              // using  size_type = types::global_dof_index
    std::vector<FullMatrix<double>> grad_phi_u_i_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_phi_v_i_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_phi_u_j_q(dim, identity);     // grad_phi_u_j container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_phi_v_j_q(dim, identity);     // grad_phi_v_j container of gradient matrics, [0, dim-1]

    /* --------------------------------------------------------------------------------
     * matrics for free energy terms: gradient terms, alpha-term, beta-term,
     * they are products of phi tensors or u/v tensors.
     * --------------------------------------------------------------------------------
     */
    FullMatrix<double>              phi_u_phi_ut(3,3), phi_v_phi_vt(3,3);
    FullMatrix<double>              old_u_old_ut(3,3), old_v_old_vt(3,3);

    FullMatrix<double>              old_u_phi_ut_i_q(3,3),
                                    old_u_phi_ut_j_q(3,3),
                                    old_v_phi_vt_i_q(3,3),
                                    old_v_phi_vt_j_q(3,3);

    FullMatrix<double>              K1grad_matrics_sum_i_j_q(3,3),
                                    rhs_K1grad_matrics_sum_i_q(3,3);                // matrix for storing K1 gradients before trace
    FullMatrix<double>              phi_phit_matrics_i_j_q(3,3);
    /*--------------------------------------------------------------------------------*/

    /* --------------------------------------------------*/
    //std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    //std::vector<double>         div_phi_u(dofs_per_cell);
    //std::vector<double>         phi_p(dofs_per_cell);
    /* --------------------------------------------------*/

    /*--------------------------------------------------*/
    /* >>>>>>>>>>  old solution for r.h.s   <<<<<<<<<<< */
    // vector to holding u^n, grad u^n on cell:
    /*std::vector<Tensor<1, dim>> old_solution_gradients_u(n_q_points);
    std::vector<double>         old_solution_u(n_q_points);

    // vector to holding v^n, grad v^n on cell:
    std::vector<Tensor<1, dim>> old_solution_gradients_v(n_q_points);
    std::vector<double>         old_solution_v(n_q_points);*/
    /* --------------------------------------------------*/    

    /*std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector     velocities(0);
    const FEValuesExtractors::Scalar     pressure(dim);*/

    // FEValuesExtractors, works for FEValues, FEFaceValues, FESubFaceValues
    //const FEValuesExtractors::Scalar u_component(0);
    //const FEValuesExtractors::Scalar v_component(1);
    
    // vector-flag for old_soluttion
    char flag_solution = 's';

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
	{
	  cell_matrix  = 0;
	  cell_rhs     = 0;

	  fe_values.reinit(cell);
	  //right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
	  //pcout << " now get on new cell!" << std::endl;
	  
	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {
	      //pcout << " q-loop starts ! q is " << q << std::endl;
	      /*--------------------------------------------------*/
	      /*for (unsigned int k = 0; k < dofs_per_cell; ++k)
		{
		  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
		  div_phi_u[k]  = fe_values[velocities].divergence(k, q);
		  phi_p[k]      = fe_values[pressure].value(k, q);
		  }*/
              /*--------------------------------------------------*/
	      
	      /*--------------------------------------------------*/
	      /*         u, v matrices cooking up                 */
	      /*--------------------------------------------------*/
	      old_solution_u = 0.0;
	      old_solution_v = 0.0;

	      for (auto it = grad_old_u_q.begin(); it != grad_old_u_q.end(); ++it) {*it = 0.0;}
	      for (auto it = grad_old_v_q.begin(); it != grad_old_v_q.end(); ++it) {*it = 0.0;}

	      //pcout << " vector-matrix function call starts !" << std::endl;
	      vector_matrix_generator(fe_values, flag_solution, q, n_q_points, old_solution_u, old_solution_v);
	      grad_vector_matrix_generator(fe_values, flag_solution, q, n_q_points, grad_old_u_q, grad_old_v_q);
	      //pcout << " vector-matrix function call ends !" << std::endl;
	      
	      // u.ut and v.vt
	      old_u_old_ut   = 0.0;
	      old_v_old_vt   = 0.0;

	      old_solution_u.mTmult(old_u_old_ut, old_solution_u);
	      old_solution_v.mTmult(old_v_old_vt, old_solution_v);
	      /*--------------------------------------------------*/            

              //pcout << " now starts i-j loop! " << std::endl;
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
		  for (unsigned int j = 0; j < dofs_per_cell; ++j)
		    {
		      /*--------------------------------------------------*/
		      /*     phi^u, phi^v matrices cooking up             */
		      /*     grad_u, grad_v matrics cookup                */
		      /*--------------------------------------------------*/

		      phi_u_i_q = 0.0; phi_u_j_q = 0.0;
		      phi_v_i_q = 0.0; phi_v_j_q = 0.0;

		      for (auto it = grad_phi_u_i_q.begin(); it != grad_phi_u_i_q.end(); ++it) {*it = 0.0;}
		      for (auto it = grad_phi_v_i_q.begin(); it != grad_phi_v_i_q.end(); ++it) {*it = 0.0;}
		      for (auto it = grad_phi_u_j_q.begin(); it != grad_phi_u_j_q.end(); ++it) {*it = 0.0;}
		      for (auto it = grad_phi_v_j_q.begin(); it != grad_phi_v_j_q.end(); ++it) {*it = 0.0;}

		      phi_matrix_generator(fe_values, i, q, phi_u_i_q, phi_v_i_q);
		      phi_matrix_generator(fe_values, j, q, phi_u_j_q, phi_v_j_q);

		      grad_phi_matrix_container_generator(fe_values, i, q, grad_phi_u_i_q, grad_phi_v_i_q);
		      grad_phi_matrix_container_generator(fe_values, j, q, grad_phi_u_j_q, grad_phi_v_j_q);
		      /*--------------------------------------------------*/
		      phi_u_phi_ut        = 0.0;
		      phi_v_phi_vt        = 0.0;

		      old_u_phi_ut_i_q    = 0.0;
		      old_u_phi_ut_j_q    = 0.0;
		      old_v_phi_vt_i_q    = 0.0;
		      old_v_phi_vt_j_q    = 0.0;

		      K1grad_matrics_sum_i_j_q = 0.0;
		      phi_phit_matrics_i_j_q   = 0.0;
		      /*--------------------------------------------------*/

		      /* --------------------------------------------------
		       * conduct matrices multiplacations
		       * --------------------------------------------------
		       */
		      // phi_u_phi_ut_ijq, phi_v_phi_vt_ijq matrics
		      phi_u_i_q.mTmult(phi_u_phi_ut, phi_u_j_q);
		      phi_v_i_q.mTmult(phi_v_phi_vt, phi_v_j_q);

		      // old_u/v_phi_ut/vt_i/jq matrics
		      old_solution_u.mTmult(old_u_phi_ut_i_q, phi_u_i_q);
		      old_solution_u.mTmult(old_u_phi_ut_j_q, phi_u_j_q);
		      old_solution_v.mTmult(old_v_phi_vt_i_q, phi_v_i_q);
		      old_solution_v.mTmult(old_v_phi_vt_j_q, phi_v_j_q);

		      // alpha terms matrics
		      phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);

		      for (unsigned int k = 0; k < dim; ++k)
			{
			  grad_phi_u_i_q[k].mTmult(K1grad_matrics_sum_i_j_q, grad_phi_u_j_q[k], true);
			  grad_phi_v_i_q[k].mTmult(K1grad_matrics_sum_i_j_q, grad_phi_v_j_q[k], true);
			}
		      
		      /*--------------------------------------------------*/		          
		      
		      cell_matrix(i,j) +=
			(((K1
			   * K1grad_matrics_sum_i_j_q.trace())
			  +(alpha_0
			    * (reduced_t-1.0)
			    * phi_phit_matrics_i_j_q.trace())
			  +(beta
			    * ((old_u_old_ut.trace() + old_v_old_vt.trace()) * phi_phit_matrics_i_j_q.trace()
			       + 2.0 * ((old_u_phi_ut_i_q.trace() + old_v_phi_vt_i_q.trace())
					* (old_u_phi_ut_j_q.trace() + old_v_phi_vt_j_q.trace()))))
			  ) * fe_values.JxW(q));
		      //pcout << " now one j loop finishes! j is " << j << std::endl;		      
		    } // cell_matrix ends here

		  rhs_K1grad_matrics_sum_i_q = 0.0;
		  for (unsigned int k = 0; k < dim; ++k)
		    {
		      grad_old_u_q[k].mTmult(rhs_K1grad_matrics_sum_i_q, grad_phi_u_i_q[k], true);
		      grad_old_v_q[k].mTmult(rhs_K1grad_matrics_sum_i_q, grad_phi_v_i_q[k], true);
		    }

		  cell_rhs(i) -=
		    (((K1
		       * rhs_K1grad_matrics_sum_i_q.trace())
		      +(alpha_0
			* (reduced_t-1.0)
			* (old_u_phi_ut_i_q.trace()
			   + old_v_phi_vt_i_q.trace()))
		      +(beta
			* (old_u_old_ut.trace() + old_v_old_vt.trace())
			* (old_u_phi_ut_i_q.trace() + old_v_phi_vt_i_q.trace()))
		      ) * fe_values.JxW(q));                      // * dx
	          // pcout << " now get one i-loop finished! i is " << i << std::endl;  
		} // i-index ends here

	    } // q-loop ends at here

       	  //pcout << " now q-loop finished!" << std::endl;
	  cell->get_dof_indices(local_dof_indices);
	  constraints_newton_update.distribute_local_to_global(cell_matrix,
				  		               cell_rhs,
						               local_dof_indices,
						               system_matrix,
						               system_rhs);
          //pcout << " just distribited cell contribution! " << std::endl;
       }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void FemGL<dim>::compute_residual(/*const LA::MPI::Vector &damped_vector*/)
  {
    TimerOutput::Scope t(computing_timer, "compute_residual");

    residual_vector = 0;

    const QGauss<dim> quadrature_formula(degree + 1);

    FEValues<dim> fe_values(fe,
			    quadrature_formula,
			    update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Vector<double>     cell_rhs(dofs_per_cell);

    /* --------------------------------------------------------------------------------
     * matrices objects for old_solution_u, old_solution_v tensors,
     * matrices objects for old_solution_gradients_u, old_solution_gradients_v.
     * --------------------------------------------------------------------------------
     */

    FullMatrix<double> old_solution_u(3,3);
    FullMatrix<double> old_solution_v(3,3);

    // containers of old_solution_gradients_u(3,3), old_solution_gradient_v(3,3);
    const FullMatrix<double>        identity (IdentityMatrix(3));
    std::vector<FullMatrix<double>> grad_old_u_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_old_v_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]

    /* --------------------------------------------------------------------------------
     * matrices objects for shape functions tensors phi^u, phi^v.    
     * These matrices depend on local Dof and Gaussian quadrature point
     * --------------------------------------------------------------------------------
     */

    FullMatrix<double> phi_u_i_q(3,3);
    FullMatrix<double> phi_v_i_q(3,3);

    //types::global_dof_index spacedim = (dim == 2)? 2 : 3;              // using  size_type = types::global_dof_index
    std::vector<FullMatrix<double>> grad_phi_u_i_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_phi_v_i_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]

    /* --------------------------------------------------------------------------------
     * matrics for free energy terms: gradient terms, alpha-term, beta-term,
     * they are products of phi tensors or u/v tensors.
     * --------------------------------------------------------------------------------
     */
    FullMatrix<double>              phi_u_phi_ut(3,3), phi_v_phi_vt(3,3);
    FullMatrix<double>              old_u_old_ut(3,3), old_v_old_vt(3,3);

    FullMatrix<double>              old_u_phi_ut_i_q(3,3),
                                    old_v_phi_vt_i_q(3,3);

    FullMatrix<double>              rhs_K1grad_matrics_sum_i_q(3,3); // matrix for storing K1 gradients before trace
    /*--------------------------------------------------------------------------------*/

    char flag_solution = 'l'; // "l" means linear search
    
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
	{
	  cell_rhs     = 0;

	  fe_values.reinit(cell);
	  //right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
	  
	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {

	      /*--------------------------------------------------*/
	      /*         u, v matrices cooking up                 */
	      /*--------------------------------------------------*/
	      old_solution_u = 0.0;
	      old_solution_v = 0.0;

	      for (auto it = grad_old_u_q.begin(); it != grad_old_u_q.end(); ++it) {*it = 0.0;}
	      for (auto it = grad_old_v_q.begin(); it != grad_old_v_q.end(); ++it) {*it = 0.0;}

	      vector_matrix_generator(fe_values, flag_solution, q, n_q_points, old_solution_u, old_solution_v);
	      grad_vector_matrix_generator(fe_values, flag_solution, q, n_q_points, grad_old_u_q, grad_old_v_q);

	      // u.ut and v.vt
	      old_u_old_ut   = 0.0;
	      old_v_old_vt   = 0.0;

	      old_solution_u.mTmult(old_u_old_ut, old_solution_u);
	      old_solution_v.mTmult(old_v_old_vt, old_solution_v);
	      /*--------------------------------------------------*/            
	      
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
		  /*--------------------------------------------------*/
		  /*     phi^u, phi^v matrices cooking up             */
		  /*     grad_u, grad_v matrics cookup                */
		  /*--------------------------------------------------*/

		  phi_u_i_q = 0.0; 
		  phi_v_i_q = 0.0; 

		  for (auto it = grad_phi_u_i_q.begin(); it != grad_phi_u_i_q.end(); ++it) {*it = 0.0;}
		  for (auto it = grad_phi_v_i_q.begin(); it != grad_phi_v_i_q.end(); ++it) {*it = 0.0;}

		  phi_matrix_generator(fe_values, i, q, phi_u_i_q, phi_v_i_q);

		  grad_phi_matrix_container_generator(fe_values, i, q, grad_phi_u_i_q, grad_phi_v_i_q);

		  /*--------------------------------------------------*/
		  old_u_phi_ut_i_q    = 0.0;
		  old_v_phi_vt_i_q    = 0.0;
		  /*--------------------------------------------------*/

		  /* --------------------------------------------------
		   * conduct matrices multiplacations
		   * --------------------------------------------------
		   */

		  // old_u/v_phi_ut/vt_i/jq matrics
		  old_solution_u.mTmult(old_u_phi_ut_i_q, phi_u_i_q);
		  old_solution_v.mTmult(old_v_phi_vt_i_q, phi_v_i_q);

		  // alpha terms matrics
		  //phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);
		  
		  rhs_K1grad_matrics_sum_i_q = 0.0;
		  for (unsigned int k = 0; k < dim; ++k)
		    {
		      grad_old_u_q[k].mTmult(rhs_K1grad_matrics_sum_i_q, grad_phi_u_i_q[k], true);
		      grad_old_v_q[k].mTmult(rhs_K1grad_matrics_sum_i_q, grad_phi_v_i_q[k], true);
		    }

		  cell_rhs(i) -=
		    (((K1
		       * rhs_K1grad_matrics_sum_i_q.trace())
		      +(alpha_0
			* (reduced_t-1.0)
			* (old_u_phi_ut_i_q.trace()
			   + old_v_phi_vt_i_q.trace()))
		      +(beta
			* (old_u_old_ut.trace() + old_v_old_vt.trace())
			* (old_u_phi_ut_i_q.trace() + old_v_phi_vt_i_q.trace()))
		      ) * fe_values.JxW(q));                      // * dx

		} // i-index ends here
	    } // q-loop ends at here

	  cell->get_dof_indices(local_dof_indices);
	  constraints_newton_update.distribute_local_to_global(cell_rhs,
						               local_dof_indices,
						               residual_vector);
	}

    residual_vector.compress(VectorOperation::add);
  }
  
  template <int dim>
  void FemGL<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::PreconditionAMG preconditioner;
    {
      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");

      pcout << " start to build AMG preconditioner." << std::endl;      
      std::vector<std::vector<bool>> constant_modes;
      DoFTools::extract_constant_modes(dof_handler,
				       ComponentMask(),
				       constant_modes);

      TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
      additional_data.constant_modes        = constant_modes;
      additional_data.elliptic              = true;
      additional_data.n_cycles              = 1;
      additional_data.w_cycle               = false;
      additional_data.output_details        = false;
      additional_data.smoother_sweeps       = 2;
      additional_data.aggregation_threshold = 1e-2;

      preconditioner.initialize(system_matrix, additional_data);
      pcout << " AMG preconditioner is built up." << std::endl;      
    }

    // With that, we can finally set up the linear solver and solve the system:
    pcout << " system_rhs.l2_norm() is " << system_rhs.l2_norm() << std::endl;
    SolverControl solver_control(10*system_matrix.m(),
                                 1e-1 * system_rhs.l2_norm());

    //SolverMinRes<LA::MPI::Vector> solver(solver_control);
    SolverFGMRES<LA::MPI::Vector> solver(solver_control);
    //SolverGMRES<LA::MPI::Vector> solver(solver_control, gmres_adddata);

    LA::MPI::Vector distributed_newton_update(locally_owned_dofs, mpi_communicator);

    // what this .set_zero() is doing ?
    // AffineContraint::set_zero() set the values of all constrained DoFs in a vector to zero. 
    // constraints_newton_update.set_zero(distributed_newton_update);

    solver.solve(system_matrix,
                 distributed_newton_update,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints_newton_update.distribute(distributed_newton_update);

    locally_relevant_newton_solution = distributed_newton_update;

  }

  template <int dim>
  void FemGL<dim>::newton_iteration()
  {
    TimerOutput::Scope t(computing_timer, "newton_iteration");    

    LA::MPI::Vector distributed_newton_update(locally_owned_dofs, mpi_communicator);
    LA::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
    
    double previous_residual = system_rhs.l2_norm();

    for (unsigned int i = 0; i < 10; ++i)
      {
	const double alpha = std::pow(0.5, static_cast<double>(i));
        distributed_newton_update = locally_relevant_newton_solution;
        distributed_solution      = local_solution;

	// damped iteration:
	distributed_solution.add(alpha, distributed_newton_update);

	// AffineConstraint::distribute call
        constraints_solution.distribute(distributed_solution);	

	// assign un-ghosted solution to ghosted solution
	locally_relevant_damped_vector = distributed_solution;

	compute_residual(/*locally_relevant_damped_vector*/);
	double current_residual = residual_vector.l2_norm();

	pcout << " step length alpha is: "  << alpha
	      << ", residual is: "          << current_residual
	      << ", previous_residual is: " << previous_residual
	      << std::endl;
	if (current_residual < previous_residual)
	  {
	    pcout << " ohh! current_residual < previous_residual, we get better solution ! "
	          << std::endl;
  	    break;
          }
	else
	  {
            pcout << " haa! current_residual >= previous_residual, more line search ! "
	          << std::endl;
	  }
      }

    local_solution = distributed_solution;
  }

  template <int dim>
  void FemGL<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    triangulation.refine_global();
  }



  template <int dim>
  void FemGL<dim>::output_results(const unsigned int cycle) const
  {

    std::vector<std::string> newton_update_components_names;
    newton_update_components_names.emplace_back("du_11");
    newton_update_components_names.emplace_back("du_12");
    newton_update_components_names.emplace_back("du_13");
    newton_update_components_names.emplace_back("du_21");
    newton_update_components_names.emplace_back("du_22");
    newton_update_components_names.emplace_back("du_23");
    newton_update_components_names.emplace_back("du_31");
    newton_update_components_names.emplace_back("du_32");
    newton_update_components_names.emplace_back("du_33");
    newton_update_components_names.emplace_back("dv_11");
    newton_update_components_names.emplace_back("dv_12");
    newton_update_components_names.emplace_back("dv_13");
    newton_update_components_names.emplace_back("dv_21");
    newton_update_components_names.emplace_back("dv_22");
    newton_update_components_names.emplace_back("dv_23");
    newton_update_components_names.emplace_back("dv_31");
    newton_update_components_names.emplace_back("dv_32");
    newton_update_components_names.emplace_back("dv_33");

    std::vector<std::string> solution_components_names;
    solution_components_names.emplace_back("u_11");
    solution_components_names.emplace_back("u_12");
    solution_components_names.emplace_back("u_13");
    solution_components_names.emplace_back("u_21");
    solution_components_names.emplace_back("u_22");
    solution_components_names.emplace_back("u_23");
    solution_components_names.emplace_back("u_31");
    solution_components_names.emplace_back("u_32");
    solution_components_names.emplace_back("u_33");
    solution_components_names.emplace_back("v_11");
    solution_components_names.emplace_back("v_12");
    solution_components_names.emplace_back("v_13");
    solution_components_names.emplace_back("v_21");
    solution_components_names.emplace_back("v_22");
    solution_components_names.emplace_back("v_23");
    solution_components_names.emplace_back("v_31");
    solution_components_names.emplace_back("v_32");
    solution_components_names.emplace_back("v_33");
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_newton_solution, newton_update_components_names,
			     DataOut<dim>::type_dof_data);
    data_out.add_data_vector(local_solution, solution_components_names,
			     DataOut<dim>::type_dof_data);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);
  }

  template <int dim>
  void FemGL<dim>::run()
  {
    pcout << "Running using Trilinos." << std::endl;

    const unsigned int n_cycles    = 1;
    const unsigned int n_iteration = 30;    
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          make_grid();
        else
          refine_grid();
	
        for (unsigned int iteration_loop = 0; iteration_loop <= n_iteration; ++iteration_loop)
	  {
	    pcout << "iteration_loop: " << iteration_loop << std::endl;
	    if (iteration_loop == 0)
	      {setup_system();}

             assemble_system();
             solve();
	     newton_iteration();

             if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 1280)
              {
               TimerOutput::Scope t(computing_timer, "output");
               //output_results(cycle);
	       output_results(iteration_loop);
              }

             computing_timer.print_summary();
             computing_timer.reset();

             pcout << std::endl;

	     if (system_rhs.l2_norm() < 1e-10)
	       break;
 
	  }

      }
     computing_timer.print_summary();	
  }

  /* --------------------------------------------------------------------------------
   * old_solution_matrix_generator
   * phi_matrix_generator
   * grad_phi_matrix_container_generator
   * --------------------------------------------------------------------------------
   */
  template <int dim>
  void FemGL<dim>::vector_matrix_generator(const FEValues<dim>  &fe_values,
					   const char &vector_flag,
					   const unsigned int   q, const unsigned int n_q_points,
					   FullMatrix<double>   &u_matrix_at_q,
					   FullMatrix<double>   &v_matrix_at_q)
  {
    LA::MPI::Vector vector_solution(locally_relevant_dofs, mpi_communicator);
    //LA::MPI::Vector *ptr_vector_solution;
    switch (vector_flag)
      {
      case 's':
	vector_solution = local_solution; break;
	//ptr_vector_solution = &local_solution; break;
      case 'd':
	vector_solution = locally_relevant_newton_solution; break;
      case 'l':
	vector_solution = locally_relevant_damped_vector; break;	
      }
    // vector to holding u_mu_i^n, grad u_mu_i^n on cell:
    /*std::vector<Tensor<1, dim>> old_solution_gradients_u11(n_q_points);
      std::vector<double>         old_solution_u11(n_q_points);*/
    std::vector<double>                  vector_solution_uxx(n_q_points),
                                         vector_solution_vxx(n_q_points);

    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	fe_values[components_u[comp_index]].get_function_values(vector_solution, vector_solution_uxx);
	fe_values[components_v[comp_index]].get_function_values(vector_solution, vector_solution_vxx);

	u_matrix_at_q.set(comp_index/3u, comp_index%3u, vector_solution_uxx[q]);
	v_matrix_at_q.set(comp_index/3u, comp_index%3u, vector_solution_vxx[q]);

      }
    /*fe_values[v33_component].get_function_gradients(old_solution, old_solution_gradients_v33);
      fe_values[v33_component].get_function_values(old_solution, old_solution_v33);*/
    /*--------------------------------------------------*/
    // old_u_matrix_at_q.set(0,0,old_solution_u_container[0][q]);
    // old_v_matrix_at_q.set(2,2,old_solution_v_container[8][q]);
    // /*--------------------------------------------------*/
    /* for (unsigned int k = 0; k < dim; ++k)
      {
      grad_old_u[k].set(0,0,old_solution_gradients_u11[q][k]);
      grad_old_v[k].set(2,2,old_solution_gradients_v33[q][k]);
      }*/
  }

  template <int dim>
  void FemGL<dim>::grad_vector_matrix_generator(const FEValues<dim>  &fe_values,
							const char &vector_flag,
							const unsigned int q, const unsigned int n_q_points,
							std::vector<FullMatrix<double>> &grad_u_at_q,
							std::vector<FullMatrix<double>> &grad_v_at_q)
  {
    LA::MPI::Vector vector_solution(locally_relevant_dofs, mpi_communicator);
    //LA::MPI::Vector *ptr_vector_solution;
    switch (vector_flag)
      {
      case 's':
	vector_solution = local_solution; break;
      case 'd':
	vector_solution = locally_relevant_newton_solution; break;
      case 'l':
	vector_solution = locally_relevant_damped_vector; break;	
      }
    // vector to holding u_mu_i^n, grad u_mu_i^n on cell:
    std::vector< Tensor<1, dim> >          container_solution_gradients_uxx(n_q_points),
                                           container_solution_gradients_vxx(n_q_points);

    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	fe_values[components_u[comp_index]].get_function_gradients(vector_solution, container_solution_gradients_uxx);
	fe_values[components_v[comp_index]].get_function_gradients(vector_solution, container_solution_gradients_vxx);

	for (unsigned int k = 0; k < dim; ++k) // loop over dim spatial derivatives [0, dim-1]
	  {
	    grad_u_at_q[k].set(comp_index/3u, comp_index%3u, container_solution_gradients_uxx[q][k]);
	    grad_v_at_q[k].set(comp_index/3u, comp_index%3u, container_solution_gradients_vxx[q][k]);
	  }
      }

  }

  template <int dim>
  void FemGL<dim>::phi_matrix_generator(const FEValues<dim> &fe_values,
					      const unsigned int  x, const unsigned int q,
					      FullMatrix<double>  &phi_u_at_x_q,
					      FullMatrix<double>  &phi_v_at_x_q)
  {
    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	phi_u_at_x_q.set(comp_index/3u, comp_index%3u, fe_values[components_u[comp_index]].value(x, q));
	phi_v_at_x_q.set(comp_index/3u, comp_index%3u, fe_values[components_v[comp_index]].value(x, q));
      }
    /*matrix_u_at_x_q.set(0,0,fe_values[u11_component].value(x, q));
      matrix_v_at_x_q.set(2,2,fe_values[v33_component].value(x, q));*/
  }

  template <int dim>
  void FemGL<dim>::grad_phi_matrix_container_generator(const FEValues<dim> &fe_values,
							     const unsigned int x, const unsigned int q,
							     std::vector<FullMatrix<double>> &container_grad_phi_u_x_q,
							     std::vector<FullMatrix<double>> &container_grad_phi_v_x_q)
  {
    /* auto grad_u11_x_q = fe_values[u11_component].gradient(x, q);
       auto grad_v33_x_q = fe_values[v33_component].gradient(x, q);*/
    for (unsigned int comp_index = 0; comp_index <= 8; ++comp_index)
      {
	auto gradient_uxx_at_q = fe_values[components_u[comp_index]].gradient(x, q);
	auto gradient_vxx_at_q = fe_values[components_v[comp_index]].gradient(x, q);

	for (unsigned int k = 0; k < dim; ++k)
	  {
	    container_grad_phi_u_x_q[k].set(comp_index/3u, comp_index%3u, gradient_uxx_at_q[k]);
	    container_grad_phi_v_x_q[k].set(comp_index/3u, comp_index%3u, gradient_vxx_at_q[k]);
	  }
      }

  }

  /*--------------------------------------------------------------------------------*/
  /*             ^^^^  All matrices genertors end at here  ^^^                      */
  /*--------------------------------------------------------------------------------*/
    
} // namespace FemGL_mpi


int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace FemGL_mpi;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      //FemGL<2> GLsolver(2);
      FemGL<3> GLsolver(1);
      GLsolver.run();
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
