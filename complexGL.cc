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
 * 11. Tammikuu. 2023.
 *
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/solver_cg.h>
//#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/solution_transfer.h>

// split global scope into ScalarGL and main() block scope
namespace complexGL
{
  using namespace dealii;

  template <int dim>
  class ComplexValuedScalarGLSolver
  {
  public:
    // calss template constructor:
    ComplexValuedScalarGLSolver();
    void run();

  private:

    // member functons of Solver class template:
    void   setup_dof_initilize_system(const bool initial_step);
    void   assemble_system();
    void   solve();
    void   newton_iteration();
    void   refine_mesh();
    void   set_boundary_values();


    // double compute_residual(const double alpha) const;
    // double determine_step_length() const;
    std::pair<double, double> line_search_residual_and_gradient(double init_stepLength) const
    double compute_residual() const;
    void   output_results(const unsigned int &refinement_cycle) const;


    // data members of solver class template.
    // those variables are necessary parts for adaptive meshed FEM and newton iteration.
    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    FESystem<dim>      fe;

    // hanging nodes constriants object:
    AffineConstraints<double> hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> newton_update;
    Vector<double> current_solution;
    Vector<double> system_rhs;

    // physical coefficients of scalar GL equation:
    const double alpha_0 = 2.0;
    const double beta = 0.5;

    // dimensinless temperature
    double t; 
  };

  
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

  // construnctor and initilizaton list
  template <int dim>
  ComplexValuedScalarGLSolver<dim>::ComplexValuedScalarGLSolver()
    : triangulation(Triangulation<dim>::maximum_smoothing)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(2), 2)
    , t(0.0)
  {}


  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::setup_dof_initilize_system(const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs(fe);
        current_solution.reinit(dof_handler.n_dofs(), /*omit_zeroing_entries*/true);
	//current_solution.print();

	for (auto it = current_solution.begin(); it != current_solution.end(); ++it)
	  *it = 4.0;

	// clear-fill-close AffineConstriants with hanging-node-constriant for new refined
        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);
        hanging_node_constraints.close();

	// print dof and cells info
	std::cout << "Number of active cells in this mesh is " << triangulation.n_active_cells()
	          << std::endl
	          << "Number of DoF in this mesh is " << dof_handler.n_dofs() << "\n"
	          << std::endl;
      }


    /*
     * system initilization
     */
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    // pass hanging_node_constraints as 3rd parameter for DSP
    DoFTools::make_sparsity_pattern(dof_handler, dsp, hanging_node_constraints); 
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    // condense dsp with constriants object, that adds these positions
    // into sparse pattern, which are required for eliminaton of constraints:
    // hanging_node_constraints.condense(dsp);
    
  }

  
  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::assemble_system()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    system_matrix = 0;
    system_rhs    = 0;

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_gradients | update_values |
			      update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    // vector to holding u^n, grad u^n on cell:
    std::vector<Tensor<1, dim>> old_solution_gradients_u(n_q_points);
    std::vector<double> old_solution_u(n_q_points);

    // vector to holding v^n, grad v^n on cell:
    std::vector<Tensor<1, dim>> old_solution_gradients_v(n_q_points);
    std::vector<double> old_solution_v(n_q_points);

    const FEValuesExtractors::Scalar u_component(0);
    const FEValuesExtractors::Scalar v_component(1);

    // local DoF indices vector for types::global_dof_index
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
	// re-fresh cell matrix, cell-rhs vector, and cell fe_values
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

	/*
         * FeValuesViews[Extractor] returns on-cell values, 
         * gradients of unkown functions corresponding to
         * given indexed component through Extractor.
         */
	fe_values[u_component].get_function_gradients(current_solution, old_solution_gradients_u);
        fe_values[u_component].get_function_values(current_solution, old_solution_u);
	fe_values[v_component].get_function_gradients(current_solution, old_solution_gradients_v);
        fe_values[v_component].get_function_values(current_solution, old_solution_v);

	
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
	    // alpha + bete (3*u^(n)^2 + v^{n}^2), u-coeff, system_matrix
            const double bulkTerm_u_coeff_sysMatr =
	      alpha_0 * (t-1.0)
	       + (beta * (3.0 * old_solution_u[q] * old_solution_u[q] + old_solution_v[q] * old_solution_v[q])); 

	    // alpha + bete (3*v^(n)^2 + u^{n}^2), v-coeff, system_matrix
            const double bulkTerm_v_coeff_sysMatr =
	      alpha_0 * (t-1.0)
	       + (beta * (3.0 * old_solution_v[q] * old_solution_v[q] + old_solution_u[q] * old_solution_u[q]));

            const double bulkTerm_uv_coeff_sysMatr =
	       (2.0 * beta * old_solution_v[q] * old_solution_u[q]);
	    
	    // alpha + bete (u^(n)^2 + v^(n)^2), u/v-coef, rhs
	    const double bulkTerm_coeff_rhs =
	      alpha_0 * (t-1.0)
       	       + beta * (old_solution_u[q] * old_solution_u[q] + old_solution_v[q] * old_solution_v[q]);

	    
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
		  {
		   cell_matrix(i, j) +=
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

		    

                cell_rhs(i) -=
		 (((fe_values[u_component].gradient(i, q)   // ((\partial_m \phi_i
		    * old_solution_gradients_u[q]           //   * \partial_m \psi^(n)
		    * 0.5)                                  //   * 1/2)
		   +                             //  +
		   (bulkTerm_coeff_rhs                      //  ((\alpha + \beta \psi^(n)2)
		    * fe_values[u_component].value(i, q)      //   * \phi_i
		    * old_solution_u[q])                   //   * \psi^(n)))
		   +
		   (fe_values[v_component].gradient(i, q)   // ((\partial_m \phi_i
		    * old_solution_gradients_v[q]           //   * \partial_m \psi^(n)
		    * 0.5)                                  //   * 1/2)
		   +
		   (bulkTerm_coeff_rhs                      //  ((\alpha + \beta \psi^(n)2)
		    * fe_values[v_component].value(i, q)      //   * \phi_i
		    * old_solution_v[q]))                   //   * \psi^(n)))		  
		  * fe_values.JxW(q));                      // * dx 
		
              }
          }

        /*
         * distribute cell contributions back to system-matrix
         */
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    // condense Vector and SparseMatrix with hanging node constriants:
    // hanging_node_constraints.condense(system_matrix);
    // hanging_node_constraints.condense(system_rhs);

    // the newton_update has zero value boundary condtion
    std::map<types::global_dof_index, double> newton_update_boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             DirichletBCs_newton_update<dim>(),
                                             newton_update_boundary_values);
    MatrixTools::apply_boundary_values(newton_update_boundary_values,
                                       system_matrix,
                                       newton_update,
                                       system_rhs);
  }


  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::solve()
  {
    SparseDirectUMFPACK inverse_system_matrix;
    inverse_system_matrix.initialize(system_matrix);

    inverse_system_matrix.vmult(newton_update,system_rhs);
    
    /*
    SolverControl            solver_control(system_rhs.size(),
                                 system_rhs.l2_norm() * 1e-6);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
    */

    hanging_node_constraints.distribute(newton_update);  // distribute the constriants

    // const double alpha = determine_step_length();
    // current_solution.add(alpha, newton_update);
  }

  /* ------------------------------------------------------------------------------------------
   * This member function returns iterative step length a^k.
   * The best way to do this is implement a line-search for the optimatical value of a^k,
   * but here we simply return 0.1, which is not the opttical value and garantee convergency.
   * ------------------------------------------------------------------------------------------
   */
  template <int dim>
  double ComplexValuedScalarGLSolver<dim>::determine_step_length() const
  {
    //   return 0.82;




    
  // The values for eta, mu are chosen such that more strict convergence
  // conditions are enforced.
  // They should be adjusted according to the problem requirements.
  const double a1        = 1.0;
  const double eta       = 0.5;
  const double mu        = 0.49;
  const double a_max     = 1.25;
  const double max_evals = 20;
  const auto   res = LineMinimization::line_search<double>(
    ls_minimization_function,
    res_0.first, res_0.second,
    LineMinimization::poly_fit<double>,
    a1, eta, mu, a_max, max_evals));
 
  return res.first; // Final stepsize

  
  }

  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::newton_iteration()
  {
    const double alpha = determine_step_length();
    current_solution.add(alpha, newton_update);
  }

 /*
  * ------------------------------------------------------------------------------------------
  * refine-mesh 
  * ------------------------------------------------------------------------------------------ 
  */

  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      // std::map<types::boundary_id, const Function<dim> *>(),
      {}, // there is no Neumann boundary, then the std::map is empty
      current_solution,
      estimated_error_per_cell);  // using default ComponentMask

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,   // refine 40%  
                                                    0.03); // de-fine/coarsen 3% 

    // preparing for SolutionTransfer class:
    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, Vector<double>> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(current_solution);

    // execute the refine and coarsen:
    triangulation.execute_coarsening_and_refinement();


    /* ------------------------------------------------------------------------------------------ */
    // distribute Dof for new mesh:
    dof_handler.distribute_dofs(fe);
    // fill hanging nodes constraints before distributing new
    // constriants of refined mesh. It is significant.
    hanging_node_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();
    /* ------------------------------------------------------------------------------------------ */

   
    // intepolate current_result on new mesh:
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(current_solution, tmp);

    
    current_solution.reinit(dof_handler.n_dofs());
    current_solution = tmp;
    // hanging_node_constraints.distribute(current_solution);

    // set the boundary values of current result correct on new mesh 
    set_boundary_values();

    // bool false will let set_up() knows Dof has been distributed
    setup_dof_initilize_system(false);
  }


  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::set_boundary_values()
  {
    // associtive contaioner for hold the BV of current_solution:
    std::map<types::global_dof_index, double> boundary_values;
    
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             boundary_values);
    for (auto &boundary_value : boundary_values)
      current_solution(boundary_value.first) = boundary_value.second;

    // distribute constriants to make old solution on new
    // mesh keeping finite elemnet field continuous
    hanging_node_constraints.distribute(current_solution);
  }


  template <int dim>
  std::pair<double, double> ComplexValuedScalarGLSolver<dim>::line_search_residual_and_gradient(double init_stepLength) const
  {

        Vector<double> evaluation_point(dof_handler.n_dofs());
    evaluation_point = current_solution;
    evaluation_point.add(alpha, newton_update);

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_gradients | update_values |
  			      update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>              cell_residual(dofs_per_cell);

    // two std::vector for holding on-cell gradients and value of FE-feild i.e., \psi
    std::vector<Tensor<1, dim>> gradients(n_q_points);
    std::vector<double> solution(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_residual = 0;
        fe_values.reinit(cell);

        // fill on-cell gradients:
        fe_values.get_function_gradients(evaluation_point, gradients);

  	// fill on-cell solution value:
        fe_values.get_function_values(evaluation_point, solution);


        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            // alpha + bete \psi^(n)^2, rhs
  	    const double bulk_term_coeff_rhs =
  	     ((alpha_0 * (t-1.0))
  	      + (beta * solution[q] * solution[q]));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_residual(i) -=
  		(((fe_values.shape_grad(i, q)    // ((\partial_m \phi_i
  		    * gradients[q]               //   * \partial_m \psi^(n)
  		    * 0.5)                       //   * 1/2)
  		   +                             //  +
  		   (bulk_term_coeff_rhs          //  ((\alpha + \beta \psi^(n)2)
  		    * fe_values.shape_value(i, q)      //   * \phi_i
  		    * solution[q]))              //   * \psi^(n)))
  		 * fe_values.JxW(q));            // * dx 
          }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          residual(local_dof_indices[i]) += cell_residual(i);
	

    for (types::global_dof_index i :
         DoFTools::extract_boundary_dofs(dof_handler))
      residual(i) = 0;


    /*
     * derivative of alpha
     */



    return std::make_pair(residual, derivative)

  }
  

  template <int dim>
  double ComplexValuedScalarGLSolver<dim>::compute_residual() const
  {
    Vector<double> residual(dof_handler.n_dofs());
    residual = system_rhs;

  //   Vector<double> evaluation_point(dof_handler.n_dofs());
  //   evaluation_point = current_solution;
  //   evaluation_point.add(alpha, newton_update);

  //   const QGauss<dim> quadrature_formula(fe.degree + 1);
  //   FEValues<dim>     fe_values(fe,
  //                           quadrature_formula,
  //                           update_gradients | update_values |
  // 			      update_quadrature_points | update_JxW_values);

  //   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  //   const unsigned int n_q_points    = quadrature_formula.size();

  //   Vector<double>              cell_residual(dofs_per_cell);

  //   // two std::vector for holding on-cell gradients and value of FE-feild i.e., \psi
  //   std::vector<Tensor<1, dim>> gradients(n_q_points);
  //   std::vector<double> solution(n_q_points);

  //   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  //   for (const auto &cell : dof_handler.active_cell_iterators())
  //     {
  //       cell_residual = 0;
  //       fe_values.reinit(cell);

  //       // fill on-cell gradients:
  //       fe_values.get_function_gradients(evaluation_point, gradients);

  // 	// fill on-cell solution value:
  //       fe_values.get_function_values(evaluation_point, solution);


  //       for (unsigned int q = 0; q < n_q_points; ++q)
  //         {
  //           // alpha + bete \psi^(n)^2, rhs
  // 	    const double bulk_term_coeff_rhs =
  // 	     ((alpha_0 * (t-1.0))
  // 	      + (beta * solution[q] * solution[q]));

  //           for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //             cell_residual(i) -=
  // 		(((fe_values.shape_grad(i, q)    // ((\partial_m \phi_i
  // 		    * gradients[q]               //   * \partial_m \psi^(n)
  // 		    * 0.5)                       //   * 1/2)
  // 		   +                             //  +
  // 		   (bulk_term_coeff_rhs          //  ((\alpha + \beta \psi^(n)2)
  // 		    * fe_values.shape_value(i, q)      //   * \phi_i
  // 		    * solution[q]))              //   * \psi^(n)))
  // 		 * fe_values.JxW(q));            // * dx 
  //         }

  //       cell->get_dof_indices(local_dof_indices);
  //       for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //         residual(local_dof_indices[i]) += cell_residual(i);
  //     }

  //   // condense redisual with hanging node constriants:
  // hanging_node_constraints.condense(residual);

  //   // setting the residuals on the boundary, which have gotten right value and no residual,
  //   // to be zero. This operation makes residuals has more zeros than system_rhs, thus the former
  //   // has smaller l2 norm.
    for (types::global_dof_index i :
         DoFTools::extract_boundary_dofs(dof_handler))
      residual(i) = 0;

    // At the end of the function, we return the norm of the residual:
    return residual.l2_norm();
  }


  // memeber function output_results() output the reaults of solution and newton update for
  // every new refined mesh. The better fromat of output file should be .hdf5. It will be implemented
  // very soon.
  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::output_results(
    const unsigned int &refinement_cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> current_solution_components_names;
    current_solution_components_names.emplace_back("Re_u");
    current_solution_components_names.emplace_back("Im_v");
    data_out.add_data_vector(current_solution, current_solution_components_names);

    std::vector<std::string> newton_update_components_names;
    newton_update_components_names.emplace_back("Re_du");
    newton_update_components_names.emplace_back("Im_dv");
    data_out.add_data_vector(newton_update, newton_update_components_names);

    // dealing frount end and back end:
    data_out.build_patches(); 

    const std::string filename =
      "iterative_solution-" + Utilities::int_to_string(refinement_cycle, 2) + "_time_mesh_refined.vtk";
    
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }



  template <int dim>
  void ComplexValuedScalarGLSolver<dim>::run()
  {

    // these two Point<> seet up the geometric size:
    const Point<dim> bottom_left(-25., -15.);
    const Point<dim> top_right(25., 15);

    const std::vector<unsigned int> repititions = {6, 5};
    const std::vector<int> n_cells_to_remove = {-4, -4};
  
    GridGenerator::subdivided_hyper_L(triangulation,
				    repititions,
				    bottom_left,
				    top_right,
				    n_cells_to_remove);    
    
    // const Point<2> center(0, 0);
    // const double radius = 100;
    // GridGenerator::hyper_ball(triangulation, center, radius);

    // const std::vector< unsigned int > sizes ={1, 1, 0, 1, 0, 0};
    
    // const std::vector< unsigned int > sizes ={2, 2, 1, 1};
    // GridGenerator::hyper_cross(triangulation, sizes);
    triangulation.refine_global(4);

    setup_dof_initilize_system(/*initial step*/ true);
    set_boundary_values();


    double       last_residual_norm = std::numeric_limits<double>::max();
    unsigned int refinement_cycle   = 0;
    do
      {
        
        if (refinement_cycle != 0)
          refine_mesh();

	std::cout << "Mesh refinement step " << refinement_cycle
		  << ", n_dofs is:" << dof_handler.n_dofs()
		  << ", n_active_cells is: " << triangulation.n_active_cells() 
		  << std::endl;


        std::cout << "  Initial residual: " << compute_residual() << std::endl;

        for (unsigned int inner_iteration = 0; inner_iteration < 30;
             ++inner_iteration)
          {
            assemble_system();
            // last_residual_norm = system_rhs.l2_norm();

            solve();
	    newton_iteration();

            std::cout << "  Residual: " << compute_residual()
	              << ", "
	              << " system_rhs.l2_norm(): " << system_rhs.l2_norm()
		      << std::endl;

	    last_residual_norm = system_rhs.l2_norm();
	    if ((refinement_cycle !=0) && (last_residual_norm < 1e-6))
	      break;
          }

        output_results(refinement_cycle);

        ++refinement_cycle;
        std::cout << std::endl;
      }
    while ((refinement_cycle !=0) && (last_residual_norm > 1e-6));
  }
} // namespace complexGL


/* ------------------------------------------------------------------------------------------
 *         Block scope of main() starts from here.
 * ------------------------------------------------------------------------------------------
 */

int main()
{
  try
    {
      using namespace complexGL;

      ComplexValuedScalarGLSolver<2> GL_2d;
      GL_2d.run();

      // RealValuedScalarGLSolver<3> GL_3d;
      // GL_3d.run();
    }
  catch (std::exception &exception)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exception.what() << std::endl
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
