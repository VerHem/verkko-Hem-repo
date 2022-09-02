/* ------------------------------------------------------------------------------------------
 * 
 * This is source code of solver of real valued scalar GL equation.
 * It is developed on the top of deal.II 9.3.3 finite element C++ library. 
 * 
 * License of this code is GNU Lesser General Public License, which been 
 * published by Free Software Fundation, either version 2.1 and later version.
 * You are free to use, modify and redistribute this program.
 *
 * ------------------------------------------------------------------------------------------
 *
 * author: Quang. Zhang (timohyva@github), Helsinki Institute of Physics, Syyskuu. 2022
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/solution_transfer.h>

// split global scope into ScalarGL and main() block scope
namespace ScalarGL
{
  using namespace dealii;

  template <int dim>
  class RealValuedScalarGLSolver
  {
  public:
    // calss template constructor:
    RealValuedScalarGLSolver();
    void run();

  private:

    // member functons of Solver class template:
    void   setup_system(const bool initial_step);
    void   assemble_system();
    void   solve();
    void   refine_mesh();
    void   set_boundary_values();
    double compute_residual(const double alpha) const;
    double determine_step_length() const;
    void   output_results(const unsigned int &refinement_cycle) const;


    // data members of solver class template.
    // those variables are necessary parts for adaptive meshed FEM and newton iteration.
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FE_Q<dim>       fe;

    // hanging nodes constriants object:
    AffineConstraints<double> hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> newton_update;
    Vector<double> current_solution;
    Vector<double> system_rhs;

    // physical coefficients of real valued scalar GL equation:
    const double alpha_0 = 2.0;
    const double beta = 0.5;
    double t; // scaled temperature
  };

  
  /* ------------------------------------------------------------------------------------------
   *
   * class template BoundaryValues inhereted from Function<dim>, in which member function value
   * is overrided as homogenous Dilicheret boundary value.
   *
   * ------------------------------------------------------------------------------------------
   */
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };


  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int /*component*/) const
  {
    return 0.0 * p[0];
    // return std::sin(6.0 * (p[0]+p[1]));
  }


  /* ------------------------------------------------------------------------------------------
   *
   * The following functions are members of RealValuedScalarGLSolver till the end of ScalarGL
   * namespace.
   * ------------------------------------------------------------------------------------------
   */

  // construnctor and initilizaton list
  template <int dim>
  RealValuedScalarGLSolver<dim>::RealValuedScalarGLSolver()
    : dof_handler(triangulation)
    , fe(2)
    , t(0.0)
  {}




  template <int dim>
  void RealValuedScalarGLSolver<dim>::setup_system(const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs(fe);
        current_solution.reinit(dof_handler.n_dofs(), /*omit_zeroing_entries*/true);
	//current_solution.print();

	for (auto it = current_solution.begin(); it != current_solution.end(); ++it)
	  *it = 2.0;

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);
        hanging_node_constraints.close();
      }
   

    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    // condense dsp with constriants object, that adds these positions
    // into sparse pattern, which are required for eliminaton of constraints:
    hanging_node_constraints.condense(dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }

  
  template <int dim>
  void RealValuedScalarGLSolver<dim>::assemble_system()
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

    // vector to holding u^n on cell:
    std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);
    std::vector<double> old_solution(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

	fe_values.get_function_gradients(current_solution, old_solution_gradients);
        fe_values.get_function_values(current_solution, old_solution);
    
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
	    // alpha + 3 bete \psi^(n)^2, system_matrix
            const double bulk_term_coeff_systemMatrix =
	      ((alpha_0 * (t-1.0))
	       + (3.0 * beta * old_solution[q] * old_solution[q]));

	    // alpha + bete \psi^(n)^2, rhs
	    const double bulk_term_coeff_rhs =
	      ((alpha_0 * (t-1.0))
	       + (beta * old_solution[q] * old_solution[q]));
 
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
		  (((fe_values.shape_grad(i, q)       // ((\partial_m \phi_i
		     * fe_values.shape_grad(j, q)     //   \partial_m \phi_j
		     * 0.5)                           //   * 1/2)
		    +                                 //  +
		    (bulk_term_coeff_systemMatrix     //  ((\alpha + 3 \beta \phi^(n)2)
		     * fe_values.shape_value(i, q)    //    * \phi_i
		     * fe_values.shape_value(j, q)))  //    * \phi_j))
		   * fe_values.JxW(q));               // * dx

		    
                    // (((fe_values.shape_grad(i, q)      // ((\nabla \phi_i
                    //    * coeff                         //   * a_n
                    //    * fe_values.shape_grad(j, q))   //   * \nabla \phi_j)
                    //   -                                //  -
                    //   (fe_values.shape_grad(i, q)      //  (\nabla \phi_i
                    //    * coeff * coeff * coeff         //   * a_n^3
                    //    * (fe_values.shape_grad(j, q)   //   * (\nabla \phi_j
                    //       * old_solution_gradients[q]) //      * \nabla u_n)
                    //    * old_solution_gradients[q]))   //   * \nabla u_n)))
                    //  * fe_values.JxW(q));              // * dx

                cell_rhs(i) -=
		 (((fe_values.shape_grad(i, q)   // ((\partial_m \phi_i
		    * old_solution_gradients[q]  //   * \partial_m \psi^(n)
		    * 0.5)                       //   * 1/2)
		   +                             //  +
		   (bulk_term_coeff_rhs          //  ((\alpha + \beta \psi^(n)2)
		    * fe_values.shape_value(i, q)      //   * \phi_i
		    * old_solution[q]))          //   * \psi^(n)))
		  * fe_values.JxW(q));           // * dx 
		
		// cell_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
                //                 * coeff                     // * a_n
                //                 * old_solution_gradients[q] // * u_n
                //                 * fe_values.JxW(q));        // * dx
              }
          }

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
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);

    // the newton iteration has zero value boundary condtion
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       newton_update,
                                       system_rhs);
  }


  template <int dim>
  void RealValuedScalarGLSolver<dim>::solve()
  {
    SolverControl            solver_control(system_rhs.size(),
                                 system_rhs.l2_norm() * 1e-6);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);

    hanging_node_constraints.distribute(newton_update);

    const double alpha = determine_step_length();
    current_solution.add(alpha, newton_update);
  }



  template <int dim>
  void RealValuedScalarGLSolver<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      // std::map<types::boundary_id, const Function<dim> *>(),
      {}, // there is no Neumann boundary, then the std::map is empty
      current_solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.35,
                                                    0.03);

    // preparing for SolutionTransfer class:
    triangulation.prepare_coarsening_and_refinement();

    SolutionTransfer<dim> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(current_solution);

    // execute the refine:
    triangulation.execute_coarsening_and_refinement();

    // distribute Dof for new mesh:
    dof_handler.distribute_dofs(fe);


    // intepolate current_result on new mesh:
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(current_solution, tmp);
    current_solution = tmp;

    // fill hanging nodes constraints before distributing new
    // constriants of refined mesh. It is significant.
    hanging_node_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    // set the boundary values of current result correct on new mesh 
    set_boundary_values();

    // bool false will let set_up() knows Dof has been distributed
    setup_system(false);
  }


  template <int dim>
  void RealValuedScalarGLSolver<dim>::set_boundary_values()
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
  double RealValuedScalarGLSolver<dim>::compute_residual(const double alpha) const
  {
    Vector<double> residual(dof_handler.n_dofs());

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
      }

    // condense redisual with hanging node constriants:
    hanging_node_constraints.condense(residual);

    // setting the residuals on the boundary, which have gotten right value and no residual,
    // to be zero. This operation makes residuals has more zeros than system_rhs, thus the former
    // has smaller l2 norm.
    for (types::global_dof_index i :
         DoFTools::extract_boundary_dofs(dof_handler))
      residual(i) = 0;

    // At the end of the function, we return the norm of the residual:
    return residual.l2_norm();
  }

  // This member function returns iterative step length a^k.
  // The best way to do this is implement a line-search for the optimatical value of a^k,
  // but here we simply return 0.1, which is not the opttical value and garantee convergency.
  template <int dim>
  double RealValuedScalarGLSolver<dim>::determine_step_length() const
  {
    return 0.1;
  }

  // memeber function output_results() output the reaults of solution and newton update for
  // every new refined mesh. The better fromat of output file should be .hdf5. It will be implemented
  // very soon.
  template <int dim>
  void RealValuedScalarGLSolver<dim>::output_results(
    const unsigned int &refinement_cycle) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "current_solution");
    data_out.add_data_vector(newton_update, "newton_update");

    // dealing frount end and back end:
    data_out.build_patches(); 

    const std::string filename =
      "iterative_solution-" + Utilities::int_to_string(refinement_cycle, 2) + "_time_mesh_refined.vtk";
    
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }



  template <int dim>
  void RealValuedScalarGLSolver<dim>::run()
  {
    const Point<2> center(0, 0);
    const double radius = 100;
    GridGenerator::hyper_ball(triangulation, center, radius);

    // const std::vector< unsigned int > sizes ={1, 1, 0, 1, 0, 0};
    
    // const std::vector< unsigned int > sizes ={2, 2, 1, 1};
    // GridGenerator::hyper_cross(triangulation, sizes);
    triangulation.refine_global(2);

    setup_system(/*first time=*/ true);
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


        std::cout << "  Initial residual: " << compute_residual(0) << std::endl;

        for (unsigned int inner_iteration = 0; inner_iteration < 41;
             ++inner_iteration)
          {
            assemble_system();
            // last_residual_norm = system_rhs.l2_norm();

            solve();

            std::cout << "  Residual: " << compute_residual(0)
	              << ", "
		      << " system_rhs.l2_norm(): " << system_rhs.l2_norm()
		      << std::endl;

	    last_residual_norm = system_rhs.l2_norm();
	    if ((refinement_cycle !=0) && (last_residual_norm < 1e-2))
	      break;
          }

        output_results(refinement_cycle);

        ++refinement_cycle;
        std::cout << std::endl;
      }
    while (last_residual_norm > 1e-2);
  }
} // namespace ScalarGL


/* ------------------------------------------------------------------------------------------
 *         Block scope of main() starts from here.
 * ------------------------------------------------------------------------------------------
 */

int main()
{
  try
    {
      using namespace ScalarGL;

      RealValuedScalarGLSolver<2> GL_2d;
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
