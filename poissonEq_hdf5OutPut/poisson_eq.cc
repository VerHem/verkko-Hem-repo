/* 
 *   A poisson equation solver.
 *     
 *     \partial_{i} \partial_{i} u = 10.
 *
 *   authors: Quang. Zhang (timohyva@github)
 */




#include <deal.II/base/utilities.h>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/hdf5.h>

#include <fstream>
#include <iostream>

using namespace dealii;


class PoissonEq_solver
{
public:
  PoissonEq_solver(); // constructor with initilizer list
  void run();


private:
  void make_merged_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  // ******************************************************************** //

  // objects involving geometry, shape functions (basis functions) and enumeration of Dofs:

  Triangulation<2> ic, channel, hec;         // three parts of domain, i.e., IC, channel and HEC  
  Triangulation<2> triangulation_HShape;     // geometry, solving domain. It is merged from ic, channel and hec

  unsigned int n_refinement_steps;
  
  FE_Q<2>          shapeFunc;                // 2D lagrange polynomials
  DoFHandler<2>    dof_handler;              // Dof handler/manager


  // objects involving sparsity partten:
  
  SparsityPattern      sparsityPattern; 
  SparseMatrix<double> systemMatrix;         // dealii built-in SparseMatrix tamplate
  

  // Vector objects of the linear system: 
  
  Vector<double> solution;                   // dealii built-in Vector tamplate
  Vector<double> system_rhs;                 // right hand side verctor of equation
};


// ************************************************************************ //
// *************        poissonEq_solver constructor         ************** //
// ************************************************************************ //

PoissonEq_solver::PoissonEq_solver()
  : shapeFunc(1)                             // linear intepolation basis functions
  , dof_handler(triangulation_HShape)
{}


// ************************************************************************ //
// *************        geometry and mesh generation         ************** //
// ************************************************************************ //

void PoissonEq_solver::make_merged_grid()
{
  // IC 
  const Point<2> upper_left1(-10., 2.);
  const Point<2> lower_right1(-5., -2.);
  GridGenerator::hyper_rectangle(ic, upper_left1, lower_right1);
  
  // channel
  const Point<2> upper_left2(-5., 0.5);
  const Point<2> lower_right2(5., -0.5);
  GridGenerator::hyper_rectangle(channel, upper_left2, lower_right2);

  // HEC
  const Point<2> upper_left3(5., 3.5);
  const Point<2> lower_right3(10., -3.5);
  GridGenerator::hyper_rectangle(hec, upper_left3, lower_right3);

  // merge IC, channel and HEC to be _HShape
  GridGenerator::merge_triangulations({&ic, &channel, &hec},
                                    triangulation_HShape,
                                    1.0e-10,
                                    false);
  
 
  // ***************         mesh refine          ************** //
  
  // triangulation_HShape.refine_global(1);
  n_refinement_steps = 3;
 
  // five/two times refinements for boundaries
  for (unsigned int step = 0; step < 2; ++step)
    {
      for (auto &cell : triangulation_HShape.active_cell_iterators())
	{
         for (const auto v : cell->vertex_indices())
	   {
	     // std::cout << " v is " << v << " vertex looks like " << cell->vertex(v) << std::endl;

	     double y_vertex = (cell->vertex(v)).operator()(1);
	     double x_vertex = (cell->vertex(v)).operator()(0);
	     
	     if (
		  ((std::fabs(y_vertex - upper_left2.operator()(1))  <= 1e-6 * std::fabs(upper_left2.operator()(1)))
		   || (std::fabs(y_vertex - lower_right2.operator()(1))  <= 1e-6 * std::fabs(lower_right2.operator()(1))))
		  && ((x_vertex >= upper_left2.operator()(0)) && (x_vertex <= lower_right2.operator()(0)))
		  
		   || (std::fabs(x_vertex) == lower_right2.operator()(0))
		   
		 )
	       {
		cell->set_refine_flag();
                break;		 
	       }

	   }
	  
	}

      triangulation_HShape.execute_coarsening_and_refinement();
    }
  triangulation_HShape.refine_global(n_refinement_steps);

  std::cout << "Number of active cells: " << triangulation_HShape.n_active_cells()
            << std::endl;

  // *********************************************************************** //
  // *********************************************************************** //
  
  
  std::ofstream out("HShape.eps");
  GridOut       grid_out;
  grid_out.write_eps(triangulation_HShape, out);

  std::cout << "Grid written to HShape.eps" << std::endl;
    
}


// ************************************************************************ //
// *************        setting up matrices and vectors      ************** //
// ************************************************************************ //


void PoissonEq_solver::setup_system()
{
  dof_handler.distribute_dofs(shapeFunc);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsityPattern.copy_from(dsp);

  // put the sparseity_pattern into the system_matrix, which is a sparse matrix
  systemMatrix.reinit(sparsityPattern);

  // initilize all Vector types, eleements set as zeros
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


// ************************************************************************ //
// *************        assemble linear system               ************** //
// ************************************************************************ //

void PoissonEq_solver::assemble_system()
{
  
  QGauss<2> quadrature_formula(shapeFunc.degree + 3u);     // 4-points Gaussian quadrature

  // FEValues object:
  FEValues<2> fe_values(shapeFunc,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
  
  const unsigned int dofs_per_cell = shapeFunc.n_dofs_per_cell();
  std::cout << " shapeFunc.n_dof_per_cell() returns " << dofs_per_cell
            << " \n " << std::endl;

 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);


  // std::vector type container for storing Dofs number in a cell, initialized with zeros
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::cout << " local_dof_indices is default initiliazed as: "
	    << std::endl;
  for (const auto &i : local_dof_indices)
    std::cout << " " << i << " ";

  std::cout << std::endl;

  // ********************************************************************* //
  
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      
      fe_values.reinit(cell);

      // operator " = " overloaded: 
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
	  // std::cout << " q_index in this cell looks like " << q_index
	  //           << " " << std::endl;

	  
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
	      {
		// std::cout << " i = " << i << " j = " << j << std::endl;
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                  fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                  fe_values.JxW(q_index));           // dx
	      }

          
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            500. *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      // std::cout << " \nfe_values.dof_indices returns: " << fe_values.dof_indices()
      // 	        << std::endl;
      // std::cout << " \n\n " << std::endl;

      
      cell->get_dof_indices(local_dof_indices);     // *cell fetches global Dof for cell vertices 
      // std::cout << " now local_dof_indices looks like " << " ";
      // for (const auto &i : local_dof_indices)
      //   std::cout << " " << i << " ";

      // std::cout << " \n\n " << std::endl;
	       

      // ***************     update system (global) matrix by cell calculation    ************** // 
       
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          systemMatrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      // ***************     update rhs system vector by cell calculation        ************** //
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  // ***************  put the Dirichlet BC into the linear system    ***************** //
  

  std::map<types::global_dof_index, double> boundary_values;              // std::map associative container
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,                             // " 0 " is boundary indicator
                                           Functions::ZeroFunction<2>(),  // zeros functions
                                           boundary_values);
  
  MatrixTools::apply_boundary_values(boundary_values,
                                     systemMatrix,
                                     solution,
                                     system_rhs);
}


// ************************************************************************ //
// *************        linear iterativesolver               ************** //
// ************************************************************************ //

void PoissonEq_solver::solve()
{
 
  SolverControl solver_control(5000, 1e-12);          // max 1000 times iteration, and tolrant 
  
  SolverCG<Vector<double>> solver(solver_control);    // Conjugate Gradient Solver type object

  solver.solve(systemMatrix, solution, system_rhs, PreconditionIdentity());
  
}


// ************************************************************************ //
// *************              output result                  ************** //
// ************************************************************************ //


void PoissonEq_solver::output_results() const
{
  
  DataOut<2> data_out;
  
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution_PoissonEq");
  
  data_out.build_patches();             // from frontend to backend

  
  std::ofstream output("solution_PoissonEq.vtk");
  data_out.write_vtk(output);

  const std::string filename_h5 = "solution_" + std::to_string(n_refinement_steps) + ".h5";
  HDF5::File poissonEq_data(filename_h5, HDF5::File::FileAccessMode::create);
  poissonEq_data.write_dataset("solution",solution);

  // DataOutBase::DataOutFilterFlags flags(true, true);
  // DataOutBase::DataOutFilter data_filter(flags);

  // data_out.write_filtered_data(data_filter);
  // data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
}



void PoissonEq_solver::run()
{
  make_merged_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}


//int main(int argc, char* argv[])
int main()
{
  //  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  deallog.depth_console(3);

  PoissonEq_solver poissonEq;
  poissonEq.run();

  return 0;
}
