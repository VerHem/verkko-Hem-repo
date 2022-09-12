/* ---------------------------------------------------------------------
 *
 * dell.II libs based FEM codes for seearching order parameters field 
 * distribution in superfluid helium-3
 * 
 * author: Quang (timohyva@github)
 *
 * ---------------------------------------------------------------------

 */



// The most fundamental class in the library is the Triangulation class, which
// is declared here:
#include <deal.II/grid/tria.h>

#include <deal.II/base/point.h>

// Here are some functions to generate standard grids:
#include <deal.II/grid/grid_generator.h>
// Output of grids in various graphics formats:
#include <deal.II/grid/grid_out.h>

 
#include <deal.II/dofs/dof_handler.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
 
#include <deal.II/dofs/dof_renumbering.h>
 
#include <fstream>





// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>

// we simply import the entire deal.II
// namespace for general use:
using namespace dealii;

// ************************************************************************



// ************************************************************************

// void merged_grid()
// {
//   Triangulation<2> recta1, recta2, recta3;
 

//   // fill it with retangular shape 1
//   const Point<2> upper_left1(-4., 2.);
//   const Point<2> lower_right1(-2.5, -2.);
//   GridGenerator::hyper_rectangle(recta1, upper_left1, lower_right1);
  
//   // fill it with retangular shape 2
//   const Point<2> upper_left2(-2.5, 0.2);
//   const Point<2> lower_right2(2.5, -0.2);
//   GridGenerator::hyper_rectangle(recta2, upper_left2, lower_right2);

//   // fill it with retangular shape 2
//   const Point<2> upper_left3(2.5, 2.);
//   const Point<2> lower_right3(4., -2.);
//   GridGenerator::hyper_rectangle(recta3, upper_left3, lower_right3);

  
//   Triangulation<2> merged_domain;
//   GridGenerator::merge_triangulations({&recta1, &recta2, &recta3},
//                                     merged_domain,
//                                     1.0e-10,
//                                     false);

//    merged_domain.refine_global(1);
//   // five times refinements for boundaries
//   for (unsigned int step = 0; step < 2; ++step)
//     {
//       for (auto &cell : merged_domain.active_cell_iterators())
// 	{
//          for (const auto v : cell->vertex_indices())
// 	   {
// 	     std::cout << " v is " << v << " vertex looks like " << cell->vertex(v) << std::endl;

// 	     double y_vertex = (cell->vertex(v)).operator()(1);
// 	     double x_vertex = (cell->vertex(v)).operator()(0);
// 	     if (
// 		  ((std::fabs(y_vertex - upper_left2.operator()(1))  <= 1e-6 * std::fabs(upper_left2.operator()(1)))
// 		   || (std::fabs(y_vertex - lower_right2.operator()(1))  <= 1e-6 * std::fabs(lower_right2.operator()(1))))
// 		  && ((x_vertex >= upper_left2.operator()(0)) && (x_vertex <= lower_right2.operator()(0)))
		  
// 		  || (std::fabs(x_vertex) == lower_right2.operator()(0))
		   
// 		 )
// 	       {
// 		cell->set_refine_flag();
//                 break;		 
// 	       }

// 	   }
	  
// 	}

//       merged_domain.execute_coarsening_and_refinement();
//     }
//    merged_domain.refine_global(2);
  
//   std::ofstream out("domain_merge.eps");
//   GridOut       grid_out;
//   grid_out.write_svg(merged_domain, out);

//   std::cout << "Grid written to domain_merge.eps" << std::endl;
  
// }

void grid_refine_channel(Triangulation<2> &channel)
{
  // fill it with retangular shape
  const Point<2> bottom_left(-25., -15.);
  const Point<2> top_right(25., 15);

  const std::vector<unsigned int> repititions = {6, 5};
  const std::vector<int> n_cells_to_remove = {-4, -4};
  
  GridGenerator::subdivided_hyper_L(channel,
				    repititions,
				    bottom_left,
				    top_right,
				    n_cells_to_remove);

  
  // // five times refinements for boundaries
  // for (unsigned int step = 0; step < 6; ++step)
  //   {
  //     for (auto &cell : channel.active_cell_iterators())
  // 	{
  //        for (const auto v : cell->vertex_indices())
  // 	   {
  // 	     std::cout << " v is " << v << " vertex looks like " << cell->vertex(v) << std::endl;

  // 	     double y_component_vertex = (cell->vertex(v)).operator()(1);
  // 	     if ((std::fabs(y_component_vertex - upper_left.operator()(1))  <= 1e-6 * std::fabs(upper_left.operator()(1)))
  // 		 || (std::fabs(y_component_vertex - lower_right.operator()(1))  <= 1e-6 * std::fabs(lower_right.operator()(1))))
  // 	       {
  //               cell->set_refine_flag();
  //               break;		 
  // 	       }

  // 	   }
	  
  // 	}

  //     channel.execute_coarsening_and_refinement();
  //   }
  
  // global refinment for main body
  // channel.refine_global(2);

  std::cout << " channel has been meshed and refined " << std::endl;


}

void distribute_dofs(DoFHandler<2> &dof_handler, Triangulation<2> &triangulation)
{
  // setting the 1st order lagrange polynomial as base function
  const FE_Q<2> finite_element(1);

  // combine bases set and their manager
  dof_handler.distribute_dofs(finite_element);

  // get the sparse pattern 
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
 
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);

  
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
 
  std::ofstream out("subdivided_L.svg");
  GridOut grid_out;
  // sparsity_pattern.print_svg(out);
  grid_out.write_svg(triangulation, out);
}


// void renumber_dofs(DoFHandler<2> &dof_handler)
// {
//   DoFRenumbering::Cuthill_McKee(dof_handler);

//   DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
//                                                   dof_handler.n_dofs());
//   DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);

//   SparsityPattern sparsity_pattern;
//   sparsity_pattern.copy_from(dynamic_sparsity_pattern);

//   std::ofstream out("s_pattern_channel_renumbered.svg");
//   sparsity_pattern.print_svg(out);
// }

// ****************************************************************************
// ***                      entrance pot:  main()                           ***
// ****************************************************************************


int main()
{
  // create domain and mesh 
  Triangulation<2> channel;
  grid_refine_channel(channel);

  // Dof handler objecct for managing the base functions (shape function, test functions)
  DoFHandler<2> dof_handler_channel(channel);

  // combine the info of base functions with their handler  
  distribute_dofs(dof_handler_channel, channel);

  // a different enumeration
  // renumber_dofs(dof_handler_channel);

  return 0;
}
