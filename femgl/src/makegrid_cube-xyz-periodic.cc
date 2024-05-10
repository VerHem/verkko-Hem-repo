/* ------------------------------------------------------------------------------------------
 *
 * Copyright (C) 2023-present by Kuang. Zhang
 *
 * This library is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU Lesser General Public License as published by the Free Software Foundation; 
 * either version 2.1 of the License, or (at your option) any later version.
 *
 * Permission is hereby granted to use or copy this program under the
 * terms of the GNU LGPL, provided that the Copyright, this License 
 * and the Availability of the original version is retained on all copies.
 * User documentation of any code that uses this code or any modified
 * version of this code must cite the Copyright, this License, the
 * Availability note, and "Used by permission." 

 * Permission to modify the code and to distribute modified code is granted, 
 * provided the Copyright, this License, and the Availability note are retained,
 * and a notice that the code was modified is included.

 * The third party libraries which are used by this library are deal.II, Triinos and few others.
 * All components involved third party supports obey their Copyrights, Licence and permissions. 
 *  
 * ------------------------------------------------------------------------------------------
 *
 * author: Quang. Zhang (timohyva@github), 
 * Helsinki Institute of Physics, University of Helsinki;
 * 27. Kesäkuu. 2023.
 *
 */


#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

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
#include <deal.II/fe/component_mask.h>

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

#include "femgl.h"
#include "dirichlet.h"
#include "confreader.h"
#include "matep.h"
#include "BinA.h"
 

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  void FemGL<dim>::make_grid()
  {
    if (dim==2)
      {
        /*const double half_length = 10.0, inner_radius = 2.0;
        GridGenerator::hyper_cube_with_cylindrical_hole(triangulation,
	inner_radius, half_length);*/

	GridGenerator::hyper_cube(triangulation, 0., 10.);
	
        triangulation.refine_global(4);
        // Dirichlet pillars centers
        //const Point<dim> p1(0., 0.);

        /*for (const auto &cell : triangulation.cell_iterators())
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
	      }*/

        //triangulation.refine_global(1);
      }
    else if (dim==3)
      {

         /*---------------------------------------*/ 
         /* loading refinements control paramters */
         /*---------------------------------------*/
         conf.enter_subsection("control parameters");
         const double number_global_refine  = conf.get_integer("Number of initial global refinments");
         const double half_length            = conf.get_double("cube half side length");	 
	 conf.leave_subsection();
         /*---------------------------------------*/
	 /*    paramters loading ends at here     */
         /*---------------------------------------*/
	
         GridGenerator::hyper_cube(triangulation,
				   -half_length,
				   half_length);

	for (const auto &cell : triangulation.cell_iterators())
	  for (const auto &face : cell->face_iterators())
	    {
	      const auto center = face->center();
	      if (
		  (std::fabs(center(0) - (-half_length)) < 1e-12 * half_length)
		 )
		face->set_boundary_id(5); // b_id1 face of periodic BC along x direction 

	      if (
		  (std::fabs(center(0) - half_length) < 1e-12 * half_length)
		 )
		face->set_boundary_id(6); // b_id2 face of periodic BC along x direction 
	      
	      if (
		  (std::fabs(center(1) - (-half_length)) < 1e-12 * half_length)
		 )
	        face->set_boundary_id(7); // b_id1 face of periodic BC along y direction 

	      if (
		  (std::fabs(center(1) - half_length) < 1e-12 * half_length)
		 )
	        face->set_boundary_id(8); // b_id2 face of periodic BC along y direction 
	      
	      if (
		  (std::fabs(center(2) - (-half_length)) < 1e-12 * half_length)
		 )
	        face->set_boundary_id(9); // b_id1 face of periodic BC along z direction 

	      if (
		  (std::fabs(center(2) - half_length) < 1e-12 * half_length)		  
		 )
	        face->set_boundary_id(10); // b_id2 face of periodic BC along y direction 
	      
	    }
	 
	 /*---------------------------------------*/
	 /* identify matched faces pairs          */
	 /* add perodicuty info into triangulation*/
	 /*---------------------------------------*/

	std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> matched_pairs_x, matched_pairs_y, matched_pairs_z;	

        GridTools::collect_periodic_faces(triangulation,
                                          5 /*b_id1 */, 6 /*b_id2*/,
                                          0, /*spatial direction of periodicity */
                                          matched_pairs_x);
	
        triangulation.add_periodicity(matched_pairs_x);

        GridTools::collect_periodic_faces(triangulation,
                                          7 /*b_id1 */, 8 /*b_id2*/,
                                          1, /*spatial direction of periodicity */
                                          matched_pairs_y);
	
        triangulation.add_periodicity(matched_pairs_y);	

        GridTools::collect_periodic_faces(triangulation,
                                          9 /*b_id1 */, 10 /*b_id2*/,
                                          2, /*spatial direction of periodicity */
                                          matched_pairs_z);
	
        triangulation.add_periodicity(matched_pairs_z);	
	
	 /*---------------------------------------*/
	 /* periodicity info handling ends here   */
	 /*---------------------------------------*/
	

	triangulation.refine_global(number_global_refine /*4*/); // The refine_global() number is somehow important.
	                                                         // If one puts just 3, DoF will be about 30K.
	                                                         // When this small mount DoF are distribited on say 64 cpu processes,
	                                                         // LAPCK rises up waring:
	                                                         // "dorgqr WARNING : performing QR on a MxN matrix where M<N".
	                                                         // To suppress this warning, one should put 4 as global refine number,
	                                                         // looks like this will make DoF disstribution smoother and beheave better.
	

      } // dim==3 block
  }

  template class FemGL<3>;
  
} // namespace FemGL_mpi

