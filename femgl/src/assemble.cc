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
  void FemGL<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    system_rhs            = 0;

    { //block of assembly starts from here, all local objects in there will be release to save memory leak
     const QGauss<dim> quadrature_formula(degree + 1);
     const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);     

     FEValues<dim> fe_values(fe,
			     quadrature_formula,
			     update_values | update_gradients | update_quadrature_points | update_JxW_values);

     FEFaceValues<dim> fe_face_values(fe,
                                      face_quadrature_formula,
                                      update_values | update_JxW_values);

     const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
     const unsigned int n_q_points    = quadrature_formula.size(),        // cell quatrature points
                        n_face_q_points = face_quadrature_formula.size();   // face quatrature points

     std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

     FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
     Vector<double>     cell_rhs(dofs_per_cell);

     /* --------------------------------------------------------------------------------
      * matrices objects for old_solution_u, old_solution_v tensors,
      * matrices objects for old_solution_gradients_u, old_solution_gradients_v.
      * --------------------------------------------------------------------------------
      */

     FullMatrix<double> old_solution_u(3,3), old_f_solution_u(3,3);
     FullMatrix<double> old_solution_v(3,3), old_f_solution_v(3,3);

     // containers of old_solution_gradients_u(3,3), old_solution_gradient_v(3,3);
     const FullMatrix<double>        identity (IdentityMatrix(3));
     std::vector<FullMatrix<double>> grad_old_u_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
     std::vector<FullMatrix<double>> grad_old_v_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]

     /* --------------------------------------------------------------------------------
      * matrices objects for shape functions tensors phi^u, phi^v.    
      * These matrices depend on local Dof and Gaussian quadrature point
      * --------------------------------------------------------------------------------
      */

     FullMatrix<double> phi_u_i_q(3,3), phi_uf_i_q(3,3);
     FullMatrix<double> phi_u_j_q(3,3), phi_uf_j_q(3,3);
     FullMatrix<double> phi_v_i_q(3,3), phi_vf_i_q(3,3);
     FullMatrix<double> phi_v_j_q(3,3), phi_vf_j_q(3,3);

     //types::global_dof_index spacedim = (dim == 2)? 2 : 3;              // using  size_type = types::global_dof_index
     std::vector<FullMatrix<double>> grad_phi_u_i_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
     std::vector<FullMatrix<double>> grad_phi_v_i_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]
     std::vector<FullMatrix<double>> grad_phi_u_j_q(dim, identity);     // grad_phi_u_j container of gradient matrics, [0, dim-1]
     std::vector<FullMatrix<double>> grad_phi_v_j_q(dim, identity);     // grad_phi_v_j container of gradient matrics, [0, dim-1]
     
     // vector-flag for old_soluttion
     char flag_solution = 's';

     for (const auto &cell : dof_handler.active_cell_iterators())
       if (cell->is_locally_owned())
	 {
	   //pcout << " I'm in assembly function now!" << std::endl;
	   cell_matrix  = 0;
	   cell_rhs     = 0;

	   fe_values.reinit(cell);

	   //pcout << " now get on new cell!" << std::endl;
	  
	   for (unsigned int q = 0; q < n_q_points; ++q)
	     {	      
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

		      
		       cell_matrix(i,j) +=
			(((K1
			   * mat_lhs_K1(grad_phi_u_i_q, grad_phi_v_i_q, grad_phi_u_j_q, grad_phi_v_j_q))
			  +((K2 + K3)
			    * mat_lhs_K2K3(grad_phi_u_i_q, grad_phi_v_i_q, grad_phi_u_j_q, grad_phi_v_j_q))
			  +(/*alpha_0*/
			    alpha/*(reduced_t-1.0)*/
			    * mat_lhs_alpha(phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			  +2.0*((beta1			    
			         * mat_lhs_beta1(old_solution_u, old_solution_v, phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			        +(beta2
			          * mat_lhs_beta2(old_solution_u, old_solution_v,
				                  phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			        +(beta3
			          * mat_lhs_beta3(old_solution_u, old_solution_v,
				                  phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			        +(beta4
			          * mat_lhs_beta4(old_solution_u, old_solution_v,
				                  phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			        +(beta5
			          * mat_lhs_beta5(old_solution_u, old_solution_v,
				                  phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q)))
			  
			  ) * fe_values.JxW(q));
		       //pcout << " now one j loop finishes! j is " << j << std::endl;		      
		     } // j-index ends at here, cell_matrix ends here


		   cell_rhs(i) -= //this "-" is that -<phi, R(u,v)>
		    (((K1
		       * vec_rhs_K1(grad_old_u_q, grad_old_v_q, grad_phi_u_i_q, grad_phi_v_i_q))
		      +((K2 + K3)
			* vec_rhs_K2K3(grad_phi_u_i_q, grad_phi_v_i_q, grad_old_u_q, grad_old_v_q))
		      +(/*alpha_0*/
			alpha/*(reduced_t-1.0)*/
			* vec_rhs_alpha(phi_u_i_q, phi_v_i_q, old_solution_u, old_solution_v))
		      +2.0*((beta1
			     * vec_rhs_beta1(old_solution_u, old_solution_v, phi_u_i_q, phi_v_i_q))
		            +(beta2
			      * vec_rhs_beta2(old_solution_u, old_solution_v, phi_u_i_q, phi_v_i_q))
		            +(beta3
			      * vec_rhs_beta3(old_solution_u, old_solution_v, phi_u_i_q, phi_v_i_q))
		            +(beta4
			      * vec_rhs_beta4(old_solution_u, old_solution_v, phi_u_i_q, phi_v_i_q))
		            +(beta5
			      * vec_rhs_beta5(old_solution_u, old_solution_v, phi_u_i_q, phi_v_i_q)))
		      
		      ) * fe_values.JxW(q));                      // * dx
	          // pcout << " now get one i-loop finished! i is " << i << std::endl;  
		 } // i-index ends here

	     } // q-loop ends at here

	   /*****************************************************/
	   /*  Homogenous Robin BC + Dirichlet,                 */
	   /*    difuse BC on boundary_id 2                     */
	   /*****************************************************/	   
           for (const auto &face : cell->face_iterators())
	    {
	      if ((face->at_boundary())
		  && (((face->boundary_id()) == 2)     // x-normal surface
		      || ((face->boundary_id()) == 3)  // y-normal surface
		      || ((face->boundary_id()) == 4)) // z-normal surface
		  && (bt < 1e10)) 
	       {
               	 fe_face_values.reinit(cell, face);

      	       for (unsigned int q_face = 0; q_face < n_face_q_points; ++q_face)
		 {
		   old_f_solution_u = 0.0;
		   old_f_solution_v = 0.0;
		   
                   vector_face_matrix_generator(fe_face_values, flag_solution, q_face, n_face_q_points,
						old_f_solution_u, old_f_solution_v, face->boundary_id());
		 
                 for (unsigned int i = 0; i < dofs_per_cell; ++i)
		   {
		     //if (fe.has_support_on_face(i, cell->face_iterator_to_index(face) /*face_index*/))
		     //{
		    
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
		      {
			// if (fe.has_support_on_face(j, cell->face_iterator_to_index(face)/*face_index*/))
			//   {
                            phi_uf_i_q = 0.0; phi_vf_i_q = 0.0;
			    phi_uf_j_q = 0.0; phi_vf_j_q = 0.0;			

			    phi_matrix_face_generator(fe_face_values, i, q_face,
						      phi_uf_i_q, phi_vf_i_q, face->boundary_id());
        		    phi_matrix_face_generator(fe_face_values, j, q_face,
						      phi_uf_j_q, phi_vf_j_q, face->boundary_id());			   

			    // this "-" may deal with normal vector issue 
                            cell_matrix(i, j) -= // this "-" comes from Stocks theorem 
			                      ((K1
				                * (-1.0/bt) // "-" sign comes from conveting out-point normal vector
			                        * mat_face_lhs_K1(phi_uf_i_q, phi_uf_j_q, phi_vf_i_q, phi_vf_j_q))
				               * fe_face_values.JxW(q_face));

			 //} // if fe.has_support_on_face(j,) block ends here 		    
			   			 
		      } // j-loop ends at here

		    /* Homogenous Robin BC RHS contribution */
		    cell_rhs(i) -= // "-" from -<phi, R(u,v)>
		     (((-K1)       
			* (-1.0/bt) // "-" for opposite normal vector
		        * vec_face_rhs_K1(phi_uf_i_q, phi_vf_i_q, old_f_solution_u, old_f_solution_v) )
		       * fe_face_values.JxW(q_face));
		       //((normal_sign == false) ? 1.0 : -1.0)
	   
		    //} // if fe.has_support_on_face(i,) block ends here 		    

		   } // i-face-loop ends here		    
                  
		 } // q-face loop ends at here
	       
	       } // if face->at_boundary() block ends here

	   } //&face loop ends at here *

	   /*****************************************************/
	   /*  Homogenous Robin BC, diffuse BC on boundary_id 2 */
	   /*                   ends at here                    */
	   /*****************************************************/	   

       	   //pcout << " now q-loop finished!" << std::endl;
	   cell->get_dof_indices(local_dof_indices);
	   constraints_newton_update.distribute_local_to_global(cell_matrix,
				  		                cell_rhs,
						                local_dof_indices,
						                system_matrix,
						                system_rhs);
	   
	} // if cell->is_locally_owned block ends at here, this is a big if-statement
	   
    } //block of whole assemble process ends here
    

    //pcout << " all local celss have done !" << std::endl;
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

  } // assemble_system() function ends at here

  template class FemGL<3>;  
} // namespace FemGL_mpi

