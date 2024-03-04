/*
 *
 */


#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
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
 

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  void FemGL<dim>::compute_residual(/*const LA::MPI::Vector &damped_vector*/)
  {
    TimerOutput::Scope t(computing_timer, "compute_residual");

    residual_vector = 0;

    { //block of residual vector assembly, starts here, all local objects will be released
    const QGauss<dim> quadrature_formula(degree + 1);
    // const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);         

    FEValues<dim> fe_values(fe,
			    quadrature_formula,
			    update_values | update_gradients | update_quadrature_points | update_JxW_values);

    // FEFaceValues<dim> fe_face_values(fe,
    //                                  face_quadrature_formula,
    //                                  update_values | update_JxW_values);
    

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size(); //,
      //n_face_q_points = face_quadrature_formula.size(); // face quatrature points                       

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Vector<double>     cell_rhs(dofs_per_cell);

    /* --------------------------------------------------------------------------------
     * matrices objects for old_solution_u, old_solution_v tensors,
     * matrices objects for old_solution_gradients_u, old_solution_gradients_v.
     * --------------------------------------------------------------------------------
     */

    FullMatrix<double> old_solution_u(3,3)/*, old_f_solution_u(3,3)*/;
    FullMatrix<double> old_solution_v(3,3)/*, old_f_solution_v(3,3)*/;

    // containers of old_solution_gradients_u(3,3), old_solution_gradient_v(3,3);
    const FullMatrix<double>        identity (IdentityMatrix(3));
    std::vector<FullMatrix<double>> grad_old_u_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_old_v_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]

    /* --------------------------------------------------------------------------------
     * matrices objects for shape functions tensors phi^u, phi^v.    
     * These matrices depend on local Dof and Gaussian quadrature point
     * --------------------------------------------------------------------------------
     */

    FullMatrix<double> phi_u_i_q(3,3)/*, phi_uf_i_q(3,3)*/;
    FullMatrix<double> phi_v_i_q(3,3)/*, phi_vf_i_q(3,3)*/;

    //types::global_dof_index spacedim = (dim == 2)? 2 : 3;              // using  size_type = types::global_dof_index
    std::vector<FullMatrix<double>> grad_phi_u_i_q(dim, identity);     // grad_phi_u_i container of gradient matrics, [0, dim-1]
    std::vector<FullMatrix<double>> grad_phi_v_i_q(dim, identity);     // grad_phi_v_i container of gradient matrics, [0, dim-1]

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

		  cell_rhs(i) -=
		    (((K1
		       * vec_rhs_K1(grad_old_u_q, grad_old_v_q, grad_phi_u_i_q, grad_phi_v_i_q))
		      +((K2 + K3)
			* vec_rhs_K2K3(grad_phi_u_i_q, grad_phi_v_i_q, grad_old_u_q, grad_old_v_q))
		      +(alpha_0
			* (reduced_t-1.0)
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

		} // i-index ends here
	    } // q-loop ends at here

	   /*****************************************************/
	   /*  Homogenous Robin BC, difuse BC on boundary_id 2  */
	   /*                                                   */	   	   
	   /*****************************************************/	   
           // for (const auto &face : cell->face_iterators())
	   //  {
	   //    if ((face->at_boundary()) && ((face->boundary_id()) == 2))
	   //     {
           //     	 fe_face_values.reinit(cell, face);

      	   //     for (unsigned int q_face = 0; q_face < n_face_q_points; ++q_face)
	   // 	 {
	   // 	   old_f_solution_u = 0.0;
	   // 	   old_f_solution_v = 0.0;
		   
           //         vector_face_matrix_generator(fe_face_values, flag_solution, q_face, n_face_q_points, old_f_solution_u, old_f_solution_v);
           //       for (unsigned int i = 0; i < dofs_per_cell; ++i)
	   // 	   {
	   // 	    phi_uf_i_q = 0.0; phi_vf_i_q = 0.0;
           //          phi_matrix_face_generator(fe_face_values, i, q_face, phi_uf_i_q, phi_vf_i_q);
		    
	   // 	    /* Homogenous Robin BC RHS contribution */
	   // 	    cell_rhs(i) -= // "-" from -<phi, R(u,v)>
	   // 	     (((-K1)       
	   // 		* (-1.0/bt) // "-" for opposite normal vector
	   // 		* vec_face_rhs_K1(phi_uf_i_q, phi_vf_i_q, old_f_solution_u, old_f_solution_v))
	   // 	       * fe_face_values.JxW(q_face));
	   // 	       //((normal_sign == false) ? 1.0 : -1.0)
	   		    
	   // 	   } // i-face-loop ends here		    

	   // 	 } // q-face loop ends at here
	       
	   //     } // if face->at_boundary() block ends here

	   // } //&face loop ends at here

	   /*****************************************************/
	   /*  Homogenous Robin BC, diffuse BC on boundary_id 2 */
	   /*                   ends at here                    */
	   /*****************************************************/	   
	  
	  
	  cell->get_dof_indices(local_dof_indices);
	  constraints_newton_update.distribute_local_to_global(cell_rhs,
						               local_dof_indices,
						               residual_vector);
	  
	} // if cell->is_locally_owned() ends at here, this is a big if-statement
    

    } // block of assemble residual vector, release the memory

    residual_vector.compress(VectorOperation::add);
  }

  template class FemGL<3>;  
} // namespace FemGL_mpi

