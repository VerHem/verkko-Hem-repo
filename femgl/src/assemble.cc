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
  void FemGL<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    system_rhs            = 0;

    { //block of assembly starts from here, all local objects in there will be release to save memory leak
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
     //FullMatrix<double>              phi_u_phi_ut(3,3), phi_v_phi_vt(3,3);
     //FullMatrix<double>              old_u_old_ut(3,3), old_v_old_vt(3,3);

     //FullMatrix<double>              old_u_phi_ut_i_q(3,3),
     //                                old_u_phi_ut_j_q(3,3),
     //                                old_v_phi_vt_i_q(3,3),
     //                                old_v_phi_vt_j_q(3,3);

     //FullMatrix<double>              K1grad_matrics_sum_i_j_q(3,3),
     //FullMatrix<double>              rhs_K1grad_matrics_sum_i_q(3,3);                // matrix for storing K1 gradients before trace
     //FullMatrix<double>              phi_phit_matrics_i_j_q(3,3);
     /*--------------------------------------------------------------------------------*/
     
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
	      
	       // u.ut and v.vt
	       // old_u_old_ut   = 0.0;
	       // old_v_old_vt   = 0.0;

	       // old_solution_u.mTmult(old_u_old_ut, old_solution_u);
	       // old_solution_v.mTmult(old_v_old_vt, old_solution_v);
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
		       //phi_u_phi_ut        = 0.0;
		       //phi_v_phi_vt        = 0.0;

		       //old_u_phi_ut_i_q    = 0.0;
		       //old_u_phi_ut_j_q    = 0.0;
		       //old_v_phi_vt_i_q    = 0.0;
		       //old_v_phi_vt_j_q    = 0.0;

		       //K1grad_matrics_sum_i_j_q = 0.0;
		       //phi_phit_matrics_i_j_q   = 0.0;
		       /*--------------------------------------------------*/

		       /* --------------------------------------------------
		        * conduct matrices multiplacations
		        * --------------------------------------------------
		        */
		       // phi_u_phi_ut_ijq, phi_v_phi_vt_ijq matrics
		       //phi_u_i_q.mTmult(phi_u_phi_ut, phi_u_j_q);
		       //phi_v_i_q.mTmult(phi_v_phi_vt, phi_v_j_q);

		       // old_u/v_phi_ut/vt_i/jq matrics
		       //old_solution_u.mTmult(old_u_phi_ut_i_q, phi_u_i_q);
		       //old_solution_u.mTmult(old_u_phi_ut_j_q, phi_u_j_q);
		       //old_solution_v.mTmult(old_v_phi_vt_i_q, phi_v_i_q);
		       //old_solution_v.mTmult(old_v_phi_vt_j_q, phi_v_j_q);

		       // alpha terms matrics
		       //phi_phit_matrics_i_j_q.add(1.0, phi_u_phi_ut, 1.0, phi_v_phi_vt);

		       /*for (unsigned int k = 0; k < dim; ++k)
			 {
			   grad_phi_u_i_q[k].mTmult(K1grad_matrics_sum_i_j_q, grad_phi_u_j_q[k], true);
			   grad_phi_v_i_q[k].mTmult(K1grad_matrics_sum_i_j_q, grad_phi_v_j_q[k], true);
			   }*/
		      
		       /*--------------------------------------------------*/		          
		      
		       cell_matrix(i,j) +=
			(((K1
			   * mat_lhs_K1(grad_phi_u_i_q, grad_phi_v_i_q, grad_phi_u_j_q, grad_phi_v_j_q))
			  +((K2 + K3)
			    * mat_lhs_K2K3(grad_phi_u_i_q, grad_phi_v_i_q, grad_phi_u_j_q, grad_phi_v_j_q))
			  +(alpha_0
			    * (reduced_t-1.0)
			    //* phi_phit_matrics_i_j_q.trace())
			    * mat_lhs_alpha(phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			  +(beta
			    //* ((old_u_old_ut.trace() + old_v_old_vt.trace()) * phi_phit_matrics_i_j_q.trace()
			    //   + 2.0 * ((old_u_phi_ut_i_q.trace() + old_v_phi_vt_i_q.trace())
			    //	* (old_u_phi_ut_j_q.trace() + old_v_phi_vt_j_q.trace())))
			    * mat_lhs_beta2(old_solution_u, old_solution_v,
					    phi_u_i_q, phi_u_j_q, phi_v_i_q, phi_v_j_q))
			  ) * fe_values.JxW(q));
		       //pcout << " now one j loop finishes! j is " << j << std::endl;		      
		     } // cell_matrix ends here

		   //rhs_K1grad_matrics_sum_i_q = 0.0;
		   /*for (unsigned int k = 0; k < dim; ++k)
		     {
		      grad_old_u_q[k].mTmult(rhs_K1grad_matrics_sum_i_q, grad_phi_u_i_q[k], true);
		      grad_old_v_q[k].mTmult(rhs_K1grad_matrics_sum_i_q, grad_phi_v_i_q[k], true);
		      }*/

		   cell_rhs(i) -=
		    (((K1
		       //* rhs_K1grad_matrics_sum_i_q.trace()
		       * vec_rhs_K1(grad_old_u_q, grad_old_v_q, grad_phi_u_i_q, grad_phi_v_i_q))
		      +((K2 + K3)
			* vec_rhs_K2K3(grad_phi_u_i_q, grad_phi_v_i_q, grad_old_u_q, grad_old_v_q))
		      +(alpha_0
			* (reduced_t-1.0)
			* vec_rhs_alpha(phi_u_i_q, phi_v_i_q, old_solution_u, old_solution_v))
		      +(beta
			//* (old_u_old_ut.trace() + old_v_old_vt.trace())
			//* (old_u_phi_ut_i_q.trace() + old_v_phi_vt_i_q.trace())
			* vec_rhs_beta2(old_solution_u, old_solution_v, phi_u_i_q, phi_v_i_q))
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

    } //block of whole assemble process ends here
    

    //pcout << " all local celss have done !" << std::endl;
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template class FemGL<3>;  
} // namespace FemGL_mpi

