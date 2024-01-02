/*
 *
 */


#include <random> // c++ std radom bumber library, for gaussian random initiation

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

// The following chunk out code is identical to step-40 and allows
// switching between PETSc and Trilinos:

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
	const double half_length = 10.0/*10.0*/, inner_radius = 2.0, z_extension =10.0/*10.0*/;
	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation,
							inner_radius, half_length, z_extension);

	triangulation.refine_global(4/*3*/); //5 will give you 30M DoF, 4 give you 5M
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

      }
  }

  template <int dim>
  void FemGL<dim>::setup_system()
  {
    {
      TimerOutput::Scope t(computing_timer, "setup");
      dof_handler.distribute_dofs(fe);

      pcout << "   Number of degrees of freedom: "
 	    << dof_handler.n_dofs()
	    << std::endl;

      locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);    
    }

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

    // initialize local_solution Vector by gaussian random numbers in cycle 0
    {
     if (cycle == 0)
       {
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

        for (auto it = distrubuted_tmp_solution.begin(); it != distrubuted_tmp_solution.end(); ++it)
         {
	  *it = gaussian_distr(gen);
         }

        // AffineConstriant::distribute call
        constraints_solution.distribute(distrubuted_tmp_solution);

        local_solution = distrubuted_tmp_solution;    
        /*---------------------------------------*/
       }
     else
       local_solution.reinit(locally_relevant_dofs, mpi_communicator/*, false*/);
    }

    {
      locally_relevant_newton_solution.reinit(locally_relevant_dofs, mpi_communicator);
      locally_relevant_damped_vector.reinit(locally_relevant_dofs, mpi_communicator);
      system_rhs.reinit(locally_owned_dofs, mpi_communicator);
      residual_vector.reinit(locally_owned_dofs, mpi_communicator);    
    }
  }



  template <int dim>
  void FemGL<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    system_rhs            = 0;

    { //block of assembly starts from here, all local objects in there will be release to same memory leak
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
	   //pcout << " I'm in assembly function now!" << std::endl;
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

    } //block of whole assemble process ends here
    

    //pcout << " all local celss have done !" << std::endl;
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void FemGL<dim>::compute_residual(/*const LA::MPI::Vector &damped_vector*/)
  {
    TimerOutput::Scope t(computing_timer, "compute_residual");

    residual_vector = 0;

    { //block of residual vector assembly, starts here, all local objects will be released
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

    } // block of assemble residual vector, release the memory

    residual_vector.compress(VectorOperation::add);
  }
  
  template <int dim>
  void FemGL<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    { //linear solver block, will be released after it finishes to save memory leak
     LA::MPI::PreconditionAMG preconditioner;
     LA::MPI::Vector distributed_newton_update(locally_owned_dofs, mpi_communicator);
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

     {
      // With that, we can finally set up the linear solver and solve the system:
      pcout << " system_rhs.l2_norm() is " << system_rhs.l2_norm() << std::endl;
      SolverControl solver_control(10*system_matrix.m(),
                                   1e-1 * system_rhs.l2_norm());

      //SolverMinRes<LA::MPI::Vector> solver(solver_control);
      SolverFGMRES<LA::MPI::Vector> solver(solver_control);
      //SolverGMRES<LA::MPI::Vector> solver(solver_control, gmres_adddata);

      // what this .set_zero() is doing ?
      // AffineContraint::set_zero() set the values of all constrained DoFs in a vector to zero. 
      // constraints_newton_update.set_zero(distributed_newton_update);

      solver.solve(system_matrix,
                   distributed_newton_update,
                   system_rhs,
                   preconditioner);

      pcout << "   Solved in " << solver_control.last_step() << " iterations."
            << std::endl;

     }

    constraints_newton_update.distribute(distributed_newton_update);

    locally_relevant_newton_solution = distributed_newton_update;

    } // solver block ends at here to save memory leak

  }

  template <int dim>
  void FemGL<dim>::newton_iteration()
  {
    TimerOutput::Scope t(computing_timer, "newton_iteration");

    { // newton iteraion block with line search, all local objects will be released to save momory leak
     LA::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
     LA::MPI::Vector distributed_newton_update(locally_owned_dofs, mpi_communicator);          
     double previous_residual = system_rhs.l2_norm();

     {// line search loop

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
       } // for loop ends at here

     } // line search block 

     local_solution = distributed_solution;
    } // newton iteraion block with line search ends from here, all local objects will be released to save momory leak


  }

  template <int dim>
  void FemGL<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    if (refinement_strategy == "global")
      {
	{
	 // solution transfer block
         parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_transfer(dof_handler);
         solution_transfer.prepare_for_coarsening_and_refinement(local_solution);

         triangulation.refine_global(1);
	 setup_system();
         TrilinosWrappers::MPI::Vector distributed_solution_tmp(locally_owned_dofs,
							        mpi_communicator);	 
  
         solution_transfer.interpolate(distributed_solution_tmp);
         constraints_solution.distribute(distributed_solution_tmp);
         local_solution = distributed_solution_tmp;
	}
      }
    else if (refinement_strategy == "adaptive")
      {
        {
         Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
         KellyErrorEstimator<dim>::estimate(dof_handler,
				         QGauss<dim - 1>(fe.degree + 1),
					 std::map<types::boundary_id, const Function<dim> *>(),
					 local_solution,
					 estimated_error_per_cell);

         parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
									     estimated_error_per_cell, 0.3, 0.0);
        }

        triangulation.prepare_coarsening_and_refinement();

        {
         parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_transfer(dof_handler);
         solution_transfer.prepare_for_coarsening_and_refinement(local_solution);
         triangulation.execute_coarsening_and_refinement();

         // after execute refining, you must do it at here otherwise local_solution will not match tmp Vector
         setup_system();
         pcout << "setup_system() call is done !" << std::endl;
      
         TrilinosWrappers::MPI::Vector distributed_solution_tmp(locally_owned_dofs,
							     mpi_communicator);
         solution_transfer.interpolate(distributed_solution_tmp);
         constraints_solution.distribute(distributed_solution_tmp);
         local_solution = distributed_solution_tmp;
        }
        //compute_nonlinear_residual(solution);
        pcout << "adaptive_refine_grid() call is done !" << std::endl;
      } // adaptive block ends here
    
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
    
    { // DataOut block starts here, release memory
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
    } // DataOut block ends here, release memory

  }

  template <int dim>
  void FemGL<dim>::run()
  {
    pcout << "Running using Trilinos." << std::endl;

    const unsigned int n_cycles    = 1;
    const unsigned int n_iteration = 50;    
    for (cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
	  {
	   make_grid();
           setup_system();	
	  }
        else
	  refine_grid();
	
        for (unsigned int iteration_loop = 0; iteration_loop <= n_iteration; ++iteration_loop)
	  {
	     pcout << "iteration_loop: " << iteration_loop << std::endl;

             assemble_system();
	     pcout << "assembly is done !" << std::endl;
             solve();
	     pcout << " AMG solve is done !" << std::endl;	     
	     newton_iteration();
	     pcout << " newton iteration is done !" << std::endl;	     	     

             if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 1280)
              {
               TimerOutput::Scope t(computing_timer, "output");
	       output_results(iteration_loop);
              }

             computing_timer.print_summary();
             computing_timer.reset();

             pcout << std::endl;

	     if ((system_rhs.l2_norm() < 1e-9) && (cycle == 0))
	       break;
	     else if ((system_rhs.l2_norm() < 1e-6) && (cycle <= 3))
	       break;
	     else if ((system_rhs.l2_norm() < 1e-9) && (cycle > 3) && (cycle <= 6))
	       break;
	  }

      }
     computing_timer.print_summary();	
  }


  template class FemGL<3>;
} // namespace FemGL_mpi

