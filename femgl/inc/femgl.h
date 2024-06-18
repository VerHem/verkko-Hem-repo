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

#ifndef FEMGL_H
#define FEMGL_H

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

#include "matep.h"

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  class FemGL
  {
  public:
    FemGL(unsigned int Q_degree, ParameterHandler &);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void compute_residual(/*const LA::MPI::Vector &*/);
    void solve();
    void newton_iteration();
    void refine_grid(std::string &);
    void output_results(const std::string &dirc) const;


    /*------------------------------------------------------------------------------------
     * declear cell-functions of K_i terms, alpha term and beta_i term, both lhs and rhs
     * they handle all cell calculations of rhl and rhs on assembly and residual fucntions
     *------------------------------------------------------------------------------------*/

    /* >>>>>>>>  lhs cell matrices terms <<<<<<<<< */
    
    double mat_lhs_alpha(FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_u_j_q,
		         FullMatrix<double> &phi_v_i_q, FullMatrix<double> &phi_v_j_q);
    
    double mat_lhs_beta1(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
	                 FullMatrix<double> &phi_u_i_q,FullMatrix<double> &phi_u_j_q,
		         FullMatrix<double> &phi_v_i_q,FullMatrix<double> &phi_v_j_q);

    double mat_lhs_beta2(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
			 FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_u_j_q,
		         FullMatrix<double> &phi_v_i_q, FullMatrix<double> &phi_v_j_q);

    double mat_lhs_beta3(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
                         FullMatrix<double> &phi_u_i_q,FullMatrix<double> &phi_u_j_q,
                         FullMatrix<double> &phi_v_i_q,FullMatrix<double> &phi_v_j_q);

    double mat_lhs_beta4(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
                         FullMatrix<double> &phi_u_i_q,FullMatrix<double> &phi_u_j_q,
                         FullMatrix<double> &phi_v_i_q,FullMatrix<double> &phi_v_j_q);

    double mat_lhs_beta5(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
                         FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_u_j_q,
                         FullMatrix<double> &phi_v_i_q, FullMatrix<double> &phi_v_j_q);

    
    
    double mat_lhs_K1(std::vector<FullMatrix<double>> &grad_phi_u_i_q, std::vector<FullMatrix<double>> &grad_phi_v_i_q,
		      std::vector<FullMatrix<double>> &grad_phi_u_j_q, std::vector<FullMatrix<double>> &grad_phi_v_j_q);

    double mat_face_lhs_K1(FullMatrix<double> &phi_uf_i_q, FullMatrix<double> &phi_uf_j_q,
		           FullMatrix<double> &phi_vf_i_q, FullMatrix<double> &phi_vf_j_q);
    
    
    /*double mat_lhs_K2K3(std::vector<FullMatrix<double>> &grad_phi_u_i_q, std::vector<FullMatrix<double>> &grad_phi_v_i_q,
      std::vector<FullMatrix<double>> &grad_phi_u_j_q, std::vector<FullMatrix<double>> &grad_phi_v_j_q);*/

    
    /*there was a hiden bug, you can't put reference into std::vector directly */
    double mat_lhs_K2K3(std::vector<FullMatrix<double>> grad_phi_u_i_q, std::vector<FullMatrix<double>> grad_phi_v_i_q,
                        std::vector<FullMatrix<double>> grad_phi_u_j_q, std::vector<FullMatrix<double>> grad_phi_v_j_q);
    
    /* >>>>>>>>  rhs cell matrices terms <<<<<<<<< */
    
    double vec_rhs_alpha(FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q,
			 FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v);
    
    double vec_rhs_beta1(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
			 FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q);    

    double vec_rhs_beta2(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
		         FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q);

    double vec_rhs_beta3(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
                         FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q);

    double vec_rhs_beta4(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
                         FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q);

    double vec_rhs_beta5(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
                         FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q);

    


    double vec_rhs_K1(std::vector<FullMatrix<double>> &grad_old_u_q, std::vector<FullMatrix<double>> &grad_old_v_q,
		      std::vector<FullMatrix<double>> &grad_phi_u_i_q, std::vector<FullMatrix<double>> &grad_phi_v_i_q);

    double vec_face_rhs_K1(FullMatrix<double> &phi_uf_i_q, FullMatrix<double> &phi_vf_i_q,
			   FullMatrix<double> &old_f_solution_u, FullMatrix<double> &old_f_solution_v);
    
    
    /*double vec_rhs_K2K3(std::vector<FullMatrix<double>> &grad_phi_u_i_q, std::vector<FullMatrix<double>> &grad_phi_v_i_q,
      std::vector<FullMatrix<double>> &grad_old_u_q, std::vector<FullMatrix<double>> &grad_old_v_q);*/

    /*there is a hiden bug, you can't put reference into std::vector directly */
    double vec_rhs_K2K3(std::vector<FullMatrix<double>> grad_phi_u_i_q, std::vector<FullMatrix<double>> grad_phi_v_i_q,
			std::vector<FullMatrix<double>> grad_old_u_q, std::vector<FullMatrix<double>> grad_old_v_q);
    
    /*------------------------------------------------------------------------------------
     * declearations of cell-functions of K_i terms, alpha term and beta_i term, end here.
     *------------------------------------------------------------------------------------*/
    
    
    // matrices constraction function for last step u, v tensors
    void   vector_matrix_generator(const FEValues<dim>  &fe_values,
                                   const char &vector_flag,
                                   const unsigned int   q, const unsigned int   n_q_point,
                                   FullMatrix<double>   &u_matrix_at_q,
                                   FullMatrix<double>   &v_matrix_at_q);

    void   vector_face_matrix_generator(const FEFaceValues<dim> &fe_face_values,
					const char &vector_flag,
					const unsigned int   q, const unsigned int n_q_points,
					FullMatrix<double>   &u_face_matrix_at_q,
					FullMatrix<double>   &v_face_matrix_at_q,
					types::boundary_id b_id);

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

    void   phi_matrix_face_generator(const FEFaceValues<dim> &fe_face_values,
   				     const unsigned int  x, const unsigned int q,
				     FullMatrix<double>  &phi_u_face_at_x_q,
				     FullMatrix<double>  &phi_v_face_at_x_q,
				     types::boundary_id b_id);
   

    // matrices construction function for gradient of shape function, grad_k_phi_u/v at local dof x and Gaussian q
    void   grad_phi_matrix_container_generator(const FEValues<dim> &fe_values,
                                               const unsigned int x, const unsigned int q,
                                               std::vector<FullMatrix<double>> &container_grad_phi_u_x_q,
                                               std::vector<FullMatrix<double>> &container_grad_phi_v_x_q);
    
    
    //std::string        refinement_strategy = "global";
    unsigned int       degree, cycle, iteration_loop;    
    MPI_Comm           mpi_communicator;
    
    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;
    ParameterHandler                          &conf;

    /* container for FEValuesExtractors::scalar
     * FEValuesExtractors, works for FEValues, FEFaceValues, FESubFaceValues
     * for he3 GL eqn, 18 copies of scalar extractors are needed, they are
     * defined by the matrix components they represents for.*/

    std::vector<FEValuesExtractors::Scalar> components_u;
    std::vector<FEValuesExtractors::Scalar> components_v;

    /* ComponentMask for marking Dirichlet components.
     * This object is necessary for interpolate_boundary_values() function call 
     * during setup() stage to built AffineConstraint object.
     * This object will be innitliized during femgl() constructor call by a list of bool.
     */
    //ComponentMask comp_mask_x; ComponentMask comp_mask_y; ComponentMask comp_mask_z;

    const std::vector< bool > Dirichlet_x_marking_list{true, false, false, true, false, false, true, false, false,
					 	       true, false, false, true, false, false, true, false, false};

    const std::vector< bool > Dirichlet_y_marking_list{false, true, false, false, true, false, false, true, false,
					 	       false, true, false, false, true, false, false, true, false};
    
    const std::vector< bool > Dirichlet_z_marking_list{false, false, true, false, false, true, false, false, true,
					 	       false, false, true, false, false, true, false, false, true};
    

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints_newton_update;
    AffineConstraints<double> constraints_solution;    

    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector       locally_relevant_newton_solution;   // this is newton update Vector
    LA::MPI::Vector       local_solution;                     // this is final solution Vector
    LA::MPI::Vector       locally_relevant_damped_vector;
    LA::MPI::Vector       system_rhs;
    LA::MPI::Vector       residual_vector;

    // the following material parameters are T=0 values
    // A = u + i v has unit of k_B T_c
    const double K1      = 0.42072;
    const double K2      = 0.42072;     // 0.42072
    const double K3      = 0.42072;     // 0.42072
    
    // const double alpha_0  = 1.0;
    // const double beta1    = -0.010657; // -0.010657
    // const double beta2    = 0.0213139; // 0.0213139   
    // const double beta3    = 0.0213139; // 0.0213139
    // const double beta4    = 0.0213139; // 0.0213139   
    // const double beta5    = -0.0213139; // -0.0213139

    Matep mat; // matep object
    
    double alpha,
           beta1, beta2, beta3, beta4, beta5;

    
    // AdGR diffuse parameter
    double bt; // 1e10 specular, 2.0 diffuse

    // reduced tmeprature t=T/T_c, pressure p in bar
    double reduced_t, p;

    // SCC switch for matep
    bool SCC_key;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
        
  };

} // namespace FemGL_mpi ends here

#endif
