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
  double FemGL<dim>::vec_rhs_beta5(FullMatrix<double> &old_solution_u, FullMatrix<double> &old_solution_v,
				   FullMatrix<double> &phi_u_i_q, FullMatrix<double> &phi_v_i_q)
  {
    //block of assembly starts from here, all local objects in there will be release to save memory leak

    FullMatrix<double>   u0_v0t(3,3) /*^*/,
                         u0t_u0(3,3) /*o*/, u0t_v0(3,3) /*s*/,
                         v0t_v0(3,3) /*x*/;

    FullMatrix<double>   v0_phi_ut_i_q(3,3), // ?5
      
                         u0t_phi_u_i_q(3,3), // ?1
                         u0t_phi_v_i_q(3,3), // ?2
                         v0t_phi_u_i_q(3,3), // ?3
                         v0t_phi_v_i_q(3,3), // ?4
                         /* ************** */
                         phi_ut_i_q_v0(3.3), // ^v1
                         phi_vt_i_q_u0(3.3); // ^v2

    FullMatrix<double>   ms1(3,3), ms2(3,3),
                         mm1(3,3), mm2(3,3);
                                   
    FullMatrix<double>   poly_I(3,3);
          
    /*-------------------------------------------------------------*/
    /* phi^u, phi^v matrices have been cooked up in other fuctions */
    /* old_u old_v matrices have been cooked up in other functions */
    /*-------------------------------------------------------------*/

    /* -------------------------------------------------------------
     * conduct matrices multiplacations: initlize matrices
     * -------------------------------------------------------------
     */
    
    u0_v0t = 0.0; /*^*/
    u0t_u0 = 0.0; /*o*/
    u0t_v0 = 0.0; /*s*/
    v0t_v0 = 0.0; /*x*/

    /* *********************/
    
    v0_phi_ut_i_q = 0.0; // ?5
    
    u0t_phi_u_i_q = 0.0; // ?1
    u0t_phi_v_i_q = 0.0; // ?2
    v0t_phi_u_i_q = 0.0; // ?3
    v0t_phi_v_i_q = 0.0; // ?4
    /* ************** */
    phi_ut_i_q_v0 = 0.0; // ^v1
    phi_vt_i_q_u0 = 0.0; // ^v2
                                       
    /* -------------------------------------------------------------
     * conduct matrices multiplacations: matrices multiplication
     * -------------------------------------------------------------
     */
   
    old_solution_u.mTmult(u0_v0t, old_solution_v);
    old_solution_u.Tmmult(u0t_u0, old_solution_u);
    old_solution_u.Tmmult(u0t_v0, old_solution_v);
    old_solution_v.Tmmult(v0t_v0, old_solution_v);                
    

    old_solution_v.mTmult(v0_phi_ut_i_q, phi_u_i_q);    

    old_solution_u.Tmmult(u0t_phi_u_i_q, phi_u_i_q);
    old_solution_u.Tmmult(u0t_phi_v_i_q, phi_v_i_q);    
    old_solution_v.Tmmult(v0t_phi_u_i_q, phi_u_i_q);
    old_solution_v.Tmmult(v0t_phi_v_i_q, phi_v_i_q);    
    /* ********************* */
    phi_u_i_q.Tmmult(phi_ut_i_q_v0, old_solution_v);
    phi_v_i_q.Tmmult(phi_vt_i_q_u0, old_solution_u);

    /**********************************************/
    /*            construct ms1, ms2,ms3          */
    /**********************************************/
    ms1 = 0.0; // ?1-?4
    ms1.add(1.0, u0t_phi_u_i_q);
    ms1.add(-1.0, v0t_phi_v_i_q);

    ms2 = 0.0; // ^v1+?3+^v2+?2
    ms2.add(1.0, phi_ut_i_q_v0);
    ms2.add(1.0, v0t_phi_u_i_q);
    ms2.add(1.0, phi_vt_i_q_u0);
    ms2.add(1.0, u0t_phi_v_i_q);

    mm1 = 0.0; //^*?5
    u0_v0t.mmult(mm1, v0_phi_ut_i_q);

    mm2 = 0.0; //x*?4
    v0t_v0.mmult(mm2, v0t_phi_v_i_q);
    
    /* ********  construct the matrices summation ******** */

    poly_I = 0.0;
    
    u0t_u0.mmult(poly_I, ms1, true);

    u0t_v0.mmult(poly_I, ms2, true);

    poly_I.add(-1.0, mm1);

    poly_I.add(1.0, mm2);    

    
    /* **  matrices summation construction ends here ***** */
    
    return  poly_I.trace();
  }

  template class FemGL<3>;

} // namespace FemGL_mpi ends at here

