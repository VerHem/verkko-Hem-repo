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


#ifndef PDWCONFIG_H
#define PDWCONFIG_H

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


namespace FemGL_mpi
{
  using namespace dealii;

  /* ------------------------------------------------------------------------------------------
   * class template BinA inhereted from Function<dim>.
   * set the reference value_list to B-in-A configuration for full-step newton iteration.
   * ------------------------------------------------------------------------------------------
   */
  
  template <int dim>
  class PDWconfig : public Function<dim>
  {
  public:
    PDWconfig(double u_para, double Q_para, double half_D_para, double gap_para)
      : Function<dim>(18) // tell base Function<dim> class I want a 2-components vector-valued function
      , u(u_para)
      , q(Q_para)
      , half_D(half_D_para)
      , gap(gap_para)	
    {}

    double u, q, half_D, gap;         // D is confiment distance
    double D = 2.*half_D;
        
    virtual void vector_value(const Point<dim> &point /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 18, ExcDimensionMismatch(values.size(), 18));

      /* assign all u components with B-state and PDW OP \phi */      
      
      values[0] = gap/std::sqrt(3.); 
      phi11(values, point(0), point(1), (point(2)+half_D)/D);
      
      values[1] = 0.0; 
      phi12(values, point(0), point(1), (point(2)+half_D)/D);
      
      values[2] = 0.0; 
      phi13(values, point(0), point(1), (point(2)+half_D)/D);      
            
      values[3] = 0.0; 
      phi21(values, point(0), point(1), (point(2)+half_D)/D);      
      
      values[4] = gap/std::sqrt(3.); 
      phi22(values, point(0), point(1), (point(2)+half_D)/D);      
      
      values[5] = 0.0; 
      phi23(values, point(0), point(1), (point(2)+half_D)/D);      
      
      values[6] = 0.0; 
      phi31(values, point(0), point(1), (point(2)+half_D)/D);            
      
      values[7] = 0.0; 
      phi32(values, point(0), point(1), (point(2)+half_D)/D);
      
      values[8] = gap/std::sqrt(3.); 
            
      /* all v components are zero */

      values[9] = 0.0; values[10] = 0.0; values[11] = 0.0;
      values[12] = 0.0; values[13] = 0.0; values[14] = 0.0;
      values[15] = 0.0; values[16] = 0.0; values[17] = 0.0;

    } // vector_value() function ends here

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        PDWconfig<dim>::vector_value(points[p], value_list[p]);
    }

    void phi11(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi12(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi13(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi21(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi22(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi23(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi31(Vector<double> &, const double &x, const double &y, const double &z) const;
    void phi32(Vector<double> &, const double &x, const double &y, const double &z) const;
    //void phi33(Vector<double> &, const double &x, const double &y, const double &z) const;        
  };

  template <int dim>
  void PDWconfig<dim>::phi11(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi11 = 0.0, z = Z/D;
    values[0] += gap * u
                * (2*(-0.010674533968153196 - 0.04928146307995591*z + 0.4123298769345518*z*z - 5.957210687551436*z*z*z + 15.697110273856326*z*z*z*z
	       - 14.963347649994844*z*z*z*z*z + 4.860888318571941*z*z*z*z*z*z)
            + 2.*(0.005381764646081998 - 0.09659207269737603*z + 3.059717163195402*std::pow(z,2.) - 35.01132629189211*std::pow(z,3.)
		  + 193.9107091164305*std::pow(z,4.) -594.3817533999686*std::pow(z,5.) + 1058.3851407510472*std::pow(z,6.)
		  -1088.4440880311813*std::pow(z,7.) + 599.3158862587954*std::pow(z,8.) - 136.74038236803835*std::pow(z,9.))
	        * (std::cos(q*x) - std::cos((q*x)/2.)*std::cos((std::sqrt(3)*q*y)/2.)));
  }

  template <int dim>
  void PDWconfig<dim>::phi12(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi12 = 0.0, z = Z/D;
    values[1] += gap * u
                     *(2.*(-0.010674533968153196 - 0.04928146307995591*z + 0.4123298769345518*z*z - 5.957210687551436*z*z*z
	        + 15.697110273856326*z*z*z*z - 14.963347649994844*z*z*z*z*z + 4.860888318571941*z*z*z*z*z*z)
            - 2.*std::sqrt(3)
                *(0.005381764646081998 - 0.09659207269737603*z + 3.059717163195402*z*z - 35.01132629189211*z*z*z + 193.9107091164305*z*z*z*z
	          - 594.3817533999686*z*z*z*z*z + 1058.3851407510472*z*z*z*z*z*z - 1088.4440880311813*z*z*z*z*z*z*z
	          + 599.3158862587954*std::pow(z,8.) - 136.74038236803835*std::pow(z,9.))
                * std::sin((q*x)/2.)
		  * std::sin((std::sqrt(3.)*q*y)/2.));
  }

  template <int dim>
  void PDWconfig<dim>::phi13(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi13 = 0.0, z = Z/D;
    values[2] += gap * u
                   * (2.*(-0.010674533968153196 - 0.04928146307995591*z + 0.4123298769345518*std::pow(z,2.) - 5.957210687551436*std::pow(z,3.)
	        + 15.697110273856326*std::pow(z,4.) - 14.963347649994844*std::pow(z,5.) + 4.860888318571941*std::pow(z,6.))
            + 2.*std::sqrt(2.)
                *(-0.008476309862633497 - 0.15843411116780842*z + 1.1880072170146772*std::pow(z,2.) - 8.79341508668322*std::pow(z,3.)
	          + 32.93035389045762*std::pow(z,4.) - 56.473628486426286*std::pow(z,5.) + 44.90526146438388*std::pow(z,6.)
		  - 13.60443812339298*std::pow(z,7.))
	        *(std::cos((std::sqrt(3)*q*y)/2.) * std::sin((q*x)/2.) + std::sin(q*x)));
  }

  template <int dim>
  void PDWconfig<dim>::phi21(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi21 = 0.0, z = Z/D;
    values[3] += gap * u
                     * (2*std::sqrt(3)
             *(-0.010674533968153196 - 0.04928146307995591*z + 0.4123298769345518*std::pow(z,2) - 5.957210687551436*std::pow(z,3)
	       + 15.697110273856326*std::pow(z,4) -14.963347649994844*std::pow(z,5) + 4.860888318571941*std::pow(z,6))
             *(std::cos(q*x) + 2.*std::cos((q*x)/2.)*std::cos((std::sqrt(3.)*q*y)/2.))
            -2*std::sqrt(3.)
              *(0.005381764646081998 - 0.09659207269737603*z + 3.059717163195402*std::pow(z,2.) - 35.01132629189211*std::pow(z,3.)
	        + 193.9107091164305*std::pow(z,4.) - 594.3817533999686*std::pow(z,5.) + 1058.3851407510472*std::pow(z,6.)
		- 1088.4440880311813*std::pow(z,7.) + 599.3158862587954*std::pow(z,8.) - 136.74038236803835*std::pow(z,9.))
              *std::sin((q*x)/2.)
	      *std::sin((std::sqrt(3)*q*y)/2.));
  }

  template <int dim>
  void PDWconfig<dim>::phi22(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi22 = 0.0, z = Z/D;
    values[4] += gap * u
                * (2*(-0.005381764646081998 + 0.09659207269737603*z - 3.059717163195402*std::pow(z,2.) + 35.01132629189211*std::pow(z,3.)
	       - 193.9107091164305*std::pow(z,4.) + 594.3817533999686*std::pow(z,5.) - 1058.3851407510472*std::pow(z,6.)
	       + 1088.4440880311813*std::pow(z,7.) - 599.3158862587954*std::pow(z,8.) + 136.74038236803835*std::pow(z,9.))
             *(std::cos(q*x) - std::cos((q*x)/2.)*std::cos((std::sqrt(3)*q*y)/2.))
            + 2*std::sqrt(3.)
               *(-0.010674533968153196 - 0.04928146307995591*z + 0.4123298769345518*std::pow(z,2.) - 5.957210687551436*std::pow(z,3.)
	         + 15.697110273856326*std::pow(z,4.) - 14.963347649994844*std::pow(z,5.) + 4.860888318571941*std::pow(z,6.))
	     *(std::cos(q*x) + 2.*std::cos((q*x)/2.)*std::cos((std::sqrt(3.)*q*y)/2.)));
  }

  template <int dim>
  void PDWconfig<dim>::phi23(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi23 = 0.0, z = Z/D;
    values[5] += gap * u
                     * (2*std::sqrt(3.)
             *(-0.010674533968153196 - 0.04928146307995591*z + 0.4123298769345518*std::pow(z,2.) - 5.957210687551436*std::pow(z,3.)
	       + 15.697110273856326*std::pow(z,4.) - 14.963347649994844*std::pow(z,5.) + 4.860888318571941*std::pow(z,6.))
             *(std::cos(q*x) + 2*std::cos((q*x)/2.)*std::cos((std::sqrt(3.)*q*y)/2.))
            + 2*std::sqrt(6.)
               *(-0.008476309862633497 - 0.15843411116780842*z + 1.1880072170146772*std::pow(z,2.) - 8.79341508668322*std::pow(z,3.)
	         + 32.93035389045762*std::pow(z,4.) - 56.473628486426286*std::pow(z,5.) + 44.90526146438388*std::pow(z,6.)
		 - 13.60443812339298*std::pow(z,7.))
               *std::cos((q*x)/2.)
	       *std::sin((std::sqrt(3.)*q*y)/2.));
  }

  template <int dim>
  void PDWconfig<dim>::phi31(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi31 = 0.0, z = Z/D;
    values[6] += gap * u
                *(2*std::sqrt(2.)
                  *(0.703221673991666 + 0.07213025175696147*z - 4.458423083418102*std::pow(z,2.) + 2.9780246543754445*std::pow(z,3.))
		  *(std::cos((std::sqrt(3.)*q*y)/2.)*std::sin((q*x)/2.) + std::sin(q*x)));
  }

  template <int dim>
  void PDWconfig<dim>::phi32(Vector<double> &values, const double &x, const double &y, const double &z) const
  {
    //double phi32 = 0.0, z =z/D;
    values[7] += gap * u
                * (2*std::sqrt(6.)
                   *(0.703221673991666 + 0.07213025175696147*z - 4.458423083418102*std::pow(z,2.) + 2.9780246543754445*std::pow(z,3.))
                   *std::cos((q*x)/2.)
                   *std::sin((std::sqrt(3.)*q*y)/2.));
  }

  // template <int dim>
  // void PDWconfig<dim>::phi33(const double &x, const double &y, const double &z)
  // { return 0.; }
  
  
} // namespace FemGL_mpi

#endif
