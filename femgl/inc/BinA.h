/*
 *
 */

#ifndef BINA_H
#define BINA_H

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

//#include "matep.h"

namespace FemGL_mpi
{
  using namespace dealii;

  /* ------------------------------------------------------------------------------------------
   * class template BinA inhereted from Function<dim>.
   * set the reference value_list to B-in-A configuration for full-step newton iteration.
   * ------------------------------------------------------------------------------------------
   */
  
  template <int dim>
  class BinA : public Function<dim>
  {
  public:
    BinA(double r, double p, double t, double matelem_A, double matelem_B)
      : Function<dim>(18) // tell base Function<dim> class I want a 2-components vector-valued function
      , radius(r)
      , p(p)
      , reduced_t(t)
      , matrix_elem_A(matelem_A)
      , matrix_elem_B(matelem_B)
    {}

    double radius;         // in unit of \xi_0    
    double p, reduced_t;
    double matrix_elem_A, matrix_elem_B;
    //Matep mat;           // matep object
    
    virtual void vector_value(const Point<dim> &point /*p*/,
                              Vector<double> &values) const override
    {
      Assert(values.size() == 18, ExcDimensionMismatch(values.size(), 18));

      if ( (point(0)*point(0) + point(1)*point(1) + point(2)*point(2)) <= (radius*radius) )     //inside sphere, B-phase
	{
	  for (unsigned int index = 0; index < values.size(); index++)
           {
	     if (
		 (index == 0)     /*u11*/
		 || (index == 4)  /*u22*/
		 || (index == 8)  /*u33*/		 
		)
	       values[index] = matrix_elem_B; //mat.gap_B_td(p, reduced_t) * 0.57735f;
             else
	       values[index] = 0.0;	       
	   } // B-phase setup loop
	}
      else if ( (point(0)*point(0) + point(1)*point(1) + point(2)*point(2)) > (radius*radius) ) //outside sphere, A-phase
	{
	  for (unsigned int index = 0; index < values.size(); index++)
           {
	     if (
		 (index == 0)      /*u11*/
		 || (index == 10)  /*v12*/
		)
	       values[index] = matrix_elem_A; //mat.gap_A_td(p, reduced_t) * 0.707107f;
             else
	       values[index] = 0.0;	       
	   } // A-phase setup loop

     	}
    } // vector_value() function ends here

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        BinA<dim>::vector_value(points[p], value_list[p]);
    }
  };

} // namespace FemGL_mpi

#endif
