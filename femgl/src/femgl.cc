/*
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

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  FemGL<dim>::FemGL(unsigned int Q_degree,
		    ParameterHandler &prmHandler)
    : degree(Q_degree)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(Q_degree), 18)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , conf(prmHandler)  
    , components_u(9, FEValuesExtractors::Scalar())
    , components_v(9, FEValuesExtractors::Scalar())
  //, reduced_t(0.0) // t = 0.3 Tc
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

    /*--------------------------------------------------*/
    /*        configuration parameters reading          */
    /*--------------------------------------------------*/
    
    /* physical parameters */
    conf.enter_subsection("physical parameters");

    reduced_t = conf.get_double("t_reduced");

    bt        = conf.get_double("AdGR diffuse length");
    
    conf.leave_subsection();

    /* Initialize the component mask object by a bool vector 
     * This is necessary for setting up AffineConstraints objects
     * in the interpolate_boundary_values() call.
     * According to AdGR BCs, 2, 5, 8, 11, 14, 17 components are zero-Dirichlet when normal vector is z
     */

    // comp_mask_z(Dirichlet_z_marking_list);
  }

  template class FemGL<3>;
  //template class FemGL<2>;      
} // namespace FemGL_mpi

