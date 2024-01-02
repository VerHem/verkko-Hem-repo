/* ------------------------------------------------------------------------------------------
 *
 * This is header of finite element solver of ternsor order parameter GL equation 
 * e.g., order parameter GL eqns of p-wave superfluid helium-3. This paticular header
 * declears the solver class template both for 2D and 3D. 
 * And it also support liner or qudartic finite elements.
 *
 * It is developed on the top of deal.II 9.3.3 finite element C++ library. 
 * 
 * License of this code is GNU Lesser General Public License, which been 
 * published by Free Software Fundation, either version 2.1 and later version.
 * You are free to use, modify and redistribute this program.
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

namespace FemGL_mpi
{
  using namespace dealii;

  template <int dim>
  class FemGL
  {
  public:
    FemGL(unsigned int Q_degree);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void compute_residual(/*const LA::MPI::Vector &*/);
    void solve();
    void newton_iteration();
    void refine_grid(std::string &);
    void output_results(const unsigned int cycle) const;

    // matrices constraction function for last step u, v tensors
    void   vector_matrix_generator(const FEValues<dim>  &fe_values,
                                   const char &vector_flag,
                                   const unsigned int   q, const unsigned int   n_q_point,
                                   FullMatrix<double>   &u_matrix_at_q,
                                   FullMatrix<double>   &v_matrix_at_q);

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

    // matrices construction function for gradient of shape function, grad_k_phi_u/v at local dof x and Gaussian q
    void   grad_phi_matrix_container_generator(const FEValues<dim> &fe_values,
                                               const unsigned int x, const unsigned int q,
                                               std::vector<FullMatrix<double>> &container_grad_phi_u_x_q,
                                               std::vector<FullMatrix<double>> &container_grad_phi_v_x_q); 

    std::string        refinement_strategy = "global";
    unsigned int       degree, cycle;    
    MPI_Comm           mpi_communicator;
    
    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    /* container for FEValuesExtractors::scalar
     * FEValuesExtractors, works for FEValues, FEFaceValues, FESubFaceValues
     * for he3 GL eqn, 18 copies of scalar extractors are needed, they are
     * defined by the matrix components they represents for.*/

    std::vector<FEValuesExtractors::Scalar> components_u;
    std::vector<FEValuesExtractors::Scalar> components_v;


    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints_newton_update;
    AffineConstraints<double> constraints_solution;    

    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector       locally_relevant_newton_solution;
    LA::MPI::Vector       local_solution;    // this is final solution Vector
    LA::MPI::Vector       locally_relevant_damped_vector;
    LA::MPI::Vector       system_rhs;
    LA::MPI::Vector       residual_vector;

    const double K1      = 0.5;
    const double alpha_0 = 2.0;
    const double beta    = 0.5;

    double reduced_t;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
        
  };

} // namespace FemGL_mpi ends here

#endif
