#ifndef LINEARGLOPERATOR_H
#define LINEARGLOPERATOR_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_gmres.h>

// #include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include "GLBackgroundCoefficients.h"
#include "LocalLinearGLOperator.h"

namespace VerHem
{
  using namespace dealii;


  template <int dim, int fe_degree, typename Number>
  class LinearGLOperator : public EnableObserverPointer
  {
  public:

    // using VectorType      = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Default>;

    // constructor
    LinearGLOperator(const DoFHandler<dim>           &dof_handler_u, const DoFHandler<dim>           &dof_handler_v,
                     const AffineConstraints<Number> &constraints_u, const AffineConstraints<Number> &constraints_v,
                     const BlockVectorType &background_u_v_sol);

    // vmult for solver
    void vmult(BlockVectorType &dst,
               const BlockVectorType &src) const;

    void initialize_dof_vector(BlockVectorType &vec) const;
    void update_background(const BlockVectorType &background_u_v_sol);

  private:
    Portable::MatrixFree<dim, Number> MF_MetaData_Engine;

    // here is the 1st time GLBackgroundCoefficients appears
    GLBackgroundCoefficients<dim, fe_degree, Number> background_coefficients;

  }; //LinearGLoperator declaretion ends here

  template <int dim, int fe_degree, typename Number>
  LinearGLOperator<dim, fe_degree, Number>::LinearGLOperator(const DoFHandler<dim> &dof_handler_u,
                                                             const DoFHandler<dim> &dof_handler_v,
                                                             const AffineConstraints<Number> &constraints_u,
                                                             const AffineConstraints<Number> &constraints_v,
                                                             const BlockVectorType &background_u_v_sol)
  {
    const MappingQ<dim> mapping(fe_degree);
    
    typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients | update_JxW_values | update_quadrature_points;
    
    const QGauss<1> quad(fe_degree + 1);
    
    /*------------------------------------------------------------
     * using similar multi-DoFHandler pattern as step-104 for block structure
     * DoFHandler 0 -> U, DoFHandler 1 -> V
     * ------------------------------------------------------------
     */
    std::vector<const DoFHandler<dim> *>           dof_handlers = {&dof_handler_u, &dof_handler_v};
    std::vector<const AffineConstraints<Number> *> constraints = {&constraints_u, &constraints_v};
    MF_MetaData_Engine.reinit(mapping, dof_handlers, constraints, quad, additional_data);


    /* --------------------------------------------------------
     * the rest of LinearGLOperator constructor is preparing background data
     * in Atribute GLBackgroundCoefficients<...> background_coefficients;
     * --------------------------------------------------------
     */    
    //Allocate the quadrature-point background storage.
    background_coefficients.reinit(MF_MetaData_Engine);

    //Interpolate the initial background into the quadrature-point coefficient arrays.
    background_coefficients.update(MF_MetaData_Engine, background_u_v_sol);
  } // LinearGLOperator() ends here

  template <int dim, int fe_degree, typename Number>
  void LinearGLOperator<dim, fe_degree, Number>::vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    dst = static_cast<Number>(0.);
    LocalLinearGLOperator<dim, fe_degree, Number> cell_LinearGLoperator(background_coefficients);

    MF_MetaData_Engine.cell_loop(cell_LinearGLoperator, src, dst);

    /* For the ordinary linear operator, constrained values of
     * the result are copied from the source.
     * There are two differenct APIs supporting distributed BlockVector copy.
     * Step-104 use the overload with parameter of type distributed BlockVector.
     *
     * Meanwhile, in portable_matrix_free.templates.h, other overload has parameter dof_handler_index
     * to take dofhandler of different block. I keep both of them here to see which one better.
     */
    // MF_MetaData_Engine.copy_constrained_values(src.block(0), dst.block(0), 0);
    // MF_MetaData_Engine.copy_constrained_values(src.block(1), dst.block(1), 1);
    MF_MetaData_Engine.copy_constrained_values(src, dst);
  } // vmult() ends here

  template <int dim, int fe_degree, typename Number>
  void LinearGLOperator<dim, fe_degree, Number>::initialize_dof_vector(BlockVectorType &vec) const
  { MF_MetaData_Engine.initialize_dof_vector(vec); }

  template <int dim, int fe_degree, typename Number>
  void LinearGLOperator<dim, fe_degree, Number>::update_background(const BlockVectorType &background_u_v_sol)
  { background_coefficients.update(MF_MetaData_Engine, background_u_v_sol); }
  
  template class LinearGLOperator<3, 1, float>;
  template class LinearGLOperator<3, 1, double>;
  
} // namespace VerHem

#endif
