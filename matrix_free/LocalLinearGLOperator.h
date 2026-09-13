#ifndef LOCALLINEARGLOPERATOR_H
#define LOCALLINEARGLOPERATOR_H

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include "GLBackgroundCoefficients.h"
#include "LocalLinearGLOperatorQuad.h"


namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  class LocalLinearGLOperator
  {
  public:

    static constexpr unsigned int n_components = 9;
    static constexpr unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    LocalLinearGLOperator(const GLBackgroundCoefficients<dim, fe_degree, Number> &background)
      : background(&background)
    {}

    DEAL_II_HOST_DEVICE
    void operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
                    const Portable::DeviceBlockVector<Number> &src,
                    Portable::DeviceBlockVector<Number> &dst) const;
  private:
    const GLBackgroundCoefficients<dim, fe_degree, Number> *background;
  };

  // functor
  template <int dim, int fe_degree, typename Number>
  DEAL_II_HOST_DEVICE
  void LocalLinearGLOperator<dim, fe_degree, Number>::operator()(
      const typename Portable::MatrixFree<dim, Number>::Data *data,
      const Portable::DeviceBlockVector<Number> &src,
      Portable::DeviceBlockVector<Number> &dst) const
  {
    //U block
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> u_eval(data, 0);
    // V block
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> v_eval(data, 1);

    //Read the two blocks of the Krylov vector.
    u_eval.read_dof_values(src.block(0));
    v_eval.read_dof_values(src.block(1));

    // I need values for the reaction term and gradients for the Laplacian.
    u_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    v_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    /* ------------------------------------------------------------
     * Physics at quadrature points.
     * ------------------------------------------------------------
     */

    LocalLinearGLOperatorQuad<dim, fe_degree, Number>
      quad_operator(data, background->get_u0_values(), background->get_v0_values());
    
    data->for_each_quad_point(
      [&](const int q_point)
      { quad_operator(&u_eval, &v_eval, q_point); });

    /* ------------------------------------------------------------
     * Integrate the two blocks.
     * ------------------------------------------------------------
     */
    u_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    v_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    /* ------------------------------------------------------------
     * Scatter to the two output blocks.
     * ------------------------------------------------------------
     */
    u_eval.distribute_local_to_global(dst.block(0));
    v_eval.distribute_local_to_global(dst.block(1));
    
  } // functor operator() ends here

} // namespace VerHem

#endif
