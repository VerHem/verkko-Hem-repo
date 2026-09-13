#ifndef LOCALLINEARGLOPERATORQUAD_H
#define LOCALLINEARGLOPERATORQUAD_H

#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  class LocalLinearGLOperatorQuad
  {
  public:

    static constexpr unsigned int n_components = 9;
    static constexpr unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    DEAL_II_HOST_DEVICE
    LocalLinearGLOperatorQuad(
      const typename Portable::MatrixFree<dim, Number>::Data *data,
      const Number *u0_coefficients,
      const Number *v0_coefficients)
      : data(data)
      , u0_coefficients(u0_coefficients)
      , v0_coefficients(v0_coefficients)
    {}


    DEAL_II_HOST_DEVICE
    void operator()(Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> *u_eval,
                    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> *v_eval,
                    const int q_point) const;

  private:
    const typename Portable::MatrixFree<dim, Number>::Data *data;
    const Number *u0_coefficients;
    const Number *v0_coefficients;
  };


  template <int dim, int fe_degree, typename Number>
  DEAL_II_HOST_DEVICE
  void LocalLinearGLOperatorQuad<dim, fe_degree, Number>::operator()(
      Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> *u_eval,
      Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> *v_eval,
      const int q_point) const
  {
    const unsigned int cell = data->cell_index;
    const unsigned int pos  = data->local_q_point_id(cell, q_point);
    /* ------------------------------------------------------------
     * Background values at this quadrature point.
     * ------------------------------------------------------------
     */
    const Number *u0 = &u0_coefficients[pos * n_components];
    const Number *v0 = &v0_coefficients[pos * n_components];
    /* ------------------------------------------------------------
     * Current Newton increment.
     *     U = delta u
     *     V = delta v
     * ------------------------------------------------------------
     */
    const auto U = u_eval->get_value(q_point);
    const auto V = v_eval->get_value(q_point);
    const auto grad_U = u_eval->get_gradient(q_point);
    const auto grad_V = v_eval->get_gradient(q_point);

    /* ------------------------------------------------------------
     * Background norms |U0|^2 |V0|^2
     * ------------------------------------------------------------
     */
    Number u0_square = 0.;
    Number v0_square = 0.;

    for (unsigned int c = 0; c < n_components; ++c)
    {
      u0_square += u0[c] * u0[c];
      v0_square += v0[c] * v0[c];
    }

    /* ------------------------------------------------------------
     * Common coefficient
     *     c(x) = alpha + beta_2 ( |U0|^2 + |V0|^2 )
     *
     * alpha and beta_2 should be supplied to the physics layer.     *
     * They are placeholders here until we connect this class
     * to existing LinearGLPhysics class.
     * ------------------------------------------------------------
     */
    const Number c = alpha + beta_2 * (u0_square + v0_square);
    /* ------------------------------------------------------------
     * Dot products: U0 . U, U0 . V, V0 . U, V0 . V
     * ------------------------------------------------------------
     */
    Number u0_dot_U = 0.;
    Number u0_dot_V = 0.;
    Number v0_dot_U = 0.;
    Number v0_dot_V = 0.;

    for (unsigned int comp = 0; comp < n_components; ++comp)
    {
      u0_dot_U += u0[comp] * U[comp];
      u0_dot_V += u0[comp] * V[comp];

      v0_dot_U += v0[comp] * U[comp];
      v0_dot_V += v0[comp] * V[comp];
    }

    /* ------------------------------------------------------------
     * Reaction terms.From the equations:
     * Ru = c U + 2 beta_2 U0 ( U0 . U + U0 . V )
     * Rv = c V + 2 beta_2 V0 ( U0 . U + V0 . V )
     * ------------------------------------------------------------
     */

    auto reaction_U = U;
    auto reaction_V = V;
    const Number common_U = u0_dot_U + u0_dot_V;
    const Number common_V = v0_dot_U + v0_dot_V;

    for (unsigned int comp = 0; comp < n_components; ++comp)
    {
      reaction_U[comp] = c * U[comp] + 2. * beta_2 * u0[comp] * common_U;
      reaction_V[comp] = c * V[comp] + 2. * beta_2 * v0[comp] * common_V;
    }

    /* ------------------------------------------------------------
     * Weak form of -K1 Laplacian:
     *     K1 (grad U, grad W_U)
     *     K1 (grad V, grad W_V)
     * ------------------------------------------------------------
     */

    auto flux_U = grad_U;
    auto flux_V = grad_V;

    flux_U *= K1;
    flux_V *= K1;
    
    /* ------------------------------------------------------------
     * Submit reaction + diffusion.
     * ------------------------------------------------------------
     */

    u_eval->submit_value(reaction_U, q_point);
    u_eval->submit_gradient(flux_U, q_point);

    v_eval->submit_value(reaction_V, q_point);
    v_eval->submit_gradient(flux_V, q_point);
  } // operator() ends here

} // namespace VerHem

#endif
