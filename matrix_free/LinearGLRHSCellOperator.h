#ifndef LINEARGLRHSCELLOPERATOR_H
#define LINEARGLRHSCELLOPERATOR_H

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  class LinearGLRHSCellOperator
    {
     public:

     static constexpr unsigned int n_components = 9;
     static constexpr unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);
      
     LinearGLRHSCellOperator(const Number K1_in,
                             const Number alpha_in,
                             const Number beta2_in)
       : K1(K1_in)
       , alpha(alpha_in)
       , beta2(beta2_in)
     {}

      DEAL_II_HOST_DEVICE
      void operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
                      const Portable::DeviceBlockVector<Number> &src,
                      Portable::DeviceBlockVector<Number> &dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> fe_u(data, 0);
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> fe_v(data, 1);

    fe_u.read_dof_values(src.block(0));
    fe_v.read_dof_values(src.block(1));

    fe_u.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    fe_v.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    data->for_each_quad_point(
      [&](const int q_point)
      {
        const auto u = fe_u.get_value(q_point);
        const auto v = fe_v.get_value(q_point);

        const auto grad_u = fe_u.get_gradient(q_point);
        const auto grad_v = fe_v.get_gradient(q_point);

        Number norm2 = Number(0.0);

        for (unsigned int a = 0; a < n_components; ++a)
          {
            norm2 += u[a] * u[a];
            norm2 += v[a] * v[a];
          }

        const Number c = alpha + beta2 * norm2;

        /*
         * R1: -K1 Delta u + c u = K1 grad(u) . grad(phi) + c u phi
         */
        fe_u.submit_gradient(K1 * grad_u, q_point);
        fe_u.submit_value(c * u, q_point);

        /*
         * R2: -K1 Delta v + c v = K1 grad(v) . grad(phi) + c v phi
         */
        fe_v.submit_gradient(K1 * grad_v, q_point);
        fe_v.submit_value(c * v, q_point);
      });

    fe_u.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    fe_v.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    fe_u.distribute_local_to_global(dst.block(0));
    fe_v.distribute_local_to_global(dst.block(1));
    
  } // LinearGLRHSCellOperator::operator() ends here

  private:
      Number K1,  alpha, beta2;
 }; // LinearGLRHSCellOperator ends here

} // VerHem namespace ends here

#endif
