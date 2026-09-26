#ifndef LAPLACEDIAGONALCELLOPERATORQUAD_H
#define LAPLACEDIAGONALCELLOPERATORQUAD_H

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace VerHem
{
  template <int dim, int fe_degree, typename Number>
  class LaplaceDiagonalCellOperatorQuad
  {
    static constexpr unsigned int n_components = 9;
    // static constexpr unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);
    
    public:
      DEAL_II_HOST_DEVICE
      void operator()(Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> *fe_eval,
                      const int q_point) const
      { fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point); } // operator() ends here
  };

} // VerHem nameespace ends here
