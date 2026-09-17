#ifndef GLBACKGROUNDCOEFFICIENTS_H
#define GLBACKGROUNDCOEFFICIENTS_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

// #include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

// #include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>


namespace VerHem
{
  using namespace dealii;

  /* ----------------------------------------------------------------
   * Device-side cell operator used to interpolate the background
   * FE solution into quadrature-point values.
   * This is NOT part of the GMRES matrix-vector product.
   * It is executed only when the Newton background is updated.
   * ----------------------------------------------------------------
   */
  template <int dim, int fe_degree, typename Number>
  class LocalGLBackgroundCoefficientOperator
  {
  public:

    static constexpr unsigned int n_components = 9;
    static constexpr unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

    LocalGLBackgroundCoefficientOperator(Number *u0_coefficients, Number *v0_coefficients)
      : u0_coefficients(u0_coefficients)
      , v0_coefficients(v0_coefficients)
    {}

    DEAL_II_HOST_DEVICE
    void operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
                    const Portable::DeviceBlockVector<Number> &src,
                    Portable::DeviceBlockVector<Number> &dst) const;
  private:

    Number *u0_coefficients;
    Number *v0_coefficients;
    
  }; // LocalGLBackgroundCoefficientOperator declaretion ends here

  template <int dim, int fe_degree, typename Number>
  DEAL_II_HOST_DEVICE
  void LocalGLBackgroundCoefficientOperator<dim, fe_degree, Number>::operator()    
    (const typename Portable::MatrixFree<dim, Number>::Data *data,
     const Portable::DeviceBlockVector<Number> &src,
     Portable::DeviceBlockVector<Number> & /*dst*/) const
  {
    // U background.
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> u0_eval(data, 0);
    // V background.
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number> v0_eval(data, 1);

    u0_eval.read_dof_values(src.block(0));
    v0_eval.read_dof_values(src.block(1));

    u0_eval.evaluate(EvaluationFlags::values);
    v0_eval.evaluate(EvaluationFlags::values);

    const unsigned int cell = data->cell_index;

    data->for_each_quad_point(
      [&](const int q_point)
      {
        const unsigned int pos = data->local_q_point_id(cell, q_point);
        
        const auto u0 = u0_eval.get_value(q_point);
        const auto v0 = v0_eval.get_value(q_point);

        for (unsigned int c = 0; c < n_components; ++c)
        {
          u0_coefficients[
            pos * n_components + c] = u0[c];

          v0_coefficients[
            pos * n_components + c] = v0[c];
        }
      });
    
  } // LocalGLBackgroundCoefficientOperator::operator() ends here
  
  /* ----------------------------------------------------------------
   * Background data stored at quadrature points.
   * u0: [quadrature point][component]
   * v0: [quadrature point][component]
   * The quadrature-point index is local to each MPI process.
   * ----------------------------------------------------------------
   */
  template <int dim, int fe_degree, typename Number>
  class GLBackgroundCoefficients
  {
  public:
    static constexpr unsigned int n_components = 9;
    static constexpr unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);
    
    using VectorType      = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Default>;

    // constructor
    GLBackgroundCoefficients() {};

    // reinit()
    void reinit(const Portable::MatrixFree<dim, Number> &mf_data);

    // update() 
    void update(const Portable::MatrixFree<dim, Number> &mf_data,
                const BlockVectorType &background);

    Number *get_u0_values();
    const Number *get_u0_values() const;

    Number *get_v0_values();
    const Number *get_v0_values() const;


  private:
    VectorType u0_coefficients;// ??? Isn't this BlockVectorType ???
    VectorType v0_coefficients;
    
  }; // GLBackgroundCoefficients declareation ends here

  // template <int dim, int fe_degree, typename Number>
  // GLBackgroundCoefficients<dim, fe_degree, Number>::GLBackgroundCoefficients() {}

  template <int dim, int fe_degree, typename Number>
  void GLBackgroundCoefficients<dim, fe_degree, Number>::reinit(
      const Portable::MatrixFree<dim, Number> &mf_data)
  {
    // &mf_data is a pointer! 
    const unsigned int n_owned_cells =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
          &mf_data.get_dof_handler(0).get_triangulation())->n_locally_owned_active_cells();

    const unsigned int n_values = n_owned_cells * n_q_points * n_components;

    // distributed vector doesn't have API to only take one unsigned int
    // ??
    u0_coefficients.reinit(n_values);
    v0_coefficients.reinit(n_values);
  } // reinit() ends here

  template <int dim, int fe_degree, typename Number>
  void GLBackgroundCoefficients<dim, fe_degree, Number>::update(
      const Portable::MatrixFree<dim, Number> &mf_data,
      const BlockVectorType &background)
  {
    LocalGLBackgroundCoefficientOperator<dim, fe_degree, Number>
      background_operator(u0_coefficients.get_values(), v0_coefficients.get_values());

    /* mf_data.cell_loop() receives a dummy destination vector background.
     * This is because the Portable MatrixFree::cell_loop interface requires one.
     * But LocalGLBackgroundCoefficientOperator does NOT write to dst.
     * The actual output is written directly into u0_coefficients and v0_coefficients.
     *
     * here the background itself is used as the dummy destination object.
     */    
    mf_data.cell_loop(background_operator, background, background);
  } // upate() 

  template <int dim, int fe_degree, typename Number>
  Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_u0_values()
  { return u0_coefficients.get_values(); }

  template <int dim, int fe_degree, typename Number>
  const Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_u0_values() const
  { return u0_coefficients.get_values(); }

  template <int dim, int fe_degree, typename Number>
  Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_v0_values()
  { return v0_coefficients.get_values(); }

  template <int dim, int fe_degree, typename Number>
  const Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_v0_values() const
  { return v0_coefficients.get_values(); }

} // namespace VerHem

#endif
