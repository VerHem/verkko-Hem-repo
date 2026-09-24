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
   * It is executed only when the Newton background is updated.
   * ----------------------------------------------------------------
   */
  template <int dim, int fe_degree, typename Number>
  class LocalGLBackgroundCoefficientOperator
  {
  public:

    static constexpr unsigned int n_components = 9;
    static constexpr unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

    LocalGLBackgroundCoefficientOperator(Number *u0_BG_ptr, Number *v0_BG_ptr)
      : u0_container_ptr(u0_BG_ptr)
      , v0_container_ptr(v0_BG_ptr)
    {}

    DEAL_II_HOST_DEVICE
    void operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
                    const Portable::DeviceBlockVector<Number> &src,
                    Portable::DeviceBlockVector<Number> &dst) const;
  private:

    Number *u0_container_ptr;
    Number *v0_container_ptr;
    
  }; // LocalGLBackgroundCoefficientOperator declaretion ends here

  template <int dim, int fe_degree, typename Number>
  DEAL_II_HOST_DEVICE
  void LocalGLBackgroundCoefficientOperator<dim, fe_degree, Number>::operator()    
    (const typename Portable::MatrixFree<dim, Number>::Data *data,
     const Portable::DeviceBlockVector<Number> &src,
     Portable::DeviceBlockVector<Number> & /*dst*/) const
  {
    // U background.
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number>
      u0_eval(data, 0);
    // V background.
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number>
      v0_eval(data, 1);

    u0_eval.read_dof_values(src.block(0));
    v0_eval.read_dof_values(src.block(1));

    u0_eval.evaluate(EvaluationFlags::values);
    v0_eval.evaluate(EvaluationFlags::values);

    const unsigned int cell = data->cell_index;

    data->for_each_quad_point(
      [&](const int q_point)
      {
        // const unsigned int pos = data->local_q_point_id(cell, q_point);
        const unsigned int pos = data->local_q_point_id(cell, n_q_points, q_point);
        
        const auto u0 = u0_eval.get_value(q_point);
        const auto v0 = v0_eval.get_value(q_point);

        for (unsigned int c = 0; c < n_components; ++c)
        {
          u0_container_ptr[
            pos * n_components + c] = u0[c];

          v0_container_ptr[
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

    // Number *get_u0_values();
    const Number *get_u0_values() const;

    // Number *get_v0_values();
    const Number *get_v0_values() const;


  private:
    VectorType u0_BGSol;
    VectorType v0_BGSol;
    
  }; // GLBackgroundCoefficients declareation ends here

  template <int dim, int fe_degree, typename Number>
  void GLBackgroundCoefficients<dim, fe_degree, Number>::reinit(
      const std::shared_ptr<Portable::MatrixFree<dim, Number>> &mf_data_ptr)
  {
    // LinearGLOperator's MF_Data_Eigine is smart pointer.
    // GLBackgroundCoefficients::reinit() should has
    // smart pointer as parameter as well.
    // source code of MatriFree's get_dof_handler() does receives dof_handler_index
    // and return DoFHandler<dim>.
    const unsigned int n_owned_cells =
      dynamic_cast<const parallel::TriangulationBase<dim> *>
      (
       // take pointer of triangulation, then run-time cast it to
       // pointer of parallel triangulation
       &((mf_data_ptr->get_dof_handler(0)).get_triangulation())
      )->n_locally_owned_active_cells();

    const unsigned int n_values = n_owned_cells * n_q_points * n_components;

    // doc says reinit() sets the global size of the vector
    // to n_values without any actual parallel distribution. 
    u0_BGSol.reinit(n_values);
    v0_BGSol.reinit(n_values);
  } // reinit() ends here

  template <int dim, int fe_degree, typename Number>
  void GLBackgroundCoefficients<dim, fe_degree, Number>::update(
      const std::shared_ptr<Portable::MatrixFree<dim, Number>> &mf_data_ptr,
      const BlockVectorType &background_UV_sol)
  {
    LocalGLBackgroundCoefficientOperator<dim, fe_degree, Number>
      background_operator(u0_BGSol.get_values(), v0_BGSol.get_values());

    /* mf_data_ptr->cell_loop() receives a dummy destination vector
     * because MatrixFree::cell_loop interface requires one.
     * LocalGLBackgroundCoefficientOperator does NOT write to this fake dst.
     * The actual output is written directly into u0_BGSol and v0_BGSol.
     *
     * here the background_UV_sol itself is used as the dummy destination object.
     */    
    mf_data_ptr->cell_loop(background_operator, background_UV_sol, background_UV_sol);
  } // upate() 


  /* ------------------------------------
   * doc of sitributed::Vector<> say get_values()
   * return the pointer to the underlying raw array
   * for Vector on given MemorySpaces.
   * ------------------------------------
   */
  template <int dim, int fe_degree, typename Number>
  const Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_u0_values() const
  { return u0_BGSol.get_values(); }

  // template <int dim, int fe_degree, typename Number>
  // Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_v0_values()
  // { return v0_coefficients.get_values(); }

  template <int dim, int fe_degree, typename Number>
  const Number *GLBackgroundCoefficients<dim, fe_degree, Number>::get_v0_values() const
  { return v0_BGSol.get_values(); }

} // namespace VerHem

#endif
