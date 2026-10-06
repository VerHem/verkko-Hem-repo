#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

// #include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/tools.h>

//#include <deal.II/multigrid/mg_coarse.h>
//#include <deal.II/multigrid/mg_matrix.h>
//#include <deal.II/multigrid/mg_smoother.h>
//#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
//#include <deal.II/multigrid/mg_transfer_matrix_free.h>
//#include <deal.II/multigrid/multigrid.h>
//#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include "roctracer/roctx.h"

#include "LinearGLProblem.h"
// #include "LinearGLOperator.h"
// #include "LocalLinearGLOperator.h"
#include "initial_UV_ConfFunction/bgSolution_U.h"
#include "initial_UV_ConfFunction/bgSolution_V.h"
#include "LinearGL_RightHandSide/LinearGLRHSCellOperator.h"
// #include "LaplaceDiagonalCellOperatorQuad.h"
// #include "preconditioner/BlockDiagonalJacobiPreconditioner.h"

namespace VerHem
{
  using namespace dealii;

  template <int dim, int fe_degree, typename Number>
  void LinearGLProblem<dim, fe_degree, Number>::setup_dofs()
  {
    roctxRangePush("ROCTX-RANGE:LinearGLProblem::setup_dofs()");
    DoFHandler_U.distribute_dofs(fe_U);
    DoFHandler_V.distribute_dofs(fe_V);

    /* U dofs */
    const IndexSet &owned_set_U   = DoFHandler_U.locally_owned_dofs();
    const IndexSet relevant_set_U = DoFTools::extract_locally_relevant_dofs(DoFHandler_U);
    constraints_U.reinit(owned_set_U, relevant_set_U);
    
    DoFTools::make_hanging_node_constraints(DoFHandler_U, constraints_U);
    VectorTools::interpolate_boundary_values(
      DoFHandler_U, 0, Functions::ZeroFunction<dim, Number>(dim), constraints_U);
    constraints_U.close();

    /* V dofs */
    const IndexSet &owned_set_V   = DoFHandler_V.locally_owned_dofs();
    const IndexSet relevant_set_V = DoFTools::extract_locally_relevant_dofs(DoFHandler_V);
    constraints_V.reinit(owned_set_V, relevant_set_V);
    
    DoFTools::make_hanging_node_constraints(DoFHandler_V, constraints_V);
    VectorTools::interpolate_boundary_values(
      DoFHandler_V, 0, Functions::ZeroFunction<dim, Number>(dim), constraints_V);
    constraints_V.close();

    /* ------------------------------------------------
     * container of DoFHandlers of U and V
     * ------------------------------------------------
     */
    std::vector<const DoFHandler<dim> *> DoFHandlers_U_V
      = {&DoFHandler_U, &DoFHandler_V};

    /* ------------------------------------------------
     * container of AffineConstraints of U and V
     * ------------------------------------------------
     */     
    std::vector<const AffineConstraints<Number> *> constraints_U_V
      = {&constraints_U, &constraints_V};

    /* ------------------------------------------------
     * allocate ssmart pointer for Portable::MatrixFree<..>
     * ------------------------------------------------
     */
    mf_data_ptr = std::make_shared<Portable::MatrixFree<dim, Number>>();

    const QGauss<1> quad(fe_degree + 1);
    // const QGauss<1> quad(degree_p + 2);
    typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values
      | update_gradients | update_JxW_values | update_quadrature_points;    
    // additional_data.mapping_update_flags = update_values | update_gradients;
    /*------------------------------------------------------------
     * using multi-DoFHandler pattern as step-104 for block structure
     * DoFHandler 0 -> U, DoFHandler 1 -> V, Here two Dofhandlers are provided
     * when Portable::Matrixfree is initialized.
     * ------------------------------------------------------------
     */    
    mf_data_ptr->reinit(mapping,
                        DoFHandlers_U_V, constraints_U_V,
                        quad, additional_data);

    /* ------------------------------------------------------------
     * create the background_solution on the host and move to device:
     * ------------------------------------------------------------
     */
    //roctxMark("Starting bgSol Construction");
    roctxRangePush("ROCTX-RANGE:Starting bgSol Construct");
    
    LinearAlgebra::distributed::BlockVector<Number, MemorySpace::Host> bgSolution_host;
    mf_data_ptr->initialize_dof_vector(bgSolution_host);

    VectorTools::interpolate(mapping, DoFHandler_U,
			     bgSolution_U<dim, Number>(0.0, 0.2, 2.0),
                             bgSolution_host.block(0));
    
    VectorTools::interpolate(mapping, DoFHandler_V,
                             bgSolution_V<dim, Number>(0.0, 0.2, 2.0),
                             bgSolution_host.block(1));

    // prepare moving host vector to device vector.
    roctxRangePush("ROCTX-RANGE:initialize_dof_vector(bg_solution)");
    mf_data_ptr->initialize_dof_vector(bg_solution);
    roctxRangePop();
    
    roctxRangePush("ROCTX-RANGE:import_elements bg_solution block0");    
    bg_solution.block(0).import_elements(bgSolution_host.block(0), VectorOperation::insert);
    roctxRangePop();

    roctxRangePush("ROCTX-RANGE:import_elements bg_solution block1");        
    bg_solution.block(1).import_elements(bgSolution_host.block(1), VectorOperation::insert);
    roctxRangePop();

    // apply Dirichlet BC constriant
    roctxRangePush("ROCTX-RANGE:setup_dofs MatrixFree->set_constr_values bg_solution");    
    mf_data_ptr->set_constrained_values(Number(0.0), bg_solution.block(0), 0);
    mf_data_ptr->set_constrained_values(Number(0.0), bg_solution.block(1), 1);    
    roctxRangePop();    
    
    //roctxMark("ending bgSol Construction");
    roctxRangePop();
    /* ------------------------------------------------------------
     * background_solution on the host and  device is done
     * ------------------------------------------------------------
     */ 
         
    {
      /* ------------------------------------------------------------
       * using the device vector bg_solution to compute rhs.
       * ------------------------------------------------------------
       */
       mf_data_ptr->initialize_dof_vector(rhs);
       rhs = 0.0; // do I need this? yes!

       const Number K1    = 0.42072;
       const Number alpha = -0.4;
       const Number beta2 = 0.1;
       
       LinearGLRHSCellOperator<dim, fe_degree, Number> rhs_operator(K1, alpha, beta2);

      /*  bg_solution.block(0) = u^0 ,bg_solution.block(1) = v^0 */
       roctxRangePush("ROCTX-RANGE:*mf_data_ptr::cell_loop() ");
       mf_data_ptr->cell_loop(rhs_operator, bg_solution /*src*/, rhs /*dst*/);
       roctxRangePop();

      /*
       * Newton updates satisfies homogeneous Dirichlet conditions.
       * Therefore constrained RHS entries must be zero.
       */
       mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(0), 0);
       mf_data_ptr->set_constrained_values(Number(0.0), rhs.block(1), 1);
      
    } // rhs setting block ends here
    roctxRangePop();
  } // LinearGLProblem<...>::setup_dofs() ends here

} // namespace VerHem ends here

template class VerHem::LinearGLProblem<3, 1, float>;
template class VerHem::LinearGLProblem<3, 1, double>;
