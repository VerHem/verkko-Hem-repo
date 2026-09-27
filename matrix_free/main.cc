#include <fstream>

#include "LinearGLProblem.h"

int main(int argc, char *argv[])
{
  try
    {
      using namespace VerHem;

      //dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
      dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

      const unsigned int dim = 3;
      const unsigned int FE_degree = 1;
      
      LinearGLProblem<dim, FE_degree, float> LinearGL_problem;
      LinearGL_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
