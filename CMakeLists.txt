cmake_minimum_required(VERSION 3.13.4)
#find_package(deal.II 9.3.0 REQUIRED)

set(DEAL_II_DIR "/projappl/project_2006155/libs/dealii-mpi-Trilinos-optimized")

FIND_PACKAGE(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

deal_ii_initialize_cached_variables()
project(FEmGL VERSION 0.1)

add_subdirectory(matep)
add_subdirectory(confreader)
add_subdirectory(femgl)
add_subdirectory(sol)


