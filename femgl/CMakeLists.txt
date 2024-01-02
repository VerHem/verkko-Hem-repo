# femgl/CMakeLists.txt

include_directories(${CMAKE_SOURCE_DIR}/femgl/include)
#include_directories(include)

add_library(femgl STATIC
  ${CMAKE_SOURCE_DIR}/femgl/src/femgl.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid2.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/setup.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/assemble.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/residual.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/solve.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/iteration.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/refine.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/io.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/s_vector2matrix.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/phi_vector2matrix.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/run.cc
  )

deal_ii_setup_target(femgl)
