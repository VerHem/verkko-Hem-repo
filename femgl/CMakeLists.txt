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
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_alpha.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta1.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta2.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_K1.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_K2K3.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_alpha.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_beta2.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_K1.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_K2K3.cc
  )

deal_ii_setup_target(femgl)
