# femgl/CMakeLists.txt

include_directories(
  ${CMAKE_SOURCE_DIR}/femgl/inc
  ${CMAKE_SOURCE_DIR}/confreader/inc
  ${CMAKE_SOURCE_DIR}/matep/inc  
  )

add_library(femgl STATIC
# main source codes of femgl class   
  ${CMAKE_SOURCE_DIR}/femgl/src/femgl.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid2.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_xz-normal_AdGR.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_cube-z-normal_AdGR.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_retangle-z-normal_AdGR.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/setup_fully_random.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/setup_uniform_A-phase.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/assemble.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/residual.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/solve.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/iteration.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/refine.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/io.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/s_vector2matrix.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/phi_vector2matrix.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/run.cc
# Matrices calculation functions for assemble_system(): cell-matrix-LHS  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_alpha.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta1.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta2.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta3.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta4.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_beta5.cc    
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_K1.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_face_lhs_K1.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_mat_lhs_K2K3.cc
# Vectors calculation functions for assemble_system(): cell-vector-RHS    
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_alpha.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_beta1.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_beta2.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_beta3.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_beta4.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_beta5.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_K1.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_face_rhs_K1.cc  
  ${CMAKE_SOURCE_DIR}/femgl/src/cell_mat_vec/cell_vec_rhs_K2K3.cc
  )

deal_ii_setup_target(femgl)
target_link_libraries(femgl conf matep)
  
