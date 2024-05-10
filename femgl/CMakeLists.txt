#------------------------------------------------------------------------------------------
#
# Copyright (C) 2023-present by Kuang. Zhang
#
# This library is free software; you can redistribute it and/or modify it under 
# the terms of the GNU Lesser General Public License as published by the Free Software Foundation; 
# either version 2.1 of the License, or (at your option) any later version.
#
# Permission is hereby granted to use or copy this program under the
# terms of the GNU LGPL, provided that the Copyright, this License 
# and the Availability of the original version is retained on all copies.
# User documentation of any code that uses this code or any modified
# version of this code must cite the Copyright, this License, the
# Availability note, and "Used by permission." 

# Permission to modify the code and to distribute modified code is granted, 
# provided the Copyright, this License, and the Availability note are retained,
# and a notice that the code was modified is included.

# The third party libraries which are used by this library are deal.II, Triinos and few others.
# All components involved third party supports obey their Copyrights, Licence and permissions. 
#  
# ------------------------------------------------------------------------------------------
#
# author: Quang. Zhang (timohyva@github), 
# Helsinki Institute of Physics, University of Helsinki;
# 27. Kesäkuu. 2023.
#
# ------------------------------------------------------------------------------------------


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
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_retangle-z-normal_AdGR.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_cube-z-normal_AdGR-xy-periodic.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_cube-xyz-periodic.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_cube-xyz-Homo-Neumann.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_retangle-xyz-periodic.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/makegrid_retangle-xyz-homogenous-Neumann.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/setup_fully_random.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/setup_uniform_A-phase.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/setup_fully_random-xy-periodic.cc
#  ${CMAKE_SOURCE_DIR}/femgl/src/setup_fully_random-xyz-periodic.cc
  ${CMAKE_SOURCE_DIR}/femgl/src/setup_uniform_BinA-bubble-like-configuration.cc
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
  
