# matep/CMakeLists.txt

include_directories(${CMAKE_SOURCE_DIR}/matep/inc)

add_library(conf STATIC
# main source codes of confreader class   
  ${CMAKE_SOURCE_DIR}/matep/src/matep.cc
  )

deal_ii_setup_target(matep)
