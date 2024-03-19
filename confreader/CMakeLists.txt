# confreader/CMakeLists.txt

include_directories(${CMAKE_SOURCE_DIR}/confreader/inc)

add_library(conf STATIC
# main source codes of confreader class   
  ${CMAKE_SOURCE_DIR}/confreader/src/confreader.cc
  )

deal_ii_setup_target(conf)
