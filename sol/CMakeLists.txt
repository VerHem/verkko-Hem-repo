include_directories(
  ${CMAKE_SOURCE_DIR}/femgl/include
  )

add_executable(FEmGL
               ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
deal_ii_setup_target(FEmGL)

target_link_libraries(FEmGL femgl)
