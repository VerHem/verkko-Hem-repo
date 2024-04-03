include_directories(
  ${CMAKE_SOURCE_DIR}/femgl/inc
  ${CMAKE_SOURCE_DIR}/confreader/inc
  ${CMAKE_SOURCE_DIR}/matep/inc    
  )

add_executable(FEmGL
               ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
deal_ii_setup_target(FEmGL)

target_link_libraries(FEmGL femgl conf matep)
