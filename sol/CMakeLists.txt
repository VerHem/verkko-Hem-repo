include_directories(
  ${CMAKE_SOURCE_DIR}/femgl/inc
  ${CMAKE_SOURCE_DIR}/confreader/inc
  ${CMAKE_SOURCE_DIR}/matep/inc    
  )

add_executable(VerHem
               ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
deal_ii_setup_target(VerHem)

target_link_libraries(VerHem femgl conf matep)
