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


include_directories(
  ${CMAKE_SOURCE_DIR}/femgl/inc
  ${CMAKE_SOURCE_DIR}/confreader/inc
  ${CMAKE_SOURCE_DIR}/matep/inc    
  )

add_executable(VerHem
               ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
deal_ii_setup_target(VerHem)

target_link_libraries(VerHem femgl conf matep)
