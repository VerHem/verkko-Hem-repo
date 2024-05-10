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


cmake_minimum_required(VERSION 3.13.4)
#find_package(deal.II 9.3.0 REQUIRED)

set(DEAL_II_DIR "/projappl/project_2006155/libs/dealii-mpi-Trilinos-optimized")

FIND_PACKAGE(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

deal_ii_initialize_cached_variables()
project(FEmGL VERSION 0.1)

add_subdirectory(matep)
add_subdirectory(confreader)
add_subdirectory(femgl)
add_subdirectory(sol)


