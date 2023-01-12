##
#  CMake script for the complex valued scalar GL equation;
#  author: Quang. Zhang (timohyva@github)
#  ----------------------------------------------------
#  QUEST-DMC project, University of Sussex;
#  Helsinki Institute of Physics;
#  11. Tammikuu. 2023
##

# name of the project and target:
SET(TARGET "complexGL")

SET(TARGET_SRC
  ${TARGET}.cc
  )

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.3.3
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
