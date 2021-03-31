##==============================================================================
## BRICK_install_packages.R
##
## Handy script to install or update the required R packages for calibrating
## and running the model.
##
## Updated by Giacomo Marangoni
## Original code: 2 April 2017
##
## Questions? Tony Wong (twong@psu.edu)
##==============================================================================
## Copyright 2016 Tony Wong, Alexander Bakker
## This file is part of BRICK (Building blocks for Relevant Ice and Climate
## Knowledge). BRICK is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## BRICK is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with BRICK.  If not, see <http://www.gnu.org/licenses/>.
##==============================================================================

install.if.needed <- function(pkg.list, force=FALSE, repos="https://cloud.r-project.org") {
  for (i in pkg.list) {
    if((!require(i, character.only=TRUE)) || (force)) {
      install.packages(i, dependencies=TRUE, repos=repos)
      if(!require(i, character.only=TRUE)) stop("Package ", i, " not found")
    }
  }
}

install.if.needed(c("devtools",
    "optparse",
    "adaptMCMC",
    "DEoptim",
    "doParallel",
    "fExtremes",
    "fields",
    "fMultivar",
    "foreach",
    "gplots",
    "lhs",
    "maps",
    "ncdf4",
    "plotrix",
    "pscl",
    "RColorBrewer",
    "sensitivity",
    "mcmcplots",
    "batchmeans",
    "coda",
    "sn"))
devtools::install_github("olafmersmann/truncnorm")

##==============================================================================
## End
##==============================================================================
