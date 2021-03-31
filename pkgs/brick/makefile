include ../../makefile_common

### SETUP ###
# Number
NITER := 20000000
NCHAIN := 10
NTHIN := 1000

DATAPATH := data
OUTPATH := ../../output/brick
BRICKPATH := $(shell pwd)

install:
	cd calibration && $(R) BRICK_install_packages.R
	cd fortran && $(MAKE) -B all


### RUN ###
BRICK_OPTS := -z 1880 -Z 1900 -t 1880 -T 2011 -H 50 -f giss -F giss -O gour -u 4 -o ../$(OUTPATH)
MCMC := cd calibration && $(R) BRICK_calib_driver.R -n $(NITER) -N $(NCHAIN) $(BRICK_OPTS)
MCMC_START := $(MCMC) -d ../$(DATAPATH)/brick_mcmc_start.rds

CLIM_PRIORS := chylek paleosens cauchy

CALIB_TARGETS := $(addprefix calib-,$(CLIM_PRIORS))
CALIB_OUT_PRE := $(OUTPATH)/brick_mcmc_fgiss_TgissOgour_s
CALIB_OUT_POST := _t18802011_z18801900_o4_h50_n$(NITER).rds

DIAG_TARGETS := $(addprefix diag-,$(CLIM_PRIORS))
DIAG_OUT_PRE := $(CALIB_OUT_PRE)
DIAG_OUT_POST := $(CALIB_OUT_POST)_t$(NTHIN)_b5.nc

MAKE_ARGS := NITER=$(NITER) NCHAIN=$(NCHAIN) NTHIN=$(NTHIN)

.PHONY: calib-all $(CALIB_TARGETS) diag-all $(DIAG_TARGETS)


calib-all: 
	$(foreach TGT,$(CALIB_TARGETS),$(call LONGRUN,$(TGT),$(NCHAIN),64,1,$(MAKE_ARGS));)


$(CALIB_TARGETS): calib-%: $(CALIB_OUT_PRE)%$(CALIB_OUT_POST)


$(CALIB_OUT_PRE)%$(CALIB_OUT_POST):
	$(MCMC_START) -s $*


diag-all:
	$(foreach TGT,$(DIAG_TARGETS),$(call LONGRUN,$(TGT),1,64,1,$(MAKE_ARGS));)


$(DIAG_TARGETS): diag-%: $(DIAG_OUT_PRE)%$(DIAG_OUT_POST)


$(DIAG_OUT_PRE)%$(DIAG_OUT_POST): $(CALIB_OUT_PRE)%$(CALIB_OUT_POST)
	cd calibration && $(R) mcmc-diagnostics.R -r ../$< -t $(NTHIN) -b 5


