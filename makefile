include makefile_common

include makefile_dicedps

up:
	$(ERSYNC) environment.yml requirements.txt makefile* misc pkgs $(REMOTE_PATH)/
	[ -d output/brick ] && $(ERSYNC) output/brick/*_n1000.csv $(REMOTE_PATH)/output/brick/

down:
	$(ERSYNC) $(REMOTE_PATH)'/output/brick/*_t1000_*nc' output/brick/
	$(ERSYNC) $(REMOTE_PATH)'/output/dicedps/*{rbf,time2}*_i1p{10,400}*{runtime,merged,rerun,metrics}.csv' output/dicedps/

xdown:
	$(ERSYNC) $(REMOTE_PATH)'/output/dicedps/*{time2,X}*runtime.csv' output/dicedps/

conda:
	@echo Run:
	@echo  conda env create -f environment.yml
	@echo 
	@echo Then activate with:
	@echo  conda activate dicedps

borg: pkgs/borg pkgs/borg/CMakeLists.txt pkgs/borg/build/lib/libborgms pkgs/borgpy/borgpy/borg.py

pkgs/borg:
	git clone git@bitbucket.org:dmh309/serial-borg-moea.git pkgs/borg && \
		cd pkgs/borg && \
		git update-index --refresh && \
		git am ../../misc/borg/borg4platypus.pat

pkgs/borg/CMakeLists.txt: misc/borg/CMakeLists.txt
	cp -v $< $@

pkgs/borg/build/lib/libborgms: pkgs/borg/borgms.c
	cd pkgs/borg && \
		rm -rfv build && \
		mkdir build && \
		cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		make

pkgs/borgpy/borgpy/borg.py: pkgs/borg/plugins/Python/borg.py
	cp -v $< $@

clean:
	find . -name "make_*.out" -type f -delete
	find . -name "make_*.pbs" -type f -delete

brick-%:
	cd pkgs/brick && $(MAKE) $*


forcings: pkgs/paradoeclim/paradoeclim/data/forcings.csv

pkgs/dicedps/dicedps/data/forcings.csv: pkgs/dicedps/dicedps/extrap_climate_forcings.py
	mkdir -p $$(dirname $@)
	$(PYTHON) -m dicedps.extrap_climate_forcings -o $@

pkgs/paradoeclim/paradoeclim/data/forcings.csv: pkgs/dicedps/dicedps/data/forcings.csv
	mkdir -p $$(dirname $@)
	cp -v $< $@

dicedps-%:
	cd pkgs/dicedps && $(MAKE) $*

venv:
	[ ! -d venv ] && $(PYTHON) -m venv venv || echo venv dir already exists
	source venv/bin/activate && pip install --ignore-installed -r requirements.txt


data:
	cp -v pkgs/brick/data/{HadCRUT.4.4.0.0.annual_ns_avg.txt,HadCRUT.4.6.0.0.annual_ns_avg.txt,forcing_hindcast_giss.csv} pkgs/paradoeclim/paradoeclim/data/


fup:
	$(ERSYNC) ./ $(REMOTE_PATH)/

#down:
#	$(RSYNC) --max-size=500m $(REMOTE_PATH)/data/ $(LOCAL_DATA_PATH)/

fdown:
	$(RSYNC) --max-size=500m $(REMOTE_PATH)/data/ $(LOCAL_DATA_PATH)/

%_last:
	$(RSYNC) "acid:dicedps/$*/*greg4{d,g,h}*{{merged,rerun}.csv,.metrics}" data/dicedps

rerun:
	$(RSYNC) "acid:dicedps/data/*rerun.csv" data/dicedps


# REMOTE
SCRATCH_PATH := /gpfs/scratch/gum184/dicedps-v2
OLD_DATA_PATH := /storage/home/gum184/group/dicedps

scratch:
	mkdir -p $(SCRATCH_PATH)
	rm -rfv scratch
	ln -sf $(SCRATCH_PATH) scratch
	ln -sf $$(pwd)/makefile_scratch $(SCRATCH_PATH)/makefile
	ln -sf $$(pwd)/misc/makefile_common $(SCRATCH_PATH)/makefile_common
	mkdir -p scratch/brick
	ln -sf $$(pwd)/pkgs/brick/makefile $(SCRATCH_PATH)/brick/
	ln -sf $$(pwd)/misc/makefile_common $(SCRATCH_PATH)/brick/makefile_common
	mkdir -p scratch/moea
	ln -sf $$(pwd)/makefile_moea $(SCRATCH_PATH)/moea/
	ln -sf $$(pwd)/misc/makefile_common $(SCRATCH_PATH)/moea/makefile_common

copy_brick_data_to_scratch:
	rsync -avP $(PROJECT_PATH)/data/brick/*20000000.rds scratch/brick/

copy_over_from_old:
	rsync -avP $(OLD_DATA_PATH)/*greg4d*runtime.csv scratch/

# OLD
#data:
#	$(RSYNC) "acid:$(BASENAME)/data/{*.nc,*4000000c4000000*csv}" sandbox/
#
#data0315:
#	$(RSYNC) "acid:$(BASENAME)/data/*greg4*.csv" sandbox/
#
#
#brick_nc:
#	PATT="^([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_*";for CLIM in ../brick/results/*f*n10000000_b5_t1000_n1.nc; do [[ $$(basename $$CLIM) =~ $$PATT ]] && cp -v $$CLIM dicedps/data/brick_$${BASH_REMATCH[3]}_$${BASH_REMATCH[4]//Ogour/}_$${BASH_REMATCH[5]//sinf/scauchy}_$${BASH_REMATCH[8]}.nc; done
#	cp -v ../brick/results/brick_mcmc_fgiss_Tgiss_sinf_t18802011_z18801900_o4_h150_n10000000_b5_t1000_n1.nc dicedps/data/brick_fgiss_tgiss_scauchy_o4.nc
#	rm -v dicedps/data/brick_*{o50,o100,t18802011,n10000000,uninf}*nc
#	rm -v dicedps/data/brick_fgiss_Tgiss_scauchy_o4.nc
#
#%_last:
#	$(RSYNC) "acid:$(BASENAME)/$*/*greg4{d,g,h}*{{merged,rerun}.csv,.metrics}" sandbox/
#
#scratch_rerun:
#	$(RSYNC) "acid:$(BASENAME)/scratch/u*rerun.csv" sandbox/
#
#scratch_con:
#	$(RSYNC) "acid:$(BASENAME)/scratch/*_c*{last,merged}.csv" sandbox/
#
