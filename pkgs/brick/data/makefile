temp: HadCRUT.4.6.0.0.annual_ns_avg.txt HadCRUT.4.6.0.0.annual_ns_avg_realisations temp_giss

forc: forcing_hindcast_giss.csv forcing_hindcast_urban.csv

temp_giss: GLB.Ts+dSST.csv

GLB.Ts+dSST.csv:
	curl 'https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.csv' | sed '1,2d;s/^\(18.*\)$$/\1 .08/;s/^\([12][90].*\)$$/\1 .05/;s/\./0./g' | tr \, ' '  > 'GLB.Ts+dSST.csv'

HadCRUT.4.6.0.0.annual_ns_avg.txt:
	curl https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.annual_ns_avg.txt -O

HadCRUT.4.6.0.0.annual_ns_avg_realisations:
	curl https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.annual_ns_avg_realisations.zip -O
	unzip HadCRUT.4.6.0.0.annual_ns_avg_realisations.zip -d HadCRUT.4.6.0.0.annual_ns_avg_realisations
	rm HadCRUT.4.6.0.0.annual_ns_avg_realisations.zip

forcing_hindcast_urban.csv:
	cp -v forcing_hindcast.csv forcing_hindcast_urban.csv

forcing_hindcast_giss.csv:
	curl https://data.giss.nasa.gov/modelforce/Fe_H11_1880-2011.txt -O
	sed '1,6d;s/ \+/,/g' Fe_H11_1880-2011.txt | sed 1i"year,ghg,o3,sh2o,refa,aie,bc,snow,stra,solar,land" | head -n -8 > forcing_hindcast_giss.csv
	rm -v Fe_H11_1880-2011.txt
	#echo '2012,3.05191,0.216,0.063,-1.44707,-1.01964,0.5986,0.202291,-0.117,0.236,-0.093' >> forcing_hindcast_giss.csv
	#echo '2013,3.08723,0.216,0.063,-1.46246,-1.0305,0.604973,0.204445,-0.123273,0.277,-0.093' >> forcing_hindcast_giss.csv
	#echo '2014,3.12255,0.216,0.063,-1.47785,-1.04136,0.611345,0.2066,-0.129545,0.259,-0.093' >> forcing_hindcast_giss.csv
	#echo '2015,3.15786,0.216,0.063,-1.49325,-1.05223,0.617718,0.208755,-0.135818,0.241,-0.093' >> forcing_hindcast_giss.csv

# OHC

ohc: IAP_OHC_estimate_update.txt TOA_OHC_errorbar_1940_2015_2.txt

TOA_OHC_errorbar_1940_2015_2.txt:
	curl http://159.226.119.60/cheng/images_files/TOA_OHC_errorbar_1940_2015_2.txt | tr \] \" | tr \[ \" > TOA_OHC_errorbar_1940_2015_2.txt

IAP_OHC_estimate_update.txt:
	curl http://159.226.119.60/cheng/images_files/IAP_OHC_estimate_update.txt | tr \] \" | tr \[ \" > IAP_OHC_estimate_update.txt
