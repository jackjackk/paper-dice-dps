#!/bin/bash

NAME="$1";
BNAME="make_$(basename ${NAME})";
NPROCS="$2";
PMEM="$3";
if [ -z "$PMEM" ]; then
    PMEM=2;
fi;
NNODES="$4";
if [ -z "$NNODES" ]; then
    NNODES=1;
fi;
NPARALL=$((NNODES*NPROCS));
QUEUE="$5";
if [ -z "$QUEUE" ]; then
    QUEUE=open;
fi;
PRECMD="$6";
MAKEARGS="${@:7}"

tee ${BNAME}.pbs  <<EOF
#!/bin/bash
#PBS -A ${QUEUE}
#PBS -l nodes=${NNODES}:ppn=${NPROCS}
#PBS -l walltime=48:00:00
#PBS -l pmem=${PMEM}gb
#PBS -j oe
#PBS -o ${BNAME}.out
#PBS -m ae
#PBS -M giacomo.marangoni@polimi.it

cd \$PBS_O_WORKDIR
pwd
sleep 0.1

${PRECMD}

/usr/bin/time -v make -j ${NPARALL} ${NAME} ${MAKEARGS}
EOF

qsub ${BNAME}.pbs

