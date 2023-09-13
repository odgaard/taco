#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

mkdir -p data/suitesparse/
cd data/suitesparse/

wget https://sparse.tamu.edu/MM/Hamm/scircuit.tar.gz
wget https://sparse.tamu.edu/MM/vanHeukelum/cage12.tar.gz
wget https://sparse.tamu.edu/MM/Raju/laminar_duct3D.tar.gz
wget https://sparse.tamu.edu/MM/Oberwolfach/filter3D.tar.gz
wget https://sparse.tamu.edu/MM/TKK/smt.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/email-Enron.tar.gz
wget https://sparse.tamu.edu/MM/Goodwin/Goodwin_040.tar.gz
wget https://sparse.tamu.edu/MM/TAMU_SmartGridCenter/ACTIVSg10K.tar.gz
