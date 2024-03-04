#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH -J "cytotox"
#SBATCH -c 20
#SBATCH --mem 100g
#SBATCH --time 200:00:00
#SBATCH -o "%j__cytotox.out"
#SBATCH -e "%j__cytotox.err"

start=`date +%s`
echo $start

## loading module
module load python/3.9
module load R/4.2.2

export RUN_ENV="biowulf"

python /home/borrela2/CIS-419/sources/py/main.py

EOT
