#!/bin/bash

## getopt:
#   https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
##

# --test returns 4 for the "enhanced" getopt.  check.
getopt --test > /dev/null
if [[ $? -ne 4 ]]; then
    echo "I’m sorry, `getopt --test` failed in this environment."
    return 1
fi

OPTIONS=d
LONG_OPTIONS=dumpdb

# -temporarily store output to be able to check for errors
# -activate advanced mode getopt quoting e.g. via “--options”
# -pass arguments only via   -- "$@"   to separate them correctly
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONG_OPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    # e.g. $? == 1
    #  then getopt has complained about wrong arguments to stdout
    return 2
fi

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -d|--dumpdb)
            dump_db=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# handle non-option arguments
echo "dump DB: $dump_db"

if [[ $dump_db -eq 1 ]]; then
    export_name="_export_$(date +"%Y%m%d_%H%M_%S").bck"
    docker exec -u postgres peaceful_saha bash -c \
        "pg_dump --format=c -v --file=/var/lib/postgresql/data/$export_name pcsml"
fi

echo "copying backup to tmp dir"
remote_pgsql_data_dir="/mnt/dkr-data-01/pcsml_ppsimport_pgsql_001/_data"
chown root:pcsdev /mnt/dkr-data-01
mkdir -p /mnt/dkr-data-01/tmp
bck_file=$(ls $remote_pgsql_data_dir/_export*.bck | sort | tail -1)
if [ -z $bck_file ]; then
    >&2 echo "ERR: no db backup file found: $bck_file"
    exit 0
fi

echo "found bck_file: $bck_file"
echo "moving out of docker volume to temp dir"
dest_dir=/mnt/dkr-data-01/tmp/$(basename $bck_file)
echo "moving to: $dest_dir"
mv $bck_file $dest_dir
# return dest path
echo $dest_dir

# sudo mv /mnt/dkr-data-01/pcsml_ppsimport_pgsql_001/_data/$export_name /mnt/dkr-data-01/tmp
# sudo chown root:pcsdev $export_name;'

# local docker instance
# dkr_setup_ubu1604_mbp