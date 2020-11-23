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

script_path="./pcsdevml009_db_export.sh"
scp $script_path [devusername]@$_dns_pcsdevml009:~/

remote_cmd="sudo bash $(basename $script_path)"
if [[ $dump_db -eq 1 ]]; then
    remote_cmd="$remote_cmd -d"
fi

echo "executing remote command: $remote_cmd"
# temp stdout dup
exec 9>&1
result=$(ssh -t [devusername]@$_dns_pcsdevml009 $remote_cmd | tee /dev/fd/9)
result=$(echo "$result" | tail -n1 | sed 's/\s*$//' | sed 's/^\s*//')

echo "db backup remote file location: $result"

rsync -v --progress [devusername]@$_dns_pcsdevml009:$result ~/
rsync -v --progress ~/$(basename $result) dfaivre@ubu1604-mbp:~/

restore_script="ubu1604-mbp-db-restore.sh"
scp ./$restore_script dfaivre@ubu1604-mbp:~/
ssh -t dfaivre@ubu1604-mbp "bash ~/$restore_script $(basename $result)"

# delete db backup from remotes
ssh [devusername]@$_dns_pcsdevml009 "rm $result"