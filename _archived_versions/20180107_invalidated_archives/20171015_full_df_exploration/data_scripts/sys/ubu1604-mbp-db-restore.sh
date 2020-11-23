#!/usr/bin/env bash

mnt_point=$(docker volume inspect pcsml_ppsimport_pgsql_001 | jp -u [0].Mountpoint)
mv ~/$1 $mnt_point/$1

docker exec -u postgres pcsml-data-pps-grps-pgsql96 \
    bash -c "\
        psql postgres -c \
            \"UPDATE pg_database SET datallowconn = 'false' WHERE datname = 'pcsml'; \
             SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'pcsml';\"\
         && psql postgres -c \"DROP DATABASE pcsml\""

docker exec -u postgres pcsml-data-pps-grps-pgsql96 \
        bash -c "\
            createdb pcsml \
            &&pg_restore --verbose --dbname=pcsml /var/lib/postgresql/data/$1"

# remove the backup
rm $mnt_point/$1
