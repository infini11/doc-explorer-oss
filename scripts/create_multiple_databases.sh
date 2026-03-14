#! /bin/bash

set -e # exit when error occurs
set -u # exit when using undeclared variables

function create_database() {
    local db_name=$1
    echo "Creating database: $db_name"
	psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $db_name;
EOSQL
}

if [ -n ${POSTGRES_MULTIPLE_DATABASES} ]; then
    echo "Multiple databases creation requested: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_database $db
    done
else
    echo "No multiple databases requested. Skipping creation."
fi