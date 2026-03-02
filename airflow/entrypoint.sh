#!/bin/bash
set -e

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db init

# Create admin user with admin/admin credentials
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start webserver in background without --daemon so logs go to stdout
# and there are no PID file conflicts on container restart
echo "Starting Airflow webserver..."
airflow webserver --port 8080 &

# Start scheduler as the main process (exec replaces the shell so Docker
# signals are delivered directly to the scheduler)
echo "Starting Airflow scheduler..."
exec airflow scheduler
