echo "Starting YACE..."
sudo docker run -d -p 5000:5000 -v $HOME/.aws/credentials:/exporter/.aws/credentials \
-v /mnt/tig-vol/volumes/tig/exporters/cw-yace/config.yml:/tmp/config.yml \
--name yace -it prom/yet-another-cloudwatch-exporter:v0.28.0-alpha

# Start Prometheus
echo "Starting Prometheus..."
sudo systemctl start prometheus

# Start Alertmanager
echo "Starting Alertmanager..."
cd /etc/prometheus/alertmanager-0.27.0.linux-amd64
./alertmanager --config.file=alertmanager.yml &

# Start redis
echo "Starting Redis..."
redis-server --daemonize yes

echo "All services started."
