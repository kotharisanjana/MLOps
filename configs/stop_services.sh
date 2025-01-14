# Stop YACE
echo "Stopping YACE..."
sudo docker stop yet-cloudwatch
sudo docker rm yet-cloudwatch

# Stop Prometheus
echo "Stopping Prometheus..."
sudo systemctl stop prometheus

# Stop Alertmanager
echo "Stopping Alertmanager..."
pkill -f alertmanager

# Stop Redis
echo "Stopping Redis"
pkill -f redis-server

echo "All services stopped."
