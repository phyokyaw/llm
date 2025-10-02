# vLLM Server Systemd Service Setup

This setup provides multiple ways to run your vLLM server as a system service with auto-start capabilities.

## Files Created

### Service Files
- `/etc/systemd/system/vllm-server.service` - Systemd service file
- `/etc/init.d/vllm-server` - SysV init script (fallback)
- `/usr/local/bin/vllm-server-wrapper` - Wrapper script for systemd
- `/usr/local/bin/vllm-autostart` - Auto-start script for cron
- `/usr/local/bin/setup-vllm-service` - Setup script

## Quick Setup

Run the setup script to automatically configure the service:

```bash
sudo /usr/local/bin/setup-vllm-service
```

## Manual Setup

### For Systemd Systems (Ubuntu 16.04+, CentOS 7+, etc.)

1. **Enable the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable vllm-server.service
   ```

2. **Start the service:**
   ```bash
   sudo systemctl start vllm-server.service
   ```

3. **Check status:**
   ```bash
   sudo systemctl status vllm-server.service
   ```

### For SysV Init Systems

1. **Enable the service:**
   ```bash
   sudo update-rc.d vllm-server defaults
   ```

2. **Start the service:**
   ```bash
   sudo service vllm-server start
   ```

3. **Disable auto-start (if needed):**
   ```bash
   sudo update-rc.d vllm-server disable
   ```

### For Systems without Init (Containers, etc.)

Add to crontab for auto-start:
```bash
sudo crontab -e
# Add this line:
@reboot /usr/local/bin/vllm-autostart
```

## Service Management Commands

### Systemd Commands
```bash
sudo systemctl start vllm-server      # Start service
sudo systemctl stop vllm-server       # Stop service
sudo systemctl restart vllm-server    # Restart service
sudo systemctl status vllm-server     # Check status
sudo systemctl enable vllm-server    # Enable auto-start
sudo systemctl disable vllm-server   # Disable auto-start
```

### SysV Init Commands
```bash
sudo service vllm-server start        # Start service
sudo service vllm-server stop         # Stop service
sudo service vllm-server restart      # Restart service
sudo service vllm-server status       # Check status
```

### Direct Script Commands
```bash
sudo /etc/init.d/vllm-server start   # Start service
sudo /etc/init.d/vllm-server stop    # Stop service
sudo /etc/init.d/vllm-server restart # Restart service
sudo /etc/init.d/vllm-server status  # Check status
```

## Configuration

### Environment Variables
Configure your server settings in `/workspace/llm/.env`:

```bash
# Server 1
MODEL1=your-model-name
PORT1=8000
NAME1=llm1

# Server 2 (optional)
MODEL2=your-second-model
PORT2=8001
NAME2=llm2

# Additional settings
HF_TOKEN=your-huggingface-token
MODEL_CACHE=/data/.cache/huggingface
```

**Note:** Poetry is expected to be available at `/venv/main/bin/poetry`. The service scripts automatically add this path to the environment.

### Service Configuration
The service runs as `root` user with the following settings:
- Working directory: `/workspace/llm`
- Auto-restart on failure
- Logs to systemd journal and `/var/log/vllm/` (instead of `~/vllm_logs`)
- GPU access enabled
- Resource limits configured

## Logs and Monitoring

### View Logs
```bash
# Systemd logs
sudo journalctl -u vllm-server -f

# Service logs
tail -f /var/log/vllm/vllm-server.log

# Application logs (vLLM server logs)
tail -f /var/log/vllm/vllm_*.log

# Note: Logs are now stored in /var/log/vllm/ instead of ~/vllm_logs
```

### Check GPU Usage
```bash
nvidia-smi
```

## Troubleshooting

### Common Issues

1. **Service fails to start:**
   ```bash
   sudo journalctl -u vllm-server --no-pager
   ```

2. **Poetry not found:**
   - Ensure Poetry is installed at `/venv/main/bin/poetry`
   - The service automatically adds this path to the environment
   - If Poetry is in a different location, update the PATH in the service scripts

3. **Permission issues:**
   - Ensure `/workspace/llm` is readable by root
   - Check GPU permissions

4. **Port conflicts:**
   - Check if ports 8000/8001 are available
   - Modify PORT1/PORT2 in `.env` file

### Manual Testing
```bash
# Test the wrapper script
sudo /usr/local/bin/vllm-server-wrapper start

# Test Poetry environment
cd /workspace/llm
sudo poetry run python run_vllm.py
```

## Security Notes

- The service runs as root for GPU access
- File system access is restricted to necessary paths
- No new privileges are granted
- Private temporary directories are used

## Uninstalling

To remove the service:

```bash
# Disable systemd service
sudo systemctl disable vllm-server.service
sudo systemctl stop vllm-server.service

# Disable SysV init service
sudo update-rc.d vllm-server disable
sudo service vllm-server stop

# Remove files
sudo rm -f /etc/systemd/system/vllm-server.service
sudo rm -f /etc/init.d/vllm-server
sudo rm -f /usr/local/bin/vllm-server-wrapper
sudo rm -f /usr/local/bin/vllm-autostart
sudo rm -f /usr/local/bin/setup-vllm-service

# Reload systemd
sudo systemctl daemon-reload
```
