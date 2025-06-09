# Gunicorn configuration for Render deployment
import multiprocessing

# Server socket
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = 1  # Use only 1 worker to save memory
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increase timeout to 2 minutes
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "waste-detection-app"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for Render)
keyfile = None
certfile = None

# Memory optimization
preload_app = True  # Preload the app to share memory between workers

# Worker timeout and memory management
worker_tmp_dir = "/dev/shm"  # Use RAM for temporary files
worker_exit_on_app_exit = True

# Graceful timeout
graceful_timeout = 30

# Memory limits
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190 