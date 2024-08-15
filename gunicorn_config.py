import os

# Number of worker processes
workers = int(os.environ.get('GUNICORN_PROCESSES', '2'))

# Number of threads per worker
threads = int(os.environ.get('GUNICORN_THREADS', '4'))

# Bind to address and port
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8080')

# Allow all IPs to forward requests
forwarded_allow_ips = '*'

# Ensure secure scheme headers are set for HTTPS
secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }

# Uncomment and set a timeout if needed
# timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))
