from prometheus_client import start_http_server, Counter
import time

REQUEST_COUNT = Counter("my_requests_total", "Total requests")

start_http_server(8000)

while True:
    REQUEST_COUNT.inc()
    time.sleep(2)