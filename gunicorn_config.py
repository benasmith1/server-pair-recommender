import os

port = int(os.environ.get("PORT", 8080))

bind = "0.0.0.0:" + port
workers = 2
