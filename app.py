from logging.config import dictConfig
from google.cloud import logging
from flask import Flask

logging_client = logging.Client()
logging_handler = logging_client.setup_logging()
dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            },
            "gcp_logging": {
                "class": "google.cloud.logging.handlers.CloudLoggingHandler",
                "client": logging_client,
            },
        },
        "root": {"level": "INFO", "handlers": ["wsgi", "gcp_logging"]},
    }
)

app = Flask(__name__)