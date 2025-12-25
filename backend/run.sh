#!/bin/env bash

uv run granian \
  --host 0.0.0.0 \
  --port 9000 \
  --interface wsgi app:app \
  --workers 4 \
  --blocking-threads 4 \
  --uds app.sock
