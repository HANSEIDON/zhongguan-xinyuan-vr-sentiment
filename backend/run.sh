#!/bin/env bash

uv run granian \
  --host 0.0.0.0 \
  --port 9000 \
  --interface wsgi app:app \
  --workers 1 \
  --blocking-threads 1
