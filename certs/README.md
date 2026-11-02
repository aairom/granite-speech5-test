# certs/

This directory holds **local-only** self-signed TLS certificates used to serve
the app over HTTPS so the browser microphone API works on localhost.

> ⚠️  `cert.pem` and `key.pem` are excluded from git (see `.gitignore`).  
> They must be regenerated locally on each machine.

## Generate

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem \
  -out   certs/cert.pem \
  -days  3650 \
  -subj  "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

Or simply run `bash scripts/setup.sh` — it generates the certs automatically
if they are missing.
