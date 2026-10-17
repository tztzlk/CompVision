# ---------------------------------------------------------------------------
# Stage 1: build the Vite SPA
# ---------------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first so the layer is cached while sources change.
# package-lock.json is optional (glob): use `npm ci` when it exists for a
# reproducible install, otherwise fall back to `npm install`.
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Base URL the frontend uses for API calls. Default "/api" is a relative path
# that nginx (stage 2) proxies to the backend service, so the browser never
# needs to know the backend host. Override at build time:
#   docker build --build-arg VITE_API_BASE_URL=https://api.example.com .
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY . .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: serve static files + reverse-proxy /api to the backend
# No node/npm in the final image - only nginx and the built /dist.
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
# Overwrites the stock default.conf shipped with the image.
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# wget is provided by busybox in alpine; /healthz is served by nginx itself
# (does not depend on the backend being up).
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/healthz >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
