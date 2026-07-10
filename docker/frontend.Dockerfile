FROM nginxinc/nginx-unprivileged:1.27-alpine

ARG BUILD_ID=dev
LABEL org.opencontainers.image.title="Real2Sim POC frontend" \
      org.opencontainers.image.version="${BUILD_ID}"
COPY deploy/poc/nginx.conf /etc/nginx/nginx.conf
COPY src/real2sim_platform/static/index.html /usr/share/nginx/html/index.html
COPY src/real2sim_platform/static /usr/share/nginx/html/static
USER 101:101
EXPOSE 8080 8443
