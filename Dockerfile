FROM docker.io/library/golang:1.27.1-alpine3.24@sha256:cf6fca6641884b8433441b2b0652976f975e1d0fdd26d177eaaf8596087f3125 AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ai-model-bom .

FROM docker.io/library/alpine:3.24.2@sha256:294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6
RUN mkdir -p /etc/ai-model-bom /models && chown 65534:65534 /models
COPY policies/container-policy.yaml /etc/ai-model-bom/policy.yaml
COPY --from=build /out/ai-model-bom /usr/local/bin/ai-model-bom
ENV AIMBOM_CONFIG=/etc/ai-model-bom/policy.yaml
ENV ALLOW_REMOTE_BIND=true
USER 65534:65534
EXPOSE 8515
ENTRYPOINT ["ai-model-bom"]
CMD ["serve"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8515/health || exit 1
