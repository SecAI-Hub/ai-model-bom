FROM docker.io/library/golang:1.27.3-alpine3.24@sha256:b8c9b82affbbb82a020bbcf998dad2ba9516944ac7e883524642a481a9505e9c AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ai-model-bom .

FROM docker.io/library/alpine:3.24.1@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b
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
