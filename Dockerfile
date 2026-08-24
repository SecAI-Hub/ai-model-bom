FROM docker.io/library/golang:1.27.0-alpine@sha256:4c9fe60190a2a3350ddc51de80d0224b8a6698d12bdfc999fee45ea9d6c46dbc AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ai-model-bom .

FROM docker.io/library/alpine:3.23@sha256:fd791d74b68913cbb027c6546007b3f0d3bc45125f797758156952bc2d6daf40
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
