FROM ghcr.io/advplyr/audiobookshelf@sha256:6432d1dc58951b0280b29144ffc69834a8ae02786a2126f6b7c9bd453801c78b AS upstream
FROM node:24.21.0-alpine3.23@sha256:9ec4a2e289874ed0d722e1772ec2de45d2801541db8612f3638b26f128c69ac2

# upgrade: the pinned Node base predates Alpine security fixes (OpenSSL, expat, …).
RUN apk upgrade --no-cache && apk add --no-cache ffmpeg tini tzdata
WORKDIR /app
COPY --from=upstream /app /app
COPY --from=upstream /usr/local/lib/nusqlite3 /usr/local/lib/nusqlite3
COPY package.json package-lock.json /app/
COPY verify.cjs /app/verify-hardening.cjs
ENV PORT=80 NODE_ENV=production CONFIG_PATH=/config METADATA_PATH=/metadata SOURCE=docker \
    NUSQLITE3_DIR=/usr/local/lib/nusqlite3 \
    NUSQLITE3_PATH=/usr/local/lib/nusqlite3/libnusqlite3.so
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund \
    && npm ls --omit=dev --all \
    && npm cache clean --force

# Keep the locked sqlite3 package; only supply upstream's native N-API binary.
COPY --from=upstream /app/node_modules/sqlite3/build/Release/node_sqlite3.node /app/node_modules/sqlite3/build/Release/node_sqlite3.node
RUN test "$(node -p 'require("./package.json").version')" = "2.37.0" \
    && node /app/verify-hardening.cjs \
    && ffmpeg -version >/dev/null \
    && tini --version

EXPOSE 80
ENTRYPOINT ["tini", "--"]
CMD ["node", "dist-server/index.js"]
