# Dockerfile for Next.js UI application

# 1. Builder stage: Install dependencies and build the application
FROM node:24-slim AS builder

# Set working directory
WORKDIR /app

# Copy package.json and lock file (if it exists)
COPY package.json package-lock.json* ./

# Install dependencies using the lock file for reproducible builds
RUN npm install # --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Disable Next.js telemetry during the build
ENV NEXT_TELEMETRY_DISABLED=1

# Build the Next.js application
# This will also trace output files if `output: 'standalone'` is set in next.config.mjs
RUN npm run build

# 2. Runner stage: Production image
FROM node:24-slim AS runner

WORKDIR /app

ENV NODE_ENV production
# Disable Next.js telemetry during runtime
ENV NEXT_TELEMETRY_DISABLED 1

# Create a non-root user for security (the 'node' user is available in node:alpine)
# and set appropriate permissions for copied files.
# Copy only the necessary artifacts from the builder stage for a lean image.
# This assumes you have `output: 'standalone'` in your next.config.mjs.
COPY --from=builder /app/public ./public
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static

# Expose port 3000 (default for Next.js)
EXPOSE 3000

# Set the Node.js user to run the application
USER node

# Command to run the Next.js application using the standalone server file
CMD ["node", "server.js"]