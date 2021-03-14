ARG IMAGE=balenalib/raspberrypi3-alpine-python
FROM ${IMAGE}

# Environment vars we can configure against
# But these are optional, so we won't define them now
#ENV HA_URL http://hass:8123
#ENV HA_KEY secret_key
#ENV DASH_URL http://hass:5050
#ENV EXTRA_CMD -D DEBUG

# API Port
EXPOSE 5050

# Mountpoints for configuration & certificates
VOLUME /conf
VOLUME /certs

# Copy appdaemon into image
WORKDIR /usr/src/app
COPY . .

# Install timezone data
RUN apk add tzdata

# Additional dependencies - line 2
RUN apk add --no-cache gcc g++ libffi-dev musl-dev  \
    libffi-dev libressl-dev python3-dev py-pip\
    && pip install --no-cache-dir .

# Install additional packages
RUN apk add --no-cache curl

# Start script
RUN chmod +x /usr/src/app/dockerStart.sh
ENTRYPOINT ["./dockerStart.sh"]
