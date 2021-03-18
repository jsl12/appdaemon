FROM python:latest

# Environment vars we can configure against
# But these are optional, so we won't define them now
#ENV HA_URL http://hass:8123
#ENV HA_KEY secret_key
#ENV DASH_URL http://hass:5050
#ENV EXTRA_CMD -D DEBUG

# API Port
EXPOSE 5050
EXPOSE 80
EXPOSE 443

# Mountpoints for configuration & certificates
VOLUME /conf
VOLUME /certs

# Copy appdaemon into image
WORKDIR /usr/src/app
COPY . .

# # Install timezone data
# RUN install_packages tzdata
#
# # Additional dependencies - line 2
# RUN install_packages --no-cache gcc g++ libffi-dev musl-dev  \
#     libffi-dev libressl-dev python3-dev py-pip\
#     && pip install --no-cache-dir .
#
# # Install additional packages
# RUN install_packages --no-cache curl
#
# # Start script
# RUN chmod +x /usr/src/app/dockerStart.sh
# CMD ["bash", "./dockerStart.sh"]
# CMD ["cat", "/etc/os-release"]
