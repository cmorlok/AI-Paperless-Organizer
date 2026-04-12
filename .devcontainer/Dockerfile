FROM python:3.14

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

# Install base dependencies + poppler (PDF processing) + build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    unzip \
    zsh \
    sqlite3 \
    sudo \
    build-essential \
    ca-certificates \
    gnupg \
    gcc \
    poppler-utils \
    psmisc \
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libdbus-1-3 \
    libcups2 \
    libxcb1 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libcairo2 \
    libpango-1.0-0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install rclone (cloud sync for cloud import feature)
RUN curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip \
    && unzip -q /tmp/rclone.zip -d /tmp/rclone \
    && mv /tmp/rclone/rclone-*/rclone /usr/bin/rclone \
    && chmod +x /usr/bin/rclone \
    && rm -rf /tmp/rclone /tmp/rclone.zip

# Install Node.js 20 from NodeSource
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Fix for rootless podman - update login.defs
RUN sed -i 's/^UID_MIN.*/UID_MIN 500/' /etc/login.defs && \
    sed -i 's/^GID_MIN.*/GID_MIN 500/' /etc/login.defs

# Create user
RUN groupadd --gid $USER_GID $USERNAME || groupmod -n $USERNAME $(getent group $USER_GID | cut -d: -f1) \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/zsh $USERNAME || usermod -l $USERNAME -d /home/$USERNAME -m $(getent passwd $USER_UID | cut -d: -f1) \
    && mkdir -p /home/$USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Create virtual environment and install dev tools
# python:3.14-trixie already provides Python 3.14 at /usr/local/bin/python3
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV \
    && $VIRTUAL_ENV/bin/pip install --upgrade pip setuptools wheel \
    && $VIRTUAL_ENV/bin/pip install black mypy \
    && chown -R $USERNAME:$USER_GID $VIRTUAL_ENV

# Add venv to PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Global npm installs
RUN npm install -g typescript typescript-language-server typescript-formatter pyright playwright

USER $USERNAME

ENV SHELL=/usr/bin/zsh

# Add devcontainer bin directory to PATH
ENV PATH="/app/.devcontainer/bin:$PATH"

WORKDIR /app
