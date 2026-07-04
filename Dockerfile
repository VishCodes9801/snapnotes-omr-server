# Audiveris ships as a self-contained Ubuntu .deb with a bundled Java
# runtime (releases stopped providing AppImages after the 5.5 line), so a
# plain Ubuntu 22.04 base is enough. Python runs the HTTP layer + music21
# post-processing.
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        wget \
        ca-certificates \
        libgomp1 \
        libglib2.0-0 \
        fontconfig \
        libfreetype6 \
        # Tesseract OCR + English language data — Audiveris uses it to
        # transcribe text annotations on the score (chord symbols,
        # lyrics, dynamics labels). Without it, those are silently
        # dropped and Audiveris logs "no OCR languages installed".
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Pinned Audiveris release, verified by SHA-256 before install — a defense
# against a tampered upstream release or CDN compromise. The default hash
# matches the pinned version; bump both together when upgrading (compute
# with: curl -sL <deb-url> | sha256sum). The .deb installs its own Java
# runtime under /opt/audiveris/lib/runtime; omr.py's JAVA_TOOL_OPTIONS
# (heap size, headless) are honored by that bundled JVM.
ARG AUDIVERIS_VERSION=5.6.2
ARG AUDIVERIS_SHA256=a3b7c1456d77ab078459a3d18746776616d61601d489d4749a65ee1ea192dd15
RUN wget -O /tmp/audiveris.deb \
        "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu22.04-x86_64.deb" \
    && echo "${AUDIVERIS_SHA256}  /tmp/audiveris.deb" | sha256sum -c - \
    && apt-get update \
    # The .deb's postinst registers desktop-menu + MIME entries via
    # xdg-desktop-menu/xdg-mime, which hard-fail headless ("No writable
    # system menu directory") and apt runs maintainer scripts with a
    # restricted PATH, so they can't be shimmed. None of that desktop
    # integration matters on a server: install the package's declared
    # dependencies ourselves, then unpack its payload directly with
    # dpkg-deb -x — same files on disk, no maintainer scripts.
    # (Dependency list mirrors the .deb's control file.)
    && apt-get install -y --no-install-recommends \
        libasound2 libbsd0 libc6 libmd0 libx11-6 libxau6 libxcb1 \
        libxdmcp6 libxext6 libxi6 libxrender1 libxtst6 zlib1g \
    && dpkg-deb -x /tmp/audiveris.deb / \
    && rm -rf /var/lib/apt/lists/* /tmp/audiveris.deb \
    # Fail the build loudly if the package layout ever changes.
    && test -x /opt/audiveris/bin/Audiveris \
    && ln -s /opt/audiveris/bin/Audiveris /usr/local/bin/audiveris

# Python deps live in a venv so we don't fight Ubuntu's externally-managed
# pip restriction.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY requirements.txt requirements.lock ./
# Install from the hash-pinned lockfile so build-time supply-chain attacks
# on transitives (a malicious release of starlette / pydantic / etc.) get
# rejected at install. Regenerate with:
#   pip-compile --generate-hashes --output-file=requirements.lock requirements.txt
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY app ./app

# Drop privileges. Audiveris/JVM, Tesseract, and uvicorn have no reason
# to run as root inside the container — a future Pillow/music21/JVM RCE
# starts from uid 1000 instead of uid 0.
RUN useradd --create-home --uid 1000 --shell /bin/bash snap \
    && chown -R snap:snap /app
USER snap

ENV PORT=8000
EXPOSE 8000

# First /extract may be slow as the JVM warms up; subsequent calls are
# faster on the same process / container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
