# SnapNotes OMR server

FastAPI service that wraps [Audiveris](https://github.com/Audiveris/audiveris)
to turn a sheet music image into the `NoteEvent` JSON the Flutter app already
consumes.

```
POST /extract            ─ multipart image upload → {bpm, notes}
GET  /health             ─ liveness probe
```

## Why Audiveris

Best-in-class open-source OMR accuracy (~90–95% on clean printed scores).
Outputs MusicXML, which the `convert.py` layer normalises into the
NoteEvent shape the Flutter app already handles. AGPLv3 licensed — fine for
private / hobby use; if you ever commercialise, the AGPL's network-use
clause requires you to publish your service's source.

## Local dev

Audiveris is a Java app (JRE 21+). On macOS the simplest path is:

```bash
brew install --cask audiveris
# now `audiveris` should be on your PATH (may need an entry under
# /Applications/Audiveris.app/Contents/MacOS/audiveris instead)
```

Or grab the AppImage from the Audiveris releases page and link it onto your
PATH manually.

Then:

```bash
cd server
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Test:

```bash
curl -X POST http://localhost:8000/extract \
  -F "image=@/path/to/sheet.jpg" | jq
```

Expected response:

```json
{
  "bpm": 120.0,
  "notes": [
    {"note": "D4", "beat": 0.0, "dur": 1.0, "hand": "right"},
    {"note": "F#4", "beat": 0.0, "dur": 1.0, "hand": "right"},
    {"note": "G3", "beat": 1.0, "dur": 2.0, "hand": "left"}
  ]
}
```

First call on a fresh JVM is slowest (~20–40s for the warmup + recognition).
Repeated calls on the same process are faster.

## Docker

Easiest way to deploy. The Dockerfile pins Audiveris and bundles a JRE +
Python venv:

```bash
docker build -t snapnotes-omr .
docker run --rm -p 8000:8000 -v snapnotes-cache:/app/cache snapnotes-omr
```

The named volume keeps cached extractions across container restarts.

If the Audiveris version pinned in the Dockerfile (`5.4.2` at time of
writing) has been replaced, check
<https://github.com/Audiveris/audiveris/releases> and bump
`AUDIVERIS_VERSION` in the Dockerfile.

## Pointing the Flutter app at it

`lib/shared/services/omr_service.dart` already POSTs a multipart upload to
`${ApiConfig.omrServerUrl}/extract`. The default is `http://localhost:8000`.
For a deployed server:

```bash
flutter run --dart-define=OMR_SERVER_URL=https://omr.example.com
```

For mobile testing against your dev machine, use your machine's LAN IP, not
`localhost`:

```bash
flutter run --dart-define=OMR_SERVER_URL=http://192.168.x.y:8000
```

## Deploy

- **Fly.io** — `fly launch`, accepts the Dockerfile as-is. JRE + Audiveris
  AppImage push the image past Fly's small free tier; you'll want a paid
  shared-cpu-2x instance (~$3–5/mo).
- **Render / Railway** — similar Docker-native flow.
- **VPS (Hetzner, DigitalOcean)** — cheapest at scale. ~$5/mo gets you a
  CPU + RAM comfortable for Audiveris.

Audiveris is memory-hungry. Give it at least 1 GB heap if you expect to
process larger scores.
