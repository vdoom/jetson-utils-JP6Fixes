# Video Codecs for RTP/RTSP Streaming

Status and plan for H.264, H.265 and AV1 streaming with `videoOutput`/`videoSource` (`gstEncoder`/`gstDecoder`).

## Status

### Phase 1 - software encoders (Orin Nano) - done

Orin Nano has no NVENC, so `--output-encoder=v4l2` falls back to the CPU encoders there.

* `--output-codec=av1` / `--input-codec=av1` (`videoOptions::CODEC_AV1`, appended so the existing enum values don't change)
* AV1 software encoder chosen at runtime from what's installed: `svtav1enc`, then `av1enc` (libaom), then `rav1enc` - when `av1enc` has no realtime mode (`usage-profile`, missing on GStreamer 1.20) `rav1enc` comes before it
* AV1 software decoder: `dav1ddec`, then `av1dec`
* encoder properties that differ between elements and GStreamer versions are only set when the element has them (`gst_element_has_property()`), since an unknown property fails the whole pipeline, and integer ones are clamped to the element's range (`gst_element_property_range()`), since an out-of-range value is ignored
* AV1 over RTP/RTSP uses `av1parse ! rtpav1pay` and `rtpav1depay`, which come from [gst-plugins-rs](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs) and aren't in stock JetPack - without them the stream fails with an error that says so. `scripts/gst-rtp-av1` builds and installs them (`--user` installs to `~/.local/share/gstreamer-1.0/plugins` without sudo, `--rav1e` also builds `rav1enc`)
* AV1 is rejected with an error for webrtc, rtmp, avi and flv outputs (mkv and mp4 work)
* fixed: software H.265 network streams never started (`x265enc` was given the x264-only `insert-vui`/`intra-refresh`)
* fixed: `x265enc` ran single-threaded on some aarch64 builds, the thread pool is now sized explicitly (8.4 -> ~24 fps at 720p on AGX Xavier)
* fixed: `appsrc max-buffers/leaky-type` only exist on GStreamer 1.20+ and broke every encoder on JetPack 5

### Phase 2 - hardware encoders (Orin NX) - done

* H.264/H.265 already used `nvv4l2h264enc`/`nvv4l2h265enc` on Orin NX
* AV1 uses `nvv4l2av1enc` on Orin and newer, except Orin Nano (no NVENC)
* AV1 decode uses `nvv4l2decoder` on Orin and newer, including Orin Nano (it has NVDEC)
* the SoC is read from `/proc/device-tree/compatible`, or from the board model in `/tmp/nv_jetson_model` inside containers (the device tree is masked there)
* `insert-sps-pps`/`insert-vui` are only passed to the H.264/H.265 hardware encoders

### Phase 3 - validation on Orin Nano - done (Orin NX still to do)

Fixed what the Orin Nano tests turned up:

* fixed: RTSP output failed with 503 for every client on JetPack 6 (also before the AV1 changes) - the encoder pipeline was started before the RTSP server linked the payloader, and the frames queued in the leaky `appsrc` failed with not-linked. Now the RTSP server starts the pipeline, both in `gstEncoder::Open()` and in the `media-configure` callback
* fixed: AV1 files were empty - `av1enc cpu-used=8` is out of range on GStreamer 1.20 (0-5), so libaom ran at `cpu-used=0` (< 0.05 fps at 720p), and `gstEncoder::Close()` only waited 1 s after EOS. `Close()` now waits for the EOS on the bus (up to 30 s for files, 1 s for network streams)
* fixed: `rav1enc` ran on one core - it only runs tiles in parallel, so it now gets a tile and a thread per CPU (0.55 -> 2.1 fps at 720p)
* fixed: AV1 hardware decoding failed for files (not-negotiated) and RTP/RTSP (couldn't link) - `nvv4l2decoder` only takes `alignment=frame`, so it now gets `av1parse ! video/x-av1,alignment=tu ! av1parse`. The TU step is needed because `av1parse` on GStreamer 1.20 merges pairs of frames when it converts the OBU-aligned `rtpav1depay` output to frames directly
* fixed: `av1dec` failed the whole pipeline on the first broken frame after packet loss - `rtpav1depay` now waits for the next keyframe (`wait-for-keyframe=true`)
* fixed: `--output-encoder=nvenc` picked `nvv4l2h264enc` on Orin Nano, which doesn't exist there - it now gets the same hardware check as `v4l2` and falls back to the CPU encoder

### What was tested

#### AGX Xavier

JetPack 5.1 (L4T R35.6.1), GStreamer 1.16, sending a 300-frame 720p clip with `video-viewer` to an independent `gst-launch-1.0` receiver.

| Test                                                  | Result                                          |
|-------------------------------------------------------|-------------------------------------------------|
| H.264 / H.265, `cpu` and `v4l2`, `rtp://`             | 291-295 of 300 frames received                  |
| H.264 / H.265, `cpu` and `v4l2`, `rtsp://`            | 222-224 frames (client joined ~3 s late)        |
| AV1 `cpu` to mkv, decoded back with `av1dec`          | valid stream                                    |
| AV1 file decode through `gstDecoder`                  | 53-54 of 60 frames                              |
| AV1 with `--output-encoder=v4l2` on Xavier            | falls back to the CPU encoder                   |
| AV1 to rtp/rtsp/webrtc/rtmp/avi/flv                   | clean errors                                    |

#### Orin Nano

Orin Nano Super developer kit, JetPack 6 (L4T R36.5), GStreamer 1.20.3, libaom 3.3, `MAXN_SUPER` power mode. The installed AV1 elements were `av1enc`, `av1dec`, `av1parse` and `nvv4l2decoder` - `rtpav1pay`, `rtpav1depay` and `rav1enc` were built with `scripts/gst-rtp-av1`, and `svtav1enc` was built by hand (see Phase 6).

Software encoder speed, 300 frames of a real 720p/1080p clip (DeepStream's `sample_720p`/`sample_1080p_h264`) from raw I420 with `gst-launch-1.0`, with the properties `gstEncoder` sets:

| Encoder                                                   | 720p      | 1080p    |
|-----------------------------------------------------------|-----------|----------|
| `x264enc`                                                 | 160 fps   | 87 fps   |
| `x265enc` (`pools=6` makes no difference here)            | 12.4 fps  | 6.8 fps  |
| `x265enc` with `frame-threads=3`                          | 16.5 fps  |          |
| `av1enc cpu-used=0` (what `cpu-used=8` turned into)       | < 0.05 fps|          |
| `av1enc cpu-used=5` (the fastest on GStreamer 1.20)       | 0.18 fps  | 0.09 fps |
| `rav1enc` `speed-preset=10`, 1 tile                       | 0.55 fps  |          |
| `rav1enc` `speed-preset=10`, 6 tiles                      | 2.1 fps   |          |
| `svtav1enc` preset 12, `rtc=1:rc=2` (SVT-AV1 4.2)         | 39.5 fps  | 25.7 fps |

At 360p, `rav1enc` does 6.5 fps and `av1enc` 1.7 fps. So in realtime at 720p30 only H.264 and AV1 with SVT-AV1 are usable on Orin Nano's CPU, H.265 isn't.

Streaming with `video-viewer` (after the fixes above), 300 frames, to an independent receiver that decodes with `nvv4l2decoder`:

| Test                                                          | Result                                                |
|---------------------------------------------------------------|-------------------------------------------------------|
| H.264, `cpu` and `v4l2`, `rtp://`                             | 298 of 300 frames                                     |
| H.265, `cpu` and `v4l2`, `rtp://`                             | 112 frames (x265 is too slow, `appsrc` drops the rest) |
| H.264, `rtsp://`, client at the start / ~3 s late             | 300 / 218 frames (0 before the RTSP fix)              |
| H.265, `rtsp://`, client at the start / ~3 s late             | 113 / 86 frames                                       |
| `--output-encoder=v4l2` / `nvenc`                             | both fall back to the CPU encoder                     |
| AV1 (`rav1enc`) to mkv and mp4 at 720p                        | valid files with 34-36 frames (0 bytes before)        |
| AV1 (`rav1enc`) to `rtp://` at 360p                           | 65 frames (the encoder does ~6.5 fps)                 |
| AV1 to `rtsp://` at 360p, client ~5 s late                    | starts decoding at a keyframe with a sequence header  |
| AV1 file input, `nvv4l2decoder` / `av1dec`                    | 28 / 27 of 30 frames                                  |
| AV1 `rtp://` input, `nvv4l2decoder` / `av1dec`                | 28 / 29 of 30 frames                                  |
| AV1 to rtp/rtsp without `rtpav1pay`, AV1 to webrtc            | clean errors                                          |

All of it ran on one board over localhost. `net.core.rmem_max` is 212 KB on JetPack, so large AV1 frames overflow the UDP receive buffer and lose packets there (`netstat -su` shows them as receive buffer errors).

Not tested yet: anything on Orin NX, JetPack 7, hardware AV1 encode, `dav1ddec` (not installed), and interop with other clients.

## Next phases

### Phase 3 - validate on Orin NX

Hardware AV1 encode, to mkv and mp4, then decode and count the frames:

```
gst-launch-1.0 -e videotestsrc num-buffers=300 ! video/x-raw,width=1280,height=720,format=I420,framerate=30/1 ! \
    nvvidconv ! 'video/x-raw(memory:NVMM)' ! nvv4l2av1enc bitrate=4000000 idrinterval=30 maxperf-enable=1 ! \
    video/x-av1 ! matroskamux ! filesink location=hw_av1.mkv
```

Then the H.264/H.265/AV1 streaming tests from the Orin Nano table with `video-viewer` (`--output-codec`, `--output-encoder=cpu|v4l2`, `rtp://` and `rtsp://`). Port 8554 may be taken by another RTSP server on the board, `video-viewer` then fails to bind it - use another port like `rtsp://@:8555/test`.

Done when:
* H.264/H.265 stream over RTP and RTSP on Orin NX
* `nvv4l2av1enc` runs on Orin NX - if it fails there, change the hardware AV1 check in `gst_select_encoder()`

### Phase 4 - AV1 over RTP/RTSP

Done on Orin Nano: the `rtp` plugin builds from gst-plugins-rs 0.15.4 for GStreamer 1.20 (`scripts/gst-rtp-av1`), `av1parse ! rtpav1pay` negotiates with `rav1enc` and `svtav1enc`, late RTSP clients get a keyframe with a sequence header, and `rtpav1depay` decodes into `av1dec` and `nvv4l2decoder`. Still to do:

* add `scripts/gst-rtp-av1` to the container build
* check that `av1parse ! rtpav1pay` negotiates with `av1enc` and `nvv4l2av1enc`
* receive side with `dav1ddec`
* interop with other clients (VLC, ffplay, GStreamer on a PC)

### Phase 5 - robustness

* fall back to the CPU encoder when a hardware encoder fails to start, not just when it's missing (e.g. `nvv4l2av1enc` exists on Xavier but has no hardware behind it)
* discovering an AV1 file on Xavier and older crashes inside `nvv4l2decoder` - skip it for AV1 on those SoCs
* on GStreamer 1.16 (no leaky `appsrc`) the input queue has no limit, since the `mNeedData` check in `encodeYUV()` is commented out - decide whether to re-enable it there
* `video-viewer` exits with 0 when it can't create the output stream (e.g. the RTSP port is taken)
* UDP receive buffer: document raising `net.core.rmem_max` (and `udpsrc buffer-size`) for AV1 over RTP, its keyframes overflow the 212 KB default

### Phase 6 - tuning and options

* SVT-AV1 on Orin Nano: it's the only software AV1 encoder that runs in realtime there (39.5 fps at 720p, 25.7 fps at 1080p with `rtc=1`), but JetPack 6 only packages SVT-AV1 0.9 and has no `svtav1enc` (it's in gst-plugins-bad 1.24+). It works when built by hand: SVT-AV1 4.2 from source, plus `ext/svtav1/gstsvtav1enc.c` from GStreamer main built against GStreamer 1.20 with two changes - skip `gst_plugin_set_static_features_flag()` (1.26+), and get one packet per picture in low-delay mode (below). Add that to `scripts/gst-rtp-av1` or a new script
* `svtav1enc` deadlocks in low-delay mode (`pred-struct=1` or `rtc=1`) with SVT-AV1 2.3 and newer: `svt_av1_enc_get_packet()` blocks in low-delay mode since 2.3, and the plugin keeps calling it until the queue is empty. This is an upstream GStreamer bug (still in main), worth reporting
* `gstEncoder` runs `svtav1enc` in random-access mode (`preset=12` gets mapped to 11, and SVT-AV1 warns that non-RTC M10+ can have visual artifacts), which adds latency for network streams. Low-delay CBR options: `parameters-string="rtc=1:rc=2"` (SVT-AV1 3.1+, 39.5 fps at 720p) or `pred-struct=1:rc=2` (older versions, 34.7 fps) - both need a `svtav1enc` without the deadlock above
* `rav1enc` overshoots its target bitrate ~2.5x at 720p (12.8 MB for 10 s at 4 Mbps)
* command-line options for the encoder preset, keyframe interval and rate control (now hardcoded: x264/x265 `ultrafast` with `key-int-max=15`, AV1 one keyframe per second, `idrinterval=30` on V4L2). The AV1 keyframe interval is in frames, so when the encoder is slower than the input it gets keyframes less often
* Orin NX low-latency settings for the hardware encoders: `preset-level`, `control-rate`, `peak-bitrate`, `vbv-size`, and `iframeinterval`/`idrinterval` for AV1
* how many concurrent encode streams one Orin NX NVENC can take

### Phase 7 - other transports, JetPack 7, docs

* AV1 over WebRTC and in MPEG-TS (`rtpmp2ts`) once the GStreamer version has it (JetPack 7 is based on Ubuntu 24.04 with GStreamer 1.24)
* JetPack 7 / Thor: check hardware AV1 encode and which software AV1 elements are installed
* update the codec list and AV1 requirements in jetson-inference `docs/aux-streaming.md`
