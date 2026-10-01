# Video Codecs for RTP/RTSP Streaming

Status and plan for H.264, H.265 and AV1 streaming with `videoOutput`/`videoSource` (`gstEncoder`/`gstDecoder`).

## Status

### Phase 1 - software encoders (Orin Nano) - done

Orin Nano has no NVENC, so `--output-encoder=v4l2` falls back to the CPU encoders there.

* `--output-codec=av1` / `--input-codec=av1` (`videoOptions::CODEC_AV1`, appended so the existing enum values don't change)
* AV1 software encoder chosen at runtime from what's installed: `svtav1enc`, then `av1enc` (libaom), then `rav1enc` - when `av1enc` has no realtime mode (`usage-profile`, missing on GStreamer 1.20) `rav1enc` comes before it
* AV1 software decoder: `dav1ddec`, then `av1dec`
* encoder properties that differ between elements and GStreamer versions are only set when the element has them (`gst_element_has_property()`), since an unknown property fails the whole pipeline, and integer ones are clamped to the element's range (`gst_element_property_range()`), since an out-of-range value is ignored
* AV1 over RTP/RTSP uses `av1parse ! rtpav1pay` and `rtpav1depay`, which come from [gst-plugins-rs](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs) and aren't in stock JetPack - without them the stream fails with an error that says so
* AV1 is rejected with an error for rtmp, avi and flv outputs (mkv and mp4 work), and was for webrtc until Phase 7
* fixed: software H.265 network streams never started (`x265enc` was given the x264-only `insert-vui`/`intra-refresh`)
* fixed: `x265enc` ran single-threaded on some aarch64 builds, the thread pool is now sized explicitly (8.4 -> ~24 fps at 720p on AGX Xavier)
* fixed: `appsrc max-buffers/leaky-type` only exist on GStreamer 1.20+ and broke every encoder on JetPack 5

### Phase 2 - hardware encoders (Orin NX) - done

* H.264/H.265 already used `nvv4l2h264enc`/`nvv4l2h265enc` on Orin NX
* AV1 uses `nvv4l2av1enc` on Orin and newer, except Orin Nano (no NVENC)
* AV1 decode uses `nvv4l2decoder` on Orin and newer, including Orin Nano (it has NVDEC)
* the SoC is read from `/proc/device-tree/compatible`, or from the board model in `/tmp/nv_jetson_model` inside containers (the device tree is masked there)
* `insert-sps-pps`/`insert-vui` are only passed to the H.264/H.265 hardware encoders

### Phase 3 - validation on Orin Nano and Orin NX - done

Fixed what the Orin Nano tests turned up:

* fixed: RTSP output failed with 503 for every client on JetPack 6 (also before the AV1 changes) - the encoder pipeline was started before the RTSP server linked the payloader, and the frames queued in the leaky `appsrc` failed with not-linked. Now the RTSP server starts the pipeline, both in `gstEncoder::Open()` and in the `media-configure` callback (which still prepares the media, so it stays prepared for the next client), and while no client is connected the frames are dropped
* fixed: AV1 files were empty - `av1enc cpu-used=8` is out of range on GStreamer 1.20 (0-5), so libaom ran at `cpu-used=0` (< 0.05 fps at 720p), and `gstEncoder::Close()` only waited 1 s after EOS. `Close()` now waits for the EOS on the bus (up to 30 s for files, 1 s for network streams)
* fixed: `rav1enc` ran on one core - it only runs tiles in parallel, so it now gets a tile and a thread per CPU
* fixed: AV1 hardware decoding failed for files (not-negotiated) and RTP/RTSP (couldn't link) - `nvv4l2decoder` only takes `alignment=frame`, so it now gets `av1parse ! video/x-av1,alignment=tu ! av1parse`. The TU step is needed because `av1parse` on GStreamer 1.20 merges pairs of frames when it converts the OBU-aligned `rtpav1depay` output to frames directly
* fixed: `av1dec` failed the whole pipeline on the first broken frame after packet loss - `rtpav1depay` now waits for the next keyframe (`wait-for-keyframe=true`)
* fixed: `--output-encoder=nvenc` picked `nvv4l2h264enc` on Orin Nano, which doesn't exist there - it now gets the same hardware check as `v4l2` and falls back to the CPU encoder

### Phase 4 - AV1 over RTP/RTSP - done on Orin Nano

* `scripts/gst-rtp-av1` builds `rtpav1pay`/`rtpav1depay` from gst-plugins-rs 0.15.4 (the newest release, it still supports GStreamer 1.20) and installs them (`--user` installs to `~/.local/share/gstreamer-1.0/plugins` without sudo). `--rav1e` also builds `rav1enc`, `--dav1d` also builds `dav1ddec` with libdav1d 1.5.4 (it needs 1.3+, JetPack 6 has 0.9.2)
* `av1parse ! rtpav1pay` works with `svtav1enc`, `rav1enc` and `av1enc`, late RTSP clients get a keyframe with a sequence header, and `rtpav1depay` decodes into `nvv4l2decoder`, `dav1ddec` and `av1dec`
* interop with FFmpeg 8 in both directions (see the tests below)
* jetson-inference's Dockerfile builds both scripts with `--build-arg AV1=on` (or `AV1=on docker/build.sh`)

### Phase 5/6 - SVT-AV1, low delay and fixes on Orin Nano - done

* `scripts/gst-svtav1` builds SVT-AV1 4.2.0 and `svtav1enc` (from GStreamer 1.28.7) against the installed GStreamer (1.20+). SVT-AV1 is the only software AV1 encoder that runs in realtime on Orin Nano, but JetPack 6 only packages SVT-AV1 0.9 and has no `svtav1enc` (it's in gst-plugins-bad 1.24+). The script fixes two bugs in `svtav1enc`'s low-delay mode, which are still in GStreamer main:
  * it deadlocks after the first frame with SVT-AV1 2.3+ - `svt_av1_enc_get_packet()` blocks in low delay since 2.3, and `svtav1enc` keeps calling it until the queue is empty
  * it reports the random-access lookahead as latency (37 frames, 1.25 s at 30 fps), so sinks that sync (`udpsink`, the RTSP server) held every frame that long and `appsrc` dropped the frames in between - 13 of 300 frames got through
* `gstEncoder` runs `svtav1enc` in low-delay CBR when it has the fixes above (`gst_svtav1_low_delay()`, the plugin built by the script says "low-delay fix" in its package name): `rtc=1:rc=2` with SVT-AV1 3.1+, `pred-struct=1:rc=2` with older versions (the version comes from `svt_av1_get_version()`). Other `svtav1enc` builds stay in random-access mode, with a warning - in low delay they deadlock with SVT-AV1 2.3+, and with any version their latency report makes network streams drop most frames. Random-access mode adds ~1.07 s of encoder latency, low delay ~10 ms
* `scripts/gst-rtp-av1 --rav1e` fixes `rav1enc`'s rate control, which used ~15x the bitrate (61 Mbps for 4 Mbps at 720p) - it passes the frame rate as rav1e's time base, which is the frame duration. Also still in gst-plugins-rs main
* `video-viewer` exits with 1 when it can't create the input or output stream (it exited with 0)
* GStreamer warnings on the bus are logged with their text (at the verbose level), not only the element they came from
* RTP input asks `udpsrc` for a 4 MB receive buffer, which the kernel caps to `net.core.rmem_max` (212 KB on JetPack) - so raising `rmem_max` on the receiver is enough to get a bigger one

### Phase 7 - Orin NX, JetPack 7, robustness and options - done

Fixed what the Orin NX / JetPack 7 tests turned up:

* fixed: hardware AV1 to mkv and mp4 wrote empty files (not-negotiated) - the muxers take a TU-aligned OBU stream and `nvv4l2av1enc` doesn't say what it outputs, so files now get `av1parse` before the muxer (when it exists, GStreamer 1.20+)
* fixed: WebRTC output aborted as soon as a client connected on JetPack 7 ("libsoup2 symbols detected") - Ubuntu 24.04's libnice uses libsoup 3 through gupnp-igd, and the two versions can't be loaded in one process. The WebRTC server builds against libsoup 3 or 2.4, CMake picks the one that libnice uses (`ldd`), and warns if that's libsoup 3 and `libsoup-3.0-dev` is missing. jetson-inference's `CMakePreBuild.sh` installs `libsoup-3.0-dev` where apt has it
* AV1 over WebRTC (it was rejected with an error) - tested with GStreamer 1.24 and a webrtcbin client, not with browsers
* fixed: H.264 clients that join a running stream (a second RTSP client, a client that reconnects, an RTP receiver started late) got 0 frames with `nvv4l2decoder` - `x264enc` now sends an IDR every second (`key-int-max` = frame rate) with a 200 ms VBV instead of intra refresh. Forcing a keyframe for new RTSP clients doesn't work with intra refresh (`x264enc` turns it into another intra refresh). The 200 ms VBV keeps the IDRs smaller than intra refresh's largest frames, at better quality (see the table below)
* `nvv4l2av1enc` for network streams: `insert-seq-hdr` (it only sends the sequence header with the first frame - `rtpav1pay` from gst-plugins-rs 0.15 re-inserts it, but other payloaders may not) and a 6-frame VBV, which halves the IDRs (a 3-frame VBV like H.264/H.265 costs AV1 ~2 dB)
* a hardware encoder that fails to start now falls back to the CPU encoder, not only a missing one: the first V4L2 encoder of each codec encodes two 320x240 frames before the real pipeline is built (~100 ms for the first, ~20 ms after). `/dev/v4l2-nvenc` is a stub that always opens on JetPack 7, so a missing or broken NVENC only shows up once frames go through. Tested by failing `VIDIOC_S_FMT` with an `LD_PRELOAD` shim
* discovering an AV1 file no longer plugs `nvv4l2decoder` on SoCs without hardware AV1 (Xavier and older, where it crashed) - the discoverer's `uridecodebin` skips it for `video/x-av1` there
* GStreamer before 1.20 (no leaky `appsrc`): `encodeYUV()` drops new frames again while `appsrc` is full, so its queue doesn't grow without a limit. 1.20+ is unchanged (`appsrc` drops the oldest frames itself)
* RTP output no longer warns "Pipeline construction is invalid, please add queues": `udpsink` gets `processing-deadline=0`, which is what it fell back to anyway. A `queue` before the payloader would give it the 20 ms deadline back, but then it holds every packet for that long (see the latency table)
* `--keyframe-interval=N` (`videoOptions::keyframeInterval`, `keyframeInterval` in Python) sets the frames between keyframes for every encoder, also for files. The default (0) keeps the earlier behavior: 30 frames for x264, the AV1 encoders and the V4L2 encoders on network streams, 15 for x265
* fixed: `svtav1enc` got a keyframe every 31 frames instead of 30 - `intra-period-length` is the frames after a keyframe. It's at least 1, since 0 is all-intra mode, which SVT-AV1 4.x can't run with CBR (and `svtav1enc` then crashes at EOS)
* `scripts/gst-rtp-av1 --dav1d` uses the installed libdav1d when it's 1.3+ (JetPack 7 has 1.4.1) instead of building one, which needed meson
* `scripts/gst-svtav1` builds on JetPack 7 (GStreamer 1.24.2), and its `svtav1enc` takes precedence over the packaged one, which is too slow on Orin (see below)

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

Orin Nano Super developer kit, JetPack 6 (L4T R36.5), GStreamer 1.20.3, libaom 3.3, `MAXN_SUPER` power mode. The installed AV1 elements were `av1enc`, `av1dec`, `av1parse` and `nvv4l2decoder` - the others were built with `scripts/gst-rtp-av1` and `scripts/gst-svtav1`.

Software encoder speed, 300 frames of a real 720p/1080p clip (DeepStream's `sample_720p`/`sample_1080p_h264`) from raw I420 with `gst-launch-1.0`, with the properties `gstEncoder` sets:

| Encoder                                                       | 720p       | 1080p    |
|---------------------------------------------------------------|------------|----------|
| `x264enc`                                                     | 160 fps    | 87 fps   |
| `x265enc` (`pools=6` makes no difference here)                | 12.4 fps   | 6.8 fps  |
| `x265enc` with `frame-threads=3`                              | 16.5 fps   |          |
| `av1enc cpu-used=0` (what `cpu-used=8` turned into)           | < 0.05 fps |          |
| `av1enc cpu-used=5` (the fastest on GStreamer 1.20)           | 0.18 fps   | 0.09 fps |
| `rav1enc` `speed-preset=10`, 6 tiles, with the rate control fix | 4.5 fps  |          |
| `svtav1enc` preset 12, low delay `rtc=1:rc=2` (SVT-AV1 4.2)   | 39.5 fps   | 25.7 fps |
| `svtav1enc` preset 13, low delay `rtc=1:rc=2`                 | 41.5 fps   | 27.1 fps |
| `svtav1enc` preset 12, low delay `pred-struct=1:rc=2`         | 34.7 fps   |          |
| `svtav1enc` preset 12, random access (the default)            | 109 fps    | 62 fps   |

So in realtime at 720p30 only H.264 and AV1 with SVT-AV1 are usable on Orin Nano's CPU, H.265 isn't. `svtav1enc` holds its bitrate (3.9 Mbps for 4 Mbps), and its encoder latency is ~10 ms in low delay (18 ms max) vs ~1.07 s in random access (1.45 s max), measured with GStreamer's latency tracer from a live 720p30 source.

AV1 decoder speed, 300 frames of SVT-AV1 at 4 Mbps:

| Decoder                    | 720p    | 1080p   |
|----------------------------|---------|---------|
| `av1dec` (libaom)          | 71 fps  | 39 fps  |
| `dav1ddec` (libdav1d 1.5.4)| 235 fps | 244 fps |
| `nvv4l2decoder` (NVDEC)    | 442 fps | 315 fps |

Streaming with `video-viewer` (after the fixes above), 300 frames, to an independent receiver that decodes with `nvv4l2decoder`:

| Test                                                          | Result                                                |
|---------------------------------------------------------------|-------------------------------------------------------|
| H.264, `cpu` and `v4l2`, `rtp://`                             | 298 of 300 frames                                     |
| H.265, `cpu` and `v4l2`, `rtp://`                             | 112 frames (x265 is too slow, `appsrc` drops the rest) |
| H.264, `rtsp://`, client at the start / ~3 s late             | 300 / 218 frames (0 before the RTSP fix)              |
| H.265, `rtsp://`, client at the start / ~3 s late             | 113 / 86 frames                                       |
| `rtsp://`, two clients (at the start + ~3 s late), H.265 / AV1 | 111 + 81 / 300 + 207 frames                          |
| `rtsp://`, a client connects 2 s after another one left, H.265 / AV1 | 37 / 145 frames                                 |
| H.264 `rtsp://`, a client joining a stream that's already running | 0 frames with `nvv4l2decoder`, 219 with FFmpeg (see Phase 5) |
| `--output-encoder=v4l2` / `nvenc`                             | both fall back to the CPU encoder                     |
| AV1 (`svtav1enc`, low delay) 720p to `rtp://`                 | 298 frames (13 before the `svtav1enc` latency fix)    |
| AV1 (`svtav1enc`, low delay) 720p to `rtsp://`, client at the start / ~5 s late | 300 / 155 frames                    |
| AV1 (`svtav1enc`, low delay) 720p to mkv                      | 298 frames                                            |
| AV1 (`svtav1enc` without the fixes, random access) 720p to `rtp://` | 229 frames (its bursts overflow the UDP buffer)  |
| AV1 (`rav1enc`) to mkv and mp4 at 720p                        | valid files with 34-36 frames (0 bytes before)        |
| AV1 (`av1enc`, 360p) through `rtpav1pay`                      | 30 of 30 frames                                       |
| AV1 file input, `nvv4l2decoder` / `dav1ddec` / `av1dec`       | 28 / 26 / 27 of 30 frames                             |
| AV1 `rtp://` input, `nvv4l2decoder` / `dav1ddec` / `av1dec`   | 28 / 29 / 29 of 30 frames                             |
| AV1 to rtp/rtsp without `rtpav1pay`, AV1 to webrtc            | clean errors                                          |

Interop with FFmpeg 8 (through PyAV 17.1, since there's no VLC or ffmpeg CLI on the board):

| Test                                                          | Result                                                |
|---------------------------------------------------------------|-------------------------------------------------------|
| `video-viewer` `rtsp://` to FFmpeg over UDP, H.264 / H.265 / AV1 | 299 / 110 / 300 frames decoded, no decode errors (AV1 with libdav1d) |
| `video-viewer` `rtsp://` to FFmpeg over TCP, H.264 / H.265 / AV1 | 300 / 114 / 300 frames decoded                      |
| FFmpeg (`libsvtav1` and its experimental AV1 RTP packetizer, needs `-strict experimental`) to `video-viewer` `rtp://` | 193 frames with `nvv4l2decoder`, 90 with `av1dec` - FFmpeg sends its random-access output in bursts, which lost 50 packets in the UDP receive buffer, and the depayloader waits for a keyframe after each loss |

All of it ran on one board over localhost. `net.core.rmem_max` is 212 KB on JetPack, so large or bursty AV1 frames overflow the UDP receive buffer and lose packets there (`netstat -su` shows them as receive buffer errors).

#### Orin NX

Orin NX developer kit, JetPack 7.2 (L4T R39.2.1), GStreamer 1.24.2, `MAXN` power mode, NVENC on its default `tegra_wmark` governor. `rtpav1pay`/`rtpav1depay`, `rav1enc`, `dav1ddec` and the low-delay `svtav1enc` (SVT-AV1 4.2) were built with `scripts/gst-rtp-av1 --user --rav1e --dav1d` and `scripts/gst-svtav1 --user`. 299-frame 720p clip (`sample_720p`), sent with `video-viewer` to an independent receiver that decodes with `nvv4l2decoder`, unless it says otherwise.

| Test                                                              | Result                                                    |
|-------------------------------------------------------------------|-----------------------------------------------------------|
| H.264 / H.265 / AV1, `v4l2` (NVENC), `rtp://`                      | 292 / 294 / 292 frames                                    |
| H.264 / H.265 / AV1, `cpu`, `rtp://`                               | 294 / 175 (x265 runs at ~18 fps) / 293 (`svtav1enc` low delay) |
| `rtsp://`, client at the start, H.264 cpu+v4l2 / H.265 v4l2 / AV1 cpu+v4l2 | 279 / 278 / 278-279                               |
| `rtsp://`, only client ~3 s late, every encoder                    | 187-189 frames (x265 112)                                 |
| `rtsp://`, second client ~3 s late, H.264 cpu / H.264, H.265, AV1 v4l2 / AV1 cpu | 279 + 160 (0 before the x264 change) / 160-161 / 188 |
| `rtsp://`, client connects 2 s after another one left             | 98-102 frames for every encoder (x264 0 before)           |
| `rtp://` receiver started ~3 s late                                | 169-175 frames for every encoder, AV1 also with `dav1ddec` and `av1dec` |
| AV1 `v4l2` to mkv / mp4                                            | 294 of 299 frames, decoded with `nvv4l2decoder`, `dav1ddec` and `av1dec` (0 bytes before) |
| AV1 file input through `gstDecoder`, `v4l2` / `cpu`               | works, 287 frames re-encoded                              |
| `--output-encoder=v4l2` with NVENC failing (`LD_PRELOAD` shim)    | falls back to `x264enc` / `x265enc` / `svtav1enc`         |
| `--keyframe-interval=15` / `60`, every encoder, to mkv             | a keyframe exactly every 15 / 60 frames                    |
| WebRTC (webrtcbin client), H.264 v4l2 / cpu, AV1 v4l2 / cpu        | 219 / 225 / 232 / 229 frames (it connects ~2 s in), AV1 also with `nvv4l2decoder` and over HTTPS/WSS - aborted before the libsoup fix |

x264 for network streams, 300 frames of the 720p clip at 4 Mbps, Y-PSNR against the source:

| `x264enc`                                        | Largest frame | Y-PSNR   | Late clients with `nvv4l2decoder` |
|--------------------------------------------------|---------------|----------|-----------------------------------|
| `intra-refresh=true key-int-max=15` (before)     | 62 KB         | 32.70 dB | no                                |
| `key-int-max=30`, default 600 ms VBV              | 105 KB        | 34.37 dB | yes                               |
| `key-int-max=30 vbv-buf-capacity=200` (now)       | 57 KB         | 33.65 dB | yes                               |
| `key-int-max=30 vbv-buf-capacity=100`             | 46 KB         | 33.13 dB | yes, but 3.6 Mbps for 4 Mbps      |

NVENC VBV size at 4 Mbps, on the 720p clip and on a still image (the worst case for IDR size):

| Encoder, VBV                                       | Clip: largest frame, Y-PSNR | Still: largest frame, Y-PSNR |
|----------------------------------------------------|-----------------------------|------------------------------|
| `nvv4l2h264enc`, default (4 Mbit)                   | 136 KB, 37.39 dB            | 163 KB, 53.13 dB             |
| `nvv4l2h264enc`, 3 frames + two-pass CBR (now)      | 42 KB, 36.53 dB             | 41 KB, 48.73 dB              |
| `nvv4l2h264enc`, 3 frames, single pass              | 42 KB, 36.34 dB             | 38 KB, 47.00 dB              |
| `nvv4l2h265enc`, 3 frames + two-pass CBR (now)      | 41 KB, 38.21 dB             | 42 KB, 51.78 dB              |
| `nvv4l2h265enc`, 3 frames, single pass              | 44 KB, 38.02 dB             | 38 KB, 50.36 dB              |
| `nvv4l2av1enc`, default (4 Mbit)                    | 94 KB, 37.07 dB             | 143 KB, 47.64 dB             |
| `nvv4l2av1enc`, 6 frames (now)                      | 75 KB, 36.55 dB             | 71 KB, 44.96 dB              |
| `nvv4l2av1enc`, 3 frames                            | 50 KB, 35.22 dB             | 54 KB, 40.74 dB              |

NVENC throughput at 1080p, raw frames from memory, with the settings `gstEncoder` uses for network streams (two-pass CBR for H.264/H.265), 4 encodes in parallel:

| Encoder                       | One encode | 4 in parallel, total | 1080p30 streams |
|-------------------------------|------------|----------------------|-----------------|
| `nvv4l2h264enc`, two-pass     | 152 fps    | 240 fps              | 8               |
| `nvv4l2h264enc`, single pass  | 309 fps    | 404 fps              | 13              |
| `nvv4l2h265enc`, two-pass     | 107 fps    | 203 fps              | 6               |
| `nvv4l2h265enc`, single pass  | 308 fps    | 398 fps              | 13              |
| `nvv4l2av1enc`                | 283 fps    | 363 fps              | 12              |

At 720p (decoded with NVDEC), 4 encodes in parallel reach 496 fps for H.264, 405 fps for H.265 (both two-pass) and 732 fps for AV1, 13-24 streams at 30 fps. NVDEC tops out at ~407 fps for four 1080p H.264 decodes, so the 1080p numbers had to come from raw frames. Two-pass CBR halves what NVENC can take for ~0.2 dB on moving scenes and ~1.5 dB on still ones.

Latency from `appsrc`/the source to `udpsink` with GStreamer's latency tracer, live 720p30 source:

| Encoder                           | Without a queue | With a queue before the payloader |
|-----------------------------------|-----------------|-----------------------------------|
| `x264enc`                         | 3.8 ms          | 20.4 ms                           |
| `nvv4l2h264enc`                   | 20.4 ms         | 20.4 ms                           |
| `nvv4l2av1enc`                    | 13.3 ms         | 13.3 ms                           |
| `svtav1enc` low delay             | 8.9 ms          | 53.4 ms                           |

Software AV1 encoders on JetPack 7, 720p clip: the packaged `svtav1enc` (SVT-AV1 1.7, random access) runs at 10.8 fps, `av1enc` (libaom 3.8) in realtime mode at 10-11 fps (`cpu-used` 8-10), `rav1enc` at 5.4 fps, and SVT-AV1 4.2 from `scripts/gst-svtav1` at 37.7 fps in low delay (166 fps in random access). So only the script's `svtav1enc` is realtime.

Interop with FFmpeg (PyAV 19, libavcodec 63.1):

| Test                                                              | Result                                                    |
|-------------------------------------------------------------------|-----------------------------------------------------------|
| `video-viewer` `rtsp://` to FFmpeg over UDP and TCP, H.264 v4l2 / H.264 cpu / H.265 v4l2 / AV1 v4l2 / AV1 cpu | 281 / 280 / 281 / 281 / 278 frames, no decode errors (AV1 with libdav1d) |
| FFmpeg `libsvtav1` low delay (`pred-struct=1`) to `video-viewer` `rtp://`, `nvv4l2decoder` / `dav1ddec` | 291 / 293 frames, no UDP receive buffer errors |
| FFmpeg `libsvtav1` random access (VBR) to `video-viewer` `rtp://`, `nvv4l2decoder` / `dav1ddec` | 266 / 272 frames, 262 packets lost in the 212 KB UDP receive buffer |

Not tested yet: interop with VLC or another machine over a real network, Thor, GStreamer 1.16 (JetPack 5) with the changes above, and browsers for AV1 over WebRTC.

## Remaining work

Needs other hardware or root:

* interop with VLC and GStreamer/FFmpeg on another machine, over a real network
* jetson-inference's `docker/build.sh` only knows base images up to JetPack 5, so `AV1=on` was only tested in the `l4t-jetpack:r36.4.0` base image, not in a full jetson-inference container build
* UDP receive buffer: test and document raising it on receivers of AV1 over RTP (`sudo sysctl -w net.core.rmem_max=4194304`) - random-access AV1 from FFmpeg lost 262 packets at the default 212 KB
* NVENC latency with the `performance` governor - `gstEncoder` only warns about it, since changing it needs root
* Xavier / JetPack 5: the AV1 discovery fix and the GStreamer 1.16 `appsrc` limit were only tested on Orin NX (the latter by forcing that code path on GStreamer 1.24)
* Thor: hardware AV1 encode

Tuning and options:

* report the bugs upstream: `svtav1enc` in low delay (deadlock with SVT-AV1 2.3+, and the latency it reports) and its crash at EOS after a configuration error to GStreamer, and `rav1enc`'s time base to gst-plugins-rs
* command-line options for the encoder preset and rate control (now hardcoded: x264/x265 `ultrafast`, `svtav1enc` preset 12, NVENC's default preset and two-pass CBR). The names and ranges differ between encoders, so they need a common scale first
* two-pass CBR halves what NVENC can take (8 vs 13 1080p30 H.264 streams) for ~0.2-1.5 dB - an option to turn it off for many streams
* NVENC `preset-level` and `peak-bitrate` for low latency - not tried yet

Other transports:

* AV1 in MPEG-TS (`rtpmp2ts`) - `mpegtsmux` in GStreamer 1.24 (JetPack 7) only takes H.264 and H.265
