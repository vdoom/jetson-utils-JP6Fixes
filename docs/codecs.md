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

Not tested yet: anything on Orin NX, JetPack 7, hardware AV1 encode, and interop with VLC or a PC.

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
* `nvv4l2av1enc` runs on Orin NX, and `av1parse ! rtpav1pay` takes its output - if it fails there, change the hardware AV1 check in `gst_select_encoder()`

### Phase 4 - AV1 over RTP/RTSP

* interop with VLC and GStreamer/FFmpeg on another machine, over a real network
* jetson-inference's `docker/build.sh` only knows base images up to JetPack 5, so `AV1=on` was only tested in the `l4t-jetpack:r36.4.0` base image, not in a full jetson-inference container build

### Phase 5 - robustness

* fall back to the CPU encoder when a hardware encoder fails to start, not just when it's missing (e.g. `nvv4l2av1enc` exists on Xavier but has no hardware behind it - Xavier already gets the CPU encoder for AV1 from the SoC check, so this is for other cases)
* discovering an AV1 file on Xavier and older crashes inside `nvv4l2decoder` - skip it for AV1 on those SoCs
* on GStreamer 1.16 (no leaky `appsrc`) the input queue has no limit, since the `mNeedData` check in `encodeYUV()` is commented out - decide whether to re-enable it there
* UDP receive buffer: document raising it on receivers of AV1 over RTP (`sudo sysctl -w net.core.rmem_max=4194304`) - not tested, it needs root
* every RTP output logs "Pipeline construction is invalid, please add queues" from `udpsink`: the encoder's reported latency isn't covered upstream, since `appsrc` and the encoder run in one streaming thread up to the sink. It's harmless with x264, but it's why `svtav1enc`'s 1.25 s latency report dropped frames - check whether a `queue` before the payloader should be added for all encoders

### Phase 6 - tuning and options

* report the bugs upstream: `svtav1enc` in low delay (deadlock with SVT-AV1 2.3+, and the latency it reports) to GStreamer, and `rav1enc`'s time base to gst-plugins-rs
* command-line options for the encoder preset, keyframe interval and rate control (now hardcoded: x264/x265 `ultrafast` with `key-int-max=15`, AV1 one keyframe per second, `idrinterval=30` on V4L2). The AV1 keyframe interval is in frames, so when the encoder is slower than the input it gets keyframes less often
* Orin NX low-latency settings for the hardware encoders: `preset-level`, `control-rate`, `peak-bitrate`, `vbv-size`, and `iframeinterval`/`idrinterval` for AV1
* how many concurrent encode streams one Orin NX NVENC can take

### Phase 7 - other transports, JetPack 7

* AV1 over WebRTC and in MPEG-TS (`rtpmp2ts`) once the GStreamer version has it (JetPack 7 is based on Ubuntu 24.04 with GStreamer 1.24)
* JetPack 7 / Thor: check hardware AV1 encode. Ubuntu 24.04 packages `svtav1enc` (gst-plugins-bad 1.24.2, SVT-AV1 1.7), which `gstEncoder` runs in random-access mode - check that `scripts/gst-svtav1` builds there for low delay
