# Video Codecs for RTP/RTSP Streaming

Status and plan for H.264, H.265 and AV1 streaming with `videoOutput`/`videoSource` (`gstEncoder`/`gstDecoder`).

## Status

### Phase 1 - software encoders (Orin Nano) - done

Orin Nano has no NVENC, so `--output-encoder=v4l2` falls back to the CPU encoders there.

* `--output-codec=av1` / `--input-codec=av1` (`videoOptions::CODEC_AV1`, appended so the existing enum values don't change)
* AV1 software encoder chosen at runtime from what's installed: `svtav1enc`, then `av1enc` (libaom), then `rav1enc`
* AV1 software decoder: `dav1ddec`, then `av1dec`
* encoder properties that differ between elements and GStreamer versions are only set when the element has them (`gst_element_has_property()`), since an unknown property fails the whole pipeline
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

### What was tested

Everything was tested on AGX Xavier, JetPack 5.1 (L4T R35.6.1), GStreamer 1.16, sending a 300-frame 720p clip with `video-viewer` to an independent `gst-launch-1.0` receiver.

| Test                                                  | Result                                          |
|-------------------------------------------------------|-------------------------------------------------|
| H.264 / H.265, `cpu` and `v4l2`, `rtp://`             | 291-295 of 300 frames received                  |
| H.264 / H.265, `cpu` and `v4l2`, `rtsp://`            | 222-224 frames (client joined ~3 s late)        |
| AV1 `cpu` to mkv, decoded back with `av1dec`          | valid stream                                    |
| AV1 file decode through `gstDecoder`                  | 53-54 of 60 frames                              |
| AV1 with `--output-encoder=v4l2` on Xavier            | falls back to the CPU encoder                   |
| AV1 to rtp/rtsp/webrtc/rtmp/avi/flv                   | clean errors                                    |

Not tested yet: anything on Orin, JetPack 6/7, AV1 over RTP/RTSP (no `rtpav1pay` available), hardware AV1 encode/decode, and software AV1 speed with a current libaom (Xavier's libaom 1.0 does ~1 fps).

## Next phases

### Phase 3 - validate on Orin Nano and Orin NX

Run on both boards (JetPack 6):

```
# which AV1 elements are installed
for e in av1enc svtav1enc rav1enc dav1ddec av1dec av1parse rtpav1pay rtpav1depay nvv4l2av1enc; do printf "%-14s" $e; gst-inspect-1.0 $e >/dev/null 2>&1 && echo yes || echo no; done
```

Orin NX - hardware AV1 encode, to mkv and mp4, then decode and count the frames:

```
gst-launch-1.0 -e videotestsrc num-buffers=300 ! video/x-raw,width=1280,height=720,format=I420,framerate=30/1 ! \
    nvvidconv ! 'video/x-raw(memory:NVMM)' ! nvv4l2av1enc bitrate=4000000 idrinterval=30 maxperf-enable=1 ! \
    video/x-av1 ! matroskamux ! filesink location=hw_av1.mkv
```

Orin Nano - software encoder speed with the properties `gstEncoder` sets (drop any that `gst-inspect-1.0 av1enc` doesn't list):

```
time gst-launch-1.0 videotestsrc num-buffers=300 ! video/x-raw,width=1280,height=720,format=I420,framerate=30/1 ! \
    av1enc usage-profile=realtime cpu-used=8 end-usage=cbr target-bitrate=4000 lag-in-frames=0 row-mt=true threads=6 keyframe-max-dist=30 ! fakesink
time gst-launch-1.0 videotestsrc num-buffers=300 ! video/x-raw,width=1280,height=720,format=I420,framerate=30/1 ! \
    x265enc bitrate=4000 speed-preset=ultrafast tune=zerolatency option-string="pools=6" key-int-max=15 ! fakesink
```

Then the H.264/H.265 matrix from the table above with `video-viewer` (`--output-codec`, `--output-encoder=cpu|v4l2`, `rtp://` and `rtsp://`).

Done when:
* H.264/H.265 stream over RTP and RTSP on both boards
* `nvv4l2av1enc` runs on Orin NX - if it fails there, change the hardware AV1 check in `gst_select_encoder()`
* the software fps for H.264, H.265 and AV1 at 720p/1080p on Orin Nano is known, to decide which codecs are usable in realtime

### Phase 4 - AV1 over RTP/RTSP

* build the `rtp` plugin from gst-plugins-rs for JetPack 6, with a gst-plugins-rs release whose bindings support GStreamer 1.20
* add it to `scripts/` or the container build, so `rtpav1pay`/`rtpav1depay` get installed
* check that `av1parse ! rtpav1pay` negotiates with each encoder (`av1enc`, `svtav1enc`, `nvv4l2av1enc`)
* late-joiner test: an RTSP client that connects ~5 s after start has to get a keyframe with a sequence header
* receive side: `rtpav1depay` into `dav1ddec` and into `nvv4l2decoder`
* interop with other clients (VLC, ffplay, GStreamer on a PC)

### Phase 5 - robustness

* `gstEncoder::Close()` waits a fixed 1 s after EOS - wait for the EOS message on the bus (with a timeout) so slow encoders finish writing files (mp4 needs its `moov` atom). RTSP server pipelines may not deliver EOS to that bus, so keep the timeout short there
* fall back to the CPU encoder when a hardware encoder fails to start, not just when it's missing (e.g. `nvv4l2av1enc` exists on Xavier but has no hardware behind it)
* discovering an AV1 file on Xavier and older crashes inside `nvv4l2decoder` - skip it for AV1 on those SoCs
* on GStreamer 1.16 (no leaky `appsrc`) the input queue has no limit, since the `mNeedData` check in `encodeYUV()` is commented out - decide whether to re-enable it there
* `--output-encoder=nvenc` is remapped to `v4l2` after the Orin Nano check in `gst_select_encoder()`, so it would pick a hardware encoder on Orin Nano

### Phase 6 - tuning and options

* command-line options for the encoder preset, keyframe interval and rate control (now hardcoded: x264/x265 `ultrafast` with `key-int-max=15`, AV1 one keyframe per second, `idrinterval=30` on V4L2)
* Orin NX low-latency settings for the hardware encoders: `preset-level`, `control-rate`, `peak-bitrate`, `vbv-size`, and `iframeinterval`/`idrinterval` for AV1
* low-delay settings for `svtav1enc` once its latency is measured
* how many concurrent encode streams one Orin NX NVENC can take

### Phase 7 - other transports, JetPack 7, docs

* AV1 over WebRTC and in MPEG-TS (`rtpmp2ts`) once the GStreamer version has it (JetPack 7 is based on Ubuntu 24.04 with GStreamer 1.24)
* JetPack 7 / Thor: check hardware AV1 encode and which software AV1 elements are installed
* update the codec list and AV1 requirements in jetson-inference `docs/aux-streaming.md`
