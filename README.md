# Vtube - Plugin

> [!WARNING]
> This is focused on be lightweight, run exclusively on iGPU or CPU.

It's a little script to make a expression detect and manipulate this on OBS-Studio.

The focused is be lightweight using:

- [OpenCV](https://opencv.org/) to capture the images.  

- [Mediapipe](https://github.com/google-ai-edge/mediapipe) to make the landmarks used to detect expression by geometric calculus.

- [PyVirtualCam](https://github.com/letmaik/pyvirtualcam) to export the avatar from OBS Studio.

## How Run?

> [!WARNING]
> Do you need configure [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback) to run on Linux.

> [!CAUTION]
> Doesn't tested on **Wayland** based distros

```bash
mise install
poetry run python3 Src/main.py
```

To run on Linux when this is cloned
