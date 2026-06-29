#!/usr/bin/env python
import argparse
import ctypes.util
import os
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TorchCodec and FFmpeg shared library visibility.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when TorchCodec cannot load.")
    parser.add_argument("--prefix", default="[torchcodec]", help="Log prefix.")
    args = parser.parse_args()

    print(f"{args.prefix} python={sys.executable}")
    print(f"{args.prefix} ld_library_path={os.environ.get('LD_LIBRARY_PATH', '')}")

    for name in ["avutil", "avcodec", "avformat", "avdevice", "avfilter", "swscale", "swresample"]:
        print(f"{args.prefix} find_library.{name}={ctypes.util.find_library(name)}")

    try:
        import av

        print(f"{args.prefix} pyav={getattr(av, '__version__', 'unknown')}")
        print(f"{args.prefix} pyav_library_versions={getattr(av, 'library_versions', {})}")
    except Exception as exc:
        print(f"{args.prefix} pyav_error={type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"{args.prefix} torch={torch.__version__}")
    except Exception as exc:
        print(f"{args.prefix} torch_error={type(exc).__name__}: {exc}")

    try:
        import torchcodec

        print(f"{args.prefix} torchcodec={getattr(torchcodec, '__version__', 'unknown')}")
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        print(f"{args.prefix} ok=1")
        return 0
    except Exception as exc:
        print(f"{args.prefix} ok=0 error={type(exc).__name__}: {exc}")
        print(f"{args.prefix} traceback_start")
        traceback.print_exc()
        print(f"{args.prefix} traceback_end")
        return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
