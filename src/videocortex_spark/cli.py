"""Command line interface.

Eight verbs:

    doctor   what's broken before you waste an hour finding out
    fetch    pull every weight up front
    render   video -> brain-map stills
    draw     saved predictions -> brain-map stills (no model needed)
    overlay  saved run + source video -> PIP animation
    sonify   saved run -> cortex.wav (|predicted BOLD| as loudness)
    export   saved predictions -> one self-contained interactive 3-D brain
    serve    loopback command deck
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

from videocortex_spark import __version__
from videocortex_spark.config import (
    HF_CHECKPOINT_REPO,
    OverlayConfig,
    RenderConfig,
    RunConfig,
    VIEW_PRESETS,
)
from videocortex_spark.spark import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FEATURE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    VALID_DEVICES,
    load_local_env,
)
from videocortex_spark.web import DEFAULT_PORT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videocortex-spark",
        description="Video in, cortical activation maps out (Meta TRIBE v2) on NVIDIA DGX Spark.",
    )
    p.add_argument("--version", action="version", version=f"videocortex-spark {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    # -- doctor ------------------------------------------------------------
    d = sub.add_parser("doctor", help="check this machine can run the model")
    d.add_argument(
        "--offline", action="store_true", help="skip the HuggingFace reachability checks"
    )
    d.add_argument(
        "--renderer",
        action="store_true",
        help="draw path only — skip torch, tribev2, ffmpeg, uvx, disk, HuggingFace",
    )

    # -- fetch -------------------------------------------------------------
    f = sub.add_parser("fetch", help="pre-download weights and surfaces")
    f.add_argument("--surfaces-only", action="store_true")
    f.add_argument("--cache-dir", type=Path, default=None)

    # -- render ------------------------------------------------------------
    r = sub.add_parser("render", help="run the model on a stimulus and draw the result")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--audio", type=Path)
    src.add_argument("--text", type=Path)
    r.add_argument("-o", "--out", type=Path, default=Path("runs"))
    r.add_argument(
        "--device", choices=VALID_DEVICES, default="auto"
    )
    # RunConfig is a slots dataclass: RunConfig.checkpoint is a member_descriptor,
    # not the default string. Passing that to argparse is how you get
    # Path(member_descriptor) at load time.
    r.add_argument("--checkpoint", default=HF_CHECKPOINT_REPO)
    r.add_argument("--cache-dir", type=Path, default=Path(".videocortex-spark-cache"))
    r.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    r.add_argument("--feature-batch-size", type=int, default=DEFAULT_FEATURE_BATCH_SIZE)
    r.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    r.add_argument("--no-save-predictions", action="store_true")
    _add_render_args(r)

    # -- draw --------------------------------------------------------------
    w = sub.add_parser("draw", help="re-render a saved predictions.npy")
    w.add_argument("predictions", type=Path)
    w.add_argument("-o", "--out", type=Path, default=None)
    w.add_argument(
        "--events", type=Path, default=None,
        help="videocortex.events.v1 JSON — prints the honesty caption on the "
             "contact sheet (draw has no ribbon, so no ticks)",
    )
    w.add_argument(
        "--no-caption", action="store_true",
        help="suppress the events caption",
    )
    _add_render_args(w)

    # -- overlay -----------------------------------------------------------
    o = sub.add_parser(
        "overlay",
        help="composite a PIP brain animation onto the source video",
    )
    o.add_argument(
        "--run", type=Path, required=True,
        help="run directory with predictions.npy + timestamps.npy",
    )
    o.add_argument(
        "--video", type=Path, default=None,
        help="source video (default: manifest.run.video)",
    )
    o.add_argument("-o", "--out", type=Path, default=None)
    o.add_argument(
        "--views", choices=sorted(VIEW_PRESETS), default="standard",
        help="PIP card views (default: standard 2×2)",
    )
    o.add_argument("--cmap", default=None)
    o.add_argument("--percentile", type=float, default=None)
    o.add_argument("--threshold-frac", type=float, default=None)
    o.add_argument(
        "--size", type=float, default=0.24,
        help="PIP width as a fraction of the frame (landscape)",
    )
    o.add_argument(
        "--position", choices=("top-right", "top-left"), default="top-right",
    )
    o.add_argument("--label", choices=("time", "tr", "none"), default="time")
    o.add_argument(
        "--lag-mode", choices=("stimulus", "scanner"), default="stimulus",
        help="stimulus = match the picture (default); scanner = +5 s BOLD delay",
    )
    o.add_argument("--stride", type=int, default=1, help="draft: every Nth TR")
    o.add_argument("--fast", action="store_true", help="NVENC encode (falls back to libx264)")
    o.add_argument("--force", action="store_true", help="rebuild PIP cards")
    o.add_argument("--light", action="store_true")
    o.add_argument(
        "--spin", action="store_true",
        help="3D inflated globe in the PIP instead of the 2×2 card",
    )
    o.add_argument(
        "--dps", type=float, default=24.0,
        help="spin yaw rate in degrees per second (12–48, default 24)",
    )
    o.add_argument(
        "--fps", type=float, default=24.0,
        help="spin overlay frame rate (default 24)",
    )
    o.add_argument(
        "--az-step", type=int, default=2, dest="az_step",
        help="atlas yaw step in degrees (default 2; smaller is smoother, slower bake)",
    )
    o.add_argument(
        "--no-monitor", action="store_true",
        help="skip the black plate and green medical-monitor bezel",
    )
    o.add_argument(
        "--ramp-frac", type=float, default=0.5, dest="ramp_frac",
        help="soft-threshold ramp width as a fraction of threshold (0 = hard edge)",
    )
    o.add_argument(
        "--no-ribbon", action="store_true",
        help="skip the energy curve + playhead under the PIP",
    )
    o.add_argument(
        "--no-regions", action="store_true",
        help="skip Destrieux region names (also skipped automatically if the "
             "atlas cannot be fetched)",
    )
    o.add_argument(
        "--events", type=Path, default=None,
        help="videocortex.events.v1 JSON — unexpected windows on the stimulus "
             "clock; ribbon ticks (spin) + the honesty caption",
    )
    o.add_argument(
        "--no-caption", action="store_true",
        help="suppress the events caption lower-third",
    )
    o.add_argument(
        "--sonify", action="store_true",
        help="mix cortex.wav under the original audio (ducked, limited)",
    )
    o.add_argument(
        "--sonify-only", action="store_true", dest="sonify_only",
        help="replace the original audio with cortex.wav",
    )

    # -- sonify ------------------------------------------------------------
    y = sub.add_parser(
        "sonify",
        help="turn a saved run into cortex.wav — |predicted BOLD| as loudness",
        epilog="Voices: occipital (L, 196 Hz), fusiform (C, 294 Hz), "
               "parahippocampal (R, 440 Hz) over a whole-brain bed — "
               "Destrieux stand-ins, not a localizer. This is |predicted "
               "BOLD| as loudness, not 'what the brain sounds like'.",
    )
    y.add_argument(
        "--run", type=Path, required=True,
        help="run directory with predictions.npy + timestamps.npy",
    )
    y.add_argument(
        "--video", type=Path, default=None,
        help="source video — sets the wav duration (default: TR clock)",
    )
    y.add_argument("-o", "--out", type=Path, default=None,
                   help="default: <run>/cortex.wav")
    y.add_argument(
        "--lag-mode", choices=("stimulus", "scanner"), default="stimulus",
        help="stimulus = match the picture (default); scanner = +5 s BOLD delay",
    )
    y.add_argument("--percentile", type=float, default=99.0)
    y.add_argument("--threshold-frac", type=float, default=0.25)

    # -- export ------------------------------------------------------------
    x = sub.add_parser(
        "export",
        help="one self-contained interactive 3-D brain (brain.html) — no model needed",
    )
    x.add_argument(
        "predictions", type=Path, nargs="?", default=None,
        help="predictions.npy (default: the --run's)",
    )
    x.add_argument(
        "--run", type=Path, default=None,
        help="run directory; reads timestamps.npy and manifest.json render defaults",
    )
    x.add_argument("-o", "--out", type=Path, default=None,
                   help="default: <run>/brain.html")
    x.add_argument("--cmap", default=None)
    x.add_argument("--percentile", type=float, default=None)
    x.add_argument("--threshold-frac", type=float, default=None)
    x.add_argument("--ramp-frac", type=float, default=None)
    x.add_argument("--no-regions", action="store_true",
                   help="skip the Destrieux region table (also skipped automatically "
                        "if the atlas cannot be fetched)")

    # -- serve -------------------------------------------------------------
    s = sub.add_parser("serve", help="loopback command deck (stdlib HTTP)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument(
        "--runs", type=Path, default=None,
        help="runs directory (default: ./runs)",
    )
    s.add_argument(
        "--no-browser", action="store_true",
        help="do not open a browser tab on start",
    )

    return p


def _add_render_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("rendering")
    g.add_argument("--views", choices=sorted(VIEW_PRESETS), default="standard")
    g.add_argument("--cmap", default="cold_hot")
    g.add_argument("--percentile", type=float, default=99.0)
    g.add_argument("--threshold-frac", type=float, default=0.25)
    g.add_argument("--stride", type=int, default=1, help="render every Nth TR")
    g.add_argument("--max-frames", type=int, default=60)
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument(
        "--cols", type=int, default=None,
        help="contact sheet columns (default: auto, targets 16:9)",
    )
    g.add_argument("--no-contact-sheet", action="store_true")
    g.add_argument("--light", action="store_true", help="light background")
    g.add_argument(
        "--ramp-frac", type=float, default=0.5, dest="ramp_frac",
        help="soft-threshold ramp width as a fraction of threshold (0 = hard edge)",
    )
    g.add_argument(
        "--no-filmstrip", action="store_true",
        help="no stimulus frames above contact-sheet tiles",
    )
    g.add_argument("--no-regions", action="store_true",
                   help="no Destrieux region names on the plates")


def _render_cfg(a: argparse.Namespace) -> RenderConfig:
    return RenderConfig(
        views=a.views,
        cmap=a.cmap,
        percentile=a.percentile,
        threshold_frac=a.threshold_frac,
        ramp_frac=a.ramp_frac,
        stride=a.stride,
        max_frames=a.max_frames,
        dpi=a.dpi,
        contact_sheet=not a.no_contact_sheet,
        contact_sheet_cols=a.cols,
        darkbg=not a.light,
        filmstrip=not a.no_filmstrip,
        regions=not a.no_regions,
    )


def cmd_doctor(a: argparse.Namespace) -> int:
    from videocortex_spark import doctor

    checks = doctor.run_all(network=not a.offline, model=not a.renderer)
    label = "preflight (renderer)" if a.renderer else "preflight"
    print(f"\nvideocortex-spark {__version__} — {label}\n")
    print(doctor.format_report(checks))
    print()
    return doctor.exit_code(checks)


def cmd_fetch(a: argparse.Namespace) -> int:
    from videocortex_spark import weights

    print("checking fsaverage5 surfaces (bundled with nilearn)...")
    print(f"  -> {weights.fetch_surfaces()}")
    if a.surfaces_only:
        return 0

    print("fetching TRIBE v2 checkpoint (~0.7 GB)...")
    for p in weights.fetch_checkpoint(a.cache_dir):
        print(f"  -> {p}")

    print("fetching frozen feature extractors (~15-20 GB)...")
    results = weights.fetch_encoders(a.cache_dir)
    failed = False
    for modality, status in results.items():
        mark = "✓" if status == "ok" else "✗"
        print(f"  {mark} {modality}: {status}")
        failed |= status != "ok"
    if failed:
        print(
            "\nSome encoders did not download. Llama-3.2-3B is gated: accept the\n"
            "licence at https://huggingface.co/meta-llama/Llama-3.2-3B, then "
            "`hf auth login`."
        )
    return 1 if failed else 0


def cmd_render(a: argparse.Namespace) -> int:
    from videocortex_spark import pipeline

    run_cfg = RunConfig(
        video=a.video,
        audio=a.audio,
        text=a.text,
        out_dir=a.out,
        device=a.device,
        checkpoint=a.checkpoint,
        cache_dir=a.cache_dir,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        feature_batch_size=a.feature_batch_size,
        save_predictions=not a.no_save_predictions,
    )
    result = pipeline.run(run_cfg, _render_cfg(a))
    _report(result)
    return 0


def cmd_draw(a: argparse.Namespace) -> int:
    from videocortex_spark import pipeline

    cfg = _render_cfg(a)
    if a.events is not None:
        from videocortex_spark.events import CAPTION, EventsError, load_events

        try:
            load_events(a.events)  # validate now — not after a render
        except EventsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if not a.no_caption:
            cfg.caption = CAPTION
    out = a.out or a.predictions.parent
    result = pipeline.render_only(a.predictions, out, cfg)
    _report(result)
    return 0


def cmd_sonify(a: argparse.Namespace) -> int:
    from videocortex_spark.sonify import SonifyError, sonify_from_run

    try:
        result = sonify_from_run(
            a.run,
            video=a.video,
            lag_mode=a.lag_mode,
            percentile=a.percentile,
            threshold_frac=a.threshold_frac,
            out=a.out,
        )
    except SonifyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    voices = ", ".join(result.voices) if result.voices else "whole-brain bed only"
    print(f"\n  cortex wav ->  {result.wav}  ({result.duration:.1f}s, {voices})")
    print(f"  tracks     ->  {result.tracks}")
    print("  |predicted BOLD| as loudness — not what a brain sounds like.\n")
    return 0


def _manifest_render_defaults(man_path: Path) -> dict:
    """The ``render`` block of a run's manifest.json, or {} — so export
    inherits the colour choices the run was actually made with."""
    if not man_path.is_file():
        return {}
    import json

    try:
        return json.loads(man_path.read_text(encoding="utf-8")).get("render") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def cmd_export(a: argparse.Namespace) -> int:
    import numpy as np

    from videocortex_spark.export import export_viewer

    if a.run is None and a.predictions is None:
        print("ERROR: give --run DIR or a predictions.npy path", file=sys.stderr)
        return 2
    side = a.run if a.run is not None else a.predictions.parent
    pred_path = (a.run / "predictions.npy") if a.run is not None else a.predictions
    if not pred_path.is_file():
        print(f"ERROR: no predictions at {pred_path}", file=sys.stderr)
        return 1

    defaults = _manifest_render_defaults(side / "manifest.json")
    cmap = a.cmap or defaults.get("cmap") or "cold_hot"
    percentile = a.percentile
    if percentile is None:
        percentile = float(defaults.get("percentile") or 99.0)
    threshold_frac = a.threshold_frac
    if threshold_frac is None:
        threshold_frac = float(defaults.get("threshold_frac") or 0.25)
    ramp_frac = a.ramp_frac
    if ramp_frac is None:
        ramp_frac = float(defaults.get("ramp_frac", 0.5))

    ts_path = side / "timestamps.npy"
    timestamps = None
    if ts_path.is_file():
        timestamps = [float(t) for t in np.load(ts_path)]

    out = a.out or side / "brain.html"
    preds = np.load(pred_path)
    result = export_viewer(
        preds,
        out,
        timestamps=timestamps,
        title=side.name,
        cmap=cmap,
        percentile=percentile,
        threshold_frac=threshold_frac,
        ramp_frac=ramp_frac,
        regions=not a.no_regions,
        progress=lambda msg: print(f"  … {msg}", flush=True),
    )
    print(f"\n  3-D viewer  ->  {result.path}  ({result.n_bytes / 1e6:.1f} MB)")
    print(f"  {result.n_tr} TRs × {result.n_vertices} vertices"
          f"{'' if result.regions else ' (no region table)'}"
          " — open it in any browser, it needs nothing else\n")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from videocortex_spark.web.server import serve

    serve(
        host=a.host,
        port=a.port,
        open_browser=not a.no_browser,
        runs_dir=a.runs,
    )
    return 0


def cmd_overlay(a: argparse.Namespace) -> int:
    from videocortex_spark.overlay import OverlayError, overlay_from_run

    cmap = a.cmap
    percentile = a.percentile
    threshold_frac = a.threshold_frac
    man = a.run / "manifest.json"
    if man.is_file() and (cmap is None or percentile is None or threshold_frac is None):
        import json

        render = json.loads(man.read_text(encoding="utf-8")).get("render") or {}
        cmap = cmap or render.get("cmap") or "cold_hot"
        if percentile is None:
            percentile = float(render.get("percentile") or 99.0)
        if threshold_frac is None:
            threshold_frac = float(render.get("threshold_frac") or 0.25)
    if a.spin and a.views != "standard":
        print(
            "warning: --views is ignored with --spin (one joined brain, not a 2×2)",
            file=sys.stderr,
        )
    dps = a.dps
    if dps < 12 or dps > 48:
        print("warning: --dps clamped to 12–48", file=sys.stderr)
        dps = min(48.0, max(12.0, dps))
    cfg = OverlayConfig(
        views=a.views,
        cmap=cmap or "cold_hot",
        percentile=99.0 if percentile is None else percentile,
        threshold_frac=0.25 if threshold_frac is None else threshold_frac,
        ramp_frac=a.ramp_frac,
        darkbg=not a.light,
        size=a.size,
        position=a.position,
        label=a.label,
        lag_mode=a.lag_mode,
        stride=a.stride,
        fast=a.fast,
        force=a.force,
        spin=a.spin,
        dps=dps,
        fps=max(8.0, float(a.fps)),
        az_step=max(1, int(a.az_step)),
        monitor=not a.no_monitor,
        ribbon=not a.no_ribbon,
        regions=not a.no_regions,
        events=a.events,
        caption=not a.no_caption,
        sonify=a.sonify,
        sonify_only=a.sonify_only,
    )
    try:
        result = overlay_from_run(
            a.run, cfg, video=a.video, out=a.out, progress=_progress_overlay,
        )
    except OverlayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    kind = "spin frames" if a.spin else "PIP cards"
    print(f"\n  {kind}  ->  {result.pip_dir}  ({result.n_cards})")
    print(f"  overlay    ->  {result.out}")
    print(f"  {result.n_cards} in {result.seconds:.1f}s\n")
    return 0


def _progress_overlay(i: int, n: int) -> None:
    end = "\n" if i == n else "\r"
    print(f"  PIP cards {i}/{n}", end=end, flush=True)


def _report(result) -> None:
    print(f"\n  {len(result.frames)} frames  ->  {result.out_dir / 'frames'}")
    if result.contact_sheet:
        print(f"  contact sheet ->  {result.contact_sheet}")
    if result.predictions:
        print(f"  predictions   ->  {result.predictions}")
    print(f"  {result.n_timesteps} TRs in {result.seconds:.1f}s\n")


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        # httpx narrates every hub request at INFO, which drowns our own output.
        for noisy in ("httpx", "urllib3", "filelock", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # tribev2 imports neuralset, which warns on import before any events exist.
        logging.getLogger("neuralset").setLevel(logging.ERROR)
        logging.getLogger("neuralset.extractors.base").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", module=r"neuralset(\.|$)")
    handlers = {
        "doctor": cmd_doctor,
        "fetch": cmd_fetch,
        "render": cmd_render,
        "draw": cmd_draw,
        "overlay": cmd_overlay,
        "sonify": cmd_sonify,
        "export": cmd_export,
        "serve": cmd_serve,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
