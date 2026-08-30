# Encoding is not attention

A counting-task demo that makes the honesty contract *visible*: TRIBE predicts
cortical drive from the pixels, and it will keep driving visual cortex while
a viewer counting passes never notices the unexpected walker. Encoding ≠
attention. The model is not modelling what anyone notices.

**Do not use the published Simons & Chabris gorilla clip, or anyone else's
inattentional-blindness footage.** Copyright aside, we don't need their
pixels. Film a local analog.

## Film the analog (30–60 s, phone, landscape)

- Two to four friends passing a ball, static framing, decent light.
- Tell a naive viewer (not the people in the clip): "count the passes."
- Mid-clip, an unexpected walker crosses the scene for ~4–6 s — a gorilla
  suit if you have one, a loud jacket if you don't.
- Afterwards, ask the viewer: "did you notice anything unusual?" Write down
  their answer yourself. `human_report` is handwritten, always; the model
  never generates it, and an absent report is not a "didn't notice".

## Run it

```bash
./setup_and_run.sh --predict      # once: tribev2 + cu130 torch
videocortex-spark render --video path/to/counting-task.mp4
```

Write `runs/counting-task/events.json` — times are on the **stimulus clock**
(the same clock `--lag-mode stimulus` shows; copy the schema below and adjust
the window):

```bash
cp examples/encode-is-not-attention/events.json runs/counting-task/events.json
$EDITOR runs/counting-task/events.json
```

Then:

```bash
videocortex-spark overlay --run runs/counting-task --events runs/counting-task/events.json
# optional: hear occupancy too
videocortex-spark sonify --run runs/counting-task --video path/to/counting-task.mp4
videocortex-spark overlay --run runs/counting-task --events runs/counting-task/events.json --sonify
```

With `--events`, the overlay gets the caption lower-third (suppress with
`--no-caption`) and the spin ribbon gets amber tick bands for the unexpected
windows. `videocortex-spark draw --events ...` puts the caption on the
contact-sheet footer.

## What you are looking at

Predicted visual-cortex drive continuing through the walker window, against
a possibly empty human report. That contrast is the entire claim. This is not
an attention map, not a "did the subject see it" score, and a quiet fusiform
does not mean nobody saw a face.

A screen-lens / VLM caption that *names* the walker in the video stream is a
deliberate join for a later PR — out of scope here. The human report stays
handwritten.
