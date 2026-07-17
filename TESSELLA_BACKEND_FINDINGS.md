# NumPy backend: RGB vs RGBX findings (from tessella benchmarking)

Measured on a mid-range laptop (not a workstation — absolute ms will differ,
relative ordering holds). All numbers median-of-N, smooth synthetic image,
level-90 JPEG / level-50 JP2.

## The decision rule (this is the whole thing)

One neutral hint, `prefer_rgbx`, threaded down the read stack. The decoder ANDs
it with its **own** capability and **never learns who the pixels are for** — the
codec decision stays local to the codec; "is this for PIL?" does not leak down.

```python
def decode_to_array(frame, *, prefer_rgbx=False):
    if prefer_rgbx and self.can_output_rgbx_cheaply:   # e.g. JPEG via EXT_RGBX
        return decode_rgbx(frame)   # (H, W, 4)
    return decode_rgb(frame)        # (H, W, 3)
```

The caller sets the hint from what it will do with the result — that is where
(and the *only* place) output-format knowledge lives:

```
read_region       (PIL out)   -> prefer_rgbx = True    # will frombuffer, 0-copy
read_region_array (numpy out) -> prefer_rgbx = False   # wants compact RGB
```

So the old `codec_can_output_rgbx_cheaply and output_is_PIL` collapses: the left
term is the decoder's internal `and`, the right term is just how the caller sets
`prefer_rgbx`. The resulting array's **channel count (3 vs 4) then drives the
boundary automatically** — `frombuffer` if 4-byte, `fromarray` if 3 — so nothing
below the decoder needs the flag either. A JP2 decoder simply reports
`can_output_rgbx_cheaply = False` and returns RGB even when `prefer_rgbx=True`;
the caller still gets a correct image, just via `fromarray`.

It's **one bool on the decode call**, not an abstraction. tessella already
supports c=3 and c=4, so no tessella change is needed — the array flows through
`compose`/`downscale` at whatever channel count the decoder produced.

## Why: the numbers

### JPEG -> PIL (2048x2048)
| path | ms |
|---|---|
| PIL decodes JPEG directly (RGBX-direct internally) | 8.40 |
| imcodecs decode -> RGB | 5.75 |
| imcodecs decode -> **RGBX direct** (`outcolorspace=EXT_RGBX`) | 6.02 (+0.27 over RGB — expand **free**, fused into libjpeg) |
| imcodecs RGB + separate expand pass | 8.27 (only *matches* PIL) |
| **imcodecs RGBX-direct -> `frombuffer`** | **6.44 (beats PIL ~23%)** |

Takeaway: for JPEG, decode directly to RGBX. libjpeg fuses the RGB->RGBX
expand into decode for ~free; a *separate* expand pass just matches PIL.

### JP2 -> PIL (2048x2048)
| path | ms |
|---|---|
| PIL decodes JP2 directly | 102.6 |
| **imcodecs decode + `Image.fromarray`** | **85.7 (beats PIL ~17%)** |
| imcodecs decode + fast expand + frombuffer | 84.5 (only ~1% more) |

Takeaway: JP2 has no RGBX decode option, but imcodecs' OpenJPEG beats PIL's
anyway. Decode dominates (~80 ms); the RGB->RGBX expand is ~2% noise, so plain
`fromarray` is fine — no fast-expand kernel worth its maintenance here.

## Concrete calls to use

```python
import imagecodecs
from PIL import Image

# JPEG + PIL out  -> RGBX-direct decode, zero-copy PIL wrap
RGBX = imagecodecs.JPEG8.CS.EXT_RGBX
rgbx = imagecodecs.jpeg8_decode(data, outcolorspace=RGBX)   # (H,W,4)
img  = Image.frombuffer("RGB", (w, h), rgbx, "raw", "RGBX", 0, 1)  # free

# JP2 + PIL out   -> RGB decode, fromarray
rgb  = imagecodecs.jpeg2k_decode(data)                      # (H,W,3)
img  = Image.fromarray(rgb)

# any + numpy out -> just return the RGB array; no PIL, no expand
```

## Assumption to keep in mind

RGBX carries a **~33% bandwidth tax** through every op (the 4th byte rides
along). The rule picks RGBX only for **shallow** pipelines (stitch = memcpy,
maybe one downscale), where the free decode-expand + free frombuffer beat the
tax. If a **deep** multi-resample RGBX path ever appears, reconsider vs
"RGB through the ops + one expand at the PIL edge".

Edge cases the rule already handles: subsampled/planar YCbCr never "outputs
RGBX cheaply" (needs color conversion first) so it falls to RGB; grayscale
degenerates to native. The `_cheaply` qualifier gates both for free.

## tessella status (dep, unchanged by this)

- Supports planar (H,W) and interleaved HWC (H,W,C), c=3 and c=4, dtypes
  uint8/uint16/float32. Pass RGB or RGBX arrays through `compose`/`downscale`
  directly.
- `tessella.rgb_to_rgbx` exists (fast RGB->RGBX, ~1.7x vs fromarray in
  isolation) but is **dead under this rule** — JPEG uses EXT_RGBX decode, JP2
  uses fromarray. Kept for now; only earns its keep if a hot *decode-less*
  raw-RGB -> PIL path shows up (WSI pipelines don't have one).

---

# Follow-up validation — full pipeline, workstation, multi-threaded (branch `numpy-backend`)

The above rule was validated end-to-end in the actual wsidicom read path, on a
16-core box, real WG26 Histech JPEG-baseline slide (**tile size 1024**, so a
compose-heavy read is **8192² = 64 tiles**, not 2048²). It holds, and the
threaded numbers make the case stronger than the laptop single-shot ones.

## What the full pipeline adds to the rule

1. **The win grows with threads** (the laptop single-thread numbers miss this).
   Isolated same-decode A/B — imagecodecs decode, PIL `paste` vs numpy
   RGBX-slice-write + `frombuffer` — 64 tiles, max pixel diff 0:

   | threads | PIL paste | numpy RGBX+frombuffer | speedup |
   |---|---|---|---|
   | 1 | 442 | 368 | 1.20× |
   | 4 | 173 | 134 | 1.29× |
   | 8 | 144 | 94 | **1.53×** |

   The compose is GIL-free (numpy slice-write / tessella), so parallel reads
   pull further ahead. `frombuffer` keeps the boundary at 0 ms at every width.

2. **In the RGB (non-RGBX) pipeline the boundary dominates and it shows as a
   regression** — exactly what the rule predicts for `output_is_PIL`. Full read
   path, default Pillow decoder, 8192², warm @ 8 threads: baseline PIL 115 ms,
   numpy-RGB-with-PIL-boundary 147 ms (1.28× **slower**). The culprit is the
   serial `fromarray(RGB 8192²)` = 70–79 ms (measured), larger than the whole
   201 MB compose (~27 ms). Confirms: RGB internal + PIL out = lose.

3. **The `numpy out -> RGB` arm is real and already wins.** Added a public
   `WsiDicom.read_region_array` (numpy sibling of `read_region`): 87 ms vs
   baseline PIL 115 ms @ 8 threads = **1.3× faster**, pixel-identical, plain RGB.
   Boundary tax avoided = 53–60 ms. A numpy consumer never pays the edge.

4. **Decoder parity caveat for the JP2 arm.** imagecodecs' OpenJPEG vs Pillow:
   JPEG-baseline and CMU-1-JP2K bit-identical, but **JP2K-33003-1 differs by ±1**
   (0.64 RMS, 41% px — lossy rounding). Fine visually, but it makes exact-md5
   goldens decoder-dependent, so the wsidicom **unit tests stay on Pillow**; the
   global decoder default is not shifted. (Doesn't affect the rule — just means
   "JP2 -> imcodecs" is opt-in, not the default, in wsidicom.)

## EXT_RGBX vs the `rgba` shortcut

The isolated numpy bench above used `jpeg_decode(outcolorspace='rgba')` (also
4-byte, +0 cost) — equivalent for the frombuffer trick, but the write-up's
`JPEG8.CS.EXT_RGBX` is the canonical fused-expand path and the one to wire.

## Integration status (what's built vs what the rule still needs)

**Built + green** (1862 unit, 449 integration, md5-identical; wsidicomizer 172
convert tests green):
- Numpy-internal pipeline: `Decoder.decode_to_array`/`array_to_image`,
  `DecodedFrameCache` holds arrays, `NumpyStitcher` (GIL-free), shared
  `WsiInstance._assemble_region` → `get_region` (PIL) + `get_region_array`
  (numpy), public `read_region_array`.
- wsidicomizer opentile/czi/bioformats expose `get_decoded_tile_array` (native
  array, no numpy↔PIL round-trip). openslide already yields 4-byte RGBA.

**NOT yet built — this is exactly the `prefer_rgbx` hint:** the decode call is
not yet parameterised by `prefer_rgbx`. Today decode always yields RGB and the
PIL boundary always `fromarray`s. Wiring it — `read_region` passes
`prefer_rgbx=True` → `decode_to_array(prefer_rgbx=True)` → EXT_RGBX for JPEG →
RGBX canvas → `array_to_image` dispatches to `frombuffer` on the 4-byte array —
is the change that flips PIL reads from 1.3× slower to ~1.5× faster. The array's
channel count carries the rest, so only the decode call and `array_to_image`
need touching. tessella needs no change — c=3 or c=4 flows through
`compose`/`downscale` unchanged.

**Remaining PIL touchpoint:** the downsampler. Still Pillow; the tessella
downscaler (2–8×) removes it and wins downscaled/thumbnail reads (today a
wash/regression because they `fromarray` the full canvas to feed Pillow).
