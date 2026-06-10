# Geodesic Spline Editor — User Manual

This manual is the practical reference for the interactive editor.  It
walks through every feature you can drive from the mouse, the keyboard,
and the command line.  If you are looking for algorithms, performance
notes, or how the codebase is organised, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) instead — that document
targets developers.

---

## 1. What is this for?

The editor lets you draw smooth curves **on top of a 3-D surface mesh**
(an OBJ / PLY / STL / VTK file).  Unlike the curve tools in Blender,
Maya, or most CAD packages, every point of the curve really lies on the
surface — not floating above concavities, not piercing through ridges,
not having an "estimated" arc-length that is a few millimetres off.
This matters when you need the curve later for:

- Cutting fabric patterns by following the curve on a 3-D scan.
- Planning a CNC tool path on a sculpted model.
- Reverse-engineering measured features (anatomical scans, scanned CAD
  pieces, archaeological artefacts).
- Carbon-fibre layup along a precise geodesic.
- Any workflow where "the distance along the curve" must be exact.

Each curve you place is **a chain of nodes**, with **tangent handles**
that give you C1-continuous control over its shape (same idea as a 2-D
Bézier path, but living on the surface).  You can have several
independent splines in the same session, save and reload them as JSON,
and export the curve to CSV / OBJ / VTK for downstream tools.

---

## 2. Installation

Install the editor's dependencies once.  Python 3.10 or newer.

```bash
pip install -r requirements.txt
# Optional, recommended (50× – 2000× speed-up on hot paths):
pip install -e .[jit]
```

The trickiest dependency is `potpourri3d`: it needs a C++17 compiler
on first install (MSVC 2022 Build Tools on Windows, gcc ≥ 9 on Linux).
Pre-built wheels exist for most platform / Python combinations and pip
will pick them automatically.

---

## 3. Launching the editor

The editor is invoked from a terminal.  Four equivalent forms cover
every starting situation:

```bash
# 1. No argument — opens fandisk.obj if present in the current
#    directory, otherwise falls back to a built-in icosahedron demo.
python geo_splines.py

# 2. Open a specific mesh, blank session.
python geo_splines.py mesh.ply

# 3. Resume a saved session.  The mesh path is read from the JSON's
#    "mesh_file" field, so the editor finds the mesh automatically.
python geo_splines.py session.json

# 4. Resume a session against a different mesh (useful for inspecting
#    the same splines on a re-meshed / higher-resolution surface).
python geo_splines.py session.json hires_mesh.vtk
```

Accepted mesh formats: `.obj`, `.ply`, `.stl`, `.vtk` (and anything
else PyVista's reader recognises).

The editor prints its version banner, the file it just loaded, and a
**console help block** listing every shortcut — handy for skimming
while you find the keys.  The same shortcuts live in a small on-screen
panel that stays visible while you work.

---

## 4. The interface at a glance

When the window opens you see:

- **The mesh**, shaded.
- A **surface cursor** at the mouse position: a small circle on the
  surface with a crosshair pointing along the local axes — this is the
  3-D pick result you can act on (double-click to add a node, etc.).
- A **HUD line** at the top-left in coloured text (status messages:
  "READY", "NODE INSERTED", "REFINED (EXACT)", "GUIDES LOADED ...",
  errors in red).
- The **help panel** at the top-right (narrow column with the same
  shortcut list the console printed).
- Three **checkboxes** at the bottom-left for the curve layers
  (blue / orange / interp) — they mirror the `b` / `o` / `k` keys.

**Camera controls** are standard VTK:

- **Left-drag** on the background: rotate around the focal point.
- **Middle-drag** or **Shift+Left-drag**: pan.
- **Mouse wheel**: zoom.
- **Right-drag**: zoom (or roll, depending on VTK build).
- **R** on the main keyboard is *not* re-bound: PyVista's "reset
  camera" is disabled because the editor uses `R` to rebuild the
  orange layer (see §10).

---

## 5. Your first spline in 5 steps

1. **Place the first node**: double-click left anywhere on the
   surface.  A red sphere appears (the **P node**), with two coloured
   arrow handles (the **A** and **B tangent handles**, green and blue
   respectively).  The HUD says "NODE INSERTED".
2. **Place the second node**: double-click left on another point.  A
   second node appears, plus a smooth curve between the two — three
   versions actually, layered:
     - **Blue** (visible by default): the interactive curve, instant.
     - **Orange** (computed in the background, may take a second to
       appear): the fully-geodesic version.  This is the "final"
       curve.
     - **Black** (toggle with `k`): a B-spline interpolation
       projected onto the surface.
3. **Adjust the node**: drag the red sphere (P).  The curve and its
   tangents follow in real time.  Releasing your mouse fires a
   ~150 ms "refinement" that replaces the live preview with the exact
   geodesic — the HUD flashes "REFINED (EXACT)".
4. **Shape the curve**: drag one of the tangent handles (A or B).
   The other handle stays symmetric (C1 continuity is enforced
   automatically), and the curve reshapes itself.
5. **Save**: press `s`.  A timestamped JSON file is written to the
   current directory; the HUD reports the filename.  Reload it any
   time with `l` (file dialog) or by passing the path on the command
   line.

That is the entire core loop.  Everything below extends or refines it.

---

## 6. Working with nodes

### Adding nodes

- **Double-click left on the surface**: appends a new node to the
  active spline.
- **Double-click left on a curve hover marker**: inserts a new node
  *into* the curve at the exact point under the cursor.  The hover
  marker is a small **telescopic-sight** indicator that appears when
  your cursor is close to any visible curve — see §11.

### Removing nodes

- **Backspace**: removes the last node of the active spline.
- If the active spline is already empty (e.g. after pressing right
  double-click to start a new one), Backspace undoes that "break" and
  returns you to the previous spline.

### Coordinate-precise editing

- **Right double-click on a P sphere**: opens a small dialog where you
  can type the exact `x`, `y`, `z` you want.  A preview shows where
  the entered point projects onto the surface; press OK to commit.

### Undo / Redo

- **Ctrl+Z**: undo the last action.  Up to 50 levels of history are
  kept across **every** kind of spline mutation — adding / removing
  nodes, drag releases, layer toggles, session loads, coordinate
  edits.
- **Ctrl+Y**: redo.

Undo uses snapshot diffing, so even large splines (hundreds of nodes)
undo in a few milliseconds.

---

## 7. Working with tangent handles (A and B)

Each node carries two arrows:

- **A** points "back" along the curve (toward the previous node).
- **B** points "forward" (toward the next node).

The arrows live on the surface, and dragging them rotates the tangent
direction *while keeping the opposite handle symmetric* — so the curve
stays C1-continuous (no kink at the node) without you having to do
anything.

### Standard handle drag

Just grab an arrow and drag.  The hovered arrow turns **black** and
grows slightly; the whole node's gizmo brightens to full opacity
**and pops in front of** any orange / blue / black spline curves
(temporarily raised in the z-buffer so it can't be clipped by an
overlapping curve) so
you can read the geometry at a glance.

### Magnitude-only drag (Shift)

Sometimes you want to change *how long* the tangent is (controls how
"sharp" the bend is at the node) without changing its **direction**.
Hold **Shift** while dragging A or B: the cursor's distance from the
node origin becomes the new tangent length, but the direction is
preserved.  Cross the origin and the tangent flips so the handle
visibly tracks your cursor.

> Vertex / edge snap modifiers (described in the next section) only
> apply to the P node, not to A / B.  Snapping the tangent length to a
> discrete vertex would defeat the smooth-scrub feel.

---

## 8. Snap modifiers — landing on exact mesh features

These modifiers help when you need a node to coincide with a precise
landmark on the mesh.

### Shift + drag P → snap to nearest vertex

Hold **Shift** while dragging a P sphere.  As you move, a **gold
sphere** appears at the closest mesh vertex.  Release the mouse over
the gold sphere and the node lands exactly on that vertex (origin =
vertex position).  The HUD reads `SNAP → vertex <idx>`.

### Ctrl + drag P → snap to nearest edge

Hold **Ctrl** while dragging.  A **cyan sphere** appears on the
closest mesh edge, at the perpendicular foot of your cursor (clamped
to the edge's endpoints).  Released, the node lands exactly on that
edge point.  The HUD shows `SNAP → edge <va>-<vb> t=<0.0–1.0>`.

Edges are real mesh edges, so the node stays exactly on the surface
by construction.  Either modifier disables the live preview
debouncing — every motion is exact.

---

## 9. Multiple splines and closed loops

### Starting a new spline

**Right double-click on empty surface** (not on a node).  An empty
"break" appears at the end of the spline list and becomes the new
active spline.  Add nodes to it as usual.  The previous spline stays
visible but its handles dim slightly to communicate "inactive".

### Switching between splines

Double-click left on any node of another spline to make that spline
active.  All other splines dim; only the active one shows full-colour
tangent handles.

### Closing a spline (looping it)

Once a spline has **3 or more nodes**, press **C**.  The first node's
A handle is reused as a closing tangent toward the last node, the
wrap-around span is drawn, and a fresh empty spline is auto-created so
you can immediately start a new shape without an extra step.  The
HUD reads `LOOP CLOSED + BREAK`.

Press **C** again on a closed spline to re-open it: the closing
tangent and the wrap-around span vanish.  The spline stays selected.

A closed spline needs at least 3 nodes — pressing `C` on shorter
splines is a no-op (the editor warns on the HUD).

---

## 10. The three curve layers

For every span (the section of curve between two consecutive nodes)
the editor maintains three independent representations:

| Layer | Colour | When updated | Purpose |
|---|---|---|---|
| **Blue** | `#a0a0b8` | Interactive (every frame of a drag) | Snappy real-time preview.  Hybrid geodesic Bézier: control polygon + geodesic legs. |
| **Orange** | `#ff8800` | Background workers, ~4–7 s per span | "Final" curve.  Fully geodesic de Casteljau, with phase-2 cascade densification and phase-3 chord-bridging so the polyline really hugs the surface even on coarse meshes. |
| **Black (interp)** | `#000000` | Immediate, debounced after edits | scipy B-spline through the node origins, projected onto the surface.  Independent of the tangent handles — good when you want a curve that just **passes through** the nodes. |

Each layer can be toggled independently:

- **`b`** — blue on/off
- **`o`** — orange on/off
- **`k`** — interp (black) on/off

The same three states are mirrored by checkboxes at the bottom-left.

**`r`** — **rebuild all orange curves** across every spline.  Useful
after toggling layers, after loading a session with many spans, or
after a worker crash.  HUD reports `ORANGE REBUILT` when finished.

While the orange layer is still computing, its spans are drawn in a
**dimmer orange with a dashed pattern**.  When a span finishes, it
flips to solid bright orange.  The HUD shows progress
(`COMPUTING ORANGE 12/40`) and `ORANGE DONE` on completion.

If a span's geodesic solver had to fall back to a straight line
(extreme mesh defects, cross-component segments), that span is
repainted **red** and the HUD warns with `GEODESIC FALLBACK on span
<sid>:<i>` — the curve there is no longer geodesic and you should
either re-route or repair the mesh.

---

## 11. Visual aids

The editor provides several **transient overlays** to make precise
work easier.  None of them are saved to the session JSON; they exist
only for live editing.

### Hover marker (cursor on a curve)

When your cursor is close to any visible curve, a **telescopic-sight
marker** appears at the closest point on that curve: a thin circle
plus a horizontal + vertical crosshair, in the curve's layer colour.
The crosshair always lines up with the screen's horizontal / vertical
axes (true optical-sight feel — independent of the curve's direction
in 3-D).  Double-click left while the marker is visible to insert a
new node at exactly that point.

The marker is drawn on top of the mesh (no z-fighting) but only
appears when the picked point on the curve is genuinely visible from
the camera — points hidden behind the mesh do not get a marker.

### Stitch preview (gray line)

A thin gray line constantly connects the **last node of the active
spline** to your **cursor position on the surface**.  This is what
the next double-click would attach.  When the cursor pauses for
~150 ms it self-refines from a fast vertex-snapped preview to the
exact topologically-inserted geodesic — no extra interaction needed.

The stitch disappears whenever:

- You hover over a node / handle (a different action would happen on
  click).
- You hover over a curve (the telescopic-sight marker is the relevant
  insertion point now).
- The active spline is closed (no "next insertion" makes sense).

### Snap indicator (gold / cyan)

The gold sphere (Shift) and the cyan sphere (Ctrl) described in §8.

### Node-index labels (hold `N`)

Hold the **`N`** key — the editor pops up 1-based number labels above
every visible node.  Single spline: just the node index (`3`).
Multi-spline: prefixed with the 1-based spline index (`s1:3`).
Labels:

- Appear instantly on press; vanish instantly on release.  This is a
  **hold-to-show** shortcut, not a toggle.
- Are drawn on an overlay layer that ignores depth, so they cannot
  get half-clipped by the mesh.
- Are still gated by **occlusion**: a node genuinely on the far side
  of the mesh gets no label, so you only see numbers for nodes the
  camera can actually see.
- Follow nodes that are being dragged.
- Update visibility as you orbit the camera.

### Didactic scaffold (`d`)

Press **`d`** to toggle a four-line scaffold that visualises the de
Casteljau cascade at a chosen parameter `t` along the most recent
span of the active spline.  A slider appears so you can sweep `t`
from 0 to 1 and watch the green construction lines collapse to the
final point on the curve.  Useful for teaching, for diagnosing weird
curve shapes, and for understanding where on the span a given `t`
lives.  Press **`d`** again to hide the scaffold and slider.

### Surface cursor crosshair

Always visible.  A small circle on the mesh under the mouse, aligned
with the local surface frame.  Tells you where a double-click would
land.

---

## 12. Guide curves — auxiliary references

Sometimes the spline you are drawing needs to **align** with curves
you have computed elsewhere: anatomical landmarks scanned separately,
isophotes from a CAD analysis, blueprint annotations, etc.  Import
them as **guide polylines**.

### Loading guides

Press **`Ctrl+X`**.  A multi-select file dialog opens (accepted:
`.vtk` / `.vtp` / `.ply` / `.stl` / `.obj`).  Pick one or several
files.  Each becomes a green polyline overlay drawn over the mesh.

- The loader extracts only **line cells**: triangles or other
  polygons in the same file are silently dropped, so you can point
  it at a mesh file that happens to also contain annotation lines.
- Both `vtkPolyData` and `vtkUnstructuredGrid` container headers are
  accepted (many tools write 1-D data as the latter).  `MultiBlock`
  files are unwrapped to their first relevant block.
- The dialog **replaces** any previously-loaded guides.  To swap a
  set of guides, press `Ctrl+X` again and pick a new selection — no
  separate "clear" command is needed.

### Hold to preview, release to toggle

Press **`x`** (no Ctrl) to **temporarily** show every guide at full
opacity for as long as the key is held — handy when you want to
verify alignment against a curve without losing the ghost-like resting
style.  When you **release**:

- if the guides were *visible* before the press → they **hide**;
- if the guides were *hidden* before the press → they **stay visible**
  and fade smoothly (~500 ms) from full opacity back down to the
  resting `GUIDE_OPACITY`.

So `x` works as a toggle *and* as a "peek" gesture in the same
keystroke — a tap behaves like the classic toggle, a hold gives you
the preview, and the release decides the resting state either way.

If you haven't loaded any guides, the HUD reminds you with `NO GUIDES
LOADED — use Ctrl+X to import`.

> Loading a new set of guides (`Ctrl+X`) always starts them visible
> at the resting opacity, even if you had toggled them off before
> importing — no more "I loaded them but the viewport is empty"
> moment.

### Styling

Guides render in **green** at the resting opacity `GUIDE_OPACITY`
(default `0.1`, ghost-like, so the underlying mesh stays readable)
with **line width 3**.  The fade duration on `x`-release is
`GUIDE_FADE_DURATION_SEC = 0.5`.  All four constants live in
`SplineConfig` (`GUIDE_COLOR_HEX`, `GUIDE_LINE_WIDTH`,
`GUIDE_OPACITY`, `GUIDE_FADE_DURATION_SEC`) if you want to tweak
them.

### Persistence

Guides are **not saved** into the session JSON — they are a "look at
this while I work" tool, not part of the spline geometry.  Re-import
them after each session load.

---

## 13. Display options

### Gizmo opacity cycle (`t`)

The handles (P / A / B) and tangent lines normally render at **0.2
opacity** so they do not occlude the spline curves.  Press **`t`** to
cycle through `0.2 → 0.4 → 0.7 → 1.0 → 0.2`.  HUD reports the new
percentage.

Hovering any handle of a node temporarily bumps **the whole node**
(both arrows + tangent line) to full opacity *and* lifts it in the
z-buffer so it draws on top of any orange / blue / black curve it
might overlap — useful when several splines cross near a node and
the handles you want to grab keep getting hidden behind a curve.
When you move the cursor away the bumped styling lingers for a 300
ms grace period before reverting (so a brief twitch off the handle
doesn't flicker), then snaps back to the cycle's opacity and the
normal z-depth.

### Wireframe overlay (`w`)

Draw the mesh's triangle edges on top of the shaded surface.  Helps
you see where vertex / edge snap will land, and diagnose mesh
density.  Press `w` again to remove.

### Surface opacity (`a`)

Cycles the mesh's own opacity through a few presets (fully opaque /
translucent / very translucent).  Useful when you need to see splines
that pass through concavities or behind folds.

---

## 14. Saving, loading, and exporting

### Save session (`s`)

Press **`s`** to write a **JSON file** to the current directory.  The
name is a timestamp (`20260513_184231.json`) on the first save of the
session; subsequent saves overwrite the same base name with a
numeric suffix (`..._01.json`) so the original is never silently
overwritten.

The save is **atomic**: written to a `.tmp` sibling, fsync'd, then
`os.replace`-d onto the target — you never see a half-written file.

### What the JSON contains

- `version` (`2`): the schema version.
- `mesh_file`: the path or label of the mesh this session was edited
  against.  Used by `l` and the CLI to find the right surface.
- `splines`: a list of objects, each with `closed` (bool) and
  `nodes` (list).  Every node records its 3-D `origin`, both tangent
  endpoints `p_a` / `p_b` (or `null` for placeholders), and an
  optional `id` (1-based, matches the labels under `N`) that the
  loader ignores — you can edit it by hand or delete it without
  consequences.

### Load session (`l`)

Press **`l`** to open a file dialog (defaults to the most recent
`.json` in the current directory).  Pick a JSON; the editor
validates its schema, rebuilds every node + tangent + path, and
launches the orange layer's background workers.

If validation fails (corrupt JSON, NaN coordinates, malformed shape),
the editor refuses the load and the HUD shows a precise error with
line / column information.  Your current state is **not** mutated —
loads are all-or-nothing.

### Export the orange curve to VTK (`v`)

Press **`v`** to write the orange (fully geodesic) curve to a binary
`.vtk` file in the current directory.  Sample count is fixed by
`SplineConfig.EXPORT_VTK_SAMPLES` (default 20 per span).  Single-node
splines are written as `VTK_VERTEX` landmarks.  If the live orange
cache already has the right sample count, the export reuses it (no
re-computation); otherwise it recomputes synchronously.

### Export geodesic paths to TXT (`e`)

Press **`e`** to write a CSV-style text file of every spline's
geodesic paths.  One row per polyline point, with the spline / node
indices alongside.

### Command-line export (`spline_export.py`)

Outside the GUI, `spline_export.py` consumes a session JSON and
prints / writes the requested curve layer.

```bash
# Default: orange curve to stdout as CSV.
python spline_export.py session.json

# Pick a different layer (b = blue, o = orange, k = interp).
python spline_export.py session.json b

# More samples per span (default 60).
python spline_export.py session.json o --samples 120

# Write to a .vtk file (same base name as the JSON).
python spline_export.py session.json --vtk

# Write to a .obj file.
python spline_export.py session.json --obj

# Use a different mesh than the one referenced by the JSON.
python spline_export.py session.json hi_res.vtk

# Combine: high-res mesh + interp layer + dense sampling.
python spline_export.py session.json hi_res.vtk k --samples 200
```

`--obj` and `--vtk` are mutually exclusive.  Without either, output
is CSV on stdout (pipe it where you want).

---

## 15. Troubleshooting

### "It just hangs after Loading mesh: …"

Heavy meshes (hundreds of thousands of triangles, plus non-manifold
defects) make the topology sanitiser take a few seconds.  Wait —
the HUD appears once the sanitiser finishes.  RVP-scale meshes load
in 2–3 s; small / clean meshes in well under a second.  If you keep
waiting more than a minute and there's no progress, kill with Ctrl+C
and check the mesh in MeshLab / Blender.

### "I pressed Ctrl+C and the terminal didn't unblock"

Should not happen on current versions — Intel MKL's Fortran runtime
was previously intercepting the interrupt before Python could clean
up.  If you see it nonetheless, set the environment variable
`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` before launching Python:

```bash
# Windows cmd.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
python geo_splines.py …
```

```bash
# Linux / macOS / Git-Bash
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
python geo_splines.py …
```

### "An orange span is red instead of orange"

The geodesic solver had to fall back to a straight line on that
span.  Causes are usually a mesh defect under the path (sliver
triangle, non-manifold ridge, disconnected components).  Inspect the
mesh in that region; consider re-meshing or moving the offending
node.

### "Orange curves never finish"

A worker may have died (rare; cross-platform multiprocessing edge
case).  Press **`r`** to rebuild all orange curves — that re-spawns
the worker pool.

### "Node labels won't appear"

The `N` key is **hold-to-show**, not a toggle.  Press and hold; the
labels appear on press, disappear on release.

### "I imported wires.vtk but nothing showed up"

If you see `GUIDES LOADED (1 file(s), N segments)` in green on the
HUD but no green lines on screen, the guides may be far from the
camera frustum.  Press the camera's "reset" middle-click or orbit
out — the wires might live in a different coordinate space than the
loaded mesh.

### "I want a smaller / larger hover marker"

`SplineConfig.HOVER_MARKER_SCREEN_SCALE` (default `0.006`) scales the
telescopic-sight radius; line widths are
`HOVER_MARKER_CIRCLE_LINE_WIDTH` (default 2) and
`HOVER_MARKER_CROSS_LINE_WIDTH` (default 1).  All are at the top of
`geo_splines.py` if you want to tweak.

---

## 16. Keyboard quick reference

### Editing

| Key | Action |
|---|---|
| **Double-click L** on surface | Add node to active spline |
| **Double-click L** on hover marker | Insert node into curve at hover point |
| **Double-click L** on another spline's node | Switch active spline |
| **Drag P** (red sphere) | Translate node |
| **Drag A / B** (handles) | Rotate tangent (other handle stays symmetric) |
| **Shift + drag P** | Snap node to nearest mesh vertex (gold indicator) |
| **Ctrl + drag P** | Snap node to nearest mesh edge (cyan indicator) |
| **Shift + drag A / B** | Magnitude-only: keep tangent direction, change length |
| **Double-click R** on surface | New (empty) spline / break |
| **Double-click R** on P | Open coordinate-edit dialog |
| **Backspace** | Remove last node, or undo last "break" |
| **Ctrl + Z** | Undo |
| **Ctrl + Y** | Redo |
| **C** | Close / re-open active spline (needs ≥ 3 nodes to close) |

### Curve layers + display

| Key | Action |
|---|---|
| **b** | Toggle blue (interactive) curve visibility |
| **o** | Toggle orange (geodesic) curve visibility |
| **k** | Toggle interp (B-spline) curve visibility |
| **r** | Rebuild all orange curves |
| **t** | Cycle gizmo opacity (20% → 40% → 70% → 100% → 20%) |
| **w** | Toggle mesh wireframe overlay |
| **a** | Cycle surface opacity |

### Visual aids

| Key | Action |
|---|---|
| **n** (hold) | Show node-index labels while pressed |
| **d** | Toggle didactic de Casteljau scaffold |
| **Ctrl + X** | Import guide polylines (file dialog).  Always loaded visible at resting opacity. |
| **x** (hold) | While held: guides at full opacity.  On release: toggles between hidden and visible (with a 500 ms fade back to resting opacity when toggling on). |

### Session + export

| Key | Action |
|---|---|
| **s** | Save session to timestamped JSON |
| **l** | Load session from JSON (file dialog) |
| **v** | Export orange curve to timestamped binary VTK |
| **e** | Export geodesic paths to TXT |

### Camera

| Action | Result |
|---|---|
| Left-drag (background) | Rotate around focal point |
| Middle-drag or Shift+Left-drag | Pan |
| Mouse wheel | Zoom |
| Right-drag | Zoom (or roll, depending on VTK build) |

---

## 17. Where to go next

- **Developers** wanting to extend the editor or understand
  algorithms: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — covers
  JIT kernels, the master-clock debounce, worker pipeline, etc.
  The JSON schema is documented in the [`README.md`](README.md).
- **CLI batch export**: see §14 above and `spline_export.py --help`.
- **Bug reports / feature requests**: file an issue with a minimal
  session JSON + the mesh, plus the HUD message at the moment things
  went wrong.

Happy splining.
