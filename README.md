# HyperPoseDemo

Experiments in pose matching and motion blending, built on
[ozz-animation](https://github.com/guillaumeblanc/ozz-animation) and
[Dear ImGui](https://github.com/ocornut/imgui).

The main app, **CommonPoses**, finds the most common poses in a set of motion
capture clips and builds a transition graph between them. A character can then
walk that graph, blending from clip to clip.

## Apps

| App (source) | What it does |
|---|---|
| **CommonPoses** (`src/common_poses.cpp`) | Clusters every frame of a set of clips into common poses (k-means on root relative, heading aligned joint positions), then builds the transitions between them. Browse poses and their frames, inspect the transition graph, and run a *pose walk*: a random walk over the graph, crossfading between clip segments. Optional root motion, and a *best match* blend that places each blend at the closest pair of frames (joint positions and velocities) near the transition. Pose sets, per clip trimming and settings are saved as `.hpproject` files. |
| **PoseBlend** (`src/pose_blend.cpp`) | Compares every frame of clip A with every frame of clip B (pose difference heat map), finds the best transition and plays A blending into B. |
| **AnimPreview** (`src/preview.cpp`) | Browses a folder tree of `.ozz` animations and plays them, optionally on a skinned character (retargeted) or as a point field. |

Tools:

| Tool | What it does |
|---|---|
| `Tools/hf_download.py` | Downloads a Hugging Face repo or dataset (used for the CMU clips). |
| `Tools/convert_fbx.py` | Batch converts FBX animations to ozz archives: `fbx2ozz`, then `ozz_fixup`. |
| `ozz_fixup` | Rescales raw ozz archives (e.g. centimeters to meters), bakes joint scales, optimizes and builds them. |
| `fbx2mesh` | Extracts skinned meshes from FBX into the framework's `mesh.ozz` format. |
| `pointfield_gen`, `pointfield_stats` | Generate and inspect point fields (points on a character's skinned surface, per frame). |

## Building

Requirements:

- CMake 3.30 or newer and a C++17 compiler. Developed on Windows with Visual
  Studio 2026; other platforms are untested (the asset scripts assume
  Windows `.exe` tool paths).
- Internet access on the first configure: ozz-animation 0.17.0, Dear ImGui
  1.93 and GLFW (bundled with ozz) are fetched by CMake.
- Optional: the [Autodesk FBX SDK](https://aps.autodesk.com/developer/overview/autodesk-fbx-sdk)
  (2020.3), only to build `fbx2ozz` and `fbx2mesh` for converting FBX assets.
  The apps build without it.

```sh
cmake -S . -B build
cmake --build build --config Release
```

Executables end up in `build/Release/`.

## Assets

No animation data or meshes are included. The apps expect them under
`Assets/` (ignored by git):

1. Download the CMU clips in FBX format (needs [uv](https://docs.astral.sh/uv/)):

   ```sh
   uv run Tools/hf_download.py gbionics/cmu-fbx --type dataset
   ```

   This writes `Assets/HuggingFace/gbionics__cmu-fbx/`, including the
   `metadata.csv` clip descriptions the apps show.

2. Convert them to ozz archives (needs `fbx2ozz`, so the FBX SDK):

   ```sh
   uv run Tools/convert_fbx.py Assets/HuggingFace/gbionics__cmu-fbx/animations --out Assets/ozz/cmu
   ```

3. Optional, a character: export a skinned FBX character with `fbx2ozz`
   (skeleton) and `fbx2mesh` (mesh) into a folder under `Assets/ozz/`
   (`skeleton.ozz` + `mesh.ozz`). Clips made for another skeleton are
   retargeted onto it.

Then run, for example:

```sh
build/Release/CommonPoses.exe
build/Release/CommonPoses.exe --project=path/to/set.hpproject --walk --root_motion
```

Command line options are listed with `--help`.

## Credits

- **[ozz-animation](https://github.com/guillaumeblanc/ozz-animation)** by
  Guillaume Blanc (MIT): animation runtime, offline tools and the sample
  framework copied into `external/ozz_framework` (with its license).
- **[Dear ImGui](https://github.com/ocornut/imgui)** by Omar Cornut (MIT): user
  interface.
- **[GLFW](https://www.glfw.org)** (zlib): windowing, via ozz-animation.
- **Motion capture data** (not included): the
  [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu), as
  converted to BVH by [cgspeed](https://sites.google.com/a/cgspeed.com/cgspeed/manifesto),
  to FBX by [RancidMilk](https://rancidmilk.itch.io/free-character-animations),
  and published on Hugging Face as
  [gbionics/cmu-fbx](https://huggingface.co/datasets/gbionics/cmu-fbx).

  > The data used in this project was obtained from mocap.cs.cmu.edu. The
  > database was created with funding from NSF EIA-0196217.

- **Autodesk FBX SDK** (not included): FBX import for the asset tools. FBX is
  a trademark of Autodesk, Inc.

## License

MIT, see [LICENSE](LICENSE). `external/ozz_framework` is under the
ozz-animation MIT license. Asset licenses are their owners'; none are
distributed here.
