# Editor fields

`editor.set`, `editor.batch_set`, and `editor.get` use these fields. The values are UI values:
the numbers that the editor panels show. A value outside the range, an unknown property, or an
unknown option is rejected with `-32602`; it is not clamped.

The **reset value** is the value of the panel reset. A photo without edits starts with the
default look of Alcedo Studio instead: exposure +1.5, contrast 15, saturation 30, and the other
fields at their reset values. Read the values with `editor.get` before you change them.

## Notes on the fields

- **Tone** (`exposure`, `contrast`, `highlights`, `shadows`, `white`, `black`): exposure is in
  stops (EV). The other tone fields are -100 to 100.
- **White balance:** `color_temp` is the white balance of the RAW decode. Its `mode` is
  `as_shot` (the camera white balance; `kelvin` and `tint` report it) or `custom` (your `kelvin`
  and `tint`). A write with `kelvin` or `tint` sets the mode to `custom`; write
  `{"mode":"as_shot"}` to go back to the camera white balance. `kelvin` is the color temperature
  of the light in the scene, so a higher `kelvin` makes the image warmer (more yellow) and a
  lower `kelvin` makes it cooler (more blue). Positive `tint` makes the image more magenta;
  negative `tint` makes it more green. Start from the as-shot values that `editor.get` returns
  and change them in steps of about 300 K and 5 tint. `grade_white_balance` is a second white
  balance in the color grade, after the RAW decode.
- **Crop and rotation:** `crop_rotate` holds the crop rectangle and the rotation:
  - `x` and `y` are the top-left corner of the crop and `width` and `height` its size, all as
    fractions of the image width and height. `y` goes down: `{"x":0.5,"y":0,"width":0.5,
    "height":0.5}` is the top-right quarter.
  - A positive `angle_degrees` turns the image clockwise. To level a horizon that rises to the
    right by 2 degrees (counterclockwise), add 2 to `angle_degrees`.
  - The editor keeps the rotated crop inside the image, so the value in the result can differ
    from the value you sent: use the returned value. A rotation makes the crop smaller; set
    the angle back to 0 and the crop to `{"x":0,"y":0,"width":1,"height":1}` to undo it.
- **Curves and LUTs** (`curve`, `lut`): a write gives the complete value.

## Field tables

These tables are generated from [editor_catalog.json](editor_catalog.json) (`editor.catalog`).

<!-- BEGIN GENERATED: editor-fields -->
### Scalar fields

| Field | Aliases | Panel | Range | Step | Decimals | Reset value |
| --- | --- | --- | --- | --- | --- | --- |
| `exposure` |  | tone | -10 to 10 | 0.01 | 2 | 0 |
| `contrast` |  | tone | -100 to 100 | 1 | 0 | 0 |
| `highlights` |  | tone | -100 to 100 | 1 | 0 | 0 |
| `shadows` |  | tone | -100 to 100 | 1 | 0 | 0 |
| `white` | `whites` | tone | -100 to 100 | 1 | 0 | 0 |
| `black` | `blacks` | tone | -100 to 100 | 1 | 0 | 0 |
| `saturation` |  | look | -100 to 100 | 1 | 0 | 0 |
| `vibrance` |  | look | -100 to 100 | 1 | 0 | 0 |
| `diffusion` |  | post | 0 to 100 | 1 | 0 | 0 |
| `clarity` |  | post | -100 to 100 | 1 | 0 | 0 |
| `sharpen` |  | post | 0 to 100 | 1 | 0 | 0 |
| `film_grain` |  | post | 0 to 100 | 1 | 0 | 0 |
| `halation` |  | post | 0 to 100 | 1 | 0 | 0 |

### Object fields

A write can give some of the properties; the others keep their values.

#### `raw_decode` (raw panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `method` | option | `default` (Default), `legacy` (Legacy), `neural_engine` (Neural Engine) |  |  |
| `highlights_reconstruct` | boolean |  |  |  |

Reset value: `{"highlights_reconstruct": true, "method": "default"}`

#### `input_profile` (raw panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `profile_override` | option | `auto` (Auto (from file)), `srgb` (sRGB), `display_p3` (Display P3), `adobe_rgb` (Adobe RGB), `rec2020` (Rec.2020), `prophoto` (ProPhoto), `linear_rec709` (Linear Rec.709) |  |  |

Reset value: `{"profile_override": "auto"}`

#### `lens_calib` (raw panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `enabled` | boolean |  |  |  |
| `lens_maker` | string |  |  |  |
| `lens_model` | string |  |  |  |

Reset value: `{"enabled": false, "lens_maker": "", "lens_model": ""}`

#### `crop_rotate` (geometry panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `x` | number | -1 to 1 | 0.001 | 3 |
| `y` | number | -1 to 1 | 0.001 | 3 |
| `width` | number | 0.0001 to 1 | 0.001 | 3 |
| `height` | number | 0.0001 to 1 | 0.001 | 3 |
| `angle_degrees` | number | -180 to 180 | 0.1 | 1 |
| `aspect_preset` | option | `free` (Free), `custom` (Custom), `ratio_2_35_1_35mm` (2.35:1 (35mm)), `ratio_1_1` (1:1), `ratio_16_9` (16:9), `ratio_1_9_1_imax` (1.9:1 (IMAX)), `ratio_1_85_1_dci` (1.85:1 (DCI)), `ratio_2_2_1_70mm` (2.2:1 (70mm)), `ratio_1_43_1_70mm_imax` (1.43:1 (70mm IMAX)), `ratio_4_3_35mm` (4:3 (35mm)), `ratio_1_5_1_native_vistavision` (1.5:1 (native or VistaVision)), `ratio_2_76_1_panavision_ultra` (2.76:1 (Panavision Ultra)) |  |  |
| `aspect_width` | number | 0.0001 to 100 | 0.01 | 2 |
| `aspect_height` | number | 0.0001 to 100 | 0.01 | 2 |

Reset value: `{"angle_degrees": 0, "aspect_height": 1, "aspect_preset": "free", "aspect_width": 1, "height": 1, "width": 1, "x": 0, "y": 0}`

#### `odt` (display panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `method` | option | `aces_2_0` (ACES 2.0), `open_drt` (OpenDRT) |  |  |
| `encoding_space` | option | `rec709` (Rec.709), `p3_d65` (P3-D65), `rec2020` (Rec.2020) |  |  |
| `encoding_eotf` | option | `bt1886` (BT.1886), `gamma_2_2` (Gamma 2.2), `srgb_piecewise` (sRGB), `st2084` (ST 2084 (PQ)), `hlg` (HLG) |  |  |
| `peak_luminance` | number | 100 to 1000 | 1 | 0 |
| `limiting_space` | option | `rec709` (Rec.709), `rec2020` (Rec.2020), `p3_d65` (P3-D65) |  |  |
| `look_preset` | option | `standard` (Standard), `arriba` (Arriba), `sylvan` (Sylvan), `colorful` (Colorful), `aery` (Aery), `dystopic` (Dystopic), `umbra` (Umbra), `custom` (Custom) |  |  |
| `tonescale_preset` | option | `use_look_preset` (Use Look Preset), `low_contrast` (Low Contrast), `medium_contrast` (Medium Contrast), `high_contrast` (High Contrast), `arriba_tonescale` (Arriba Tonescale), `sylvan_tonescale` (Sylvan Tonescale), `colorful_tonescale` (Colorful Tonescale), `aery_tonescale` (Aery Tonescale), `dystopic_tonescale` (Dystopic Tonescale), `umbra_tonescale` (Umbra Tonescale), `aces_1_x` (ACES 1.x), `aces_2_0` (ACES 2.0), `marvelous_tonscape` (Marvelous Tonscape), `dagrinchi_tonegroan` (Dagrinchi Tonegroan), `custom` (Custom) |  |  |
| `creative_white` | option | `use_look_preset` (Use Look Preset), `d93` (D93), `d75` (D75), `d65` (D65), `d60` (D60), `d55` (D55), `d50` (D50) |  |  |

Reset value: `{"creative_white": "use_look_preset", "encoding_eotf": "gamma_2_2", "encoding_space": "rec709", "limiting_space": "rec709", "look_preset": "standard", "method": "open_drt", "peak_luminance": 100, "tonescale_preset": "use_look_preset"}`

Valid `encoding_eotf` values of each `encoding_space`:

- `p3_d65`: `gamma_2_2`, `srgb_piecewise`, `st2084`
- `rec2020`: `st2084`, `hlg`
- `rec709`: `bt1886`, `gamma_2_2`, `srgb_piecewise`

#### `color_temp` (raw panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `mode` | option | `as_shot` (As Shot), `custom` (Custom) |  |  |
| `kelvin` | number | 2000 to 15000; slider scale `kelvin_pivot`, position 0 to 4096, 6000 K at 2048 | 1 | 0 |
| `tint` | number | -150 to 150 | 1 | 0 |

Reset value: `{"kelvin": 6500, "mode": "as_shot", "tint": 0}`

#### `grade_white_balance` (look panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `temperature` | number | 2000 to 15000; slider scale `kelvin_pivot`, position 0 to 4096, 6000 K at 2048 | 1 | 0 |
| `tint` | number | -150 to 150 | 1 | 0 |

Reset value: `{"temperature": 5999.9931640625, "tint": 9.711277961730957}`

#### `hls` (look panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `hue` | number | 0 to 315 | 45 | 0 |
| `hue_shift` | number_list | 8 numbers, -30 to 30 | 1 | 0 |
| `lightness` | number_list | 8 numbers, -100 to 100 | 1 | 0 |
| `chroma` | number_list | 8 numbers, -100 to 100 | 1 | 0 |
| `hue_smoothness` | number_list | 8 numbers, 1 to 180 | 1 | 0 |

Reset value: `{"chroma": [0, 0, 0, 0, 0, 0, 0, 0], "hue": 0, "hue_shift": [0, 0, 0, 0, 0, 0, 0, 0], "hue_smoothness": [45, 45, 45, 45, 45, 45, 45, 45], "lightness": [0, 0, 0, 0, 0, 0, 0, 0]}`

#### `color_wheel` (look panel)

| Property | Type | Range or values | Step | Decimals |
| --- | --- | --- | --- | --- |
| `lift_x` | number | -1 to 1 | 0.001 | 3 |
| `lift_y` | number | -1 to 1 | 0.001 | 3 |
| `lift_master` | number | -800 to 800 | 1 | 0 |
| `gamma_x` | number | -1 to 1 | 0.001 | 3 |
| `gamma_y` | number | -1 to 1 | 0.001 | 3 |
| `gamma_master` | number | -800 to 800 | 1 | 0 |
| `gain_x` | number | -1 to 1 | 0.001 | 3 |
| `gain_y` | number | -1 to 1 | 0.001 | 3 |
| `gain_master` | number | -800 to 800 | 1 | 0 |

Reset value: `{"gain_master": 0, "gain_x": 0, "gain_y": 0, "gamma_master": 0, "gamma_x": 0, "gamma_y": 0, "lift_master": 0, "lift_x": 0, "lift_y": 0}`

### Model fields

A write gives the complete value.

#### `curve` (tone panel)

{"points": [{"x": number, "y": number}, ...]}: at least two control points in the unit square

Reset value: `{"points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}`

#### `lut` (lut panel)

{"cube_path": string} or a tagged {"reference": object}, and optional "name", "strength" (0 to 1), "input_encoding", and "output_encoding" (color encoding ids). {} is no LUT

Reset value: `{}`
<!-- END GENERATED: editor-fields -->
