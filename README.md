# gst-parallaxdof

GStreamer plugin z dwoma elementami:

| Element | Rola |
|---------|------|
| **depthanalyzer** | Wstępna analiza głębi → plik `.raw` (uint8, W×H na klatkę) |
| **parallaxdof**   | Paralaksa 2.5D + DoF (realtime lub z preanalizy) |

Logika 1:1 z `3dplayer1.py` (różnica klatek + Sobel, bez filtracji skupisk).

## Workflow dwuprzebiegowy (zalecany)

### 1) Preanaliza głębi

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! depthanalyzer location=depths.raw proc-scale=0.5 \
  ! fakesink
```

Zapisuje sekwencję map głębi do `depths.raw` (format jak w Pythonie: 1 bajt/piksel, kolejność klatek).

### 2) Render z gotowymi mapami

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! parallaxdof depth-file=depths.raw parallax-shift=15 fg-threshold=0.4 bg-blur=8 \
  ! videoconvert \
  ! x264enc bitrate=4000 ! h264parse ! mp4mux \
  ! filesink location=3d.mp4
```

### Alternatywa: wszystko w jednym przebiegu (wolniej)

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! parallaxdof parallax-shift=15 bg-blur=8 \
  ! videoconvert ! autovideosink
```

(bez `depth-file` → estymacja on-the-fly)

## Build & Install

```bash
meson setup build && ninja -C build
DESTDIR=~/.local ninja -C build install
export GST_PLUGIN_PATH=~/.local/$(pkg-config gstreamer-1.0 --variable=pluginsdir)
# lub: export GST_PLUGIN_PATH=$PWD/build
```

```bash
gst-inspect-1.0 depthanalyzer
gst-inspect-1.0 parallaxdof
```

## Właściwości

### depthanalyzer

| Właściwość     | Domyślna | Opis |
|----------------|----------|------|
| `location`     | (wymagane) | Ścieżka pliku `.raw` z mapami głębi |
| `proc-scale`   | 0.5      | Skala estymacji |
| `smooth-ksize` | 21       | Wygładzanie mapy głębi |
| `flow-weight`  | 0.7      | Waga optical flow |
| `edge-weight`  | 0.3      | Waga krawędzi Sobela |

### parallaxdof

| Właściwość        | Domyślna | Opis |
|-------------------|----------|------|
| `depth-file`      | (brak)   | Plik z preanalizy — jeśli ustawiony, pomija estymację |
| `parallax-shift`  | 15       | Przesunięcie FG (px) |
| `fg-threshold`    | 0.4      | Próg pierwszego planu |
| `bg-blur`         | 8.0      | Rozmycie tła |
| `fg-sharpness`    | 1.5      | Wyostrzenie FG |
| `proc-scale`      | 0.5      | Skala (tylko gdy brak depth-file) |
| `max-fg-clusters` | 1        | Ile skupisk FG |
| `smooth-ksize`    | 21       | Wygładzanie (tylko realtime) |
| `flow-weight`     | 0.7      | — |
| `edge-weight`     | 0.3      | — |

## Format pliku depths.raw

Identyczny jak w `3dplayer.py` / projekcie `.def`:

```
dla każdej klatki:
  width * height bajtów uint8  (depth * 255)
```

Wymiary muszą być takie same jak wideo (pełna rozdzielczość).
Kolejność klatek = kolejność w pliku źródłowym.
