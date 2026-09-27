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
| `depth-bands`     | 1        | Liczba pasm tonalnych głębi z osobnym wzmocnieniem (1 = wyłączone; wymaga `depth-file`) |
| `near-gain`       | 1.6      | Mnożnik residual-gain dla najjaśniejszego (najbliższego) pasma |
| `far-gain`        | 0.4      | Mnożnik residual-gain dla najciemniejszego (najdalszego) pasma |
| `band-blur`       | 3.0      | Rozmycie granic pasm (sigma), zapobiega schodkowaniu |
| `wiggle-amount`   | 0.0      | Amplituda (px) realnego przesunięcia zależnego od głębi, Smooth3D-style (0 = wyłączone; wymaga `depth-file`) |
| `wiggle-freq-hz`  | 4.0      | Częstotliwość oscylacji wiggle w Hz |

### Wzmocnienie per-pasmo głębi (`depth-bands`)

Domyślnie `gain` (mnożnik nałożonego residualu) jest jeden dla całej klatki.
Ustawiając `depth-bands > 1` (wymaga `depth-file`), mapa głębi zostaje
podzielona na N równych przedziałów tonalnych (0–255), a każdy piksel
dostaje mnożnik interpolowany liniowo od `far-gain` (najciemniejsze/
najdalsze pasmo) do `near-gain` (najjaśniejsze/najbliższe). Efekt:
pierwszy plan może „pulsować” znacznie mocniej niż tło zamiast jednolicie
w całym kadrze — silniejsza percepcyjna separacja planów niż z samego
kierunku pseudo-flow.

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! parallaxdof depth-file=depths.raw parallax-shift=15 \
      depth-bands=4 near-gain=2.2 far-gain=0.15 band-blur=4 \
  ! videoconvert \
  ! x264enc bitrate=4000 ! h264parse ! mp4mux \
  ! filesink location=3d.mp4
```

Uwaga: konwencja `depth-file` zakłada jaśniej = bliżej. Jeśli Twoje mapy
głębi mają odwrotną konwencję, po prostu zamień wartości `near-gain` i
`far-gain`.

### Saliency-weighted local detail boost (`saliency-boost`)

Domyślnie wyłączony (`saliency-boost=0`). Po włączeniu dokłada osobny etap:
lokalne wzmocnienie kontrastu/detalu (unsharp na wysokich częstotliwościach
klatki), ważone center-weighted maską "uwagi" — środek kadru dostaje pełne
wzmocnienie, peryferie zanikają Gaussem. Działa identycznie w trybie
realtime i z `depth-file`, niezależnie od `depth-bands`.

| Właściwość        | Domyślna | Opis |
|-------------------|----------|------|
| `saliency-boost`  | 0.0      | Siła wzmocnienia (0 = wyłączone) |
| `saliency-radius` | 0.6      | Promień maski jako ułamek połowy przekątnej kadru |
| `saliency-sigma`  | 2.0      | Sigma Gaussa do wydzielenia detalu (mniejsza = drobniejsze detale) |

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! parallaxdof depth-file=depths.raw parallax-shift=15 \
      depth-bands=4 near-gain=2.2 far-gain=0.15 band-blur=4 \
      saliency-boost=1.5 saliency-radius=0.5 saliency-sigma=2.5 \
  ! videoconvert \
  ! x264enc bitrate=4000 ! h264parse ! mp4mux \
  ! filesink location=3d.mp4
```

To NIE jest HDR (łączenie ekspozycji) — to lokalny kontrast/clarity boost,
percepcyjnie podobny do suwaka "Clarity" w edytorach zdjęć. Jeśli
faktyczna kompozycja kadru nie ma obiektu zainteresowania w centrum,
`saliency-radius` warto zwiększyć albo ustawić właściwość na przyszłość:
prawdziwa saliency (detekcja twarzy/obiektów) wymagałaby dodatkowego
modelu — to jest tani, spatial-only proxy.

### Smooth3D-style wiggle (`wiggle-amount`)

Domyślnie wyłączony (`wiggle-amount=0`). W przeciwieństwie do bazowego
efektu "mimoza" (przesunięcie residualu o ułamek px wzdłuż wektora ruchu/
gradientu głębi — czyli kierunkowy podbicie mikrokontrastu), `wiggle-amount`
dokłada **realne przesunięcie poziome pikseli** proporcjonalne do głębi,
oscylujące w czasie sinusoidalnie (jak w Smooth3D/Wiggle3D). Bliższe piksele
przesuwają się mocniej niż tło — to prawdziwa kinetyczna paralaksa, nie tylko
wskazówka kontrastowa, więc percepcyjnie wypada dużo mocniej niż sama
"mimoza". Wymaga `depth-file` (potrzebna mapa głębi per-piksel).

| Właściwość        | Domyślna | Opis |
|-------------------|----------|------|
| `wiggle-amount`   | 0.0      | Amplituda przesunięcia (px) dla najbliższego piksela; 0 = wyłączone |
| `wiggle-freq-hz`  | 4.0      | Częstotliwość oscylacji (Hz); typowo 2–6 |

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! parallaxdof depth-file=depths.raw parallax-shift=15 \
      wiggle-amount=6 wiggle-freq-hz=4 \
  ! videoconvert \
  ! x264enc bitrate=4000 ! h264parse ! mp4mux \
  ! filesink location=3d.mp4
```

Uwagi:
- Działa niezależnie od "mimozy" i `depth-bands`/`saliency-boost` — można
  łączyć wszystkie warstwy naraz.
- Duża amplituda (>15–20px) przy ostrych krawędziach FG/BG ujawnia
  rozciągnięte piksele na brzegach (prosty `BORDER_REPLICATE`, bez
  wypełniania odsłonięć) — przy 3–8px praktycznie niezauważalne.
- Zbyt niska częstotliwość (<1 Hz) daje wrażenie "kołysania" kadru zamiast
  głębi; zbyt wysoka (>10 Hz) rozmywa się w migotanie.

### Realny flow + kontrast od prędkości (`motion-file`)

Domyślnie kierunek "mimozy" w trybie `depth-file` jest zgadywany z gradientu
mapy głębi (Sobel) — tani, ale to przybliżenie, nie prawdziwy wektor ruchu.
`motion-file` to osobny plik zapisywany przez `depthanalyzer`, z **realnym
optical-flow** (ten sam Farneback, który w `parallaxdof` liczony jest tylko
w trybie realtime) policzonym w preanalizie i zapisanym na dysk — 2. przebieg
czyta go zamiast liczyć na nowo.

Dwie niezależne korzyści, można włączyć osobno lub razem:

1. **Realny kierunek "mimozy"** — gdy `motion-file` jest ustawiony razem z
   `depth-file` (albo nawet samodzielnie, bez `depth-file`), residual jest
   próbkowany wzdłuż prawdziwego wektora ruchu zamiast pseudo-flow z
   gradientu. Dokładniejszy, bardziej "przyczepiony" do faktycznego ruchu
   w kadrze.
2. **Kontrast zależny od prędkości** (`motion-saliency-weight`) — szybciej
   poruszające się piksele (wyższy magnitude w `motion.raw`) dostają
   mocniejszy lokalny detail-boost (ta sama mechanika co `saliency-boost`/
   `depth-saliency-weight`, tylko maska pochodzi z prędkości ruchu zamiast
   z centrum kadru / głębi). Wymaga `saliency-boost > 0`.

#### 1) Preanaliza: głębia + realny flow

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! depthanalyzer location=depths.raw motion-file=motion.raw \
      proc-scale=0.5 motion-max-mag=20 \
  ! fakesink
```

#### 2) Render z realnym flow + kontrastem od prędkości

```bash
gst-launch-1.0 filesrc location=in.mp4 ! decodebin \
  ! videoconvert ! video/x-raw,format=BGR \
  ! parallaxdof depth-file=depths.raw motion-file=motion.raw motion-max-mag=20 \
      parallax-shift=15 depth-bands=4 near-gain=2.2 far-gain=0.15 \
      saliency-boost=1.5 motion-saliency-weight=0.7 motion-saliency-gamma=1.3 \
  ! videoconvert \
  ! x264enc bitrate=4000 ! h264parse ! mp4mux \
  ! filesink location=3d.mp4
```

`motion-file` działa też bez `depth-file` (napędza samą mimozę realnym flow,
bez pasm głębi/wiggle — te dwa wymagają `depth-file`).

| Właściwość (depthanalyzer) | Domyślna | Opis |
|-----------------------------|----------|------|
| `motion-file`                | (brak)   | Ścieżka zapisu realnego flow (Farneback), 2 bajty/px/klatkę |
| `motion-max-mag`             | 20.0     | Clamp (px/klatkę) do kwantyzacji magnitude na 0–255 |

| Właściwość (parallaxdof)      | Domyślna | Opis |
|--------------------------------|----------|------|
| `motion-file`                  | (brak)   | Odczyt realnego flow — musi być z tego samego przebiegu `depthanalyzer` |
| `motion-max-mag`               | 20.0     | **Musi być identyczny** jak przy zapisie, inaczej dekwantyzacja magnitude jest błędna |
| `motion-saliency-weight`       | 0.0      | Udział mapy prędkości w masce saliency (0 = wyłączone) |
| `motion-saliency-gamma`        | 1.0      | Krzywa mapowania magnitude→waga (>1 = tylko najszybsze piksele, <1 = spłaszcza różnice) |

Uwaga: `motion-max-mag` musi być takie samo po obu stronach (zapis/odczyt) —
to jest zakres kwantyzacji, nie luźny suwak.

⚠️ Koszt: `motion-file` dokłada Farneback do preanalizy (pierwszy przebieg
wolniejszy — teraz liczy diff+Sobel **i** optical flow), oraz podwaja
rozmiar pliku pomocniczego (2 bajty/px zamiast 1). Drugi przebieg
(`parallaxdof`) pozostaje szybki — flow jest czytany z pliku, nie liczony.

## Format pliku depths.raw

Identyczny jak w `3dplayer.py` / projekcie `.def`:

```
dla każdej klatki:
  width * height bajtów uint8  (depth * 255)
```

Wymiary muszą być takie same jak wideo (pełna rozdzielczość).
Kolejność klatek = kolejność w pliku źródłowym.

## Format pliku motion.raw

```
dla każdej klatki:
  width * height * 2 bajtów uint8:
    bajt 0: magnitude, 0..255 względem clampu motion-max-mag (px/klatkę)
    bajt 1: kąt, 0..255 względem 0..2π (atan2(vy, vx), znormalizowany do [0, 2π))
```

Wymiary i kolejność klatek jak w `depths.raw`. `motion-max-mag` musi być
taki sam przy zapisie (`depthanalyzer`) i odczycie (`parallaxdof`).
