/*
 * gstparallaxdof.cpp — GStreamer plugin: parallaxdof + depthanalyzer
 *
 * Efekt: mimoza transwektorowa
 *   - porównanie klatki z oryginalną poprzednią
 *   - odejmowanie poprzednio nałożonych zmian (residual)
 *   - delikatne podkreślenie ruchu wzdłuż wypadkowej wektorów flow
 *   - auto-adaptacja siły do typowego |flow| w klatce (niezależnie od filmu)
 *   - NIE jest to bokeh/DoF — żadnego stałego rozmycia tła
 *
 * Elementy:
 *   depthanalyzer  — opcjonalna preanaliza map ruchu → .raw
 *   parallaxdof    — mimoza transwektorowa (realtime lub z pliku)
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <gst/base/gstbasetransform.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

/* ═══════════════════════════════════════════════════════════════════════════
 * Wspólne stałe / helpers (DepthEstimator + ParallaxDoFRenderer)
 * ═════════════════════════════════════════════════════════════════════════ */

#define DEFAULT_PARALLAX_SHIFT    15
#define DEFAULT_FG_THRESHOLD      0.4
#define DEFAULT_BG_BLUR           8.0
#define DEFAULT_FG_SHARPNESS      1.5
#define DEFAULT_PROC_SCALE        0.5
#define DEFAULT_MAX_FG_CLUSTERS   1
#define DEFAULT_BG_LEARN_RATE     0.05f
#define DEFAULT_SMOOTH_KSIZE      21
#define DEFAULT_FLOW_WEIGHT       0.7f
#define DEFAULT_EDGE_WEIGHT       0.3f
#define DEFAULT_MID_BLUR_RADIUS   2.0
#define DEFAULT_MIN_CLUSTER_RATIO 0.01f
#define DEFAULT_DEPTH_BANDS       1
#define DEFAULT_NEAR_GAIN         1.6
#define DEFAULT_FAR_GAIN          0.4
#define DEFAULT_BAND_BLUR         3.0
#define DEFAULT_SALIENCY_BOOST    0.0
#define DEFAULT_SALIENCY_RADIUS   0.6
#define DEFAULT_SALIENCY_SIGMA    2.0
#define DEFAULT_DEPTH_SALIENCY_WEIGHT 0.0
#define DEFAULT_DEPTH_SALIENCY_GAMMA  1.0
#define DEFAULT_WIGGLE_AMOUNT     0.0
#define DEFAULT_WIGGLE_FREQ_HZ    4.0
#define DEFAULT_MOTION_MAX_MAG    20.0
#define DEFAULT_MOTION_SALIENCY_WEIGHT 0.0
#define DEFAULT_MOTION_SALIENCY_GAMMA  1.0

/* ── Flow + residual helpers ─────────────────────────────────────────────── */

/* Lekki Farneback — wektory ruchu między prev a current */
static void
compute_flow (const cv::Mat &prev_gray, const cv::Mat &gray, cv::Mat &flow)
{
  /* pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags */
  cv::calcOpticalFlowFarneback (
      prev_gray, gray, flow,
      0.5, 2, 13, 2, 5, 1.1, 0);
}

/* ── Maska wzmocnienia per-pasmo głębi tonalnej ──────────────────────────
 * Zamiast jednego globalnego `gain` dla całej klatki, dzielimy zakres
 * mapy głębi (0..255, jaśniej = bliżej — zgodnie z konwencją depth-file)
 * na `bands` równych przedziałów tonalnych i przypisujemy każdemu
 * przedziałowi inne wzmocnienie: od `far_gain` (najciemniejsze/najdalsze
 * pasmo) do `near_gain` (najjaśniejsze/najbliższe pasmo), liniowo.
 *
 * `band_blur` rozmywa granice pasm, żeby przejścia między nimi nie były
 * twardymi krawędziami (uniknięcie widocznego "schodkowania" w obrazie).
 *
 * bands == 1 → maska stała (średnia near/far) = zachowanie identyczne
 * jak globalny gain sprzed tej zmiany (pełna kompatybilność wsteczna
 * przy domyślnym depth-bands=1).
 * ─────────────────────────────────────────────────────────────────────────── */
static cv::Mat
build_depth_band_gain (const cv::Mat &depth_u8, int bands,
                       double near_gain, double far_gain, double band_blur)
{
  cv::Mat gain (depth_u8.size (), CV_32FC1);
  bands = std::max (1, bands);

  if (bands == 1) {
    gain.setTo ((float)((near_gain + far_gain) * 0.5));
    return gain;
  }

  const int step = std::max (1, 256 / bands);
  for (int y = 0; y < depth_u8.rows; ++y) {
    const guint8 *d = depth_u8.ptr<guint8> (y);
    float *g = gain.ptr<float> (y);
    for (int x = 0; x < depth_u8.cols; ++x) {
      int band = d[x] / step;
      if (band >= bands) band = bands - 1;
      /* t=0 → najdalsze pasmo (far_gain), t=1 → najbliższe (near_gain) */
      float t = (float)band / (float)(bands - 1);
      g[x] = (float)(far_gain + t * (near_gain - far_gain));
    }
  }

  if (band_blur > 0.0)
    cv::GaussianBlur (gain, gain, cv::Size (0, 0), band_blur);

  return gain;
}

/* ── Rdzeń: mając już gotowe pole 'flow' (CV_32FC2, px pełnej rozdz.),
 * nakłada mimozę transwektorową. Współdzielony przez tryb realtime
 * (Farneback) i tryb z pliku (depth-file → gradient jako pseudo-flow).
 *
 * 1. residual_raw = current − prev_original
 * 2. flow (vx, vy) — z Farnebacka LUB z gradientu mapy głębi
 * 3. Próbkuj residual wzdłuż wektora (x − k·vx, y − k·vy)
 *    → podkreśla ruch „zbliżający się” wzdłuż wypadkowej
 * 4. out = current + gain · residual_sampled · soft(|flow|)
 *         − decay · prev_applied_delta
 * 5. Zapisz deltę do odejmowania w następnej klatce
 *
 * Bez rozmycia tła, bez maski FG/BG, bez stałego bokeh.
 * ─────────────────────────────────────────────────────────────────────────── */
static void
render_mimosa_core (cv::Mat &frame_bgr,          /* in/out */
                    const cv::Mat &prev_bgr,     /* oryginał poprzedniej */
                    cv::Mat *prev_delta,
                    cv::Mat &flow,                /* CV_32FC2, może być zmodyfikowane in-place (blur) */
                    int strength_px,
                    double residual_gain,
                    double decay,
                    const cv::Mat *gain_mask = nullptr /* CV_32FC1 pełnej rozdz., mnoży residual_gain per-piksel */)
{
  const int h = frame_bgr.rows, w = frame_bgr.cols;

  if (!flow.isContinuous ())
    flow = flow.clone ();
  cv::GaussianBlur (flow, flow, cv::Size (0, 0), 1.5);

  /* residual = current − prev (signed float) */
  cv::Mat cur_f, prev_f, residual;
  frame_bgr.convertTo (cur_f, CV_32FC3);
  prev_bgr.convertTo (prev_f, CV_32FC3);
  residual = cur_f - prev_f;

  /* delta do nałożenia */
  cv::Mat delta = cv::Mat::zeros (h, w, CV_32FC3);

  /* ── Auto-adaptacja do filmu ───────────────────────────────────────────
   * Szacujemy typową wielkość ruchu (percentyl ~70 z podpróbki),
   * potem normalizujemy k i soft względem niej.
   * → ten sam „wygląd” na spokojnym i dynamicznym materiale.
   */
  const float *flow_data0 = flow.ptr<float> (0);
  const int flow_step0 = (int)(flow.step / sizeof (float));
  std::vector<float> mags;
  mags.reserve ((size_t)(h * w / 16));
  for (int y = 0; y < h; y += 4) {
    const float *fl = flow_data0 + y * flow_step0;
    for (int x = 0; x < w; x += 4) {
      float vx = fl[x * 2], vy = fl[x * 2 + 1];
      if (!std::isfinite (vx) || !std::isfinite (vy)) continue;
      float m = std::sqrt (vx * vx + vy * vy);
      if (m > 0.01f) mags.push_back (m);
    }
  }
  float ref_mag = 2.0f; /* fallback */
  if (!mags.empty ()) {
    size_t idx = (size_t)(mags.size () * 0.70);
    if (idx >= mags.size ()) idx = mags.size () - 1;
    std::nth_element (mags.begin (), mags.begin () + (long)idx, mags.end ());
    ref_mag = std::max (0.5f, mags[idx]);
  }

  /* k ~ 0.5..2 px w jednostkach „typowego” ruchu; strength_px skaluje 1..30 → 0.3..1.5 */
  const float k_base = 0.35f + 0.04f * (float)std::max (0, strength_px);
  const float k = k_base / ref_mag;           /* przesunięcie wzdłuż wektora znormalizowane */
  const float soft_tau = ref_mag * 1.2f;      /* próg soft dopasowany do sceny */
  const float max_mag = ref_mag * 4.0f;       /* clamp outlierów flow (gwałtowny pan/kurz) */
  const float gain = (float)residual_gain;

  /* Bezpieczne próbkowanie residualu wzdłuż wektora (bez OOB) */
  const float *flow_data = flow_data0;
  const int flow_step = flow_step0;

  for (int y = 0; y < h; ++y) {
    const float *fl = flow_data + y * flow_step;
    float *dlt = delta.ptr<float> (y);
    const float *gm_row = (gain_mask && gain_mask->rows == h && gain_mask->cols == w)
        ? gain_mask->ptr<float> (y) : nullptr;

    for (int x = 0; x < w; ++x) {
      float vx = fl[x * 2 + 0];
      float vy = fl[x * 2 + 1];
      if (!std::isfinite (vx) || !std::isfinite (vy))
        continue;

      float mag = std::sqrt (vx * vx + vy * vy);

      /* Outlier clamp: pojedynczy piksel z rozjechanym Farnebackiem
       * (gwałtowny pan kamery + kurz) nie może próbkować residualu
       * z absurdalnie odległego miejsca — obcinamy długość wektora,
       * kierunek zostaje bez zmian. Normalny ruch (mag < max_mag)
       * przechodzi nietknięty. */
      if (mag > max_mag) {
        float sc = max_mag / mag;
        vx *= sc;
        vy *= sc;
        mag = max_mag;
      }
      float soft = mag / (mag + soft_tau);
      if (soft < 0.04f)
        continue;

      /* źródło residualu „wzdłuż wektora wstecz” */
      float sx = (float)x - k * vx;
      float sy = (float)y - k * vy;

      /* clamp do [0, w-1] / [0, h-1] przed floor */
      if (sx < 0.f) sx = 0.f;
      if (sy < 0.f) sy = 0.f;
      if (sx > (float)(w - 1)) sx = (float)(w - 1);
      if (sy > (float)(h - 1)) sy = (float)(h - 1);

      int x0 = (int)sx;
      int y0 = (int)sy;
      int x1 = x0 + 1;
      int y1 = y0 + 1;
      if (x1 >= w) x1 = w - 1;
      if (y1 >= h) y1 = h - 1;

      float fx = sx - (float)x0;
      float fy = sy - (float)y0;

      const float *r00 = residual.ptr<float> (y0) + x0 * 3;
      const float *r10 = residual.ptr<float> (y0) + x1 * 3;
      const float *r01 = residual.ptr<float> (y1) + x0 * 3;
      const float *r11 = residual.ptr<float> (y1) + x1 * 3;

      float ww00 = (1.f - fx) * (1.f - fy);
      float ww10 = fx * (1.f - fy);
      float ww01 = (1.f - fx) * fy;
      float ww11 = fx * fy;

      float px_gain = gm_row ? (gain * gm_row[x]) : gain;

      for (int c = 0; c < 3; ++c) {
        float rv = r00[c] * ww00 + r10[c] * ww10 + r01[c] * ww01 + r11[c] * ww11;
        dlt[x * 3 + c] = rv * px_gain * soft;
      }
    }
  }

  /* Wygładzenie przestrzenne delty — bez ostrych łączeń */
  cv::GaussianBlur (delta, delta, cv::Size (0, 0), 1.2);

  /* Odejmij poprzednio nałożone zmiany + temporal blend (spójność klatek) */
  if (prev_delta && !prev_delta->empty () &&
      prev_delta->size () == delta.size () &&
      prev_delta->type () == delta.type ()) {
    /* residual anti-stack */
    delta -= (*prev_delta) * (float)decay;
    /* temporal low-pass: nowa_delta = 0.55*nowa + 0.45*stara  */
    cv::addWeighted (delta, 0.55, *prev_delta, 0.45, 0.0, delta);
  }

  /* out = current + delta */
  cv::Mat out_f = cur_f + delta;
  out_f.convertTo (frame_bgr, CV_8U);

  /* zapamiętaj deltę */
  if (prev_delta)
    delta.copyTo (*prev_delta);
}

/* ── Mimoza transwektorowa — tryb REALTIME (Farneback) ─────────────────── */
static void
render_mimosa (cv::Mat &frame_bgr,           /* in/out */
               const cv::Mat *prev_bgr,      /* oryginał poprzedniej (może null) */
               cv::Mat *prev_delta,          /* poprzednio nałożone zmiany */
               cv::Mat *prev_gray_out,       /* aktualizowany gray */
               int strength_px,              /* skalar przesunięcia wzdłuż wektora */
               double residual_gain,         /* wzmocnienie residualu */
               double decay,                 /* ile odejmować z poprzedniej delty */
               double proc_scale)
{
  const int h = frame_bgr.rows, w = frame_bgr.cols;

  cv::Mat gray;
  cv::cvtColor (frame_bgr, gray, cv::COLOR_BGR2GRAY);

  /* Pierwsza klatka — tylko zapamiętaj, zero efektu */
  if (!prev_bgr || prev_bgr->empty () ||
      prev_bgr->size () != frame_bgr.size ()) {
    if (prev_gray_out)
      gray.copyTo (*prev_gray_out);
    if (prev_delta)
      *prev_delta = cv::Mat::zeros (h, w, CV_32FC3);
    return;
  }

  cv::Mat prev_gray;
  cv::cvtColor (*prev_bgr, prev_gray, cv::COLOR_BGR2GRAY);

  /* Flow na mniejszej rozdzielczości dla szybkości */
  const int pw = std::max (1, (int)(w * proc_scale));
  const int ph = std::max (1, (int)(h * proc_scale));

  cv::Mat g0, g1;
  cv::resize (prev_gray, g0, cv::Size (pw, ph), 0, 0, cv::INTER_AREA);
  cv::resize (gray,      g1, cv::Size (pw, ph), 0, 0, cv::INTER_AREA);

  cv::Mat flow_s;
  compute_flow (g0, g1, flow_s);

  /* Upscale flow do pełnej rozdzielczości */
  cv::Mat flow;
  cv::resize (flow_s, flow, cv::Size (w, h), 0, 0, cv::INTER_CUBIC);
  const float scale_xy = (float)(w / (double)pw);
  flow *= scale_xy;

  render_mimosa_core (frame_bgr, *prev_bgr, prev_delta, flow,
      strength_px, residual_gain, decay);

  if (prev_gray_out)
    gray.copyTo (*prev_gray_out);
}

/* ── Prawdziwy "wiggle" (Smooth3D-style) — tryb Z PLIKU (depth-file) ────────
 *
 * W przeciwieństwie do "mimozy" (residual przesuwany o ułamek px wzdłuż
 * wektora) to jest REALNE przesunięcie poziome pikseli, proporcjonalne do
 * głębi, oscylujące w czasie sinusoidalnie. To właśnie ten mechanizm daje
 * Smooth3D/Wiggle3D swój mocny efekt: mózg dostaje prawdziwą kinetyczną
 * paralaksę (bliższe obiekty przesuwają się bardziej niż tło), a nie tylko
 * podbity mikrokontrast na krawędziach ruchu.
 *
 * amount_px  — amplituda przesunięcia (px) dla najbliższego piksela (depth=1)
 * phase      — 2π · freq_hz · frame_count / fps (radiany)
 * Wymaga mapy głębi per-piksel, więc działa tylko z depth-file.
 * ─────────────────────────────────────────────────────────────────────────── */
static void
apply_depth_wiggle (cv::Mat &frame_bgr,          /* in/out */
                    const cv::Mat &depth_u8,     /* CV_8UC1, jaśniej = bliżej */
                    double amount_px,
                    double phase)
{
  if (amount_px <= 0.0)
    return;

  const int h = frame_bgr.rows, w = frame_bgr.cols;

  cv::Mat depth_f;
  depth_u8.convertTo (depth_f, CV_32F, 1.0 / 255.0);   /* 0..1, im jaśniej tym bliżej */

  const float osc = (float)(std::sin (phase) * amount_px);

  cv::Mat map_x (h, w, CV_32F), map_y (h, w, CV_32F);
  for (int y = 0; y < h; ++y) {
    const float *d  = depth_f.ptr<float> (y);
    float *mx = map_x.ptr<float> (y);
    float *my = map_y.ptr<float> (y);
    for (int x = 0; x < w; ++x) {
      mx[x] = (float)x - osc * d[x];   /* bliższe piksele przesuwają się mocniej */
      my[x] = (float)y;
    }
  }

  cv::remap (frame_bgr, frame_bgr, map_x, map_y,
             cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

/* ── Pseudo-flow z gradientu mapy głębi (Sobel) ──────────────────────────
 * W miejscach gdzie mapa się zmienia przestrzennie, gradient wskazuje
 * kierunek "w stronę" większej wartości — tani (O(N), jeden przebieg)
 * zamiennik kierunku optical-flow. To przybliżenie, nie prawdziwy wektor
 * ruchu. Używane, gdy nie ma dostępnego `motion-file` (realnego Farneback
 * z depthanalyzer). */
static cv::Mat
build_pseudo_flow_from_depth (const cv::Mat &depth_u8)
{
  cv::Mat depth_f;
  depth_u8.convertTo (depth_f, CV_32F, 1.0 / 255.0);

  cv::Mat gx, gy;
  cv::Sobel (depth_f, gx, CV_32F, 1, 0, 5);
  cv::Sobel (depth_f, gy, CV_32F, 0, 1, 5);

  /* Skala dobrana tak, żeby rząd wielkości pseudo-flow odpowiadał
   * typowemu |flow| z Farneback (kilka px) przy domyślnym smooth-ksize. */
  const float grad_scale = 60.0f;
  gx *= grad_scale;
  gy *= grad_scale;

  cv::Mat flow;
  std::vector<cv::Mat> ch = { gx, gy };
  cv::merge (ch, flow);
  return flow;
}

/* ── Dekwantyzacja realnego flow z motion-file ───────────────────────────
 * motion_u8: CV_8UC2, pełna rozdz. — kanał 0 = magnitude (0..255 względem
 * max_mag), kanał 1 = kąt (0..255 → 0..2π). Odwraca kwantyzację zapisaną
 * przez depthanalyzer (patrz niżej, sekcja motion-file). */
static cv::Mat
dequantize_motion_flow (const cv::Mat &motion_u8, double max_mag)
{
  cv::Mat flow (motion_u8.rows, motion_u8.cols, CV_32FC2);
  const float mm = (float) std::max (1e-3, max_mag);

  for (int y = 0; y < motion_u8.rows; ++y) {
    const guint8 *mo = motion_u8.ptr<guint8> (y);
    float *fl = flow.ptr<float> (y);
    for (int x = 0; x < motion_u8.cols; ++x) {
      float mag = ((float) mo[x * 2 + 0] / 255.0f) * mm;
      float ang = ((float) mo[x * 2 + 1] / 255.0f) * 2.0f * (float) M_PI;
      fl[x * 2 + 0] = mag * std::cos (ang);
      fl[x * 2 + 1] = mag * std::sin (ang);
    }
  }
  return flow;
}

/* ── Maska saliency zależna od prędkości ruchu ───────────────────────────
 * Analogicznie do build_depth_saliency, ale z kanału magnitude motion.raw:
 * szybciej poruszające się piksele → wyższa waga → mocniejszy unsharp/
 * detail boost w apply_saliency_detail_boost. */
static cv::Mat
build_motion_saliency (const cv::Mat &motion_u8, double gamma)
{
  cv::Mat mag8 (motion_u8.rows, motion_u8.cols, CV_8UC1);
  int from_to[] = { 0, 0 };
  cv::mixChannels (&motion_u8, 1, &mag8, 1, from_to, 1);

  cv::Mat m32;
  mag8.convertTo (m32, CV_32FC1, 1.0 / 255.0);

  if (std::abs (gamma - 1.0) > 1e-4)
    cv::pow (m32, gamma, m32);

  cv::GaussianBlur (m32, m32, cv::Size (0, 0), 1.5);
  return m32;
}

/* ── Mimoza transwektorowa — tryb Z PLIKU (depth-file, opcjonalnie + motion-file)
 *
 * To jest właściwe uzupełnienie brakującej części z v0.80: `depth_file`/
 * `depth_fp` były tam otwierane w `start()` i zamykane w `stop()`, ale
 * `transform_frame` nigdy z nich nie czytał — property fizycznie nic nie
 * robiła. Tutaj faktycznie czytamy jedną klatkę mapy (W×H, uint8) na klatkę
 * wideo i pomijamy Farneback (najdroższy krok trybu realtime).
 *
 * Kierunek flow: jeśli podano `real_flow` (zdekwantyzowany z motion-file —
 * prawdziwy Farneback policzony wcześniej w depthanalyzer), używamy go
 * bezpośrednio. W przeciwnym razie kierunek jest zgadywany z gradientu
 * mapy głębi (pseudo-flow, przybliżenie).
 * ─────────────────────────────────────────────────────────────────────────── */
static void
render_mimosa_from_depth (cv::Mat &frame_bgr,        /* in/out */
                          const cv::Mat *prev_bgr,
                          cv::Mat *prev_delta,
                          const cv::Mat &depth_u8,    /* CV_8UC1, pełna rozdz. */
                          const cv::Mat *real_flow,   /* opcjonalnie: CV_32FC2 z motion-file; nadpisuje pseudo-flow */
                          int strength_px,
                          double residual_gain,
                          double decay,
                          int depth_bands,            /* liczba pasm tonalnych (1 = wyłączone) */
                          double near_gain,
                          double far_gain,
                          double band_blur)
{
  const int h = frame_bgr.rows, w = frame_bgr.cols;

  /* Pierwsza klatka — tylko zapamiętaj, zero efektu */
  if (!prev_bgr || prev_bgr->empty () ||
      prev_bgr->size () != frame_bgr.size ()) {
    if (prev_delta)
      *prev_delta = cv::Mat::zeros (h, w, CV_32FC3);
    return;
  }

  cv::Mat flow = (real_flow && !real_flow->empty ())
      ? real_flow->clone ()
      : build_pseudo_flow_from_depth (depth_u8);

  cv::Mat gain_mask = build_depth_band_gain (depth_u8, depth_bands,
      near_gain, far_gain, band_blur);

  render_mimosa_core (frame_bgr, *prev_bgr, prev_delta, flow,
      strength_px, residual_gain, decay, &gain_mask);
}

/* ── Mimoza transwektorowa — tryb Z PLIKU, motion-file BEZ depth-file ────
 * Jak wyżej, ale bez mapy głębi (więc bez per-pasmowego gain_mask —
 * identyczne matematycznie do depth-bands=1 z domyślnymi near/far-gain,
 * patrz build_depth_band_gain). Flow zawsze realny (z motion.raw). */
static void
render_mimosa_from_flow (cv::Mat &frame_bgr,         /* in/out */
                         const cv::Mat *prev_bgr,
                         cv::Mat *prev_delta,
                         cv::Mat &flow,               /* CV_32FC2, zdekwantyzowany real flow */
                         int strength_px,
                         double residual_gain,
                         double decay)
{
  const int h = frame_bgr.rows, w = frame_bgr.cols;

  if (!prev_bgr || prev_bgr->empty () ||
      prev_bgr->size () != frame_bgr.size ()) {
    if (prev_delta)
      *prev_delta = cv::Mat::zeros (h, w, CV_32FC3);
    return;
  }

  render_mimosa_core (frame_bgr, *prev_bgr, prev_delta, flow,
      strength_px, residual_gain, decay);
}

/* Helper: BGR frame from GstVideoFrame (handles stride) */
static cv::Mat
frame_to_mat (GstVideoFrame *frame, bool *owned)
{
  guint8 *data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (frame, 0);
  const int w  = GST_VIDEO_FRAME_WIDTH (frame);
  const int h  = GST_VIDEO_FRAME_HEIGHT (frame);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE (frame, 0);
  /* Zawsze ciągła kopia — bezpieczne dla OpenCV ops */
  cv::Mat m (h, w, CV_8UC3);
  for (int y = 0; y < h; ++y)
    std::memcpy (m.ptr (y), data + y * stride, w * 3);
  *owned = true;
  return m;
}

static void
mat_to_frame (const cv::Mat &m, GstVideoFrame *frame)
{
  guint8 *data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (frame, 0);
  const int w  = GST_VIDEO_FRAME_WIDTH (frame);
  const int h  = GST_VIDEO_FRAME_HEIGHT (frame);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE (frame, 0);
  if (stride == w * 3 && m.isContinuous ()) {
    std::memcpy (data, m.data, (size_t)w * h * 3);
  } else {
    for (int y = 0; y < h; ++y)
      std::memcpy (data + y * stride, m.ptr (y), w * 3);
  }
}

/* ── Saliency-weighted local detail boost ("clarity") ────────────────────
 * Nie jest to prawdziwe HDR (łączenie ekspozycji) — to lokalny kontrast/
 * detail boost (unsharp na wysokich częstotliwościach) ważony maską
 * "uwagi": obszary centralne kadru dostają silniejsze wzmocnienie
 * szczegółu niż peryferie. Tani proxy prawdziwej saliency (bez detekcji
 * twarzy/obiektów) — w praktyce dobrze pokrywa się z tym, gdzie widz
 * naturalnie patrzy w większości ujęć.
 *
 * Maska jest budowana raz per rozdzielczość (cache w GstParallaxDoF) —
 * per-klatkowy koszt to tylko GaussianBlur + odejmowanie + mnożenie.
 * ─────────────────────────────────────────────────────────────────────────── */
static cv::Mat
build_center_saliency (int w, int h, double radius_frac)
{
  cv::Mat m (h, w, CV_32FC1);
  const float cx = w * 0.5f, cy = h * 0.5f;
  const float half_diag = std::sqrt ((float)(w * w + h * h)) * 0.5f;
  const float sigma = std::max (1.0f, (float)(radius_frac * half_diag));

  for (int y = 0; y < h; ++y) {
    float *row = m.ptr<float> (y);
    float dy = (float)y - cy;
    for (int x = 0; x < w; ++x) {
      float dx = (float)x - cx;
      float d2 = dx * dx + dy * dy;
      row[x] = std::exp (-d2 / (2.0f * sigma * sigma));
    }
  }
  return m;
}

static void
apply_saliency_detail_boost (cv::Mat &frame_bgr,     /* in/out */
                             const cv::Mat &saliency, /* CV_32FC1, pełna rozdz., 0..1 */
                             double boost,
                             double detail_sigma)
{
  if (boost <= 0.0)
    return;

  cv::Mat cur_f, blur_f, high;
  frame_bgr.convertTo (cur_f, CV_32FC3);
  cv::GaussianBlur (cur_f, blur_f, cv::Size (0, 0), detail_sigma);
  high = cur_f - blur_f;   /* wysokoczęstotliwościowy detal */

  cv::Mat w = saliency * (float)boost;
  std::vector<cv::Mat> wch = { w, w, w };
  cv::Mat w3;
  cv::merge (wch, w3);

  cv::Mat out_f = cur_f + high.mul (w3);
  out_f.convertTo (frame_bgr, CV_8U);
}

/* ── Depth-modulated saliency ────────────────────────────────────────────
 * Buduje maskę z mapy głębi (jaśniej = bliżej → wyższa waga) i miesza ją
 * z center-weighted maską. weight=0 → czysty center (kompatybilność
 * wsteczna); weight=1 → czysta mapa głębi.
 * ─────────────────────────────────────────────────────────────────────────── */
static cv::Mat
build_depth_saliency (const cv::Mat &depth_u8, double gamma)
{
  cv::Mat d32;
  depth_u8.convertTo (d32, CV_32FC1, 1.0 / 255.0);   /* 0..1 */

  if (std::abs (gamma - 1.0) > 1e-4)
    cv::pow (d32, gamma, d32);

  /* lekkie wygładzenie granic — unikamy twardych krawędzi pasm */
  cv::GaussianBlur (d32, d32, cv::Size (0, 0), 1.5);
  return d32;
}

static cv::Mat
build_combined_saliency (const cv::Mat &center,
                         const cv::Mat &depth_mask,
                         double weight)
{
  if (weight <= 0.0 || depth_mask.empty ())
    return center.clone ();

  weight = std::max (0.0, std::min (1.0, weight));
  cv::Mat out;
  cv::addWeighted (center, 1.0 - weight, depth_mask, weight, 0.0, out);
  return out;
}

/* ── Mapa ruchu dla depthanalyzer: różnica klatek + Sobel ─────────────────
 * Zgodnie z README ("logika 1:1 z 3dplayer1.py: różnica klatek + Sobel"),
 * ale `flow_weight`/`edge_weight` były tylko przyjmowane i od razu wyciszane
 * `(void)` — Sobel w ogóle nie istniał, więc oba suwaki nie miały żadnego
 * wpływu na wynik. Dokładamy brakującą składową krawędziową i faktycznie
 * ważymy nią kombinację z różnicą klatek. */
static cv::Mat
estimate_depth (cv::Mat *prev_gray,
                const cv::Mat &frame_bgr,
                double proc_scale,
                int smooth_ksize,
                float flow_weight,
                float edge_weight)
{
  const int fw = frame_bgr.cols, fh = frame_bgr.rows;
  const int pw = std::max (1, (int)(fw * proc_scale));
  const int ph = std::max (1, (int)(fh * proc_scale));

  cv::Mat small, gray;
  cv::resize (frame_bgr, small, cv::Size (pw, ph), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor (small, gray, cv::COLOR_BGR2GRAY);

  /* Składowa 1: różnica klatek — tani proxy ruchu/„flow” (jak dotychczas) */
  cv::Mat diff = cv::Mat::zeros (ph, pw, CV_32F);
  if (prev_gray && !prev_gray->empty () && prev_gray->size () == gray.size ()) {
    cv::Mat g0, g1;
    prev_gray->convertTo (g0, CV_32F);
    gray.convertTo (g1, CV_32F);
    cv::absdiff (g1, g0, diff);
    diff /= 255.0f;
  }
  if (prev_gray)
    gray.copyTo (*prev_gray);

  /* Składowa 2: krawędzie (Sobel) — brakująca część z README/3dplayer1.py.
   * Mocne krawędzie zwykle leżą na granicach obiektów/pierwszego planu,
   * więc to niezależny (od ruchu) sygnał "bliskości". */
  cv::Mat gray_f, gx, gy, edge;
  gray.convertTo (gray_f, CV_32F);
  cv::Sobel (gray_f, gx, CV_32F, 1, 0, 3);
  cv::Sobel (gray_f, gy, CV_32F, 0, 1, 3);
  cv::magnitude (gx, gy, edge);
  edge /= 255.0f * 4.0f;   /* Sobel 3x3 na obrazie 0..255 → rząd wielkości do ~[0,~kilka] */
  cv::min (edge, 1.0f, edge);

  /* Ważona kombinacja — flow_weight/edge_weight teraz realnie coś robią.
   * Suma wag normalizuje wynik niezależnie od tego, jak użytkownik je ustawi. */
  const float wsum = std::max (1e-3f, flow_weight + edge_weight);
  cv::Mat combined = (diff * flow_weight + edge * edge_weight) / wsum;

  int ks = smooth_ksize | 1;
  if (ks < 3) ks = 3;
  cv::GaussianBlur (combined, combined, cv::Size (ks, ks), ks / 6.0);

  double dmx;
  cv::minMaxLoc (combined, nullptr, &dmx);
  if (dmx > 0) combined /= (float)dmx;

  if (pw != fw || ph != fh)
    cv::resize (combined, combined, cv::Size (fw, fh), 0, 0, cv::INTER_LINEAR);
  return combined;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ELEMENT 1: depthanalyzer
 * Zapisuje mapy głębi (uint8, W×H) sekwencyjnie do pliku .raw
 * Video przechodzi bez zmian (identity).
 * ═════════════════════════════════════════════════════════════════════════ */

GST_DEBUG_CATEGORY_STATIC (gst_depth_analyzer_debug);
#define GST_CAT_DEFAULT gst_depth_analyzer_debug

enum {
  DA_PROP_0,
  DA_PROP_LOCATION,
  DA_PROP_PROC_SCALE,
  DA_PROP_SMOOTH_KSIZE,
  DA_PROP_FLOW_WEIGHT,
  DA_PROP_EDGE_WEIGHT,
  DA_PROP_MOTION_FILE,
  DA_PROP_MOTION_MAX_MAG,
};

struct _GstDepthAnalyzer {
  GstVideoFilter parent;

  gchar  *location;
  gdouble proc_scale;
  gint    smooth_ksize;
  gdouble flow_weight;
  gdouble edge_weight;
  gchar  *motion_file;
  gdouble motion_max_mag;

  cv::Mat *prev_gray;
  cv::Mat *prev_gray_full;  /* pełna rozdz., osobno od prev_gray — tylko dla motion-file */
  FILE    *fp;
  FILE    *motion_fp;
  gint     frame_count;
  gint     width, height;
};

#define GST_TYPE_DEPTH_ANALYZER (gst_depth_analyzer_get_type ())
G_DECLARE_FINAL_TYPE (GstDepthAnalyzer, gst_depth_analyzer,
                      GST, DEPTH_ANALYZER, GstVideoFilter)
G_DEFINE_TYPE (GstDepthAnalyzer, gst_depth_analyzer, GST_TYPE_VIDEO_FILTER)

static GstStaticPadTemplate da_sink_tmpl = GST_STATIC_PAD_TEMPLATE (
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)BGR"));
static GstStaticPadTemplate da_src_tmpl = GST_STATIC_PAD_TEMPLATE (
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)BGR"));

static gboolean
gst_depth_analyzer_set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info)
{
  (void)incaps; (void)outcaps; (void)out_info;
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (filter);
  self->width  = GST_VIDEO_INFO_WIDTH (in_info);
  self->height = GST_VIDEO_INFO_HEIGHT (in_info);
  self->prev_gray->release ();
  self->prev_gray_full->release ();
  self->frame_count = 0;
  GST_INFO_OBJECT (self, "set_info: %dx%d", self->width, self->height);
  return TRUE;
}

static gboolean
gst_depth_analyzer_start (GstBaseTransform *trans)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (trans);
  if (self->fp) {
    fclose (self->fp);
    self->fp = nullptr;
  }
  if (self->motion_fp) {
    fclose (self->motion_fp);
    self->motion_fp = nullptr;
  }
  if (!self->location || !self->location[0]) {
    GST_ERROR_OBJECT (self, "property 'location' is required");
    return FALSE;
  }
  self->fp = fopen (self->location, "wb");
  if (!self->fp) {
    GST_ERROR_OBJECT (self, "cannot open depth file for writing: %s", self->location);
    return FALSE;
  }
  if (self->motion_file && self->motion_file[0]) {
    self->motion_fp = fopen (self->motion_file, "wb");
    if (!self->motion_fp) {
      GST_ERROR_OBJECT (self, "cannot open motion file for writing: %s", self->motion_file);
      return FALSE;
    }
    GST_INFO_OBJECT (self, "writing motion vectors to %s (max-mag=%.2f)",
        self->motion_file, self->motion_max_mag);
  }
  self->frame_count = 0;
  self->prev_gray->release ();
  self->prev_gray_full->release ();
  GST_INFO_OBJECT (self, "writing depths to %s", self->location);
  return TRUE;
}

static gboolean
gst_depth_analyzer_stop (GstBaseTransform *trans)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (trans);
  if (self->fp) {
    fclose (self->fp);
    self->fp = nullptr;
    GST_INFO_OBJECT (self, "closed depth file after %d frames (%dx%d)",
        self->frame_count, self->width, self->height);
  }
  if (self->motion_fp) {
    fclose (self->motion_fp);
    self->motion_fp = nullptr;
    GST_INFO_OBJECT (self, "closed motion file after %d frames (%dx%d)",
        self->frame_count, self->width, self->height);
  }
  return TRUE;
}

static GstFlowReturn
gst_depth_analyzer_transform_frame_ip (GstVideoFilter *filter, GstVideoFrame *frame)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (filter);
  if (!self->fp)
    return GST_FLOW_ERROR;

  bool owned = false;
  cv::Mat bgr = frame_to_mat (frame, &owned);

  cv::Mat depth = estimate_depth (
      self->prev_gray, bgr,
      self->proc_scale, self->smooth_ksize,
      (float)self->flow_weight, (float)self->edge_weight);

  /* Zapis uint8 jak w Python: (d * 255).astype(np.uint8) */
  cv::Mat depth_u8;
  depth.convertTo (depth_u8, CV_8U, 255.0);

  size_t nbytes = (size_t)self->width * self->height;
  if (fwrite (depth_u8.data, 1, nbytes, self->fp) != nbytes) {
    GST_ERROR_OBJECT (self, "write error on depth file");
    return GST_FLOW_ERROR;
  }

  /* ── motion-file: realny Farneback (vx,vy), niezależny od estimate_depth ──
   * estimate_depth() liczy diff+Sobel na obniżonej rozdzielczości i sam
   * zarządza self->prev_gray — nie ma tam prawdziwego kierunku ruchu.
   * Tu liczymy osobny, realny optical flow (ten sam Farneback co realtime
   * w parallaxdof) i kwantyzujemy go do 2 bajtów/piksel: magnitude + kąt. */
  if (self->motion_fp) {
    cv::Mat gray_full;
    cv::cvtColor (bgr, gray_full, cv::COLOR_BGR2GRAY);

    cv::Mat flow_full;
    if (!self->prev_gray_full->empty () &&
        self->prev_gray_full->size () == gray_full.size ()) {
      const int pw = std::max (1, (int)(self->width * self->proc_scale));
      const int ph = std::max (1, (int)(self->height * self->proc_scale));

      cv::Mat g0, g1;
      cv::resize (*self->prev_gray_full, g0, cv::Size (pw, ph), 0, 0, cv::INTER_AREA);
      cv::resize (gray_full, g1, cv::Size (pw, ph), 0, 0, cv::INTER_AREA);

      cv::Mat flow_s;
      compute_flow (g0, g1, flow_s);

      cv::resize (flow_s, flow_full, cv::Size (self->width, self->height), 0, 0, cv::INTER_CUBIC);
      const float scale_xy = (float)(self->width / (double)pw);
      flow_full *= scale_xy;
    } else {
      flow_full = cv::Mat::zeros (self->height, self->width, CV_32FC2);
    }
    gray_full.copyTo (*self->prev_gray_full);

    cv::Mat motion_u8 (self->height, self->width, CV_8UC2);
    const float max_mag = (float) std::max (1e-3, self->motion_max_mag);
    const float two_pi = 2.0f * (float) M_PI;

    for (int y = 0; y < self->height; ++y) {
      const float *fl = flow_full.ptr<float> (y);
      guint8 *mo = motion_u8.ptr<guint8> (y);
      for (int x = 0; x < self->width; ++x) {
        float vx = fl[x * 2 + 0], vy = fl[x * 2 + 1];
        float mag = (std::isfinite (vx) && std::isfinite (vy))
            ? std::sqrt (vx * vx + vy * vy) : 0.f;
        if (mag > max_mag) mag = max_mag;

        float ang = std::atan2 (vy, vx);       /* (-pi, pi] */
        if (ang < 0.f) ang += two_pi;

        mo[x * 2 + 0] = (guint8) std::lround ((mag / max_mag) * 255.0f);
        mo[x * 2 + 1] = (guint8) std::lround ((ang / two_pi) * 255.0f);
      }
    }

    size_t mbytes = (size_t) self->width * self->height * 2;
    if (fwrite (motion_u8.data, 1, mbytes, self->motion_fp) != mbytes) {
      GST_ERROR_OBJECT (self, "write error on motion file");
      return GST_FLOW_ERROR;
    }
  }

  self->frame_count++;

  /* Video pass-through — nic nie zmieniamy w klatce */
  return GST_FLOW_OK;
}

static void
gst_depth_analyzer_set_property (GObject *obj, guint id,
    const GValue *v, GParamSpec *ps)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (obj);
  switch (id) {
    case DA_PROP_LOCATION:
      g_free (self->location);
      self->location = g_value_dup_string (v);
      break;
    case DA_PROP_PROC_SCALE:
      self->proc_scale = g_value_get_double (v);
      self->prev_gray->release ();
      break;
    case DA_PROP_SMOOTH_KSIZE:
      self->smooth_ksize = g_value_get_int (v);
      break;
    case DA_PROP_FLOW_WEIGHT:
      self->flow_weight = g_value_get_double (v);
      break;
    case DA_PROP_EDGE_WEIGHT:
      self->edge_weight = g_value_get_double (v);
      break;
    case DA_PROP_MOTION_FILE:
      g_free (self->motion_file);
      self->motion_file = g_value_dup_string (v);
      break;
    case DA_PROP_MOTION_MAX_MAG:
      self->motion_max_mag = g_value_get_double (v);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, id, ps);
  }
}

static void
gst_depth_analyzer_get_property (GObject *obj, guint id,
    GValue *v, GParamSpec *ps)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (obj);
  switch (id) {
    case DA_PROP_LOCATION:     g_value_set_string (v, self->location);     break;
    case DA_PROP_PROC_SCALE:   g_value_set_double (v, self->proc_scale);   break;
    case DA_PROP_SMOOTH_KSIZE: g_value_set_int    (v, self->smooth_ksize); break;
    case DA_PROP_FLOW_WEIGHT:  g_value_set_double (v, self->flow_weight);  break;
    case DA_PROP_EDGE_WEIGHT:  g_value_set_double (v, self->edge_weight);  break;
    case DA_PROP_MOTION_FILE:      g_value_set_string (v, self->motion_file);     break;
    case DA_PROP_MOTION_MAX_MAG:   g_value_set_double (v, self->motion_max_mag);  break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, id, ps);
  }
}

static void
gst_depth_analyzer_finalize (GObject *obj)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (obj);
  if (self->fp) { fclose (self->fp); self->fp = nullptr; }
  if (self->motion_fp) { fclose (self->motion_fp); self->motion_fp = nullptr; }
  g_free (self->location);
  g_free (self->motion_file);
  delete self->prev_gray;
  delete self->prev_gray_full;
  G_OBJECT_CLASS (gst_depth_analyzer_parent_class)->finalize (obj);
}

static void
gst_depth_analyzer_init (GstDepthAnalyzer *self)
{
  self->location     = nullptr;
  self->proc_scale   = DEFAULT_PROC_SCALE;
  self->smooth_ksize = DEFAULT_SMOOTH_KSIZE;
  self->flow_weight  = DEFAULT_FLOW_WEIGHT;
  self->edge_weight  = DEFAULT_EDGE_WEIGHT;
  self->motion_file     = nullptr;
  self->motion_max_mag  = DEFAULT_MOTION_MAX_MAG;
  self->prev_gray    = new cv::Mat ();
  self->prev_gray_full = new cv::Mat ();
  self->fp           = nullptr;
  self->motion_fp    = nullptr;
  self->frame_count  = 0;
  self->width = self->height = 0;
}

static void
gst_depth_analyzer_class_init (GstDepthAnalyzerClass *klass)
{
  GObjectClass        *obj_cls = G_OBJECT_CLASS (klass);
  GstElementClass     *el_cls  = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_cls = GST_BASE_TRANSFORM_CLASS (klass);
  GstVideoFilterClass *vf_cls  = GST_VIDEO_FILTER_CLASS (klass);

  obj_cls->set_property = gst_depth_analyzer_set_property;
  obj_cls->get_property = gst_depth_analyzer_get_property;
  obj_cls->finalize     = gst_depth_analyzer_finalize;

  g_object_class_install_property (obj_cls, DA_PROP_LOCATION,
      g_param_spec_string ("location", "Depth file",
          "Path to write sequential depth maps (uint8, W×H per frame)",
          nullptr,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, DA_PROP_PROC_SCALE,
      g_param_spec_double ("proc-scale", "Processing scale",
          "Downscale for depth estimation",
          0.1, 1.0, DEFAULT_PROC_SCALE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, DA_PROP_SMOOTH_KSIZE,
      g_param_spec_int ("smooth-ksize", "Smooth kernel size",
          "Gaussian kernel size for depth smoothing",
          3, 51, DEFAULT_SMOOTH_KSIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, DA_PROP_FLOW_WEIGHT,
      g_param_spec_double ("flow-weight", "Flow weight",
          "Weight of optical-flow magnitude",
          0.0, 1.0, DEFAULT_FLOW_WEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, DA_PROP_EDGE_WEIGHT,
      g_param_spec_double ("edge-weight", "Edge weight",
          "Weight of Sobel edges",
          0.0, 1.0, DEFAULT_EDGE_WEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, DA_PROP_MOTION_FILE,
      g_param_spec_string ("motion-file", "Motion file",
          "Ścieżka do zapisu realnego optical-flow (Farneback), 2 bajty/piksel "
          "(magnitude, kąt) na klatkę. Opcjonalne, niezależne od 'location'.",
          nullptr,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, DA_PROP_MOTION_MAX_MAG,
      g_param_spec_double ("motion-max-mag", "Motion max magnitude",
          "Clamp (px/klatkę) do kwantyzacji magnitude na 0..255 w motion-file. "
          "Musi być taki sam przy odczycie w parallaxdof (motion-max-mag).",
          0.1, 200.0, DEFAULT_MOTION_MAX_MAG,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata (el_cls,
      "DepthAnalyzer",
      "Filter/Effect/Video",
      "Pre-analyze depth (flow+Sobel) and optionally real Farneback motion, write to .raw files",
      "Tomasz / stpf99");

  gst_element_class_add_static_pad_template (el_cls, &da_sink_tmpl);
  gst_element_class_add_static_pad_template (el_cls, &da_src_tmpl);

  bt_cls->start = GST_DEBUG_FUNCPTR (gst_depth_analyzer_start);
  bt_cls->stop  = GST_DEBUG_FUNCPTR (gst_depth_analyzer_stop);
  vf_cls->set_info           = GST_DEBUG_FUNCPTR (gst_depth_analyzer_set_info);
  vf_cls->transform_frame_ip = GST_DEBUG_FUNCPTR (gst_depth_analyzer_transform_frame_ip);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ELEMENT 2: parallaxdof
 * ═════════════════════════════════════════════════════════════════════════ */

GST_DEBUG_CATEGORY_STATIC (gst_parallax_dof_debug);
#undef GST_CAT_DEFAULT
#define GST_CAT_DEFAULT gst_parallax_dof_debug

enum {
  PROP_0,
  PROP_PARALLAX_SHIFT,
  PROP_FG_THRESHOLD,
  PROP_BG_BLUR,
  PROP_FG_SHARPNESS,
  PROP_PROC_SCALE,
  PROP_MAX_FG_CLUSTERS,
  PROP_SMOOTH_KSIZE,
  PROP_FLOW_WEIGHT,
  PROP_EDGE_WEIGHT,
  PROP_DEPTH_FILE,
  PROP_DEPTH_BANDS,
  PROP_NEAR_GAIN,
  PROP_FAR_GAIN,
  PROP_BAND_BLUR,
  PROP_SALIENCY_BOOST,
  PROP_SALIENCY_RADIUS,
  PROP_SALIENCY_SIGMA,
  PROP_DEPTH_SALIENCY_WEIGHT,
  PROP_DEPTH_SALIENCY_GAMMA,
  PROP_WIGGLE_AMOUNT,
  PROP_WIGGLE_FREQ_HZ,
  PROP_MOTION_FILE,
  PROP_MOTION_MAX_MAG,
  PROP_MOTION_SALIENCY_WEIGHT,
  PROP_MOTION_SALIENCY_GAMMA,
};

struct _GstParallaxDoF {
  GstVideoFilter parent;

  gint    parallax_shift;
  gdouble fg_threshold;
  gdouble bg_blur;
  gdouble fg_sharpness;
  gdouble proc_scale;
  gint    max_fg_clusters;
  gint    smooth_ksize;
  gdouble flow_weight;
  gdouble edge_weight;
  gchar  *depth_file;
  gint    depth_bands;
  gdouble near_gain;
  gdouble far_gain;
  gdouble band_blur;
  gdouble saliency_boost;
  gdouble saliency_radius;
  gdouble saliency_sigma;
  gdouble depth_saliency_weight;
  gdouble depth_saliency_gamma;
  gdouble wiggle_amount;
  gdouble wiggle_freq_hz;
  gchar  *motion_file;
  gdouble motion_max_mag;
  gdouble motion_saliency_weight;
  gdouble motion_saliency_gamma;

  cv::Mat *prev_gray;
  cv::Mat *prev_bgr;      /* oryginał poprzedniej klatki */
  cv::Mat *prev_delta;    /* poprzednio nałożone zmiany (do odejmowania) */
  cv::Mat *saliency_mask; /* cache center-weighted maski, per rozdzielczość */
  FILE    *depth_fp;
  FILE    *motion_fp;
  gint     width, height;
  gint     fps_n, fps_d;   /* do fazy oscylacji wiggle */
  gint     frame_count;    /* do fazy oscylacji wiggle */
};

#define GST_TYPE_PARALLAX_DOF (gst_parallax_dof_get_type ())
G_DECLARE_FINAL_TYPE (GstParallaxDoF, gst_parallax_dof,
                      GST, PARALLAX_DOF, GstVideoFilter)
G_DEFINE_TYPE (GstParallaxDoF, gst_parallax_dof, GST_TYPE_VIDEO_FILTER)

static GstStaticPadTemplate pd_sink_tmpl = GST_STATIC_PAD_TEMPLATE (
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)BGR"));
static GstStaticPadTemplate pd_src_tmpl = GST_STATIC_PAD_TEMPLATE (
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)BGR"));

static gboolean
gst_parallax_dof_set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info)
{
  (void)incaps; (void)outcaps; (void)out_info;
  GstParallaxDoF *self = GST_PARALLAX_DOF (filter);
  self->width  = GST_VIDEO_INFO_WIDTH (in_info);
  self->height = GST_VIDEO_INFO_HEIGHT (in_info);
  self->fps_n  = GST_VIDEO_INFO_FPS_N (in_info);
  self->fps_d  = GST_VIDEO_INFO_FPS_D (in_info);
  self->prev_gray->release ();
  self->prev_bgr->release ();
  self->prev_delta->release ();
  self->saliency_mask->release ();
  GST_INFO_OBJECT (self, "set_info: %dx%d", self->width, self->height);
  return TRUE;
}

static gboolean
gst_parallax_dof_start (GstBaseTransform *trans)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (trans);
  if (self->depth_fp) {
    fclose (self->depth_fp);
    self->depth_fp = nullptr;
  }
  if (self->motion_fp) {
    fclose (self->motion_fp);
    self->motion_fp = nullptr;
  }
  if (self->depth_file && self->depth_file[0]) {
    self->depth_fp = fopen (self->depth_file, "rb");
    if (!self->depth_fp) {
      GST_ERROR_OBJECT (self, "cannot open depth-file: %s", self->depth_file);
      return FALSE;
    }
    GST_INFO_OBJECT (self, "using precomputed depths from %s", self->depth_file);
  } else {
    GST_INFO_OBJECT (self, "realtime depth estimation (no depth-file)");
  }
  if (self->motion_file && self->motion_file[0]) {
    self->motion_fp = fopen (self->motion_file, "rb");
    if (!self->motion_fp) {
      GST_ERROR_OBJECT (self, "cannot open motion-file: %s", self->motion_file);
      return FALSE;
    }
    GST_INFO_OBJECT (self, "using precomputed real flow from %s (max-mag=%.2f)",
        self->motion_file, self->motion_max_mag);
  }
  self->prev_gray->release ();
  self->prev_bgr->release ();
  self->prev_delta->release ();
  self->frame_count = 0;
  return TRUE;
}

static gboolean
gst_parallax_dof_stop (GstBaseTransform *trans)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (trans);
  if (self->depth_fp) {
    fclose (self->depth_fp);
    self->depth_fp = nullptr;
  }
  if (self->motion_fp) {
    fclose (self->motion_fp);
    self->motion_fp = nullptr;
  }
  return TRUE;
}

static GstFlowReturn
gst_parallax_dof_transform_frame (GstVideoFilter *filter,
    GstVideoFrame *inframe, GstVideoFrame *outframe)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (filter);

  bool owned = false;
  cv::Mat bgr = frame_to_mat (inframe, &owned);
  cv::Mat orig_for_next = bgr.clone ();  /* zanim render_mimosa* zmodyfikuje bgr in-place */
  self->frame_count++;

  /* Mimoza transwektorowa:
   * strength  ← parallax_shift
   * gain      ← fg_sharpness (reużycie suwaka jako residual-gain)
   * decay     ← 0.85 stałe (odejmowanie poprzednich zmian)
   */
  /* fg_sharpness 1.5 default → gain ~0.4 (delikatnie) */
  double gain = self->fg_sharpness * 0.25;
  if (gain < 0.05) gain = 0.05;
  if (gain > 2.0) gain = 2.0;

  /* Mapa głębi / realny flow bieżącej klatki — dostępne dla
   * render_mimosa_from_depth/render_mimosa_from_flow oraz dla
   * depth/motion-modulated saliency. Oba pliki są niezależne od siebie:
   * albo, oba, albo żaden. */
  cv::Mat depth_u8;
  bool have_depth = false;
  cv::Mat motion_u8;
  bool have_motion = false;

  if (self->depth_fp) {
    /* Tryb Z PLIKU: czytamy jedną klatkę mapy (W×H, uint8) z depth-file
     * i pomijamy Farneback (najdroższy krok trybu realtime). */
    size_t nbytes = (size_t) self->width * (size_t) self->height;
    depth_u8.create (self->height, self->width, CV_8UC1);
    size_t got = fread (depth_u8.data, 1, nbytes, self->depth_fp);

    if (got != nbytes) {
      /* Plik krótszy niż wideo (np. inny frame count) — od tej klatki
       * po prostu wyłączamy depth (spadamy na motion-only albo realtime
       * poniżej), zamiast wywalać pipeline. */
      GST_WARNING_OBJECT (self,
          "depth-file exhausted after short read (%zu/%zu bytes) — "
          "disabling depth for the rest of the stream",
          got, nbytes);
      fclose (self->depth_fp);
      self->depth_fp = nullptr;
      depth_u8.release ();
    } else {
      have_depth = true;
    }
  }

  if (self->motion_fp) {
    /* motion-file: 2 bajty/piksel (magnitude, kąt) — realny Farneback
     * policzony wcześniej w depthanalyzer. */
    size_t mbytes = (size_t) self->width * (size_t) self->height * 2;
    motion_u8.create (self->height, self->width, CV_8UC2);
    size_t got = fread (motion_u8.data, 1, mbytes, self->motion_fp);

    if (got != mbytes) {
      GST_WARNING_OBJECT (self,
          "motion-file exhausted after short read (%zu/%zu bytes) — "
          "disabling real-flow/motion-saliency for the rest of the stream",
          got, mbytes);
      fclose (self->motion_fp);
      self->motion_fp = nullptr;
      motion_u8.release ();
    } else {
      have_motion = true;
    }
  }

  /* Kolejność preferencji źródła kierunku flow dla mimozy:
   *   motion-file (realny Farneback, jeśli dostępny)
   *   > depth-file (pseudo-flow z gradientu głębi)
   *   > realtime Farneback (live, najdroższy). */
  if (have_depth) {
    cv::Mat real_flow;
    const cv::Mat *rf = nullptr;
    if (have_motion) {
      real_flow = dequantize_motion_flow (motion_u8, self->motion_max_mag);
      rf = &real_flow;
    }
    render_mimosa_from_depth (bgr, self->prev_bgr, self->prev_delta,
        depth_u8, rf, self->parallax_shift, gain, 0.92,
        self->depth_bands, self->near_gain, self->far_gain, self->band_blur);

    /* Smooth3D-style wiggle: realne przesunięcie px zależne od głębi,
     * oscylujące sinusoidalnie. Niezależne od "mimozy" — dodaje
     * prawdziwą kinetyczną paralaksę. Wyłączone domyślnie (amount=0). */
    if (self->wiggle_amount > 0.0) {
      double fps = (self->fps_d > 0)
          ? (double)self->fps_n / (double)self->fps_d : 25.0;
      double phase = 2.0 * M_PI * self->wiggle_freq_hz *
          ((double)self->frame_count / fps);
      apply_depth_wiggle (bgr, depth_u8, self->wiggle_amount, phase);
    }
  } else if (have_motion) {
    cv::Mat real_flow = dequantize_motion_flow (motion_u8, self->motion_max_mag);
    render_mimosa_from_flow (bgr, self->prev_bgr, self->prev_delta,
        real_flow, self->parallax_shift, gain, 0.92);
  } else {
    render_mimosa (bgr,
        self->prev_bgr,
        self->prev_delta,
        self->prev_gray,
        self->parallax_shift,
        gain,
        0.92,
        self->proc_scale);
  }

  /* zapamiętaj oryginał bieżącej jako prev do następnej klatki
   * (bierzemy z inframe, nie z przetworzonego bgr) */
  {
    orig_for_next.copyTo (*self->prev_bgr);
  }

  /* Saliency-weighted local detail boost — niezależny etap post-processingu.
   * Gdy depth-saliency-weight > 0 i mamy mapę głębi, mieszamy center-maskę
   * z maską zbudowaną z depth (bliżej = mocniejszy detal).
   * Wyłączony domyślnie (saliency-boost=0) — zero kosztu i zero zmiany
   * zachowania, jeśli użytkownik go nie włączy. */
  if (self->saliency_boost > 0.0) {
    if (self->saliency_mask->empty () ||
        self->saliency_mask->rows != self->height ||
        self->saliency_mask->cols != self->width) {
      *self->saliency_mask = build_center_saliency (
          self->width, self->height, self->saliency_radius);
    }

    cv::Mat final_mask = *self->saliency_mask;

    if (self->depth_saliency_weight > 0.0 && have_depth && !depth_u8.empty ()) {
      cv::Mat depth_mask = build_depth_saliency (
          depth_u8, self->depth_saliency_gamma);
      final_mask = build_combined_saliency (
          final_mask, depth_mask, self->depth_saliency_weight);
    }

    /* Kontrast zależny od prędkości: szybciej poruszające się piksele
     * (wyższy magnitude w motion.raw) dostają mocniejszy unsharp/detail
     * boost. Miksowane z istniejącą maską (center i/lub depth) — można
     * łączyć wszystkie trzy źródła naraz. */
    if (self->motion_saliency_weight > 0.0 && have_motion && !motion_u8.empty ()) {
      cv::Mat motion_mask = build_motion_saliency (
          motion_u8, self->motion_saliency_gamma);
      final_mask = build_combined_saliency (
          final_mask, motion_mask, self->motion_saliency_weight);
    }

    apply_saliency_detail_boost (bgr, final_mask,
        self->saliency_boost, self->saliency_sigma);
  }

  mat_to_frame (bgr, outframe);
  return GST_FLOW_OK;
}
static void
gst_parallax_dof_set_property (GObject *obj, guint id,
    const GValue *v, GParamSpec *ps)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (obj);
  switch (id) {
    case PROP_PARALLAX_SHIFT:  self->parallax_shift  = g_value_get_int (v);    break;
    case PROP_FG_THRESHOLD:    self->fg_threshold    = g_value_get_double (v); break;
    case PROP_BG_BLUR:         self->bg_blur         = g_value_get_double (v); break;
    case PROP_FG_SHARPNESS:    self->fg_sharpness    = g_value_get_double (v); break;
    case PROP_PROC_SCALE:
      self->proc_scale = g_value_get_double (v);
      self->prev_gray->release ();
      break;
    case PROP_MAX_FG_CLUSTERS: self->max_fg_clusters = g_value_get_int (v);    break;
    case PROP_SMOOTH_KSIZE:    self->smooth_ksize    = g_value_get_int (v);    break;
    case PROP_FLOW_WEIGHT:     self->flow_weight     = g_value_get_double (v); break;
    case PROP_EDGE_WEIGHT:     self->edge_weight     = g_value_get_double (v); break;
    case PROP_DEPTH_FILE:
      g_free (self->depth_file);
      self->depth_file = g_value_dup_string (v);
      break;
    case PROP_DEPTH_BANDS:     self->depth_bands     = g_value_get_int (v);    break;
    case PROP_NEAR_GAIN:       self->near_gain        = g_value_get_double (v); break;
    case PROP_FAR_GAIN:        self->far_gain         = g_value_get_double (v); break;
    case PROP_BAND_BLUR:       self->band_blur        = g_value_get_double (v); break;
    case PROP_SALIENCY_BOOST:  self->saliency_boost   = g_value_get_double (v); break;
    case PROP_SALIENCY_RADIUS: self->saliency_radius  = g_value_get_double (v); break;
    case PROP_SALIENCY_SIGMA:  self->saliency_sigma   = g_value_get_double (v); break;
    case PROP_DEPTH_SALIENCY_WEIGHT:
      self->depth_saliency_weight = g_value_get_double (v); break;
    case PROP_DEPTH_SALIENCY_GAMMA:
      self->depth_saliency_gamma  = g_value_get_double (v); break;
    case PROP_WIGGLE_AMOUNT:  self->wiggle_amount   = g_value_get_double (v); break;
    case PROP_WIGGLE_FREQ_HZ: self->wiggle_freq_hz  = g_value_get_double (v); break;
    case PROP_MOTION_FILE:
      g_free (self->motion_file);
      self->motion_file = g_value_dup_string (v);
      break;
    case PROP_MOTION_MAX_MAG:          self->motion_max_mag          = g_value_get_double (v); break;
    case PROP_MOTION_SALIENCY_WEIGHT:  self->motion_saliency_weight  = g_value_get_double (v); break;
    case PROP_MOTION_SALIENCY_GAMMA:   self->motion_saliency_gamma   = g_value_get_double (v); break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, id, ps);
  }
}

static void
gst_parallax_dof_get_property (GObject *obj, guint id,
    GValue *v, GParamSpec *ps)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (obj);
  switch (id) {
    case PROP_PARALLAX_SHIFT:  g_value_set_int    (v, self->parallax_shift);  break;
    case PROP_FG_THRESHOLD:    g_value_set_double (v, self->fg_threshold);    break;
    case PROP_BG_BLUR:         g_value_set_double (v, self->bg_blur);         break;
    case PROP_FG_SHARPNESS:    g_value_set_double (v, self->fg_sharpness);    break;
    case PROP_PROC_SCALE:      g_value_set_double (v, self->proc_scale);      break;
    case PROP_MAX_FG_CLUSTERS: g_value_set_int    (v, self->max_fg_clusters); break;
    case PROP_SMOOTH_KSIZE:    g_value_set_int    (v, self->smooth_ksize);    break;
    case PROP_FLOW_WEIGHT:     g_value_set_double (v, self->flow_weight);     break;
    case PROP_EDGE_WEIGHT:     g_value_set_double (v, self->edge_weight);     break;
    case PROP_DEPTH_FILE:      g_value_set_string (v, self->depth_file);      break;
    case PROP_DEPTH_BANDS:     g_value_set_int    (v, self->depth_bands);     break;
    case PROP_NEAR_GAIN:       g_value_set_double (v, self->near_gain);       break;
    case PROP_FAR_GAIN:        g_value_set_double (v, self->far_gain);        break;
    case PROP_BAND_BLUR:       g_value_set_double (v, self->band_blur);       break;
    case PROP_SALIENCY_BOOST:  g_value_set_double (v, self->saliency_boost);  break;
    case PROP_SALIENCY_RADIUS: g_value_set_double (v, self->saliency_radius); break;
    case PROP_SALIENCY_SIGMA:  g_value_set_double (v, self->saliency_sigma);  break;
    case PROP_DEPTH_SALIENCY_WEIGHT:
      g_value_set_double (v, self->depth_saliency_weight); break;
    case PROP_DEPTH_SALIENCY_GAMMA:
      g_value_set_double (v, self->depth_saliency_gamma);  break;
    case PROP_WIGGLE_AMOUNT:  g_value_set_double (v, self->wiggle_amount);  break;
    case PROP_WIGGLE_FREQ_HZ: g_value_set_double (v, self->wiggle_freq_hz); break;
    case PROP_MOTION_FILE:             g_value_set_string (v, self->motion_file);            break;
    case PROP_MOTION_MAX_MAG:          g_value_set_double (v, self->motion_max_mag);         break;
    case PROP_MOTION_SALIENCY_WEIGHT:  g_value_set_double (v, self->motion_saliency_weight); break;
    case PROP_MOTION_SALIENCY_GAMMA:   g_value_set_double (v, self->motion_saliency_gamma);  break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, id, ps);
  }
}

static void
gst_parallax_dof_finalize (GObject *obj)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (obj);
  if (self->depth_fp) { fclose (self->depth_fp); self->depth_fp = nullptr; }
  if (self->motion_fp) { fclose (self->motion_fp); self->motion_fp = nullptr; }
  g_free (self->depth_file);
  g_free (self->motion_file);
  delete self->prev_gray;
  delete self->prev_bgr;
  delete self->prev_delta;
  delete self->saliency_mask;
  G_OBJECT_CLASS (gst_parallax_dof_parent_class)->finalize (obj);
}

static void
gst_parallax_dof_init (GstParallaxDoF *self)
{
  self->parallax_shift  = DEFAULT_PARALLAX_SHIFT;
  self->fg_threshold    = DEFAULT_FG_THRESHOLD;
  self->bg_blur         = DEFAULT_BG_BLUR;
  self->fg_sharpness    = DEFAULT_FG_SHARPNESS;
  self->proc_scale      = DEFAULT_PROC_SCALE;
  self->max_fg_clusters = DEFAULT_MAX_FG_CLUSTERS;
  self->smooth_ksize    = DEFAULT_SMOOTH_KSIZE;
  self->flow_weight     = DEFAULT_FLOW_WEIGHT;
  self->edge_weight     = DEFAULT_EDGE_WEIGHT;
  self->depth_file      = nullptr;
  self->depth_bands     = DEFAULT_DEPTH_BANDS;
  self->near_gain       = DEFAULT_NEAR_GAIN;
  self->far_gain        = DEFAULT_FAR_GAIN;
  self->band_blur       = DEFAULT_BAND_BLUR;
  self->saliency_boost  = DEFAULT_SALIENCY_BOOST;
  self->saliency_radius = DEFAULT_SALIENCY_RADIUS;
  self->saliency_sigma  = DEFAULT_SALIENCY_SIGMA;
  self->depth_saliency_weight = DEFAULT_DEPTH_SALIENCY_WEIGHT;
  self->depth_saliency_gamma  = DEFAULT_DEPTH_SALIENCY_GAMMA;
  self->wiggle_amount   = DEFAULT_WIGGLE_AMOUNT;
  self->wiggle_freq_hz  = DEFAULT_WIGGLE_FREQ_HZ;
  self->motion_file     = nullptr;
  self->motion_max_mag  = DEFAULT_MOTION_MAX_MAG;
  self->motion_saliency_weight = DEFAULT_MOTION_SALIENCY_WEIGHT;
  self->motion_saliency_gamma  = DEFAULT_MOTION_SALIENCY_GAMMA;
  self->prev_gray       = new cv::Mat ();
  self->prev_bgr        = new cv::Mat ();
  self->prev_delta      = new cv::Mat ();
  self->saliency_mask   = new cv::Mat ();
  self->depth_fp        = nullptr;
  self->motion_fp       = nullptr;
  self->width = self->height = 0;
  self->fps_n = 25; self->fps_d = 1;
  self->frame_count = 0;
}

static void
gst_parallax_dof_class_init (GstParallaxDoFClass *klass)
{
  GObjectClass          *obj_cls = G_OBJECT_CLASS (klass);
  GstElementClass       *el_cls  = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_cls  = GST_BASE_TRANSFORM_CLASS (klass);
  GstVideoFilterClass   *vf_cls  = GST_VIDEO_FILTER_CLASS (klass);

  obj_cls->set_property = gst_parallax_dof_set_property;
  obj_cls->get_property = gst_parallax_dof_get_property;
  obj_cls->finalize     = gst_parallax_dof_finalize;

  g_object_class_install_property (obj_cls, PROP_PARALLAX_SHIFT,
      g_param_spec_int ("parallax-shift", "Parallax shift",
          "Horizontal pixel shift for foreground (px)",
          0, 200, DEFAULT_PARALLAX_SHIFT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_FG_THRESHOLD,
      g_param_spec_double ("fg-threshold", "FG threshold",
          "Depth threshold for foreground (0.0–1.0)",
          0.0, 1.0, DEFAULT_FG_THRESHOLD,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_BG_BLUR,
      g_param_spec_double ("bg-blur", "Background blur",
          "Gaussian radius/sigma for background blur",
          0.0, 50.0, DEFAULT_BG_BLUR,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_FG_SHARPNESS,
      g_param_spec_double ("fg-sharpness", "FG sharpness",
          "Unsharp mask multiplier for foreground (1.0 = off)",
          1.0, 5.0, DEFAULT_FG_SHARPNESS,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_PROC_SCALE,
      g_param_spec_double ("proc-scale", "Processing scale",
          "Downscale for depth estimation (ignored if depth-file is set)",
          0.1, 1.0, DEFAULT_PROC_SCALE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_MAX_FG_CLUSTERS,
      g_param_spec_int ("max-fg-clusters", "Max FG clusters",
          "Max foreground connected components to keep",
          1, 8, DEFAULT_MAX_FG_CLUSTERS,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_SMOOTH_KSIZE,
      g_param_spec_int ("smooth-ksize", "Smooth kernel size",
          "Gaussian kernel size for depth smoothing",
          3, 51, DEFAULT_SMOOTH_KSIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_FLOW_WEIGHT,
      g_param_spec_double ("flow-weight", "Flow weight",
          "Weight of optical-flow magnitude in depth",
          0.0, 1.0, DEFAULT_FLOW_WEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_EDGE_WEIGHT,
      g_param_spec_double ("edge-weight", "Edge weight",
          "Weight of Sobel edges in depth",
          0.0, 1.0, DEFAULT_EDGE_WEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_DEPTH_FILE,
      g_param_spec_string ("depth-file", "Depth file",
          "Precomputed depth maps (.raw from depthanalyzer). "
          "If set, skips realtime estimation.",
          nullptr,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_DEPTH_BANDS,
      g_param_spec_int ("depth-bands", "Depth bands",
          "Liczba pasm tonalnych głębi do osobnego wzmacniania efektu "
          "(1 = wyłączone, zachowanie jak globalny gain). Wymaga depth-file.",
          1, 16, DEFAULT_DEPTH_BANDS,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_NEAR_GAIN,
      g_param_spec_double ("near-gain", "Near band gain",
          "Mnożnik residual-gain dla najjaśniejszego (najbliższego) pasma głębi",
          0.0, 5.0, DEFAULT_NEAR_GAIN,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_FAR_GAIN,
      g_param_spec_double ("far-gain", "Far band gain",
          "Mnożnik residual-gain dla najciemniejszego (najdalszego) pasma głębi",
          0.0, 5.0, DEFAULT_FAR_GAIN,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_BAND_BLUR,
      g_param_spec_double ("band-blur", "Band gain blur",
          "Rozmycie (sigma) maski wzmocnienia pasm — zapobiega schodkowaniu na granicach",
          0.0, 30.0, DEFAULT_BAND_BLUR,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_SALIENCY_BOOST,
      g_param_spec_double ("saliency-boost", "Saliency detail boost",
          "Siła lokalnego wzmocnienia kontrastu/detalu (unsharp) w obszarze "
          "uwagi (0 = wyłączone)",
          0.0, 5.0, DEFAULT_SALIENCY_BOOST,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_SALIENCY_RADIUS,
      g_param_spec_double ("saliency-radius", "Saliency radius",
          "Promień center-weighted maski uwagi, jako ułamek połowy przekątnej kadru",
          0.05, 2.0, DEFAULT_SALIENCY_RADIUS,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_SALIENCY_SIGMA,
      g_param_spec_double ("saliency-sigma", "Saliency detail sigma",
          "Sigma Gaussa użyta do wydzielenia wysokich częstotliwości (detalu) do wzmocnienia",
          0.3, 10.0, DEFAULT_SALIENCY_SIGMA,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_DEPTH_SALIENCY_WEIGHT,
      g_param_spec_double ("depth-saliency-weight", "Depth saliency weight",
          "Udział mapy głębi w masce saliency (0 = czysty center-weighting, "
          "1 = czysta mapa głębi). Wymaga depth-file.",
          0.0, 1.0, DEFAULT_DEPTH_SALIENCY_WEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_DEPTH_SALIENCY_GAMMA,
      g_param_spec_double ("depth-saliency-gamma", "Depth saliency gamma",
          "Krzywa mapowania głębi na wagę saliency (1.0 = liniowo, >1 wzmacnia "
          "bliskie obiekty, <1 wzmacnia dalsze)",
          0.3, 3.0, DEFAULT_DEPTH_SALIENCY_GAMMA,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_WIGGLE_AMOUNT,
      g_param_spec_double ("wiggle-amount", "Wiggle amount",
          "Amplituda (px) realnego, zależnego od głębi przesunięcia poziomego "
          "oscylującego w czasie (Smooth3D-style). 0 = wyłączone. Wymaga depth-file.",
          0.0, 40.0, DEFAULT_WIGGLE_AMOUNT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_WIGGLE_FREQ_HZ,
      g_param_spec_double ("wiggle-freq-hz", "Wiggle frequency",
          "Częstotliwość oscylacji wiggle w Hz (typowo 2–6)",
          0.1, 20.0, DEFAULT_WIGGLE_FREQ_HZ,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_MOTION_FILE,
      g_param_spec_string ("motion-file", "Motion file",
          "Precomputed real optical-flow (Farneback) from depthanalyzer's "
          "motion-file — 2 bytes/px (magnitude, angle). Jeśli ustawiony, "
          "nadpisuje kierunek pseudo-flow z depth-file gradientu, i/lub "
          "napędza mimozę samodzielnie (bez depth-file).",
          nullptr,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_MOTION_MAX_MAG,
      g_param_spec_double ("motion-max-mag", "Motion max magnitude",
          "Clamp (px/klatkę) użyty przy zapisie motion-file w depthanalyzer — "
          "musi być identyczny, inaczej dekwantyzacja magnitude będzie błędna.",
          0.1, 200.0, DEFAULT_MOTION_MAX_MAG,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_MOTION_SALIENCY_WEIGHT,
      g_param_spec_double ("motion-saliency-weight", "Motion saliency weight",
          "Udział mapy prędkości (magnitude z motion-file) w masce saliency — "
          "szybciej poruszające się piksele dostają mocniejszy detail-boost. "
          "Miksowane z center-weighting i/lub depth-saliency-weight. "
          "0 = wyłączone. Wymaga motion-file i saliency-boost > 0.",
          0.0, 1.0, DEFAULT_MOTION_SALIENCY_WEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (obj_cls, PROP_MOTION_SALIENCY_GAMMA,
      g_param_spec_double ("motion-saliency-gamma", "Motion saliency gamma",
          "Krzywa mapowania magnitude na wagę saliency (1.0 = liniowo, "
          ">1 wzmacnia tylko najszybsze piksele, <1 spłaszcza różnice)",
          0.3, 3.0, DEFAULT_MOTION_SALIENCY_GAMMA,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata (el_cls,
      "ParallaxDoF",
      "Filter/Effect/Video",
      "Transvector mimicry: residual motion along flow vectors (no bokeh)",
      "Tomasz / stpf99");

  gst_element_class_add_static_pad_template (el_cls, &pd_sink_tmpl);
  gst_element_class_add_static_pad_template (el_cls, &pd_src_tmpl);

  bt_cls->start = GST_DEBUG_FUNCPTR (gst_parallax_dof_start);
  bt_cls->stop  = GST_DEBUG_FUNCPTR (gst_parallax_dof_stop);
  vf_cls->set_info        = GST_DEBUG_FUNCPTR (gst_parallax_dof_set_info);
  vf_cls->transform_frame = GST_DEBUG_FUNCPTR (gst_parallax_dof_transform_frame);
  /* transform_frame (out-of-place) — wynik zawsze w osobnym buforze */
  (void)klass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Plugin entry
 * ═════════════════════════════════════════════════════════════════════════ */
static gboolean
plugin_init (GstPlugin *plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_depth_analyzer_debug, "depthanalyzer", 0,
      "Depth pre-analysis (optical flow + Sobel)");
  GST_DEBUG_CATEGORY_INIT (gst_parallax_dof_debug, "parallaxdof", 0,
      "2.5D Parallax DoF video filter");

  if (!gst_element_register (plugin, "depthanalyzer",
          GST_RANK_NONE, GST_TYPE_DEPTH_ANALYZER))
    return FALSE;
  if (!gst_element_register (plugin, "parallaxdof",
          GST_RANK_NONE, GST_TYPE_PARALLAX_DOF))
    return FALSE;
  return TRUE;
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    parallaxdof,
    "2.5D parallax + depth analyzer (synced with 3dplayer.py)",
    plugin_init,
    "0.8.0",
    "LGPL",
    "gst-parallaxdof",
    "https://github.com/stpf99")
