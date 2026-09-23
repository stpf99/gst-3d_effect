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

/* ── Mimoza transwektorowa ─────────────────────────────────────────────────
 *
 * 1. residual_raw = current − prev_original
 * 2. flow (vx, vy) między prev a current
 * 3. Próbkuj residual wzdłuż wektora (x − k·vx, y − k·vy)
 *    → podkreśla ruch „zbliżający się” wzdłuż wypadkowej
 * 4. out = current + gain · residual_sampled · soft(|flow|)
 *         − decay · prev_applied_delta
 * 5. Zapisz delta do odejmowania w następnej klatce
 *
 * Bez rozmycia tła, bez maski FG/BG, bez stałego bokeh.
 * ─────────────────────────────────────────────────────────────────────────── */
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
  if (!flow.isContinuous ())
    flow = flow.clone ();
  const float scale_xy = (float)(w / (double)pw);
  flow *= scale_xy;
  /* wygładź pole wektorów — mniej szwów */
  cv::GaussianBlur (flow, flow, cv::Size (0, 0), 1.5);

  /* residual = current − prev (signed float) */
  cv::Mat cur_f, prev_f, residual;
  frame_bgr.convertTo (cur_f, CV_32FC3);
  prev_bgr->convertTo (prev_f, CV_32FC3);
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
  const float gain = (float)residual_gain;

  /* Bezpieczne próbkowanie residualu wzdłuż wektora (bez OOB) */
  const float *flow_data = flow_data0;
  const int flow_step = flow_step0;

  for (int y = 0; y < h; ++y) {
    const float *fl = flow_data + y * flow_step;
    float *dlt = delta.ptr<float> (y);

    for (int x = 0; x < w; ++x) {
      float vx = fl[x * 2 + 0];
      float vy = fl[x * 2 + 1];
      if (!std::isfinite (vx) || !std::isfinite (vy))
        continue;

      float mag = std::sqrt (vx * vx + vy * vy);
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

      for (int c = 0; c < 3; ++c) {
        float rv = r00[c] * ww00 + r10[c] * ww10 + r01[c] * ww01 + r11[c] * ww11;
        dlt[x * 3 + c] = rv * gain * soft;
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

  /* zapamiętaj deltę i gray */
  if (prev_delta)
    delta.copyTo (*prev_delta);
  if (prev_gray_out)
    gray.copyTo (*prev_gray_out);
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

/* ── Prosta mapa ruchu (dla depthanalyzer, opcjonalnie) ──────────────────── */
static cv::Mat
estimate_depth (cv::Mat *prev_gray,
                const cv::Mat &frame_bgr,
                double proc_scale,
                int smooth_ksize,
                float flow_weight,
                float edge_weight)
{
  (void)flow_weight; (void)edge_weight;
  const int fw = frame_bgr.cols, fh = frame_bgr.rows;
  const int pw = std::max (1, (int)(fw * proc_scale));
  const int ph = std::max (1, (int)(fh * proc_scale));

  cv::Mat small, gray;
  cv::resize (frame_bgr, small, cv::Size (pw, ph), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor (small, gray, cv::COLOR_BGR2GRAY);

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

  int ks = smooth_ksize | 1;
  if (ks < 3) ks = 3;
  cv::GaussianBlur (diff, diff, cv::Size (ks, ks), ks / 6.0);

  double dmx;
  cv::minMaxLoc (diff, nullptr, &dmx);
  if (dmx > 0) diff /= (float)dmx;

  if (pw != fw || ph != fh)
    cv::resize (diff, diff, cv::Size (fw, fh), 0, 0, cv::INTER_LINEAR);
  return diff;
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
};

struct _GstDepthAnalyzer {
  GstVideoFilter parent;

  gchar  *location;
  gdouble proc_scale;
  gint    smooth_ksize;
  gdouble flow_weight;
  gdouble edge_weight;

  cv::Mat *prev_gray;
  FILE    *fp;
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
  if (!self->location || !self->location[0]) {
    GST_ERROR_OBJECT (self, "property 'location' is required");
    return FALSE;
  }
  self->fp = fopen (self->location, "wb");
  if (!self->fp) {
    GST_ERROR_OBJECT (self, "cannot open depth file for writing: %s", self->location);
    return FALSE;
  }
  self->frame_count = 0;
  self->prev_gray->release ();
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
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, id, ps);
  }
}

static void
gst_depth_analyzer_finalize (GObject *obj)
{
  GstDepthAnalyzer *self = GST_DEPTH_ANALYZER (obj);
  if (self->fp) { fclose (self->fp); self->fp = nullptr; }
  g_free (self->location);
  delete self->prev_gray;
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
  self->prev_gray    = new cv::Mat ();
  self->fp           = nullptr;
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

  gst_element_class_set_static_metadata (el_cls,
      "DepthAnalyzer",
      "Filter/Effect/Video",
      "Pre-analyze depth maps (optical flow + Sobel) and write to .raw file",
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

  cv::Mat *prev_gray;
  cv::Mat *prev_bgr;      /* oryginał poprzedniej klatki */
  cv::Mat *prev_delta;    /* poprzednio nałożone zmiany (do odejmowania) */
  FILE    *depth_fp;
  gint     width, height;
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
  self->prev_gray->release ();
  self->prev_bgr->release ();
  self->prev_delta->release ();
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
  self->prev_gray->release ();
  self->prev_bgr->release ();
  self->prev_delta->release ();
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
  return TRUE;
}

static GstFlowReturn
gst_parallax_dof_transform_frame (GstVideoFilter *filter,
    GstVideoFrame *inframe, GstVideoFrame *outframe)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (filter);

  bool owned = false;
  cv::Mat bgr = frame_to_mat (inframe, &owned);

  /* Mimoza transwektorowa:
   * strength  ← parallax_shift
   * gain      ← fg_sharpness (reużycie suwaka jako residual-gain)
   * decay     ← 0.85 stałe (odejmowanie poprzednich zmian)
   */
  /* fg_sharpness 1.5 default → gain ~0.4 (delikatnie) */
  double gain = self->fg_sharpness * 0.25;
  if (gain < 0.05) gain = 0.05;
  if (gain > 2.0) gain = 2.0;

  render_mimosa (bgr,
      self->prev_bgr,
      self->prev_delta,
      self->prev_gray,
      self->parallax_shift,
      gain,
      0.92,
      self->proc_scale);

  /* zapamiętaj oryginał bieżącej jako prev do następnej klatki
   * (bierzemy z inframe, nie z przetworzonego bgr) */
  {
    bool o2 = false;
    cv::Mat orig = frame_to_mat (inframe, &o2);
    orig.copyTo (*self->prev_bgr);
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
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, id, ps);
  }
}

static void
gst_parallax_dof_finalize (GObject *obj)
{
  GstParallaxDoF *self = GST_PARALLAX_DOF (obj);
  if (self->depth_fp) { fclose (self->depth_fp); self->depth_fp = nullptr; }
  g_free (self->depth_file);
  delete self->prev_gray;
  delete self->prev_bgr;
  delete self->prev_delta;
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
  self->prev_gray       = new cv::Mat ();
  self->prev_bgr        = new cv::Mat ();
  self->prev_delta      = new cv::Mat ();
  self->depth_fp        = nullptr;
  self->width = self->height = 0;
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
