project('gst-parallaxdof', 'cpp',
  version         : '0.2.0',
  default_options : ['cpp_std=c++17', 'warning_level=2'])

# ── Zależności ────────────────────────────────────────────────────────────────
gst_dep   = dependency('gstreamer-1.0',       version : '>= 1.16', required : true)
gstvid    = dependency('gstreamer-video-1.0', version : '>= 1.16', required : true)

# OpenCV: próbuj najpierw opencv4, potem opencv (Solus używa opencv4)
opencv    = dependency('opencv4', required : false)
if not opencv.found()
  opencv  = dependency('opencv',  required : true)
endif

message('GStreamer: ' + gst_dep.version())
message('OpenCV:    ' + opencv.version())

# ── Katalog instalacji pluginów ───────────────────────────────────────────────
# pkg-config gstreamer-1.0 --variable=pluginsdir  →  /usr/lib64/gstreamer-1.0 (lub podobny)
plugins_install_dir = gst_dep.get_variable(pkgconfig : 'pluginsdir',
                                           default_value : get_option('libdir') / 'gstreamer-1.0')

# ── Budowanie biblioteki współdzielonej ───────────────────────────────────────
cdata = configuration_data()
cdata.set_quoted('PACKAGE', 'gst-parallaxdof')
cdata.set_quoted('VERSION', meson.project_version())
configure_file(output : 'config.h', configuration : cdata)

shared_library(
  'gstparallaxdof',
  'src/gstparallaxdof.cpp',
  dependencies       : [gst_dep, gstvid, opencv],
  install            : true,
  install_dir        : plugins_install_dir,
  name_prefix        : 'lib',
  cpp_args           : ['-DHAVE_CONFIG_H'],
  include_directories: include_directories('.'),
)

message('Install dir: ' + plugins_install_dir)
