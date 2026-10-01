/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstUtility.h"
#include "filesystem.h"

#include "NvInfer.h"
#include "logging.h"

#include <stdint.h>
#include <stdio.h>
#include <strings.h>
#include <algorithm>
#include <map>
#include <string>

#include <dlfcn.h>
#include <link.h>


//---------------------------------------------------------------------------------------------
imageFormat gst_parse_format( GstStructure* caps )
{
	const char* format = gst_structure_get_string(caps, "format");
	
	if( !format )
		return IMAGE_UNKNOWN;
	
	if( strcasecmp(format, "rgb") == 0 )
		return IMAGE_RGB8;
	else if( strcasecmp(format, "yuy2") == 0 )
		return IMAGE_YUY2;
	else if( strcasecmp(format, "i420") == 0 )
		return IMAGE_I420;
	else if( strcasecmp(format, "nv12") == 0 )
		return IMAGE_NV12;
	else if( strcasecmp(format, "yv12") == 0 )
		return IMAGE_YV12;
	else if( strcasecmp(format, "yuyv") == 0 )
		return IMAGE_YUYV;
	else if( strcasecmp(format, "yvyu") == 0 )
		return IMAGE_YVYU;
	else if( strcasecmp(format, "uyvy") == 0 )
		return IMAGE_UYVY;
	else if( strcasecmp(format, "bggr") == 0 )
		return IMAGE_BAYER_BGGR;
	else if( strcasecmp(format, "gbrg") == 0 )
		return IMAGE_BAYER_GBRG;
	else if( strcasecmp(format, "grgb") == 0 )
		return IMAGE_BAYER_GRBG;
	else if( strcasecmp(format, "rggb") == 0 )
		return IMAGE_BAYER_RGGB;
	else if( strcasecmp(format, "gray8") == 0 )
		return IMAGE_GRAY8;
	
	return IMAGE_UNKNOWN;
}

const char* gst_format_to_string( imageFormat format )
{
	switch(format)
	{
		case IMAGE_RGB8:	return "RGB";
		case IMAGE_YUY2:	return "YUY2";
		case IMAGE_I420:	return "I420";
		case IMAGE_NV12:	return "NV12";
		case IMAGE_YV12:	return "YV12";
		case IMAGE_YVYU:	return "YVYU";
		case IMAGE_UYVY:	return "UYVY";
		case IMAGE_BAYER_BGGR:	return "bggr";
		case IMAGE_BAYER_GBRG:	return "gbrg";
		case IMAGE_BAYER_GRBG:	return "grbg";
		case IMAGE_BAYER_RGGB:	return "rggb";
		case IMAGE_GRAY8:	return "GRAY8";
	}
	
	return " ";
}

videoOptions::Codec gst_parse_codec( GstStructure* caps )
{
	const char* codec = gst_structure_get_name(caps);
	
	if( !codec )
		return videoOptions::CODEC_UNKNOWN;
	
	if( strcasecmp(codec, "video/x-raw") == 0 || strcasecmp(codec, "video/x-bayer") == 0 )
		return videoOptions::CODEC_RAW;
	else if( strcasecmp(codec, "video/x-h264") == 0 )
		return videoOptions::CODEC_H264;
	else if( strcasecmp(codec, "video/x-h265") == 0 )
		return videoOptions::CODEC_H265;
	else if( strcasecmp(codec, "video/x-vp8") == 0 )
		return videoOptions::CODEC_VP8;
	else if( strcasecmp(codec, "video/x-vp9") == 0 )
		return videoOptions::CODEC_VP9;
	else if( strcasecmp(codec, "video/x-av1") == 0 )
		return videoOptions::CODEC_AV1;
	else if( strcasecmp(codec, "image/jpeg") == 0 )
		return videoOptions::CODEC_MJPEG;
	else if( strcasecmp(codec, "video/mpeg") == 0 )
	{
		int mpegVersion = 0;
	
		if( !gst_structure_get_int(caps, "mpegversion", &mpegVersion) )
		{
			LogError(LOG_GSTREAMER "MPEG codec, but failed to get MPEG version from caps\n");
			return videoOptions::CODEC_UNKNOWN;
		}
		
		if( mpegVersion == 2 )
			return videoOptions::CODEC_MPEG2;
		else if( mpegVersion == 4 )
			return videoOptions::CODEC_MPEG4;
		else
		{
			LogError(LOG_GSTREAMER "invalid MPEG codec version:  %i (MPEG-2 and MPEG-4 are supported)\n", mpegVersion);
			return videoOptions::CODEC_UNKNOWN;
		}
	}
	
	LogError(LOG_GSTREAMER "unrecognized codec - %s\n", codec);
	return videoOptions::CODEC_UNKNOWN;
}

const char* gst_codec_to_string( videoOptions::Codec codec )
{
	switch(codec)
	{
		case videoOptions::CODEC_RAW: 	return "video/x-raw";
		case videoOptions::CODEC_H264:	return "video/x-h264";
		case videoOptions::CODEC_H265:	return "video/x-h265";
		case videoOptions::CODEC_VP8:	return "video/x-vp8";
		case videoOptions::CODEC_VP9:	return "video/x-vp9";
		case videoOptions::CODEC_AV1:	return "video/x-av1";
		case videoOptions::CODEC_MJPEG:	return "image/jpeg";
		case videoOptions::CODEC_MPEG2:	return "video/mpeg, mpegversion=(int)2";
		case videoOptions::CODEC_MPEG4:	return "video/mpeg, mpegversion=(int)4";
	}
	
	return " ";
}


//---------------------------------------------------------------------------------------------
inline const char* gst_debug_level_str( GstDebugLevel level )
{
	switch (level)
	{
		case GST_LEVEL_NONE:	return "GST_LEVEL_NONE   ";
		case GST_LEVEL_ERROR:	return "GST_LEVEL_ERROR  ";
		case GST_LEVEL_WARNING:	return "GST_LEVEL_WARNING";
		case GST_LEVEL_INFO:	return "GST_LEVEL_INFO   ";
		case GST_LEVEL_DEBUG:	return "GST_LEVEL_DEBUG  ";
		case GST_LEVEL_LOG:		return "GST_LEVEL_LOG    ";
		case GST_LEVEL_FIXME:	return "GST_LEVEL_FIXME  ";
#ifdef GST_LEVEL_TRACE
		case GST_LEVEL_TRACE:	return "GST_LEVEL_TRACE  ";
#endif
		case GST_LEVEL_MEMDUMP:	return "GST_LEVEL_MEMDUMP";
    		default:				return "<unknown>        ";
    }
}

#define SEP "              "

void rilog_debug_function(GstDebugCategory* category, GstDebugLevel level,
                          const gchar* file, const char* function,
                          gint line, GObject* object, GstDebugMessage* message,
                          gpointer data)
{
	if( level > GST_LEVEL_WARNING /*GST_LEVEL_INFO*/ )
		return;

	//gchar* name = NULL;
	//if( object != NULL )
	//	g_object_get(object, "name", &name, NULL);

	const char* typeName  = " ";
	const char* className = " ";

	if( object != NULL )
	{
		typeName  = G_OBJECT_TYPE_NAME(object);
		className = G_OBJECT_CLASS_NAME(object);
	}

	LogVerbose(LOG_GSTREAMER "%s %s %s\n" SEP "%s:%i  %s\n" SEP "%s\n", 
		  	 gst_debug_level_str(level), typeName,
		  	 gst_debug_category_get_name(category), file, line, function, 
            	 gst_debug_message_get(message));

}


// gstreamerInit
bool gstreamerInit()
{
	static bool gstreamer_initialized = false;

	if( gstreamer_initialized )
		return true;

	int argc = 0;
	//char* argv[] = { "none" };

	if( !gst_init_check(&argc, NULL, NULL) )
	{
		LogError(LOG_GSTREAMER "failed to initialize gstreamer library with gst_init()\n");
		return false;
	}

	gstreamer_initialized = true;

	uint32_t ver[] = { 0, 0, 0, 0 };
	gst_version( &ver[0], &ver[1], &ver[2], &ver[3] );

	LogInfo(LOG_GSTREAMER "initialized gstreamer, version %u.%u.%u.%u\n", ver[0], ver[1], ver[2], ver[3]);


	// debugging
	gst_debug_remove_log_function(gst_debug_log_default);
	
	if( true )
	{
		gst_debug_add_log_function(rilog_debug_function, NULL, NULL);

		gst_debug_set_active(true);
		gst_debug_set_colored(false);
	}
	
	return true;
}

//---------------------------------------------------------------------------------------------
static void gst_print_one_tag(const GstTagList * list, const gchar * tag, gpointer user_data)
{
  int i, num;

  num = gst_tag_list_get_tag_size (list, tag);
  for (i = 0; i < num; ++i) {
    const GValue *val;

    /* Note: when looking for specific tags, use the gst_tag_list_get_xyz() API,
     * we only use the GValue approach here because it is more generic */
    val = gst_tag_list_get_value_index (list, tag, i);
    if (G_VALUE_HOLDS_STRING (val)) {
      LogVerbose("\t%20s : %s\n", tag, g_value_get_string (val));
    } else if (G_VALUE_HOLDS_UINT (val)) {
      LogVerbose("\t%20s : %u\n", tag, g_value_get_uint (val));
    } else if (G_VALUE_HOLDS_DOUBLE (val)) {
      LogVerbose("\t%20s : %g\n", tag, g_value_get_double (val));
    } else if (G_VALUE_HOLDS_BOOLEAN (val)) {
      LogVerbose("\t%20s : %s\n", tag,
          (g_value_get_boolean (val)) ? "true" : "false");
    } else if (GST_VALUE_HOLDS_BUFFER (val)) {
      //GstBuffer *buf = gst_value_get_buffer (val);
      //guint buffer_size = GST_BUFFER_SIZE(buf);

      LogVerbose("\t%20s : buffer of size %u\n", tag, /*buffer_size*/0);
    } /*else if (GST_VALUE_HOLDS_DATE_TIME (val)) {
      GstDateTime *dt = (GstDateTime*)g_value_get_boxed (val);
      gchar *dt_str = gst_date_time_to_iso8601_string (dt);

      printf("\t%20s : %s\n", tag, dt_str);
      g_free (dt_str);
    }*/ else {
      LogVerbose("\t%20s : tag of type '%s'\n", tag, G_VALUE_TYPE_NAME (val));
    }
  }
}

static const char* gst_stream_status_string( GstStreamStatusType status )
{
	switch(status)
	{
		case GST_STREAM_STATUS_TYPE_CREATE:	return "CREATE";
		case GST_STREAM_STATUS_TYPE_ENTER:		return "ENTER";
		case GST_STREAM_STATUS_TYPE_LEAVE:		return "LEAVE";
		case GST_STREAM_STATUS_TYPE_DESTROY:	return "DESTROY";
		case GST_STREAM_STATUS_TYPE_START:		return "START";
		case GST_STREAM_STATUS_TYPE_PAUSE:		return "PAUSE";
		case GST_STREAM_STATUS_TYPE_STOP:		return "STOP";
		default:							return "UNKNOWN";
	}
}

// gst_message_print
gboolean gst_message_print(GstBus* bus, GstMessage* message, gpointer user_data)
{
	switch (GST_MESSAGE_TYPE (message)) 
	{
		case GST_MESSAGE_ERROR: 
		{
			GError *err = NULL;
			gchar *dbg_info = NULL;
 
			gst_message_parse_error (message, &err, &dbg_info);
			LogVerbose(LOG_GSTREAMER "gstreamer %s ERROR %s\n", GST_OBJECT_NAME (message->src), err->message);
        		LogVerbose(LOG_GSTREAMER "gstreamer Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
        
			g_error_free(err);
        		g_free(dbg_info);
			//g_main_loop_quit (app->loop);
        		break;
		}
		case GST_MESSAGE_WARNING:
		{
			GError *err = NULL;
			gchar *dbg_info = NULL;

			gst_message_parse_warning (message, &err, &dbg_info);
			LogVerbose(LOG_GSTREAMER "gstreamer %s WARNING %s\n", GST_OBJECT_NAME (message->src), err->message);
			LogVerbose(LOG_GSTREAMER "gstreamer Debugging info: %s\n", (dbg_info) ? dbg_info : "none");

			g_error_free(err);
			g_free(dbg_info);
			break;
		}
		case GST_MESSAGE_EOS:
		{
			LogVerbose(LOG_GSTREAMER "gstreamer %s recieved EOS signal...\n", GST_OBJECT_NAME(message->src));
			//g_main_loop_quit (app->loop);		// TODO trigger plugin Close() upon error
			break;
		}
		case GST_MESSAGE_STATE_CHANGED:
		{
			GstState old_state, new_state;
    
			gst_message_parse_state_changed(message, &old_state, &new_state, NULL);
			
			LogVerbose(LOG_GSTREAMER "gstreamer changed state from %s to %s ==> %s\n",
							gst_element_state_get_name(old_state),
							gst_element_state_get_name(new_state),
						     GST_OBJECT_NAME(message->src));
			break;
		}
		case GST_MESSAGE_STREAM_STATUS:
		{
			GstStreamStatusType streamStatus;
			gst_message_parse_stream_status(message, &streamStatus, NULL);
			
			LogVerbose(LOG_GSTREAMER "gstreamer stream status %s ==> %s\n",
							gst_stream_status_string(streamStatus), 
							GST_OBJECT_NAME(message->src));
			break;
		}
		case GST_MESSAGE_TAG: 
		{
			GstTagList *tags = NULL;
			gst_message_parse_tag(message, &tags);
			gchar* txt = gst_tag_list_to_string(tags);

			if( txt != NULL )
			{
				LogVerbose(LOG_GSTREAMER "gstreamer %s %s\n", GST_OBJECT_NAME(message->src), txt);		
				g_free(txt);	
			}
		
			//gst_tag_list_foreach(tags, gst_print_one_tag, NULL);

			if( tags != NULL )			
				gst_tag_list_free(tags);
			
			break;
		}
		default:
		{
			LogVerbose(LOG_GSTREAMER "gstreamer message %s ==> %s\n", gst_message_type_get_name(GST_MESSAGE_TYPE(message)), GST_OBJECT_NAME(message->src));
			break;
		}
	}

	return TRUE;
}


// gst_build_filesink
bool gst_build_filesink( const URI& uri, videoOptions::Codec codec, std::ostringstream& pipeline )
{
	if( uri.path.length() <= 0 || uri.protocol != "file" )
	{
		LogError(LOG_GSTREAMER "invalid file path -- unable to build filesink pipeline\n");
		return false;
	}
	
	// the muxers take AV1 as a TU-aligned OBU stream, and nvv4l2av1enc doesn't say which format it outputs
	// (av1parse requires GStreamer 1.20, the software encoders already output that format)
	#define ADD_CODEC_PARSER() \
		if( codec == videoOptions::CODEC_H264 ) \
			pipeline << "h264parse ! "; \
		else if( codec == videoOptions::CODEC_H265 ) \
			pipeline << "h265parse ! "; \
		else if( codec == videoOptions::CODEC_AV1 && gst_element_exists("av1parse") ) \
			pipeline << "av1parse ! ";
		
	if( uri.extension == "mkv" )
	{
		ADD_CODEC_PARSER();
		pipeline << "matroskamux ! ";
	}
	else if( uri.extension == "flv" )
	{
		if( codec == videoOptions::CODEC_AV1 )
		{
			LogError(LOG_GSTREAMER "FLV format doesn't support codec %s (use mkv or mp4 instead)\n", videoOptions::CodecToStr(codec));
			return false;
		}

		ADD_CODEC_PARSER();
		pipeline << "flvmux ! ";
	}
	else if( uri.extension == "avi" )
	{
		if( codec == videoOptions::CODEC_H265 || codec == videoOptions::CODEC_VP9 || codec == videoOptions::CODEC_AV1 )
		{
			LogError(LOG_GSTREAMER "AVI format doesn't support codec %s\n", videoOptions::CodecToStr(codec));
			LogError(LOG_GSTREAMER "supported AVI codecs are:\n");
			LogError(LOG_GSTREAMER "   * h264\n");
			LogError(LOG_GSTREAMER "   * vp8\n");
			LogError(LOG_GSTREAMER "   * mjpeg\n");

			return false;
		}

		pipeline << "avimux ! ";
	}
	else if( uri.extension == "mp4" || uri.extension == "qt" )
	{
		ADD_CODEC_PARSER();
		pipeline << "qtmux ! ";
	}
	else if( uri.extension != "h264" && uri.extension != "h265" )
	{
		printf(LOG_GSTREAMER "unsupported video file extension (%s)\n", uri.extension.c_str());
		printf(LOG_GSTREAMER "supported video extensions are:\n");
		printf(LOG_GSTREAMER "   * mkv\n");
		printf(LOG_GSTREAMER "   * mp4, qt\n");
		printf(LOG_GSTREAMER "   * flv\n");
		printf(LOG_GSTREAMER "   * avi\n");
		printf(LOG_GSTREAMER "   * h264, h265\n");

		return false;
	}

	pipeline << "filesink location=" << uri.location << " ";
	return true;
}


// gst_element_exists
bool gst_element_exists( const char* name )
{
	if( !name )
		return false;

	GstElementFactory* factory = gst_element_factory_find(name);

	if( !factory )
		return false;

	gst_object_unref(factory);
	return true;
}


// gst_element_has_property
bool gst_element_has_property( const char* name, const char* property )
{
	if( !name || !property )
		return false;

	GstElement* element = gst_element_factory_make(name, NULL);

	if( !element )
		return false;

	const bool found = (g_object_class_find_property(G_OBJECT_GET_CLASS(element), property) != NULL);

	gst_object_unref(element);
	return found;
}


// gst_element_property_range
bool gst_element_property_range( const char* name, const char* property, int64_t* min, int64_t* max )
{
	if( !name || !property || !min || !max )
		return false;

	GstElement* element = gst_element_factory_make(name, NULL);

	if( !element )
		return false;

	GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), property);
	bool found = true;

	if( spec != NULL && G_IS_PARAM_SPEC_INT(spec) )
	{
		*min = G_PARAM_SPEC_INT(spec)->minimum;
		*max = G_PARAM_SPEC_INT(spec)->maximum;
	}
	else if( spec != NULL && G_IS_PARAM_SPEC_UINT(spec) )
	{
		*min = G_PARAM_SPEC_UINT(spec)->minimum;
		*max = G_PARAM_SPEC_UINT(spec)->maximum;
	}
	else if( spec != NULL && G_IS_PARAM_SPEC_INT64(spec) )
	{
		*min = G_PARAM_SPEC_INT64(spec)->minimum;
		*max = G_PARAM_SPEC_INT64(spec)->maximum;
	}
	else
	{
		found = false;
	}

	gst_object_unref(element);
	return found;
}


// return the first element from a NULL-terminated list that is installed
static const char* gst_first_element( const char** names )
{
	for( uint32_t n=0; names[n] != NULL; n++ )
	{
		if( gst_element_exists(names[n]) )
			return names[n];
	}

	return NULL;
}


// select a software AV1 decoder
static const char* gst_select_av1_decoder()
{
	static const char* decoders[] = { "dav1ddec", "av1dec", NULL };
	const char* decoder = gst_first_element(decoders);

	if( !decoder )
		LogError(LOG_GSTREAMER "no AV1 software decoder found (dav1ddec or av1dec from gstreamer1.0-plugins-bad)\n");

	return decoder;
}


// select a software AV1 encoder (in order of preference for realtime encoding speed)
static const char* gst_select_av1_encoder()
{
	// av1enc doesn't have its realtime mode (usage-profile) on older GStreamer like 1.20 - without it libaom runs
	// its good-quality mode, which is much slower than rav1enc (0.18 vs 2-4.5 fps at 720p on Orin Nano)
	static const char* realtime[]    = { "svtav1enc", "av1enc", "rav1enc", NULL };
	static const char* no_realtime[] = { "svtav1enc", "rav1enc", "av1enc", NULL };

	const char* encoder = gst_first_element(gst_element_has_property("av1enc", "usage-profile") ? realtime : no_realtime);

	if( !encoder )
		LogError(LOG_GSTREAMER "no AV1 software encoder found (svtav1enc, av1enc from gstreamer1.0-plugins-bad, or rav1enc from gst-plugins-rs)\n");

	return encoder;
}


// find the path of a loaded shared library (by part of its name)
static int gst_find_library( struct dl_phdr_info* info, size_t size, void* user )
{
	std::pair<const char*, std::string>* search = (std::pair<const char*, std::string>*)user;

	if( !info->dlpi_name || !strstr(info->dlpi_name, search->first) )
		return 0;

	search->second = info->dlpi_name;
	return 1;
}


// gst_svtav1_low_delay
const char* gst_svtav1_low_delay()
{
	static bool checked = false;
	static const char* params = NULL;

	if( checked )
		return params;

	checked = true;

	// creating the element loads the plugin and SVT-AV1
	GstElement* element = gst_element_factory_make("svtav1enc", NULL);

	if( !element )
		return NULL;

	// the plugin built by scripts/gst-svtav1 has the fix for low delay
	GstPlugin* plugin = gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(gst_element_get_factory(element)));
	const bool fixed = plugin != NULL && gst_plugin_get_package(plugin) != NULL && strstr(gst_plugin_get_package(plugin), "low-delay fix") != NULL;

	if( plugin != NULL )
		gst_object_unref(plugin);

	gst_object_unref(element);

	// get the version from the SVT-AV1 library that the plugin loaded
	std::pair<const char*, std::string> library("libSvtAv1Enc.so", "");
	dl_iterate_phdr(gst_find_library, &library);

	int major = 0, minor = 0, patch = 0;

	if( library.second.length() > 0 )
	{
		void* handle = dlopen(library.second.c_str(), RTLD_LAZY | RTLD_NOLOAD);

		if( handle != NULL )
		{
			typedef const char* (*svt_av1_get_version_t)(void);
			svt_av1_get_version_t svt_av1_get_version = (svt_av1_get_version_t)dlsym(handle, "svt_av1_get_version");

			if( svt_av1_get_version != NULL && svt_av1_get_version() != NULL )
			{
				const char* version = svt_av1_get_version();
				sscanf(version[0] == 'v' ? version + 1 : version, "%d.%d.%d", &major, &minor, &patch);
			}

			dlclose(handle);
		}
	}

	LogVerbose(LOG_GSTREAMER "gstEncoder -- svtav1enc uses SVT-AV1 %d.%d.%d%s\n", major, minor, patch, fixed ? " (with the low-delay fix)" : "");

	// upstream svtav1enc deadlocks in low delay with SVT-AV1 2.3+, and with any version it reports
	// the random-access latency (1.25 s), so sinks that sync hold every frame and appsrc drops the rest
	if( !fixed || major == 0 )
		return NULL;

	// rtc is the fastest low-delay mode, it was added in SVT-AV1 3.1
	if( major > 3 || (major == 3 && minor >= 1) )
		params = "rtc=1:rc=2";
	else
		params = "pred-struct=1:rc=2";

	return params;
}


// check if the hardware codecs support AV1 (Orin and newer)
static bool gst_query_hw_av1()
{
#if defined(__aarch64__)
	if( fileExists("/proc/device-tree/compatible") )
	{
		const std::string soc = readFile("/proc/device-tree/compatible");

		if( soc.length() == 0 )
			return false;

		// Nano/TX1 (tegra210), TX2 (tegra186), and Xavier (tegra194) don't have AV1 in NVENC/NVDEC
		if( soc.find("nvidia,tegra210") != std::string::npos ||
		    soc.find("nvidia,tegra186") != std::string::npos ||
		    soc.find("nvidia,tegra194") != std::string::npos )
			return false;

		return true;
	}

	// the device tree is masked inside containers, where only the board model gets mounted
	if( !fileExists("/tmp/nv_jetson_model") )
		return false;

	std::string board = readFile("/tmp/nv_jetson_model");
	std::transform(board.begin(), board.end(), board.begin(), [](unsigned char c){ return std::tolower(c); });

	return board.find("orin") != std::string::npos || board.find("thor") != std::string::npos;
#else
	return false;
#endif
}


// gst_select_decoder
const char* gst_select_decoder( videoOptions::Codec codec, videoOptions::CodecType& type )
{
#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
	if( type == videoOptions::CODEC_OMX )  // JetPack 5 doesn't have OMX
		type = gst_default_codec();
#endif
#elif defined(__x86_64__) || defined(__amd64__)
	if( type == videoOptions::CODEC_OMX || type == videoOptions::CODEC_V4L2 )
		type = gst_default_codec();
#endif

	if( type == videoOptions::CODEC_NVENC || type == videoOptions::CODEC_NVDEC )
		type = gst_default_codec();
	
	if( codec == videoOptions::CODEC_MJPEG )
		type = videoOptions::CODEC_CPU;
	
	if( codec == videoOptions::CODEC_RAW )
		type = videoOptions::CODEC_CPU;

	if( codec == videoOptions::CODEC_AV1 && type != videoOptions::CODEC_CPU && !(type == videoOptions::CODEC_V4L2 && gst_query_hw_av1()) )
	{
		LogWarning(LOG_GSTREAMER "hardware AV1 decoder requires Orin or newer, reverting to CPU decoder\n");
		type = videoOptions::CODEC_CPU;
	}

	if( type == videoOptions::CODEC_CPU )
	{
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "avdec_h264";
			case videoOptions::CODEC_H265:   return "avdec_h265";
			case videoOptions::CODEC_VP8:	   return "vp8dec";
			case videoOptions::CODEC_VP9:    return "vp9dec";
			case videoOptions::CODEC_AV1:    return gst_select_av1_decoder();
			case videoOptions::CODEC_MPEG2:  return "avdec_mpeg2video";
			case videoOptions::CODEC_MPEG4:  return "avdec_mpeg4";
			case videoOptions::CODEC_MJPEG:  return "jpegdec";
		}
	}
	else if( type == videoOptions::CODEC_OMX )
	{
	#if GST_CHECK_VERSION(1,0,0)
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "omxh264dec";
			case videoOptions::CODEC_H265:   return "omxh265dec";
			case videoOptions::CODEC_VP8:	   return "omxvp8dec";
			case videoOptions::CODEC_VP9:    return "omxvp9dec";
			case videoOptions::CODEC_MPEG2:  return "omxmpeg2videodec";
			case videoOptions::CODEC_MPEG4:  return "omxmpeg4videodec";
			case videoOptions::CODEC_MJPEG:  return "nvjpegdec";
		}
	#else
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "nv_omx_h264dec";
			case videoOptions::CODEC_H265:   return "nv_omx_h265dec";
			case videoOptions::CODEC_VP8:	   return "nv_omx_vp8dec";
			case videoOptions::CODEC_VP9:    return "nv_omx_vp9dec";
			case videoOptions::CODEC_MPEG2:  return "nx_omx_mpeg2videodec";
			case videoOptions::CODEC_MPEG4:  return "nx_omx_mpeg4videodec";
			case videoOptions::CODEC_MJPEG:  return "nvjpegdec";
		}
	#endif
	}
	else if( type == videoOptions::CODEC_V4L2 )
	{
		if( codec == videoOptions::CODEC_MJPEG )
			return "nvjpegdec";
		
		return "nvv4l2decoder";
	}
	
	return NULL;
}


// check for hardware-accelerated encoder support
static bool gst_query_hw_encoder()
{
#if defined(__aarch64__)
	std::string board = readFile("/proc/device-tree/model");
	
	if( board.length() == 0 )
		board = readFile("/tmp/nv_jetson_model");  // this is where it gets mounted in the container
	
	if( board.length() == 0 )
		return false;
	
	LogVerbose(LOG_GSTREAMER "gstEncoder -- detected board '%s'\n", board.c_str());
	
	// convert to lowercase for robustness
	std::transform(board.begin(), board.end(), board.begin(), [](unsigned char c){ return std::tolower(c); });
	
	// look for specific boards that don't have encoder hw
	if( board.find("orin nano") != std::string::npos )
		return false;
	
	return true;
#else
	// TODO check for NVENC/NVDEC
	return false;
#endif
}


// gst_hw_encoder_works
static bool gst_hw_encoder_works( const char* encoder )
{
	// the V4L2 encoders can exist without working hardware behind them (in a container without the
	// devices, or when the GPU failed to start) and /dev/v4l2-nvenc always opens, so that only shows
	// up once frames go through - encode a couple of frames once, before the real pipeline is built
	static std::map<std::string, bool> results;

	const std::map<std::string, bool>::iterator cached = results.find(encoder);

	if( cached != results.end() )
		return cached->second;

	bool works = false;

	if( gst_element_exists(encoder) )
	{
		const std::string launchStr = std::string("videotestsrc num-buffers=2 ! video/x-raw,width=320,height=240,format=I420,framerate=30/1 ! "
										  "nvvidconv ! video/x-raw(memory:NVMM) ! ") + encoder + " ! fakesink";

		GError* err = NULL;
		GstElement* pipeline = gst_parse_launch(launchStr.c_str(), &err);

		if( err != NULL )
		{
			LogVerbose(LOG_GSTREAMER "%s test pipeline couldn't be created (%s)\n", encoder, err->message);
			g_error_free(err);
		}

		if( pipeline != NULL )
		{
			GstBus* bus = gst_element_get_bus(pipeline);

			if( gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE )
			{
				GstMessage* msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND, (GstMessageType)(GST_MESSAGE_EOS|GST_MESSAGE_ERROR));

				if( msg != NULL && GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS )
				{
					works = true;
				}
				else if( msg != NULL )
				{
					GError* error = NULL;
					gst_message_parse_error(msg, &error, NULL);
					LogWarning(LOG_GSTREAMER "%s failed to encode test frames: %s\n", encoder, error != NULL ? error->message : "unknown error");
					g_clear_error(&error);
				}
				else
				{
					LogWarning(LOG_GSTREAMER "%s timed out encoding test frames\n", encoder);
				}

				if( msg != NULL )
					gst_message_unref(msg);
			}

			gst_element_set_state(pipeline, GST_STATE_NULL);
			gst_object_unref(bus);
			gst_object_unref(pipeline);
		}
	}

	results[encoder] = works;
	return works;
}


// gst_select_encoder
const char* gst_select_encoder( videoOptions::Codec codec, videoOptions::CodecType& type )
{
	// remap nvenc before the platform checks, so it gets the same hardware checks as v4l2
	// (otherwise it picked the V4L2 hardware encoders on Orin Nano, which doesn't have NVENC)
	if( type == videoOptions::CODEC_NVENC || type == videoOptions::CODEC_NVDEC )
		type = gst_default_codec();  // TODO NVENC/NVDEC support

#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
	if( type == videoOptions::CODEC_OMX )
	{
		// JetPack 5 doesn't have OMX
		type = gst_default_codec();
	}
	else if( type == videoOptions::CODEC_V4L2 )
	{
		const bool has_hw_encoder = gst_query_hw_encoder();
		
		if( !has_hw_encoder )
		{
			LogWarning(LOG_GSTREAMER "gstEncoder -- hardware encoder not detected, reverting to CPU encoder\n");
			type = videoOptions::CODEC_CPU;
		}
	}
#endif
#elif defined(__x86_64__) || defined(__amd64__)
	if( type == videoOptions::CODEC_OMX || type == videoOptions::CODEC_V4L2 )
		type = gst_default_codec();
#endif

	if( codec == videoOptions::CODEC_RAW )
		type = videoOptions::CODEC_CPU;

	if( codec == videoOptions::CODEC_AV1 && type != videoOptions::CODEC_CPU && !(type == videoOptions::CODEC_V4L2 && gst_query_hw_av1() && gst_element_exists("nvv4l2av1enc")) )
	{
		LogWarning(LOG_GSTREAMER "gstEncoder -- hardware AV1 encoder requires Orin or newer (except Orin Nano), reverting to CPU encoder\n");
		type = videoOptions::CODEC_CPU;
	}

	if( type == videoOptions::CODEC_CPU )
	{
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "x264enc";
			case videoOptions::CODEC_H265:   return "x265enc";
			case videoOptions::CODEC_VP8:	   return "vp8enc";
			case videoOptions::CODEC_VP9:    return "vp9enc";
			case videoOptions::CODEC_AV1:    return gst_select_av1_encoder();
			case videoOptions::CODEC_MJPEG:  return "jpegenc";
		}
	}
	else if( type == videoOptions::CODEC_OMX )
	{
	#if GST_CHECK_VERSION(1,0,0)
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "omxh264enc";
			case videoOptions::CODEC_H265:   return "omxh265enc";
			case videoOptions::CODEC_VP8:	   return "omxvp8enc";
			case videoOptions::CODEC_VP9:    return "omxvp9enc";
			case videoOptions::CODEC_MJPEG:  return "nvjpegenc";
		}
	#else
		switch(codec)
		{
			case videoOptions::CODEC_H264:   return "nv_omx_h264enc";
			case videoOptions::CODEC_H265:   return "nv_omx_h265enc";
			case videoOptions::CODEC_VP8:	   return "nv_omx_vp8enc";
			case videoOptions::CODEC_VP9:    return "nv_omx_vp9enc";
			case videoOptions::CODEC_MJPEG:  return "nvjpegenc";
		}
	#endif
	}
	else if( type == videoOptions::CODEC_V4L2 )
	{
		const char* encoder = NULL;

		switch(codec)
		{
			case videoOptions::CODEC_H264:   encoder = "nvv4l2h264enc"; break;
			case videoOptions::CODEC_H265:   encoder = "nvv4l2h265enc"; break;
			case videoOptions::CODEC_VP8:	   encoder = "nvv4l2vp8enc"; break;
			case videoOptions::CODEC_VP9:    encoder = "nvv4l2vp9enc"; break;
			case videoOptions::CODEC_AV1:    encoder = "nvv4l2av1enc"; break;
			case videoOptions::CODEC_MJPEG:  return "nvjpegenc";
		}

		if( encoder != NULL && !gst_hw_encoder_works(encoder) )
		{
			LogWarning(LOG_GSTREAMER "gstEncoder -- %s failed to start, reverting to CPU encoder\n", encoder);
			type = videoOptions::CODEC_CPU;
			return gst_select_encoder(codec, type);
		}

		return encoder;
	}
	
	return NULL;
}


// gst_default_codec_type
videoOptions::CodecType gst_default_codec()
{
#if defined(__aarch64__)
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
	return videoOptions::CODEC_V4L2;	// JetPack 5
#else
	return videoOptions::CODEC_OMX;	// JetPack 4
#endif
#elif defined(__x86_64__) || defined(__amd64__)
	return videoOptions::CODEC_CPU;	// x86
#endif
}

