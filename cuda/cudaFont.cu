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

#include "cudaFont.h"
#include "cudaVector.h"
#include "cudaOverlay.h"
#include "cudaMappedMemory.h"

#include "imageIO.h"
#include "filesystem.h"
#include "logging.h"

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "../cpp/image/stb/stb_truetype.h"


//#define DEBUG_FONT


// Decode the next UTF-8 codepoint from str, advancing the pointer.
// Returns 0 when the end of string is reached.
static uint32_t nextUTF8( const char*& str )
{
	if( !*str )
		return 0;

	uint8_t c = (uint8_t)*str++;

	if( c < 0x80 )
		return c;

	if( (c & 0xE0) == 0xC0 )
	{
		uint32_t cp = (c & 0x1F) << 6;
		if( (*str & 0xC0) == 0x80 ) cp |= (*str++ & 0x3F);
		return cp;
	}

	if( (c & 0xF0) == 0xE0 )
	{
		uint32_t cp = (c & 0x0F) << 12;
		if( (*str & 0xC0) == 0x80 ) cp |= (*str++ & 0x3F) << 6;
		if( (*str & 0xC0) == 0x80 ) cp |= (*str++ & 0x3F);
		return cp;
	}

	if( (c & 0xF8) == 0xF0 )
	{
		uint32_t cp = (c & 0x07) << 18;
		if( (*str & 0xC0) == 0x80 ) cp |= (*str++ & 0x3F) << 12;
		if( (*str & 0xC0) == 0x80 ) cp |= (*str++ & 0x3F) << 6;
		if( (*str & 0xC0) == 0x80 ) cp |= (*str++ & 0x3F);
		return cp;
	}

	return 0xFFFD; // replacement character
}


// Struct for one character to render
struct __align__(16) GlyphCommand
{
	short x;		// x coordinate origin in output image to begin drawing the glyph at 
	short y;		// y coordinate origin in output image to begin drawing the glyph at 
	short u;		// x texture coordinate in the baked font map where the glyph resides
	short v;		// y texture coordinate in the baked font map where the glyph resides 
	short width;	// width of the glyph in pixels
	short height;	// height of the glyph in pixels
};


// adaptFontSize
float adaptFontSize( uint32_t dimension )
{
	const float max_font = 32.0f;
	const float min_font = 28.0f;

	const uint32_t max_dim = 1536;
	const uint32_t min_dim = 768;

	if( dimension > max_dim )
		dimension = max_dim;

	if( dimension < min_dim )
		dimension = min_dim;

	const float dim_ratio = float(dimension - min_dim) / float(max_dim - min_dim);

	return min_font + dim_ratio * (max_font - min_font);
}


// constructor
cudaFont::cudaFont()
{
	mSize = 0.0f;
	
	mCommandCPU = NULL;
	mCommandGPU = NULL;
	mCmdIndex   = 0;

	mFontMapCPU = NULL;
	mFontMapGPU = NULL;

	mRectsCPU   = NULL;
	mRectsGPU   = NULL;
	mRectIndex  = 0;

	mFontMapWidth  = 512;
	mFontMapHeight = 512;

	mMaxGlyphHeight = 0;
}



// destructor
cudaFont::~cudaFont()
{
	if( mRectsCPU != NULL )
	{
		CUDA(cudaFreeHost(mRectsCPU));
		
		mRectsCPU = NULL; 
		mRectsGPU = NULL;
	}

	if( mCommandCPU != NULL )
	{
		CUDA(cudaFreeHost(mCommandCPU));
		
		mCommandCPU = NULL; 
		mCommandGPU = NULL;
	}

	if( mFontMapCPU != NULL )
	{
		CUDA(cudaFreeHost(mFontMapCPU));
		
		mFontMapCPU = NULL; 
		mFontMapGPU = NULL;
	}
}


// Create
cudaFont* cudaFont::Create( float size )
{
	// default fonts	
	std::vector<std::string> fonts;
	
	fonts.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
	fonts.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

	return Create(fonts, size);
}


// Create
cudaFont* cudaFont::Create( const std::vector<std::string>& fonts, float size )
{
	const uint32_t numFonts = fonts.size();

	for( uint32_t n=0; n < numFonts; n++ )
	{
		cudaFont* font = Create(fonts[n].c_str(), size);

		if( font != NULL )
			return font;
	}

	return NULL;
}


// Create
cudaFont* cudaFont::Create( const char* font, float size )
{
	// verify parameters
	if( !font )
		return Create(size);

	// create new font
	cudaFont* c = new cudaFont();
	
	if( !c )
		return NULL;
		
	if( !c->init(font, size) )
	{
		delete c;
		return NULL;
	}

	return c;
}


// init
bool cudaFont::init( const char* filename, float size )
{
	// validate parameters
	if( !filename )
		return NULL;

	// verify that the font file exists and get its size
	const size_t ttf_size = fileSize(filename);

	if( !ttf_size )
	{
		LogError(LOG_CUDA "font doesn't exist or empty file '%s'\n", filename);
 		return false;
	}

	// allocate memory to store the font file
	void* ttf_buffer = malloc(ttf_size);

	if( !ttf_buffer )
	{
		LogError(LOG_CUDA "failed to allocate %zu byte buffer for reading '%s'\n", ttf_size, filename);
		return false;
	}

	// open the font file
	FILE* ttf_file = fopen(filename, "rb");

	if( !ttf_file )
	{
		LogError(LOG_CUDA "failed to open '%s' for reading\n", filename);
		free(ttf_buffer);
		return false;
	}

	// read the font file
	const size_t ttf_read = fread(ttf_buffer, 1, ttf_size, ttf_file);

	fclose(ttf_file);

	if( ttf_read != ttf_size )
	{
		LogError(LOG_CUDA "failed to read contents of '%s'\n", filename);
		LogError(LOG_CUDA "(read %zu bytes, expected %zu bytes)\n", ttf_read, ttf_size);

		free(ttf_buffer);
		return false;
	}

	// buffers that store the coordinates of the baked glyphs
	stbtt_packedchar latinCoords[LatinCount];
	stbtt_packedchar cyrillicCoords[CyrillicCount];

	// define the Unicode ranges to bake
	stbtt_pack_range ranges[2] = {};

	ranges[0].font_size = size;
	ranges[0].first_unicode_codepoint_in_range = LatinFirst;
	ranges[0].num_chars = LatinCount;
	ranges[0].chardata_for_range = latinCoords;

	ranges[1].font_size = size;
	ranges[1].first_unicode_codepoint_in_range = CyrillicFirst;
	ranges[1].num_chars = CyrillicCount;
	ranges[1].chardata_for_range = cyrillicCoords;

	// increase the size of the bitmap until all the glyphs fit
	while(true)
	{
		// allocate memory for the packed font texture (alpha only)
		const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

		if( !cudaAllocMapped((void**)&mFontMapCPU, (void**)&mFontMapGPU, fontMapSize) )
		{
			LogError(LOG_CUDA "failed to allocate %zu bytes to store %ix%i font map\n", fontMapSize, mFontMapWidth, mFontMapHeight);
			free(ttf_buffer);
			return false;
		}

		// initialize packing context
		stbtt_pack_context spc;

		if( !stbtt_PackBegin(&spc, mFontMapCPU, mFontMapWidth, mFontMapHeight, 0, 1, NULL) )
		{
			LogError(LOG_CUDA "failed to initialize font packing context\n");
			free(ttf_buffer);
			return false;
		}

		stbtt_PackSetSkipMissingCodepoints(&spc, 1);

		// attempt to pack all glyph ranges
		stbtt_PackFontRanges(&spc, (uint8_t*)ttf_buffer, 0, ranges, 2);
		stbtt_PackEnd(&spc);

		// stbtt_PackFontRanges returns 0 if any codepoint is missing from the
		// font, even with skip_missing set.  Instead of trusting that return
		// value, verify packing succeeded by checking a known glyph ('A').
		const int testIdx = 'A' - LatinFirst;
		if( latinCoords[testIdx].x1 > latinCoords[testIdx].x0 )
		{
			LogVerbose(LOG_CUDA "packed %u glyphs in %ux%u bitmap (font size=%.0fpx)\n", NumGlyphs, mFontMapWidth, mFontMapHeight, size);
			break;
		}

	#ifdef DEBUG_FONT
		LogDebug(LOG_CUDA "failed to fit all glyphs in %ux%u bitmap\n", mFontMapWidth, mFontMapHeight);
	#endif

		CUDA(cudaFreeHost(mFontMapCPU));

		mFontMapCPU = NULL;
		mFontMapGPU = NULL;

		mFontMapWidth *= 2;
		mFontMapHeight *= 2;

		if( mFontMapWidth > 4096 )
		{
			LogError(LOG_CUDA "failed to pack font glyphs (atlas exceeded 4096x4096)\n");
			free(ttf_buffer);
			return false;
		}

	#ifdef DEBUG_FONT
		LogDebug(LOG_CUDA "attempting to pack font with %ux%u bitmap...\n", mFontMapWidth, mFontMapHeight);
	#endif
	}

	// free the TTF font data
	free(ttf_buffer);

	// store glyph info from Latin range
	for( uint32_t n=0; n < LatinCount; n++ )
	{
		mGlyphInfo[n].x = latinCoords[n].x0;
		mGlyphInfo[n].y = latinCoords[n].y0;

		mGlyphInfo[n].width  = latinCoords[n].x1 - latinCoords[n].x0;
		mGlyphInfo[n].height = latinCoords[n].y1 - latinCoords[n].y0;

		mGlyphInfo[n].xAdvance = latinCoords[n].xadvance;
		mGlyphInfo[n].xOffset  = latinCoords[n].xoff;
		mGlyphInfo[n].yOffset  = latinCoords[n].yoff;
	}

	// store glyph info from Cyrillic range
	for( uint32_t n=0; n < CyrillicCount; n++ )
	{
		const uint32_t idx = LatinCount + n;

		mGlyphInfo[idx].x = cyrillicCoords[n].x0;
		mGlyphInfo[idx].y = cyrillicCoords[n].y0;

		mGlyphInfo[idx].width  = cyrillicCoords[n].x1 - cyrillicCoords[n].x0;
		mGlyphInfo[idx].height = cyrillicCoords[n].y1 - cyrillicCoords[n].y0;

		mGlyphInfo[idx].xAdvance = cyrillicCoords[n].xadvance;
		mGlyphInfo[idx].xOffset  = cyrillicCoords[n].xoff;
		mGlyphInfo[idx].yOffset  = cyrillicCoords[n].yoff;
	}

	// compute uniform glyph cell height
	mMaxGlyphHeight = 0;
	for( uint32_t i = 0; i < NumGlyphs; i++ )
	{
		const int h = abs((int)mGlyphInfo[i].yOffset);
		if( h > mMaxGlyphHeight )
			mMaxGlyphHeight = h;
	}

	// allocate memory for GPU command buffer
	if( !cudaAllocMapped(&mCommandCPU, &mCommandGPU, sizeof(GlyphCommand) * MaxCommands) )
		return false;
	
	// allocate memory for background rect buffers
	if( !cudaAllocMapped((void**)&mRectsCPU, (void**)&mRectsGPU, sizeof(float4) * MaxCommands) )
		return false;

	mSize = size;
	return true;
}


/*inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}*/

inline __host__ __device__ float4 alpha_blend( const float4& bg, const float4& fg )
{
	const float alpha = fg.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	return make_float4(alpha * fg.x + ialph * bg.x,
				    alpha * fg.y + ialph * bg.y,
				    alpha * fg.z + ialph * bg.z,
				    bg.w);
} 


template<typename T>
__global__ void gpuOverlayText( unsigned char* font, int fontWidth, GlyphCommand* commands,
                                T* input, T* output, int imgWidth, int imgHeight, float4 color ) 
{
	const GlyphCommand cmd = commands[blockIdx.x];

	if( threadIdx.x >= cmd.width || threadIdx.y >= cmd.height )
		return;

	const int x = cmd.x + threadIdx.x;
	const int y = cmd.y + threadIdx.y;

	if( x < 0 || y < 0 || x >= imgWidth || y >= imgHeight )
		return;

	const int u = cmd.u + threadIdx.x;
	const int v = cmd.v + threadIdx.y;

	const float px_glyph = font[v * fontWidth + u];

	const float4 px_font = make_float4(px_glyph * color.x, px_glyph * color.y, px_glyph * color.z, px_glyph * color.w);
	const float4 px_in   = cast_vec<float4>(input[y * imgWidth + x]);

	output[y * imgWidth + x] = cast_vec<T>(alpha_blend(px_in, px_font));	 
}


// cudaOverlayText
cudaError_t cudaOverlayText( unsigned char* font, const int2& maxGlyphSize, size_t fontMapWidth,
                             GlyphCommand* commands, size_t numCommands, const float4& fontColor, 
                             void* input, void* output, imageFormat format, size_t imgWidth, size_t imgHeight,
                             cudaStream_t stream )	
{
	if( !font || !commands || !input || !output || numCommands == 0 || fontMapWidth == 0 || imgWidth == 0 || imgHeight == 0 )
		return cudaErrorInvalidValue;

	const float4 color_scaled = make_float4( fontColor.x / 255.0f, fontColor.y / 255.0f, fontColor.z / 255.0f, fontColor.w / 255.0f );
	
	// setup arguments
	const dim3 block(maxGlyphSize.x, maxGlyphSize.y);
	const dim3 grid(numCommands);

	if( format == IMAGE_RGB8 )
		gpuOverlayText<uchar3><<<grid, block, 0, stream>>>(font, fontMapWidth, commands, (uchar3*)input, (uchar3*)output, imgWidth, imgHeight, color_scaled); 
	else if( format == IMAGE_RGBA8 )
		gpuOverlayText<uchar4><<<grid, block, 0, stream>>>(font, fontMapWidth, commands, (uchar4*)input, (uchar4*)output, imgWidth, imgHeight, color_scaled); 
	else if( format == IMAGE_RGB32F )
		gpuOverlayText<float3><<<grid, block, 0, stream>>>(font, fontMapWidth, commands, (float3*)input, (float3*)output, imgWidth, imgHeight, color_scaled); 
	else if( format == IMAGE_RGBA32F )
		gpuOverlayText<float4><<<grid, block, 0, stream>>>(font, fontMapWidth, commands, (float4*)input, (float4*)output, imgWidth, imgHeight, color_scaled); 
	else
		return cudaErrorInvalidValue;

	return cudaGetLastError();
}


// Overlay
bool cudaFont::OverlayText( void* image, imageFormat format, uint32_t width, uint32_t height, 
                            const std::vector< std::pair< std::string, int2 > >& strings, 
                            const float4& color, const float4& bg_color, int bg_padding,
                            cudaStream_t stream )
{
	const uint32_t numStrings = strings.size();

	if( !image || width == 0 || height == 0 || numStrings == 0 )
		return false;

	if( format != IMAGE_RGB8 && format != IMAGE_RGBA8 && format != IMAGE_RGB32F && format != IMAGE_RGBA32F )
	{
		LogError(LOG_CUDA "cudaFont::OverlayText() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_CUDA "                           supported formats are:\n");
		LogError(LOG_CUDA "                              * rgb8\n");		
		LogError(LOG_CUDA "                              * rgba8\n");		
		LogError(LOG_CUDA "                              * rgb32f\n");		
		LogError(LOG_CUDA "                              * rgba32f\n");

		return false;
	}

	
	const bool has_bg = bg_color.w > 0.0f;
	int2 maxGlyphSize = make_int2(0,0);

	int numCommands = 0;
	int numRects = 0;
	int maxChars = 0;

	// find the bg rects and total char count
	for( uint32_t s=0; s < numStrings; s++ )
		maxChars += strings[s].first.size();

	// reset the buffer indices if we need the space
	if( mCmdIndex + maxChars >= MaxCommands )
		mCmdIndex = 0;

	if( has_bg && mRectIndex + numStrings >= MaxCommands )
		mRectIndex = 0;

	// generate glyph commands and bg rects
	for( uint32_t s=0; s < numStrings; s++ )
	{
		const uint32_t numChars = strings[s].first.size();
		
		if( numChars == 0 )
			continue;

		// use precomputed uniform glyph cell height
		const int maxHeight = mMaxGlyphHeight;

	#ifdef DEBUG_FONT
		LogDebug(LOG_CUDA "max glyph height:  %i\n", maxHeight);
	#endif

		// get the starting position of the string
		int2 pos = strings[s].second;

		if( pos.x < 0 )
			pos.x = 0;

		if( pos.y < 0 )
			pos.y = 0;
		
		pos.y += maxHeight;

		// reset the background rect if needed
		if( has_bg )
			mRectsCPU[mRectIndex] = make_float4(width, height, 0, 0);

		// make a glyph command for each character
		const char* p = strings[s].first.c_str();
		while( *p )
		{
			const uint32_t cp = nextUTF8(p);
			const int gi = GlyphIndex(cp);

			// make sure the character is in range
			if( gi < 0 )
				continue;

			// fill the next command
			GlyphCommand* cmd = ((GlyphCommand*)mCommandCPU) + mCmdIndex + numCommands;

			cmd->x = pos.x;
			cmd->y = pos.y + mGlyphInfo[gi].yOffset;
			cmd->u = mGlyphInfo[gi].x;
			cmd->v = mGlyphInfo[gi].y;

			cmd->width  = mGlyphInfo[gi].width;
			cmd->height = mGlyphInfo[gi].height;

			// advance the text position
			pos.x += mGlyphInfo[gi].xAdvance;

			// track the maximum glyph size
			if( maxGlyphSize.x < mGlyphInfo[gi].width )
				maxGlyphSize.x = mGlyphInfo[gi].width;

			if( maxGlyphSize.y < mGlyphInfo[gi].height )
				maxGlyphSize.y = mGlyphInfo[gi].height;

			// expand the background rect
			if( has_bg )
			{
				float4* rect = mRectsCPU + mRectIndex + numRects;

				if( cmd->x < rect->x )
					rect->x = cmd->x;

				if( cmd->y < rect->y )
					rect->y = cmd->y;

				const float x2 = cmd->x + cmd->width;
				const float y2 = cmd->y + cmd->height;

				if( x2 > rect->z )
					rect->z = x2;

				if( y2 > rect->w )
					rect->w = y2;
			}

			numCommands++;
		}

		if( has_bg )
		{
			float4* rect = mRectsCPU + mRectIndex + numRects;

			// apply padding
			rect->x -= bg_padding;
			rect->y -= bg_padding;
			rect->z += bg_padding;
			rect->w += bg_padding;

			numRects++;
		}
	}

#ifdef DEBUG_FONT
	LogDebug(LOG_CUDA "max glyph size is %ix%i\n", maxGlyphSize.x, maxGlyphSize.y);
#endif

	// draw background rects
	if( has_bg && numRects > 0 )
		CUDA(cudaRectFill(image, image, width, height, format, mRectsGPU + mRectIndex, numRects, bg_color, stream));

	// draw text characters
	CUDA(cudaOverlayText(mFontMapGPU, maxGlyphSize, mFontMapWidth,
                         ((GlyphCommand*)mCommandGPU) + mCmdIndex, numCommands, 
                         color, image, image, format, width, height, stream));
			
	// advance the buffer indices
	mCmdIndex += numCommands;
	mRectIndex += numRects;
		   
	return true;
}


// Overlay
bool cudaFont::OverlayText( void* image, imageFormat format, uint32_t width, uint32_t height, 
                            const char* str, int x, int y, const float4& color, const float4& bg_color, 
                            int bg_padding, cudaStream_t stream )
{
	if( !str )
		return NULL;
		
	std::vector< std::pair< std::string, int2 > > list;
	
	list.push_back( std::pair< std::string, int2 >( str, make_int2(x,y) ));

	return OverlayText(image, format, width, height, list, color, bg_color, bg_padding, stream);
}


// TextExtents
int4 cudaFont::TextExtents( const char* str, int x, int y )
{
	if( !str )
		return make_int4(0,0,0,0);

	// use precomputed uniform glyph cell height
	const int maxHeight = mMaxGlyphHeight;

	// get the starting position of the string
	int2 pos = make_int2(x,y);

	if( pos.x < 0 )
		pos.x = 0;

	if( pos.y < 0 )
		pos.y = 0;

	pos.y += maxHeight;


	// find the extents of the string
	const char* p = str;
	while( *p )
	{
		const uint32_t cp = nextUTF8(p);
		const int gi = GlyphIndex(cp);

		// make sure the character is in range
		if( gi < 0 )
			continue;

		// advance the text position
		pos.x += mGlyphInfo[gi].xAdvance;
	}

	return make_int4(x, y, pos.x, pos.y);
}
	


				
	
