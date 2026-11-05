// Apple's compression library: used only by the shader capture tool (nvn_capture.c) for LZFSE. Linux builds answer 0 (nothing written / decoded), which the
// caller treats as a failed compression and keeps the data raw.
#pragma once
#include "dm_linux.h"
typedef enum { COMPRESSION_LZ4 = 0x100, COMPRESSION_ZLIB = 0x205, COMPRESSION_LZMA = 0x306, COMPRESSION_LZ4_RAW = 0x101, COMPRESSION_LZFSE = 0x801 } compression_algorithm;
static inline size_t compression_encode_scratch_buffer_size(compression_algorithm a) { (void)a; return 0; }
static inline size_t compression_decode_scratch_buffer_size(compression_algorithm a) { (void)a; return 0; }
static inline size_t compression_encode_buffer(uint8_t *d, size_t dn, const uint8_t *s, size_t sn, void *scratch, compression_algorithm a) { (void)d; (void)dn; (void)s; (void)sn; (void)scratch; (void)a; return 0; }
static inline size_t compression_decode_buffer(uint8_t *d, size_t dn, const uint8_t *s, size_t sn, void *scratch, compression_algorithm a) { (void)d; (void)dn; (void)s; (void)sn; (void)scratch; (void)a; return 0; }
