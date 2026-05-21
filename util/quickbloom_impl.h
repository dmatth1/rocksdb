//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// QuickBloom: Split Block Bloom Filter probe kernel ported from
// https://github.com/dmatth1/quickbloom (`bloom_sbbf.c`, single_key
// variant) into RocksDB's filter machinery. SBBF was originally
// described by the Apache Parquet spec (Putze, Sanders, Singler 2007).
// The same 256-bit block + fixed K=8 salt layout is used by arrow-cpp
// / arrow-rs / Velox / DuckDB; this is the AVX2-optimised single-key
// kernel.
//
// Differences from upstream quickbloom (kept the kernel intact, adapted
// only the addressing to RocksDB conventions):
//   - Block index uses `FastRange32(Lower32of64(h), num_blocks)` rather
//     than bitmasking a power-of-2 block count. This matches the
//     RocksDB `FastLocalBloomImpl` sizing model and removes the
//     power-of-2 constraint on filter size.
//   - The 32-bit probe hash is `Upper32of64(h)` (RocksDB convention)
//     rather than the low half of the 64-bit hash.
//   - Loads/stores use the unaligned AVX2 intrinsics; RocksDB filter
//     buffers are not guaranteed 32-byte aligned.
//
// Differences from `FastLocalBloomImpl` (the win we're benchmarking):
//   - 256-bit blocks (32 bytes) instead of 512-bit cache-line blocks,
//     so half the bytes per probe at the cost of two blocks per cache
//     line.
//   - K=8 fixed, materialised in one AVX2 mask via SBBF salts:
//     `vpmullo + vpsrli + vpsllv`. Single `vptest` settles the probe.
//   - No num_probes parameter, no per-probe loop, no permute/blend
//     ladder for splitting an 8-way mask across the two halves of a
//     64-byte cache line.

#pragma once
#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "port/port.h"        // for PREFETCH
#include "util/bloom_impl.h"  // for BloomMath
#include "util/fastrange.h"
#include "util/hash.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ROCKSDB_NAMESPACE {

// 256-bit SBBF probe kernel, K=8, AVX2 mask compute when available
// with a scalar fallback that preserves the on-disk format. Mirrors
// the static-method shape of `FastLocalBloomImpl` so it slots into the
// same builder/reader plumbing.
class QuickBloomImpl {
 public:
  // Block layout constants.
  static constexpr int kBlockBytes = 32;  // 256-bit blocks
  static constexpr int kNumProbes = 8;    // K=8, one bit per 32-bit lane

  // Eight SBBF salts (one per 32-bit lane). Same constants as
  // quickbloom/bloom_sbbf.c. Independent multipliers (rather than a
  // chained multiply like `FastLocalBloomImpl`) get us K=8 in one
  // mask compute with no cross-lane dependencies, at the cost of
  // baking the salt vector into the kernel.
  static constexpr uint32_t kSalts[8] = {
      0x47b6137bu, 0x44974d91u, 0x8824ad5bu, 0xa2b7289du,
      0x705495c7u, 0x2df1424bu, 0x9efc4947u, 0x5c6bfb31u};

  // Estimated false-positive rate for SBBF at the given sizing. Reuses
  // the BloomMath cache-local estimate against a 256-bit "cache line"
  // and the 64-bit hash fingerprint term, mirroring the shape of
  // `FastLocalBloomImpl::EstimatedFpRate`. Validated only to "useful
  // for warnings / user feedback" accuracy.
  static double EstimatedFpRate(size_t keys, size_t bytes, int hash_bits) {
    return BloomMath::IndependentProbabilitySum(
        BloomMath::CacheLocalFpRate(
            8.0 * static_cast<double>(bytes) / static_cast<double>(keys),
            kNumProbes,
            /*cache line bits*/ kBlockBytes * 8),
        BloomMath::FingerprintFpRate(keys, hash_bits));
  }

  // Insert a 64-bit hash into the filter. `len_bytes` is the size of
  // the bloom region (excluding metadata).
  static inline void AddHash(uint64_t h, uint32_t len_bytes, char* data) {
    uint32_t block_offset = BlockOffset(h, len_bytes);
    AddHashPrepared(h, data + block_offset);
  }

  // Insert when the block offset has already been computed by
  // PrepareHash. `data_at_block` must point at the start of a 32-byte
  // block.
  static inline void AddHashPrepared(uint64_t h, char* data_at_block) {
#ifdef __AVX2__
    const __m256i mask = MaskFor(Upper32of64(h));
    __m256i* p = reinterpret_cast<__m256i*>(data_at_block);
    __m256i cur = _mm256_loadu_si256(p);
    _mm256_storeu_si256(p, _mm256_or_si256(cur, mask));
#else
    ScalarAdd(Upper32of64(h), data_at_block);
#endif
  }

  // Compute the 32-byte block offset for `h` within `len_bytes` of
  // filter data, prefetch the block, and store the offset for the
  // caller to use later.
  static inline void PrepareHash(uint64_t h, uint32_t len_bytes,
                                 const char* data,
                                 uint32_t /*out*/ * block_offset) {
    uint32_t off = BlockOffset(h, len_bytes);
    // 32-byte block fits in a single cache line, so one prefetch is
    // sufficient (vs. two for the 64-byte FastLocalBloom block).
    PREFETCH(data + off, 0 /* rw */, 1 /* locality */);
    *block_offset = off;
  }

  // Probe path. Returns true if all 8 SBBF bits demanded by `h` are
  // set in the chosen block (the key may be present).
  static inline bool HashMayMatch(uint64_t h, uint32_t len_bytes,
                                  const char* data) {
    uint32_t block_offset = BlockOffset(h, len_bytes);
    return HashMayMatchPrepared(h, data + block_offset);
  }

  static inline bool HashMayMatchPrepared(uint64_t h,
                                          const char* data_at_block) {
#ifdef __AVX2__
    const __m256i mask = MaskFor(Upper32of64(h));
    const __m256i cur = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(data_at_block));
    // testc(cur, mask) is true iff (~cur & mask) == 0, i.e. every bit
    // demanded by the mask is already set in the block.
    return _mm256_testc_si256(cur, mask) != 0;
#else
    return ScalarMatch(Upper32of64(h), data_at_block);
#endif
  }

 private:
  // Map a 64-bit hash to a 32-byte block offset within the filter.
  // Mirrors `FastLocalBloomImpl`'s FastRange32 trick, but with a
  // 32-byte (rather than 64-byte) block stride: nblocks = len_bytes /
  // 32.
  static inline uint32_t BlockOffset(uint64_t h, uint32_t len_bytes) {
    uint32_t num_blocks = len_bytes >> 5;  // 32-byte blocks
    return FastRange32(Lower32of64(h), num_blocks) << 5;
  }

#ifdef __AVX2__
  static inline __m256i MaskFor(uint32_t h32) {
    const __m256i salt = _mm256_set_epi32(
        static_cast<int>(kSalts[0]), static_cast<int>(kSalts[1]),
        static_cast<int>(kSalts[2]), static_cast<int>(kSalts[3]),
        static_cast<int>(kSalts[4]), static_cast<int>(kSalts[5]),
        static_cast<int>(kSalts[6]), static_cast<int>(kSalts[7]));
    const __m256i hbcast = _mm256_set1_epi32(static_cast<int>(h32));
    const __m256i prod = _mm256_mullo_epi32(hbcast, salt);
    // Top 5 bits of each 32-bit product pick the bit within the lane.
    const __m256i shift = _mm256_srli_epi32(prod, 27);
    const __m256i ones = _mm256_set1_epi32(1);
    return _mm256_sllv_epi32(ones, shift);
  }
#endif

  // Portable scalar paths used when AVX2 isn't available. Same
  // algorithm (K=8 fixed, SBBF salts, top-5-bits-as-bitpos within each
  // 32-bit lane) so the on-disk format is identical across builds.
  static inline void ScalarAdd(uint32_t h32, char* block) {
    for (int i = 0; i < 8; ++i) {
      uint32_t bit = uint32_t{1} << ((h32 * kSalts[i]) >> 27);
      uint32_t word;
      std::memcpy(&word, block + i * 4, 4);
      word |= bit;
      std::memcpy(block + i * 4, &word, 4);
    }
  }

  static inline bool ScalarMatch(uint32_t h32, const char* block) {
    for (int i = 0; i < 8; ++i) {
      uint32_t bit = uint32_t{1} << ((h32 * kSalts[i]) >> 27);
      uint32_t word;
      std::memcpy(&word, block + i * 4, 4);
      if ((word & bit) == 0) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace ROCKSDB_NAMESPACE
