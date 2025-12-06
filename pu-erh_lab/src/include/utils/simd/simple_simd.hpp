// simple_simd.h -- minimal cross-arch SIMD abstraction
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// --- arch detection ---
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define SIMPLE_SIMD_X86 1
#else
#define SIMPLE_SIMD_X86 0
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
#define SIMPLE_SIMD_ARM 1
#else
#define SIMPLE_SIMD_ARM 0
#endif

#if SIMPLE_SIMD_X86
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif SIMPLE_SIMD_ARM
#include <arm_neon.h>
#endif

namespace simple_simd {

// -------- feature detection (x86) ----------
struct CPUFeatures {
  bool sse41 = false;
  bool sse42 = false;
  bool avx2  = false;
  bool avx   = false;
};

inline CPUFeatures detect_cpu() {
  CPUFeatures f;
#if SIMPLE_SIMD_X86 && defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int info[4];
  __cpuid(info, 0);
  int nIds = info[0];
  if (nIds >= 1) {
    __cpuid(info, 1);
    int ecx = info[2];
    f.sse42 = (ecx & (1 << 20)) != 0;
    f.sse41 = (ecx & (1 << 19)) != 0;
    f.avx   = (ecx & (1 << 28)) != 0;
  }
  if (nIds >= 7) {
    __cpuidex(info, 7, 0);
    int ebx = info[1];
    f.avx2  = (ebx & (1 << 5)) != 0;
  }
#elif SIMPLE_SIMD_X86 && (defined(__GNUC__) || defined(__clang__))
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid_max(0, nullptr) >= 1) {
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    f.sse41 = (ecx & (1 << 19)) != 0;
    f.sse42 = (ecx & (1 << 20)) != 0;
    f.avx   = (ecx & (1 << 28)) != 0;
  }
  if (__get_cpuid_max(0, nullptr) >= 7) {
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    f.avx2 = (ebx & (1 << 5)) != 0;
  }
#endif
  return f;
}

// -------- types ----------
#if SIMPLE_SIMD_ARM
using f32x4 = float32x4_t;  // ARM NEON 128-bit vector
#else
using f32x4 = __m128;  // consistent 128-bit API
#endif

// -------- scalar fallback ----------
#if SIMPLE_SIMD_X86
namespace impl_scalar {
inline f32x4 load(const float* p) { return _mm_set_ps(p[3], p[2], p[1], p[0]); }
inline void  store(float* dst, const f32x4& v) { _mm_storeu_ps(dst, v); }
inline f32x4 set1(float x) { return _mm_set1_ps(x); }
inline f32x4 mul(const f32x4& a, const f32x4& b) {
  alignas(16) float A[4], B[4];
  _mm_store_ps(A, a);
  _mm_store_ps(B, b);
  return _mm_set_ps(A[3] * B[3], A[2] * B[2], A[1] * B[1], A[0] * B[0]);
}
inline f32x4 add(const f32x4& a, const f32x4& b) {
  alignas(16) float A[4], B[4];
  _mm_store_ps(A, a);
  _mm_store_ps(B, b);
  return _mm_set_ps(A[3] + B[3], A[2] + B[2], A[1] + B[1], A[0] + B[0]);
}
inline f32x4 sub(const f32x4& a, const f32x4& b) {
  alignas(16) float A[4], B[4];
  _mm_store_ps(A, a);
  _mm_store_ps(B, b);
  return _mm_set_ps(A[3] - B[3], A[2] - B[2], A[1] - B[1], A[0] - B[0]);
}
inline f32x4 div(const f32x4& a, const f32x4& b) {
  alignas(16) float A[4], B[4];
  _mm_store_ps(A, a);
  _mm_store_ps(B, b);
  return _mm_set_ps(A[3] / B[3], A[2] / B[2], A[1] / B[1], A[0] / B[0]);
}
inline f32x4 mul_add(const f32x4& a, const f32x4& b, const f32x4& c) {
  alignas(16) float A[4], B[4], C[4];
  _mm_store_ps(A, a);
  _mm_store_ps(B, b);
  _mm_store_ps(C, c);
  return _mm_set_ps(A[3] * B[3] + C[3], A[2] * B[2] + C[2], A[1] * B[1] + C[1], A[0] * B[0] + C[0]);
}
}  // namespace impl_scalar
#endif

// -------- SSE4.1 ----------
#if SIMPLE_SIMD_X86 &&                                                                       \
    (defined(__SSE4_1__) || (defined(_MSC_VER) && defined(_M_IX86_FP) && _M_IX86_FP >= 2) || \
     defined(_M_X64))
namespace impl_sse {
inline f32x4 load(const float* p) { return _mm_loadu_ps(p); }
inline void  store(float* dst, const f32x4& v) { _mm_storeu_ps(dst, v); }
inline f32x4 set1(float x) { return _mm_set1_ps(x); }
inline f32x4 mul(const f32x4& a, const f32x4& b) { return _mm_mul_ps(a, b); }
inline f32x4 add(const f32x4& a, const f32x4& b) { return _mm_add_ps(a, b); }
inline f32x4 sub(const f32x4& a, const f32x4& b) { return _mm_sub_ps(a, b); }
inline f32x4 div(const f32x4& a, const f32x4& b) { return _mm_div_ps(a, b); }
inline f32x4 mul_add(const f32x4& a, const f32x4& b, const f32x4& c) {
  return _mm_fmadd_ps(a, b, c);
}
}  // namespace impl_sse
#endif

// -------- AVX2 (still using 128-bit lanes for API uniformity) ----------
#if SIMPLE_SIMD_X86 && defined(__AVX2__)
namespace impl_avx2 {
inline f32x4 load(const float* p) { return _mm_loadu_ps(p); }
inline void  store(float* dst, const f32x4& v) { _mm_storeu_ps(dst, v); }
inline f32x4 set1(float x) { return _mm_set1_ps(x); }
inline f32x4 mul(const f32x4& a, const f32x4& b) { return _mm_mul_ps(a, b); }
inline f32x4 add(const f32x4& a, const f32x4& b) { return _mm_add_ps(a, b); }
inline f32x4 sub(const f32x4& a, const f32x4& b) { return _mm_sub_ps(a, b); }
inline f32x4 div(const f32x4& a, const f32x4& b) { return _mm_div_ps(a, b); }
inline f32x4 mul_add(const f32x4& a, const f32x4& b, const f32x4& c) {
  return _mm_fmadd_ps(a, b, c);
}
}  // namespace impl_avx2
#endif

// -------- NEON (ARM64) ----------
#if SIMPLE_SIMD_ARM
namespace impl_neon {
inline f32x4 load(const float* p) { return vld1q_f32(p); }
inline void  store(float* dst, const f32x4& v) { vst1q_f32(dst, v); }
inline f32x4 set1(float x) { return vdupq_n_f32(x); }
inline f32x4 mul(const f32x4& a, const f32x4& b) { return vmulq_f32(a, b); }
inline f32x4 add(const f32x4& a, const f32x4& b) { return vaddq_f32(a, b); }
inline f32x4 sub(const f32x4& a, const f32x4& b) { return vsubq_f32(a, b); }
inline f32x4 div(const f32x4& a, const f32x4& b) {
  // NEON does not have native division; use reciprocal approximation
  f32x4 reciprocal = vrecpeq_f32(b);
  reciprocal       = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);  // refine
  return vmulq_f32(a, reciprocal);
}
inline f32x4 mul_add(const f32x4& a, const f32x4& b, const f32x4& c) {
  return vmlaq_f32(c, a, b);  // c + a * b
}
}  // namespace impl_neon
#endif

// -------- dispatcher ----------
using load_fn_t    = f32x4 (*)(const float*);
using store_fn_t   = void (*)(float*, const f32x4&);
using set1_fn_t    = f32x4 (*)(float);
using mul_fn_t     = f32x4 (*)(const f32x4&, const f32x4&);
using add_fn_t     = f32x4 (*)(const f32x4&, const f32x4&);
using sub_fn_t     = f32x4 (*)(const f32x4&, const f32x4&);
using div_fn_t     = f32x4 (*)(const f32x4&, const f32x4&);
using mul_add_fn_t = f32x4 (*)(const f32x4&, const f32x4&, const f32x4&);

struct Dispatch {
#if SIMPLE_SIMD_ARM
  load_fn_t    load    = impl_neon::load;
  store_fn_t   store   = impl_neon::store;
  set1_fn_t    set1    = impl_neon::set1;
  mul_fn_t     mul     = impl_neon::mul;
  add_fn_t     add     = impl_neon::add;
  sub_fn_t     sub     = impl_neon::sub;
  div_fn_t     div     = impl_neon::div;
  mul_add_fn_t mul_add = impl_neon::mul_add;
#else
  load_fn_t    load    = impl_scalar::load;
  store_fn_t   store   = impl_scalar::store;
  set1_fn_t    set1    = impl_scalar::set1;
  mul_fn_t     mul     = impl_scalar::mul;
  add_fn_t     add     = impl_scalar::add;
  sub_fn_t     sub     = impl_scalar::sub;
  div_fn_t     div     = impl_scalar::div;
  mul_add_fn_t mul_add = impl_scalar::mul_add;
#endif

  void init() {
#if SIMPLE_SIMD_X86
    CPUFeatures f = detect_cpu();
#if defined(__AVX2__)
    if (f.avx2) {
      load    = impl_avx2::load;
      store   = impl_avx2::store;
      set1    = impl_avx2::set1;
      mul     = impl_avx2::mul;
      add     = impl_avx2::add;
      sub     = impl_avx2::sub;
      div     = impl_avx2::div;
      mul_add = impl_avx2::mul_add;
      return;
    }
#endif
#if defined(__SSE4_1__)
    if (f.sse41) {
      load    = impl_sse::load;
      store   = impl_sse::store;
      set1    = impl_sse::set1;
      mul     = impl_sse::mul;
      add     = impl_sse::add;
      sub     = impl_sse::sub;
      div     = impl_sse::div;
      mul_add = impl_sse::mul_add;
      return;
    }
#endif
    // fallback scalar already set on x86
#elif SIMPLE_SIMD_ARM
    // NEON is baseline on AArch64; defaults already set.
    return;
#endif
  }
};

inline Dispatch& global_dispatch() {
  static Dispatch d;
  static bool     initialized = false;
  if (!initialized) {
    d.init();
    initialized = true;
  }
  return d;
}

// -------- public API ----------
inline f32x4 load_f32x4(const float* p) { return global_dispatch().load(p); }
inline void  store_f32x4(float* dst, const f32x4& v) { return global_dispatch().store(dst, v); }
inline f32x4 set1_f32(float x) { return global_dispatch().set1(x); }
inline f32x4 mul_f32x4(const f32x4& a, const f32x4& b) { return global_dispatch().mul(a, b); }
inline f32x4 add_f32x4(const f32x4& a, const f32x4& b) { return global_dispatch().add(a, b); }
inline f32x4 sub_f32x4(const f32x4& a, const f32x4& b) { return global_dispatch().sub(a, b); }
inline f32x4 div_f32x4(const f32x4& a, const f32x4& b) { return global_dispatch().div(a, b); }
inline f32x4 mul_add_f32x4(const f32x4& a, const f32x4& b, const f32x4& c) {
  return global_dispatch().mul_add(a, b, c);
}

}  // namespace simple_simd