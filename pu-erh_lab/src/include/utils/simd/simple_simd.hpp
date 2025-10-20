// simple_simd.h -- minimal cross-arch SIMD abstraction
#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h> // harmless on MSVC/non-x86
#if defined(_MSC_VER)
  #include <intrin.h>
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
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
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
        f.avx2 = (ebx & (1 << 5)) != 0;
    }
#elif defined(__GNUC__) || defined(__clang__)
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
using f32x4 = __m128; // consistent 128-bit API

// -------- scalar fallback ----------
namespace impl_scalar {
    inline f32x4 load(const float* p) {
        return _mm_set_ps(p[3], p[2], p[1], p[0]);
    }
    inline void store(float* dst, f32x4 v) {
        _mm_storeu_ps(dst, v);
    }
    inline f32x4 set1(float x) { return _mm_set1_ps(x); }
    inline f32x4 mul(f32x4 a, f32x4 b) {
        alignas(16) float A[4], B[4];
        _mm_store_ps(A, a);
        _mm_store_ps(B, b);
        return _mm_set_ps(A[3]*B[3], A[2]*B[2], A[1]*B[1], A[0]*B[0]);
    }
    inline f32x4 add(f32x4 a, f32x4 b) {
        alignas(16) float A[4], B[4];
        _mm_store_ps(A, a);
        _mm_store_ps(B, b);
        return _mm_set_ps(A[3]+B[3], A[2]+B[2], A[1]+B[1], A[0]+B[0]);
    }
}

// -------- SSE4.1 ----------
#if (defined(__SSE4_1__) || (defined(_MSC_VER) && defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_X64))
namespace impl_sse {
    inline f32x4 load(const float* p) { return _mm_loadu_ps(p); }
    inline void store(float* dst, f32x4 v) { _mm_storeu_ps(dst, v); }
    inline f32x4 set1(float x) { return _mm_set1_ps(x); }
    inline f32x4 mul(f32x4 a, f32x4 b) { return _mm_mul_ps(a, b); }
    inline f32x4 add(f32x4 a, f32x4 b) { return _mm_add_ps(a, b); }
}
#endif

// -------- AVX2 (still using 128-bit lanes for API uniformity) ----------
#if defined(__AVX2__)
namespace impl_avx2 {
    inline f32x4 load(const float* p) { return _mm_loadu_ps(p); }
    inline void store(float* dst, f32x4 v) { _mm_storeu_ps(dst, v); }
    inline f32x4 set1(float x) { return _mm_set1_ps(x); }
    inline f32x4 mul(f32x4 a, f32x4 b) { return _mm_mul_ps(a, b); }
    inline f32x4 add(f32x4 a, f32x4 b) { return _mm_add_ps(a, b); }
}
#endif

// -------- dispatcher ----------
using load_fn_t = f32x4(*)(const float*);
using store_fn_t = void(*)(float*, f32x4);
using set1_fn_t = f32x4(*)(float);
using mul_fn_t = f32x4(*)(f32x4, f32x4);
using add_fn_t = f32x4(*)(f32x4, f32x4);

struct Dispatch {
    load_fn_t load  = impl_scalar::load;
    store_fn_t store = impl_scalar::store;
    set1_fn_t set1  = impl_scalar::set1;
    mul_fn_t mul    = impl_scalar::mul;
    add_fn_t add    = impl_scalar::add;

    void init() {
        CPUFeatures f = detect_cpu();
#if defined(__AVX2__)
        if (f.avx2) {
            load = impl_avx2::load;
            store = impl_avx2::store;
            set1 = impl_avx2::set1;
            mul = impl_avx2::mul;
            add = impl_avx2::add;
            return;
        }
#endif
#if defined(__SSE4_1__)
        if (f.sse41) {
            load = impl_sse::load;
            store = impl_sse::store;
            set1 = impl_sse::set1;
            mul = impl_sse::mul;
            add = impl_sse::add;
            return;
        }
#endif
        // fallback scalar already set
    }
};

inline Dispatch& global_dispatch() {
    static Dispatch d;
    static bool initialized = false;
    if (!initialized) {
        d.init();
        initialized = true;
    }
    return d;
}

// -------- public API ----------
inline f32x4 load_f32x4(const float* p) { return global_dispatch().load(p); }
inline void store_f32x4(float* dst, f32x4 v) { return global_dispatch().store(dst, v); }
inline f32x4 set1_f32(float x) { return global_dispatch().set1(x); }
inline f32x4 mul_f32x4(f32x4 a, f32x4 b) { return global_dispatch().mul(a, b); }
inline f32x4 add_f32x4(f32x4 a, f32x4 b) { return global_dispatch().add(a, b); }

} // namespace simple_simd