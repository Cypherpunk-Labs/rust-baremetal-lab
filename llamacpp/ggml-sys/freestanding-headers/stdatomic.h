#ifndef FH_STDATOMIC_H
#define FH_STDATOMIC_H
#include <stddef.h>
#include <stdint.h>
typedef _Atomic(_Bool) atomic_bool;
typedef _Atomic(int_least8_t) atomic_int_least8_t;
typedef _Atomic(uint_least8_t) atomic_uint_least8_t;
typedef _Atomic(int_least16_t) atomic_int_least16_t;
typedef _Atomic(uint_least16_t) atomic_uint_least16_t;
typedef _Atomic(int_least32_t) atomic_int_least32_t;
typedef _Atomic(uint_least32_t) atomic_uint_least32_t;
typedef _Atomic(int_least64_t) atomic_int_least64_t;
typedef _Atomic(uint_least64_t) atomic_uint_least64_t;
typedef _Atomic(intptr_t) atomic_intptr_t;
typedef _Atomic(uintptr_t) atomic_uintptr_t;
#define atomic_store_explicit(p, v, o) __atomic_store_n(p, v, o)
#define atomic_load_explicit(p, o) __atomic_load_n(p, o)
#define atomic_fetch_add_explicit(p, v, o) __atomic_fetch_add(p, v, o)
#define atomic_fetch_sub_explicit(p, v, o) __atomic_fetch_sub(p, v, o)
#define atomic_exchange_explicit(p, v, o) __atomic_exchange_n(p, v, o)
#define atomic_compare_exchange_strong_explicit(p, e, v, so, fo) __atomic_compare_exchange_n(p, e, v, 0, so, fo)
#define atomic_compare_exchange_weak_explicit(p, e, v, so, fo) __atomic_compare_exchange_n(p, e, v, 1, so, fo)
#define memory_order_relaxed __ATOMIC_RELAXED
#define memory_order_consume __ATOMIC_CONSUME
#define memory_order_acquire __ATOMIC_ACQUIRE
#define memory_order_release __ATOMIC_RELEASE
#define memory_order_acq_rel __ATOMIC_ACQ_REL
#define memory_order_seq_cst __ATOMIC_SEQ_CST
#define atomic_store(p, v) atomic_store_explicit(p, v, memory_order_seq_cst)
#define atomic_load(p) atomic_load_explicit(p, memory_order_seq_cst)
#define atomic_fetch_add(p, v) atomic_fetch_add_explicit(p, v, memory_order_seq_cst)
#define atomic_fetch_sub(p, v) atomic_fetch_sub_explicit(p, v, memory_order_seq_cst)
#define atomic_thread_fence(o) __atomic_thread_fence(o)
#define ATOMIC_VAR_INIT(v) (v)
#endif
