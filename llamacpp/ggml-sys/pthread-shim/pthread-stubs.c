/*
 * Single-threaded pthread shim for the freestanding ggml CPU backend.
 *
 * ggml's CPU backend is written against POSIX threads. On the bare-metal
 * kernel we run with n_threads == 1 (one physical core, no OS scheduler), so
 * the thread pool never actually spawns worker threads. These stubs implement
 * the symbols ggml references as harmless no-ops:
 *
 *   - mutex/cond operations are no-ops (there is no contention),
 *   - pthread_create is never called for n_threads == 1 (it returns failure
 *     just in case),
 *   - affinity/sched calls are ignored.
 *
 * This file is compiled ONLY into the freestanding libggmlcpu.a; the host
 * build uses the real system pthread.
 */

#include <pthread.h>
#include <stddef.h>

int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
    (void)m; (void)a;
    return 0;
}
int pthread_mutex_destroy(pthread_mutex_t *m) {
    (void)m;
    return 0;
}
int pthread_mutex_lock(pthread_mutex_t *m) {
    (void)m;
    return 0;
}
int pthread_mutex_unlock(pthread_mutex_t *m) {
    (void)m;
    return 0;
}
int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) {
    (void)c; (void)a;
    return 0;
}
int pthread_cond_destroy(pthread_cond_t *c) {
    (void)c;
    return 0;
}
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    (void)c; (void)m;
    return 0;
}
int pthread_cond_signal(pthread_cond_t *c) {
    (void)c;
    return 0;
}
int pthread_cond_broadcast(pthread_cond_t *c) {
    (void)c;
    return 0;
}
int pthread_create(pthread_t *t, const pthread_attr_t *a,
                   void *(*fn)(void *), void *arg) {
    (void)t; (void)a; (void)fn; (void)arg;
    return -1;
}
int pthread_join(pthread_t t, void **r) {
    (void)t; (void)r;
    return 0;
}
pthread_t pthread_self(void) {
    return 0;
}
int pthread_setaffinity_np(pthread_t t, size_t n, const void *s) {
    (void)t; (void)n; (void)s;
    return 0;
}
int pthread_getaffinity_np(pthread_t t, size_t n, void *s) {
    (void)t; (void)n; (void)s;
    return 0;
}
int pthread_setschedparam(pthread_t t, int p, const struct sched_param *sp) {
    (void)t; (void)p; (void)sp;
    return 0;
}
