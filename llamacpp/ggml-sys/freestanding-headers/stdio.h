#ifndef FH_STDIO_H
#define FH_STDIO_H
#include <stdarg.h>
#include <stddef.h>
typedef struct { int _n; } FILE;
extern FILE *stdout;
extern FILE *stderr;
int snprintf(char *s, size_t n, const char *fmt, ...);
int vsnprintf(char *s, size_t n, const char *fmt, va_list ap);
#endif
