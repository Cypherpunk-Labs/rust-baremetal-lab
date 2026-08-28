#ifndef FH_STDDEF_H
#define FH_STDDEF_H
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
#endif
#define NULL ((void*)0)
#define offsetof(type, member) __builtin_offsetof(type, member)
#endif
