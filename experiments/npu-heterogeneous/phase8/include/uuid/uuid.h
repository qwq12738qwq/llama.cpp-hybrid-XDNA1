#ifndef UUID_UUID_H
#define UUID_UUID_H
#include <stdint.h>
#include <cstring>
typedef unsigned char uuid_t[16];
typedef unsigned char uuid_le[16];
static inline void uuid_clear(uuid_t u) { memset(u, 0, 16); }
static inline void uuid_copy(uuid_t dst, const uuid_t src) { memcpy(dst, src, 16); }
static inline int uuid_compare(const uuid_t a, const uuid_t b) { return memcmp(a, b, 16); }
static inline void uuid_generate(uuid_t u) { for(int i=0;i<16;i++) u[i]=0; }
static inline int uuid_parse(const char *s, uuid_t u) { (void)s; memset(u, 0, 16); return -1; }
static inline void uuid_unparse(const uuid_t u, char *s) { (void)u; s[0]=0; }
static inline void uuid_unparse_lower(const uuid_t u, char *s) { uuid_unparse(u, s); }
static inline int uuid_is_null(const uuid_t u) { for(int i=0;i<16;i++) if(u[i]) return 0; return 1; }
static inline int uuid_type(const uuid_t u) { (void)u; return 0; }
static inline void uuid_generate_random(uuid_t u) { uuid_generate(u); }
static inline void uuid_generate_time(uuid_t u) { uuid_generate(u); }
#endif
