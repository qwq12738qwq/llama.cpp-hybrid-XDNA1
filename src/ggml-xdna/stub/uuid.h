#ifndef UUID_UUID_H
#define UUID_UUID_H
#include <stdint.h>
typedef struct { uint8_t b[16]; } uuid_t;
typedef struct { uint8_t b[16]; } uuid_le;
static inline void uuid_generate(uuid_t u) { for(int i=0;i<16;i++) u.b[i]=0; }
static inline int uuid_parse(const char *s, uuid_t *u) { (void)s; (void)u; return -1; }
static inline void uuid_unparse(const uuid_t *u, char *s) { (void)u; s[0]=0; }
#endif
