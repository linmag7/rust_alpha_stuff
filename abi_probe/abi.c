#include <stdint.h>
#include <stddef.h>

struct pair { uint64_t a, b; };
struct triple { uint64_t a, b, c; };
struct aligned { uint8_t tag; uint64_t value; };

uint64_t c_scalars(uint32_t u, int32_t s, uint8_t b, int16_t h,
                   double f, uint64_t x, uint64_t y, double g) {
    return u == UINT32_C(0xf1234567) && s == -1234567 && b == 241 &&
           h == -1234 && f == 1.25 && x == 17 && y == 29 && g == -2.5;
}
uint64_t c_layout(void) {
    return sizeof(struct aligned) == 16 && offsetof(struct aligned, value) == 8 &&
           _Alignof(uint64_t) == 8;
}
uint64_t c_pair_arg(struct pair p) { return p.a == 17 && p.b == 29; }
uint64_t c_triple_arg(struct triple p) { return p.a == 17 && p.b == 29 && p.c == 43; }
struct pair c_pair_ret(void) { return (struct pair){17, 29}; }

extern uint64_t rust_scalars(uint32_t, int32_t, uint8_t, int16_t, double,
                             uint64_t, uint64_t, double);
extern uint64_t rust_pair_arg(struct pair);
extern uint64_t rust_triple_arg(struct triple);
extern struct pair rust_pair_ret(void);

uint64_t c_call_rust_scalars(void) {
    return rust_scalars(UINT32_C(0xf1234567), -1234567, 241, -1234, 1.25, 17, 29, -2.5);
}
uint64_t c_call_rust_pair_arg(void) { return rust_pair_arg((struct pair){17,29}); }
uint64_t c_call_rust_triple_arg(void) { return rust_triple_arg((struct triple){17,29,43}); }
uint64_t c_call_rust_pair_ret(void) {
    struct pair p = rust_pair_ret();
    return p.a == 17 && p.b == 29;
}

struct tiny { uint32_t value; };
struct odd { uint8_t bytes[3]; };
struct large { uint64_t words[8]; };
struct mixed { double real; uint32_t word; };
struct tiny c_tiny_ret(void) { return (struct tiny){UINT32_C(0xf1234567)}; }
struct odd c_odd_ret(void) { return (struct odd){{17, 29, 43}}; }
struct triple c_triple_ret(void) { return (struct triple){17,29,43}; }
uint64_t c_large_arg(struct large p) {
    for (unsigned i=0; i<8; ++i) if (p.words[i] != 17+i) return 0;
    return 1;
}
uint64_t c_split_arg(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                     uint64_t e, struct triple p, uint64_t tail) {
    return a==1 && b==2 && c==3 && d==4 && e==5 && p.a==17 && p.b==29 && p.c==43 && tail==71;
}
uint64_t c_odd_arg(struct odd p) { return p.bytes[0]==17 && p.bytes[1]==29 && p.bytes[2]==43; }
uint64_t c_mixed_arg(struct mixed p) { return p.real==1.25 && p.word==UINT32_C(0xf1234567); }
extern struct tiny rust_tiny_ret(void);
extern struct odd rust_odd_ret(void);
extern struct triple rust_triple_ret(void);
extern uint64_t rust_large_arg(struct large);
extern uint64_t rust_split_arg(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, struct triple, uint64_t);
extern uint64_t rust_odd_arg(struct odd);
extern uint64_t rust_mixed_arg(struct mixed);
uint64_t c_call_rust_extended(void) {
    struct tiny tiny=rust_tiny_ret();
    if (tiny.value!=UINT32_C(0xf1234567)) return 1;
    struct odd odd=rust_odd_ret();
    if (!c_odd_arg(odd)) return 2;
    struct triple triple=rust_triple_ret();
    if (!c_triple_arg(triple)) return 3;
    struct large large={{17,18,19,20,21,22,23,24}};
    if (!rust_large_arg(large)) return 4;
    if (!rust_split_arg(1,2,3,4,5,triple,71)) return 5;
    if (!rust_odd_arg(odd)) return 6;
    if (!rust_mixed_arg((struct mixed){1.25, UINT32_C(0xf1234567)})) return 7;
    return 0;
}
