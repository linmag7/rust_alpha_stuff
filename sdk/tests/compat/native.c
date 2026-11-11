#include <stdint.h>
#include <string.h>
struct triple { uint64_t a, b, c; };
uint64_t sdk_native_sum(struct triple value, uint32_t mask) {
    return value.a + value.b + value.c + mask;
}
size_t sdk_native_length(const char *text) { return strlen(text); }
