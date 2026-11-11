#include "layouts.c"
int main(void) {
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i)
        printf("%u %llu\n", i, (unsigned long long)c_layout_value(i));
    return 0;
}
