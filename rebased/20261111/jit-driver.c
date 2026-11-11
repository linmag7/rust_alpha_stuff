#include <stdio.h>
#include <stdlib.h>
long alpha_global = 97;
long alpha_external(long x) { return x * 3 + 11; }
#define DECLARE(N) extern long alpha_jit_eval##N(long);
DECLARE(0) DECLARE(1) DECLARE(2) DECLARE(3)
DECLARE(4) DECLARE(5) DECLARE(6) DECLARE(7)
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const long iteration = strtol(argv[1], NULL, 10);
    long (*functions[])(long) = {
        alpha_jit_eval0, alpha_jit_eval1, alpha_jit_eval2, alpha_jit_eval3,
        alpha_jit_eval4, alpha_jit_eval5, alpha_jit_eval6, alpha_jit_eval7
    };
    const long inputs[] = {0, 1, -1, 0x7fffffffL, 0x80000000L, -0x80000000L,
                           0x123456789L, -0x123456789L};
    for (int f = 0; f < 8; ++f) {
        for (int i = 0; i < 8; ++i) {
            long expected = inputs[i] * 3 + 108 + iteration + f;
            long actual = functions[f](inputs[i]);
            if (actual != expected) {
                fprintf(stderr, "context %ld function %d input %ld: %ld != %ld\n",
                        iteration, f, inputs[i], actual, expected);
                return 1;
            }
        }
    }
    return 0;
}
