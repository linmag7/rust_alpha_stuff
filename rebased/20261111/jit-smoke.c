#include <stdbool.h>
#include <libgccjit.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    for (int iteration = 0; iteration < 16; ++iteration) {
        gcc_jit_context *ctx = gcc_jit_context_acquire();
        if (!ctx) return 3;
        gcc_jit_context_set_int_option(ctx, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL,
                                      iteration % 2 ? 3 : 0);
        gcc_jit_context_add_command_line_option(ctx, "-fPIC");
        gcc_jit_context_add_command_line_option(ctx, "-mcpu=ev6");
        gcc_jit_context_add_command_line_option(ctx, "--param=ggc-min-expand=0");
        gcc_jit_context_add_command_line_option(ctx, "--param=ggc-min-heapsize=0");
        gcc_jit_type *ty = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_LONG);
        gcc_jit_param *ext_arg = gcc_jit_context_new_param(ctx, NULL, ty, "x");
        gcc_jit_function *ext = gcc_jit_context_new_function(
            ctx, NULL, GCC_JIT_FUNCTION_IMPORTED, ty, "alpha_external", 1, &ext_arg, 0);
        gcc_jit_lvalue *global = gcc_jit_context_new_global(
            ctx, NULL, GCC_JIT_GLOBAL_IMPORTED, ty, "alpha_global");
        for (int function = 0; function < 8; ++function) {
            char name[64];
            snprintf(name, sizeof(name), "alpha_jit_eval%d", function);
            gcc_jit_param *arg = gcc_jit_context_new_param(ctx, NULL, ty, "x");
            gcc_jit_function *fn = gcc_jit_context_new_function(
                ctx, NULL, GCC_JIT_FUNCTION_EXPORTED, ty, name, 1, &arg, 0);
            gcc_jit_rvalue *x = gcc_jit_param_as_rvalue(arg);
            gcc_jit_rvalue *call = gcc_jit_context_new_call(ctx, NULL, ext, 1, &x);
            gcc_jit_rvalue *sum = gcc_jit_context_new_binary_op(
                ctx, NULL, GCC_JIT_BINARY_OP_PLUS, ty, call, gcc_jit_lvalue_as_rvalue(global));
            gcc_jit_rvalue *result = gcc_jit_context_new_binary_op(
                ctx, NULL, GCC_JIT_BINARY_OP_PLUS, ty, sum,
                gcc_jit_context_new_rvalue_from_long(ctx, ty, iteration + function));
            gcc_jit_block_end_with_return(gcc_jit_function_new_block(fn, "entry"), NULL, result);
        }
        char path[4096];
        snprintf(path, sizeof(path), "%s/jit-%d.o", argv[1], iteration);
        gcc_jit_context_compile_to_file(ctx, GCC_JIT_OUTPUT_KIND_OBJECT_FILE, path);
        const char *error = gcc_jit_context_get_first_error(ctx);
        if (error) {
            fprintf(stderr, "context %d: %s\n", iteration, error);
            gcc_jit_context_release(ctx);
            return 1;
        }
        gcc_jit_context_release(ctx);
    }
    puts("libgccjit: 16 repeated contexts, 128 functions compiled");
    return 0;
}
