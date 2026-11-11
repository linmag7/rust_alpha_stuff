#include <stdbool.h>
#include <libgccjit.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    gcc_jit_context *c = gcc_jit_context_acquire();
    gcc_jit_type *i = gcc_jit_context_get_type(c, GCC_JIT_TYPE_INT);
    gcc_jit_function *f = gcc_jit_context_new_function(c, 0, GCC_JIT_FUNCTION_EXPORTED, i, "answer", 0, 0, 0);
    gcc_jit_block *b = gcc_jit_function_new_block(f, "entry");
    gcc_jit_block_end_with_return(b, 0, gcc_jit_context_new_rvalue_from_int(c, i, 42));
    gcc_jit_context_compile_to_file(c, GCC_JIT_OUTPUT_KIND_OBJECT_FILE, argv[1]);
    int failed = gcc_jit_context_get_first_error(c) != 0;
    gcc_jit_context_release(c);
    return failed;
}
