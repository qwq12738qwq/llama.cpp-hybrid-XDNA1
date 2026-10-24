#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-impl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>

static const char* op_name(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE: return "NONE";
        case GGML_OP_DUP: return "DUP";
        case GGML_OP_ADD: return "ADD";
        case GGML_OP_ADD_ID: return "ADD_ID";
        case GGML_OP_ADD1: return "ADD1";
        case GGML_OP_ACC: return "ACC";
        case GGML_OP_SUB: return "SUB";
        case GGML_OP_MUL: return "MUL";
        case GGML_OP_DIV: return "DIV";
        case GGML_OP_SQR: return "SQR";
        case GGML_OP_SQRT: return "SQRT";
        case GGML_OP_LOG: return "LOG";
        case GGML_OP_SIN: return "SIN";
        case GGML_OP_COS: return "COS";
        case GGML_OP_SUM: return "SUM";
        case GGML_OP_SUM_ROWS: return "SUM_ROWS";
        case GGML_OP_CUMSUM: return "CUMSUM";
        case GGML_OP_MEAN: return "MEAN";
        case GGML_OP_ARGMAX: return "ARGMAX";
        case GGML_OP_COUNT_EQUAL: return "COUNT_EQUAL";
        case GGML_OP_REPEAT: return "REPEAT";
        case GGML_OP_REPEAT_BACK: return "REPEAT_BACK";
        case GGML_OP_CONCAT: return "CONCAT";
        case GGML_OP_SILU_BACK: return "SILU_BACK";
        case GGML_OP_NORM: return "NORM";
        case GGML_OP_RMS_NORM: return "RMS_NORM";
        case GGML_OP_RMS_NORM_BACK: return "RMS_NORM_BACK";
        case GGML_OP_GROUP_NORM: return "GROUP_NORM";
        case GGML_OP_L2_NORM: return "L2_NORM";
        case GGML_OP_MUL_MAT: return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        case GGML_OP_OUT_PROD: return "OUT_PROD";
        case GGML_OP_SCALE: return "SCALE";
        case GGML_OP_SET: return "SET";
        case GGML_OP_CPY: return "CPY";
        case GGML_OP_CONT: return "CONT";
        case GGML_OP_RESHAPE: return "RESHAPE";
        case GGML_OP_VIEW: return "VIEW";
        case GGML_OP_PERMUTE: return "PERMUTE";
        case GGML_OP_TRANSPOSE: return "TRANSPOSE";
        case GGML_OP_GET_ROWS: return "GET_ROWS";
        case GGML_OP_GET_ROWS_BACK: return "GET_ROWS_BACK";
        case GGML_OP_SET_ROWS: return "SET_ROWS";
        case GGML_OP_DIAG: return "DIAG";
        case GGML_OP_DIAG_MASK_INF: return "DIAG_MASK_INF";
        case GGML_OP_DIAG_MASK_ZERO: return "DIAG_MASK_ZERO";
        case GGML_OP_SOFT_MAX: return "SOFT_MAX";
        case GGML_OP_SOFT_MAX_BACK: return "SOFT_MAX_BACK";
        case GGML_OP_ROPE: return "ROPE";
        case GGML_OP_ROPE_BACK: return "ROPE_BACK";
        case GGML_OP_CLAMP: return "CLAMP";
        case GGML_OP_CONV_TRANSPOSE_1D: return "CONV_TRANSPOSE_1D";
        case GGML_OP_IM2COL: return "IM2COL";
        case GGML_OP_IM2COL_BACK: return "IM2COL_BACK";
        case GGML_OP_IM2COL_3D: return "IM2COL_3D";
        case GGML_OP_COL2IM_1D: return "COL2IM_1D";
        case GGML_OP_CONV_2D: return "CONV_2D";
        case GGML_OP_CONV_3D: return "CONV_3D";
        case GGML_OP_CONV_2D_DW: return "CONV_2D_DW";
        case GGML_OP_CONV_TRANSPOSE_2D: return "CONV_TRANSPOSE_2D";
        case GGML_OP_POOL_1D: return "POOL_1D";
        case GGML_OP_POOL_2D: return "POOL_2D";
        case GGML_OP_POOL_2D_BACK: return "POOL_2D_BACK";
        case GGML_OP_UPSCALE: return "UPSCALE";
        case GGML_OP_PAD: return "PAD";
        case GGML_OP_PAD_REFLECT_1D: return "PAD_REFLECT_1D";
        case GGML_OP_ROLL: return "ROLL";
        case GGML_OP_ARANGE: return "ARANGE";
        case GGML_OP_TIMESTEP_EMBEDDING: return "TIMESTEP_EMBEDDING";
        case GGML_OP_ARGSORT: return "ARGSORT";
        case GGML_OP_TOP_K: return "TOP_K";
        case GGML_OP_LEAKY_RELU: return "LEAKY_RELU";
        case GGML_OP_TRI: return "TRI";
        case GGML_OP_FILL: return "FILL";
        case GGML_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        case GGML_OP_FLASH_ATTN_BACK: return "FLASH_ATTN_BACK";
        case GGML_OP_SSM_CONV: return "SSM_CONV";
        case GGML_OP_SSM_SCAN: return "SSM_SCAN";
        case GGML_OP_WIN_PART: return "WIN_PART";
        case GGML_OP_WIN_UNPART: return "WIN_UNPART";
        default: return "UNKNOWN";
    }
}

static const char* type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS: return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ1_S: return "IQ1_S";
        case GGML_TYPE_IQ4_NL: return "IQ4_NL";
        case GGML_TYPE_IQ3_S: return "IQ3_S";
        case GGML_TYPE_IQ2_S: return "IQ2_S";
        case GGML_TYPE_IQ4_XS: return "IQ4_XS";
        case GGML_TYPE_I8: return "I8";
        case GGML_TYPE_I16: return "I16";
        case GGML_TYPE_I32: return "I32";
        case GGML_TYPE_I64: return "I64";
        case GGML_TYPE_F64: return "F64";
        case GGML_TYPE_BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

void print_tensor_info(const struct ggml_tensor* t, const char* prefix) {
    if (!t) return;
    printf("%s  name=%s op=%s(%d) type=%s ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu]\n",
           prefix,
           t->name ? t->name : "unnamed",
           op_name(t->op), t->op,
           type_name(t->type),
           t->ne[0], t->ne[1], t->ne[2], t->ne[3],
           t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    if (t->src[0]) printf("%s    src0=%s\n", prefix, t->src[0]->name ? t->src[0]->name : "unnamed");
    if (t->src[1]) printf("%s    src1=%s\n", prefix, t->src[1]->name ? t->src[1]->name : "unnamed");
}

struct NodeInfo {
    int idx;
    std::string name;
    enum ggml_op op;
    enum ggml_type type;
    int64_t ne[4];
    size_t nb[4];
    std::string src0_name;
    std::string src1_name;
    bool is_view_op;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt_size>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int prompt_size = atoi(argv[2]);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.use_extra_bufts = true;
    mparams.flash_attn = true;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = prompt_size + 128;
    cparams.flash_attn = true;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    // Build the graph for prompt processing using llama internal API
    llama_batch batch = llama_batch_init(prompt_size, 0, 1);
    for (int i = 0; i < prompt_size; i++) {
        batch.token[i] = i % 32000;
    }
    batch.n_tokens = prompt_size;

    // Decode to trigger graph build
    int res = llama_decode(ctx, batch);
    if (res != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", res);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}