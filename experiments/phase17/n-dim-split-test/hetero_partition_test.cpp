/**
 * hetero_partition_test.cpp - Test scheduler N-dimension partition transformation
 */

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

int main() {
    printf("[TEST] Heterogeneous partition scheduler test\n");
    fflush(stdout);

    ggml_backend_load_all();

    ggml_backend_t backend_cpu = ggml_backend_init_by_name("CPU", "");
    if (!backend_cpu) {
        printf("ERROR: CPU backend not available\n");
        return 1;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_cpu);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend_cpu, &buft, 1, 1024, false, false);

    if (!sched) {
        printf("ERROR: Failed to create scheduler\n");
        return 1;
    }

    // Create a simple graph with MUL_MAT N=4096
    struct ggml_init_params params = { 0 };
    params.mem_size = 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    int M = 512, K = 1024, N = 4096;
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);

    ggml_build_forward_expand(gf, c);

    printf("Original graph: %d nodes\n", ggml_graph_n_nodes(gf));
    fflush(stdout);

    // Fill with test data
    std::vector<float> weight_data(K * N);
    std::vector<float> input_data(K * M);
    for (size_t i = 0; i < weight_data.size(); i++) weight_data[i] = (float)(i % 256) / 256.0f;
    for (size_t i = 0; i < input_data.size(); i++) input_data[i] = (float)((i * 7) % 256) / 256.0f;

    // Allocate buffers
    ggml_backend_buffer_t buf_a = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(a));
    ggml_backend_buffer_t buf_b = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(b));
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(c));

    a->buffer = buf_a;
    a->data = ggml_backend_buffer_get_base(buf_a);
    b->buffer = buf_b;
    b->data = ggml_backend_buffer_get_base(buf_b);
    c->buffer = buf_c;
    c->data = ggml_backend_buffer_get_base(buf_c);

    ggml_backend_tensor_set(a, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, input_data.data(), 0, input_data.size() * sizeof(float));

    // Reserve scheduler (this triggers the transformation)
    bool reserved = ggml_backend_sched_reserve(sched, gf);

    // Count MUL_MAT nodes
    int mul_mat_count = 0;
    int concat_count = 0;
    int graph_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < graph_nodes; i++) {
        ggml_tensor* n = ggml_graph_node(gf, i);
        if (n->op == GGML_OP_MUL_MAT) mul_mat_count++;
        if (n->op == GGML_OP_CONCAT) concat_count++;
    }
    printf("MUL_MAT nodes: %d, CONCAT nodes: %d\n", mul_mat_count, concat_count);

if (mul_mat_count >= 2 && concat_count >= 1) {
        printf("PASS: Graph was partitioned into %d MUL_MAT + %d CONCAT\n", mul_mat_count, concat_count);
    } else {
        printf("FAIL: Expected partitioned graph, got %d MUL_MAT + %d CONCAT\n", mul_mat_count, concat_count);
    }
    
    // Compute with scheduler and verify correctness
    enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    fflush(stdout);
    
    // Check all nodes
    for (int i = 0; i < graph_nodes; i++) {
        ggml_tensor* n = ggml_graph_node(gf, i);
        if (n->op == GGML_OP_MUL_MAT) mul_mat_count++;
        if (n->op == GGML_OP_CONCAT) concat_count++;
    }
    ggml_tensor* output = ggml_graph_node(gf, graph_nodes - 1);
    if (output->op != GGML_OP_CONCAT) {
        printf("ERROR: Expected CONCAT as final node, got %s\n", ggml_op_name(output->op));
    } else {
        std::vector<float> scheduler_out(ggml_nbytes(output) / sizeof(float));
        ggml_backend_tensor_get(output, scheduler_out.data(), 0, scheduler_out.size() * sizeof(float));
        
        // Compute reference directly on CPU
        struct ggml_init_params ref_params = { 0 };
        ref_params.mem_size = 256 * 1024 * 1024;
        ref_params.mem_buffer = nullptr;
        ref_params.no_alloc = false;
        struct ggml_context* ref_ctx = ggml_init(ref_params);
        struct ggml_cgraph* ref_gf = ggml_new_graph(ref_ctx);
        
        struct ggml_tensor* ref_a = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor* ref_b = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor* ref_c = ggml_mul_mat(ref_ctx, ref_a, ref_b);
        ggml_build_forward_expand(ref_gf, ref_c);
        
        ggml_backend_buffer_t ref_buf_a = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(ref_a));
        ggml_backend_buffer_t ref_buf_b = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(ref_b));
        ggml_backend_buffer_t ref_buf_c = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(ref_c));
        
        ref_a->buffer = ref_buf_a;
        ref_a->data = ggml_backend_buffer_get_base(ref_buf_a);
        ref_b->buffer = ref_buf_b;
        ref_b->data = ggml_backend_buffer_get_base(ref_buf_b);
        ref_c->buffer = ref_buf_c;
        ref_c->data = ggml_backend_buffer_get_base(ref_buf_c);
        
        ggml_backend_tensor_set(ref_a, weight_data.data(), 0, weight_data.size() * sizeof(float));
        ggml_backend_tensor_set(ref_b, input_data.data(), 0, input_data.size() * sizeof(float));
        
        ggml_backend_graph_compute(backend_cpu, ref_gf);
        
        std::vector<float> ref_out(ggml_nbytes(ref_c) / sizeof(float));
        ggml_backend_tensor_get(ref_c, ref_out.data(), 0, ref_out.size() * sizeof(float));
        
        float max_diff = 0;
        int max_diff_idx = -1;
        for (size_t i = 0; i < ref_out.size(); i++) {
            float diff = fabs(ref_out[i] - scheduler_out[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = i;
            }
        }
        printf("Max diff (scheduler vs CPU reference): %.6f at index %d\n", max_diff, max_diff_idx);
        printf("ref_out[max_diff_idx] = %.6f, scheduler_out[max_diff_idx] = %.6f\n", ref_out[max_diff_idx], scheduler_out[max_diff_idx]);
        printf("First 10 ref_out: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", ref_out[i]);
        printf("\nFirst 10 scheduler_out: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", scheduler_out[i]);
        printf("\n");
        if (max_diff < 1e-5) {
            printf("PASS: Numerical correctness verified\n");
        } else {
            printf("FAIL: Numerical mismatch detected\n");
        }
        
        ggml_free(ref_ctx);
    }

    // Print node details
    printf("\nGraph nodes after transformation:\n");
    for (int i = 0; i < graph_nodes; i++) {
        ggml_tensor* n = ggml_graph_node(gf, i);
        printf("  Node %d: op=%s ne=[%ld,%ld,%ld,%ld]\n", i, ggml_op_name(n->op),
               n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend_cpu);

    return 0;
}
