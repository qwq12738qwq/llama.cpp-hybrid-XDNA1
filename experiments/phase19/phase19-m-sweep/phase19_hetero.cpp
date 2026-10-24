// phase19_hetero.cpp - Vulkan(GPU) + XDNA(NPU) N-partition heterogeneous GEMM
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <sys/stat.h>
#include <sys/time.h>
#include "ggml.h"
#include "ggml-backend.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static bool file_exists(const char* p) {
    struct stat st; return stat(p, &st) == 0;
}
struct XdnaKernel {
    std::string xclbin_path, inst_path;
    int M_tile, K_tile, N_tile;
    size_t inst_bytes = 0;
    xrt::device* dev=nullptr; xrt::xclbin* xcl=nullptr;
    xrt::hw_context* hw_ctx=nullptr; xrt::kernel* krnl=nullptr;
    xrt::bo* inst_bo=nullptr; xrt::bo* input_bo=nullptr;
    xrt::bo* weight_bo=nullptr; xrt::bo* output_bo=nullptr;
    xrt::run* pre_run=nullptr;
};
static bool xdna_init(XdnaKernel& k) {
    try {
        k.dev = new xrt::device(0);
        k.xcl = new xrt::xclbin(k.xclbin_path);
        k.dev->register_xclbin(*k.xcl);
        k.hw_ctx = new xrt::hw_context(*k.dev, k.xcl->get_uuid());
        k.krnl = new xrt::kernel(*k.hw_ctx, "MLIR_AIE");
        std::ifstream ifs(k.inst_path, std::ios::binary);
        if (!ifs) return false;
        ifs.seekg(0, std::ios::end); k.inst_bytes = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<uint32_t> insts(k.inst_bytes/4);
        ifs.read(reinterpret_cast<char*>(insts.data()), k.inst_bytes); ifs.close();
        uint32_t gid1 = k.krnl->group_id(1), gid3 = k.krnl->group_id(3);
        k.inst_bo = new xrt::bo(*k.dev, k.inst_bytes, xrt::bo::flags::cacheable, gid1);
        std::memcpy(k.inst_bo->map(), insts.data(), k.inst_bytes);
        k.inst_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k.input_bo  = new xrt::bo(*k.dev, (size_t)k.M_tile*k.K_tile, xrt::bo::flags::host_only, gid3);
        k.weight_bo = new xrt::bo(*k.dev, (size_t)k.K_tile*k.N_tile, xrt::bo::flags::host_only, gid3);
        k.output_bo = new xrt::bo(*k.dev, (size_t)k.M_tile*k.N_tile*4, xrt::bo::flags::host_only, gid3);
        k.pre_run = new xrt::run(*k.krnl);
        k.pre_run->set_arg(0,(uint64_t)3); k.pre_run->set_arg(1,*k.inst_bo);
        k.pre_run->set_arg(2,(uint32_t)k.inst_bytes);
        k.pre_run->set_arg(3,*k.input_bo); k.pre_run->set_arg(4,*k.weight_bo); k.pre_run->set_arg(5,*k.output_bo);
        fprintf(stderr,"[XDNA] Init OK: M=%d K=%d N=%d\n",k.M_tile,k.K_tile,k.N_tile);
        return true;
    } catch(const std::exception& e) { fprintf(stderr,"[XDNA] Init FAILED: %s\n",e.what()); return false; }
}
static void xdna_fini(XdnaKernel& k) {
    delete k.pre_run; delete k.output_bo; delete k.weight_bo;
    delete k.input_bo; delete k.inst_bo; delete k.krnl;
    delete k.hw_ctx; delete k.xcl; delete k.dev;
}
static void xdna_gemm(XdnaKernel& k, const int8_t* A, const int8_t* B,
                       int32_t* C, int M, int K, int N) {
    int Mt=k.M_tile, tiles=(M+Mt-1)/Mt;
    std::vector<int8_t> padded((size_t)Mt*K);
    std::vector<int32_t> kd((size_t)tiles*Mt*N);
    for(int t=0;t<tiles;t++){
        int m0=t*Mt, mr=std::min(Mt,M-m0);
        std::memset(padded.data(),0,padded.size());
        std::memcpy(padded.data(),A+(size_t)m0*K,(size_t)mr*K);
        size_t ib=(size_t)Mt*K,wb=(size_t)K*N,ob=(size_t)Mt*N*4;
        std::memcpy(k.input_bo->map(),padded.data(),ib); k.input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memcpy(k.weight_bo->map(),B,wb); k.weight_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memset(k.output_bo->map(),0,ob); k.output_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k.pre_run->start(); k.pre_run->wait();
        k.output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(kd.data()+(size_t)t*Mt*N, k.output_bo->map(), ob);
    }
    for(int m=0;m<M;m++){ int t=m/Mt,r=m%Mt;
        std::memcpy(C+(size_t)m*N, kd.data()+(size_t)(t*Mt+r)*N, N*sizeof(int32_t));
    }
}
// Vulkan GEMM via ggml: F32 A[M,K] * Q8_0 B[K,N] -> F32 C[M,N]
static double vulkan_gemm(ggml_backend_t backend, const float* A_f32,
                           const uint8_t* B_q8, float* C_f32, int M, int K, int N) {
    struct ggml_init_params params = {};
    params.mem_size = (size_t)K*N*34/32 + (size_t)K*M*4 + 64*1024*1024;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return -1;
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, w, b);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return -1; }
    ggml_backend_tensor_set(w, B_q8, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(b, A_f32, 0, (size_t)M*K*sizeof(float));
    double t0 = now_ms();
    ggml_backend_graph_compute(backend, gf);
    double ms = now_ms() - t0;
    std::vector<float> vk_out((size_t)N*M);
    ggml_backend_tensor_get(c, vk_out.data(), 0, vk_out.size()*sizeof(float));
    for(int m=0;m<M;m++) for(int n=0;n<N;n++)
        C_f32[(size_t)m*N+n] = vk_out[(size_t)n*M+m];
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return ms;
}
static std::string find_kernel(int Mt, int K, int N) {
    const char* fmts[]={"M%d_K%d_N%d_C1_Tm64_Tk64_Tn64","M%d_K%d_N%d_C1_Tm64_Tk32_Tn64","M%d_K%d_N%d_C1","M%d_K%d_N%d"};
    for(auto fmt:fmts){ char dir[512]; snprintf(dir,sizeof(dir),"/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/%s",fmt);
        char full[512]; snprintf(full,sizeof(full),dir,Mt,K,N);
        if(file_exists((std::string(full)+"/design.xclbin").c_str())&&file_exists((std::string(full)+"/design.insts.bin").c_str())) return std::string(full);
    } return "";
}
int main() {
    setvbuf(stdout,NULL,_IONBF,0); setvbuf(stderr,NULL,_IONBF,0);
    printf("=== Phase 19: Vulkan+XDNA Heterogeneous N-Partition ===\n\n");
    ggml_backend_load_all();
    ggml_backend_t backend_vk = ggml_backend_init_by_name("Vulkan0", "");
    if (!backend_vk) backend_vk = ggml_backend_init_by_name("Vulkan", "");
    if (!backend_vk) { fprintf(stderr,"[FATAL] No Vulkan backend\n"); return 1; }
    fprintf(stderr,"[VK] Vulkan backend ready\n");
    std::ofstream csv("/workspace/llama.cpp/phase19_hetero.csv");
    csv<<"M,K,N,Nv,VK_ms,XDNA_ms,HETERO_ms,Speedup\n";
    struct TC{int M,K,N;};
    TC cases[]={{512,1024,3072},{1024,1024,3072},{2048,1024,3072},{4096,1024,3072},{8192,1024,3072},{16384,1024,3072},{4096,2048,3072},{4096,4096,3072}};
    printf("%-8s %-6s %-6s %-6s %-12s %-12s %-12s %-8s\n","M","K","N","Nv","VK_ms","XDNA_ms","HETERO_ms","Speedup");
    printf("-------- ------ ------ ------ ------------ ------------ ------------ --------\n");
    for(auto& tc:cases){ int M=tc.M,K=tc.K,N=tc.N;
        int best_mt=-1; for(int mt:{512,256,128,64}) if(!find_kernel(mt,K,N).empty()){best_mt=mt;break;}
        if(best_mt<0)continue;
        size_t mem=(size_t)M*K+(size_t)K*N+(size_t)M*N*4;
        if(mem>1500ULL*1024*1024){printf("M=%d K=%d N=%d SKIPPED\n",M,K,N);continue;}
        std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f,1.0f);
        std::vector<float> A_f32((size_t)M*K), B_f32((size_t)N*K);
        for(auto&v:A_f32)v=dist(rng); for(auto&v:B_f32)v=dist(rng);
        size_t q8_sz=(size_t)(K*N/32)*34;
        std::vector<uint8_t> B_q8(q8_sz);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, B_f32.data(), B_q8.data(), 0, N, K, nullptr);
        // XDNA weight: [K, N] row-major int8 (qs values from Q8_0)
        std::vector<int8_t> B_i8((size_t)K*N);
        { int nb=K/32;
          for(int n=0;n<N;n++) for(int b=0;b<nb;b++){
              size_t src=(size_t)n*nb*34+b*34+2;
              for(int j=0;j<32;j++) B_i8[(size_t)(b*32+j)*N+n]=(int8_t)B_q8[src+j]; }
        }
        std::vector<int8_t> A_i8((size_t)M*K);
        for(size_t i=0;i<A_i8.size();i++) A_i8[i]=(int8_t)std::clamp(A_f32[i]*127.0f,-127.0f,127.0f);
        // Vulkan-only
        std::vector<float> C_vk((size_t)M*N);
        double t_vk=vulkan_gemm(backend_vk,A_f32.data(),B_q8.data(),C_vk.data(),M,K,N);
        // XDNA-only
        XdnaKernel xd; std::string kdr=find_kernel(best_mt,K,N);
        xd.xclbin_path=kdr+"/design.xclbin"; xd.inst_path=kdr+"/design.insts.bin";
        xd.M_tile=best_mt; xd.K_tile=K; xd.N_tile=N;
        if(!xdna_init(xd))continue;
        std::vector<int32_t> C_xd((size_t)M*N);
        double t0=now_ms(); xdna_gemm(xd,A_i8.data(),B_i8.data(),C_xd.data(),M,K,N); double t_xdna=now_ms()-t0;
        printf("\nM=%d K=%d N=%d  VK=%.1fms XDNA=%.1fms\n",M,K,N,t_vk,t_xdna);
        printf("  Nv   Nx   HETERO_ms   Speedup\n");
        double best_het=1e18; int best_nv=0;
        for(int pct=0;pct<=100;pct+=10){ int Nv=(int)((double)N*pct/100); Nv=(Nv/32)*32;
            if(Nv<0)Nv=0; if(Nv>N)Nv=N; int Nx=N-Nv; if(Nv==0&&Nx==0)continue;
            std::vector<uint8_t> Bq_vk;
            if(Nv>0){ int nb=K/32; Bq_vk.resize((size_t)nb*Nv*34);
                for(int n=0;n<Nv;n++) std::memcpy(Bq_vk.data()+(size_t)n*nb*34,B_q8.data()+(size_t)n*nb*34,nb*34); }
            std::vector<int8_t> Bi_xd;
            if(Nx>0){ Bi_xd.resize((size_t)K*Nx);
                for(int k=0;k<K;k++) std::memcpy(Bi_xd.data()+(size_t)k*Nx,B_i8.data()+(size_t)k*N+Nv,Nx); }
            double ts=now_ms();
            auto vf=std::async(std::launch::async,[&](){ if(Nv>0){ std::vector<float> Cp((size_t)M*Nv); vulkan_gemm(backend_vk,A_f32.data(),Bq_vk.data(),Cp.data(),M,K,Nv);} });
            auto xf=std::async(std::launch::async,[&](){ if(Nx>0){ std::vector<int32_t> Cp((size_t)M*Nx); xdna_gemm(xd,A_i8.data(),Bi_xd.data(),Cp.data(),M,K,Nx);} });
            vf.wait(); xf.wait(); double t_het=now_ms()-ts;
            if(t_het<best_het){best_het=t_het;best_nv=Nv;}
            printf("  %-4d %-4d %-11.2f %.2f\n",Nv,Nx,t_het,t_vk/t_het);
            csv<<M<<","<<K<<","<<N<<","<<Nv<<","<<t_vk<<","<<t_xdna<<","<<t_het<<","<<(t_vk/t_het)<<"\n";
            csv.flush();
        }
        printf("  => Best: Nv=%d speedup=%.2f\n",best_nv,t_vk/best_het);
        xdna_fini(xd);
    }
    csv.close(); printf("\n=== Phase 19 complete. CSV: phase19_hetero.csv ===\n");
    ggml_backend_free(backend_vk); return 0;
}
