// Phase 48: heterogeneous PP scaling benchmark (Vulkan + XDNA).
//
// One process per (config, PP) point. Loads the model once, creates a context
// with the requested n_batch/n_ubatch, then runs the prompt phase warmup+measure
// times and prints machine-readable CSV lines.
//
// Modes:
//   bench    : warmup + measured PP runs, reports pp t/s and total ms
//   verify   : greedy decode of N tokens, reports the token ids and their md5
//   ubatch   : sweep n_ubatch for a fixed n_batch, to expose ubatch
//              fragmentation of the NPU GEMMs
//
// Usage:
//   bench_pp_scaling --model M.gguf --dev Vulkan0,XDNA --pp 1024 \
//       --batch 1024 --ubatch 1024 --warmup 2 --measure 5 --mode bench
//
// Every field of the phase 48 CSV that needs ggml-level instrumentation is
// emitted by the backends themselves under GGML_SCHED_TRACE_STATS; this tool
// only prints what it can measure locally.

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

long peak_rss_kb() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return -1;
    }
    return ru.ru_maxrss;
}

std::string md5_of(const std::vector<llama_token> & tokens) {
    // compact, reproducible digest of the greedy token sequence
    std::string s;
    for (llama_token t : tokens) {
        s += std::to_string(t);
        s.push_back(',');
    }
    uint64_t h = 1469598103934665603ull; // FNV-1a
    for (char c : s) {
        h ^= (unsigned char) c;
        h *= 1099511628211ull;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

struct args {
    std::string model;
    std::vector<std::string> dev_list;
    std::string mode = "bench";
    int pp          = 512;
    int batch       = 0;   // defaults to pp
    int ubatch      = 0;   // defaults to pp
    int warmup      = 2;
    int measure     = 5;
    int n_predict   = 32;
    int n_threads   = 0;   // 0 -> leave the default
    int n_gpu_layers = -1; // -1 all, 0 keeps weights on the CPU buffers
    std::string ubatch_sweep;
    std::string tag;
    // "pattern=DevName;pattern2=DevName2" -> pin matching tensors to a device
    std::vector<std::string> tensor_overrides;
};

void usage() {
    fprintf(stderr,
            "usage: bench_pp_scaling --model M.gguf [--dev Vulkan0,XDNA] [--pp N]\n"
            "       [--batch N] [--ubatch N] [--warmup N] [--measure N]\n"
            "       [--mode bench|verify|ubatch] [--n-predict N] [--threads N]\n"
            "       [--ubatch-sweep 512,1024,2048] [--ngl N] [--tag NAME]\n"
            "       [--ot pattern=DeviceName[,pattern2=DeviceName2]]\n");
}

std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> out;
    size_t p = 0;
    while (p < s.size()) {
        size_t c = s.find(',', p);
        if (c == std::string::npos) {
            c = s.size();
        }
        out.push_back(atoi(s.substr(p, c - p).c_str()));
        p = c + 1;
    }
    return out;
}

std::vector<std::string> split_commas(const std::string & s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p < s.size()) {
        size_t c = s.find(',', p);
        if (c == std::string::npos) {
            c = s.size();
        }
        out.push_back(s.substr(p, c - p));
        p = c + 1;
    }
    return out;
}

// deterministic token stream: repeated vocab-sized ids
std::vector<llama_token> make_prompt(llama_context * ctx, int n) {
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<llama_token> toks;
    toks.reserve(n);
    for (int i = 0; i < n; i++) {
        toks.push_back((llama_token) (1 + (i * 7919) % (vocab - 1)));
    }
    return toks;
}

llama_context * make_ctx(llama_model * model, const common_params & mp, int n_ctx, int n_batch, int n_ubatch) {
    common_params cp = mp;
    cp.n_ctx        = (uint32_t) n_ctx;
    cp.n_batch      = (uint32_t) n_batch;
    cp.n_ubatch     = (uint32_t) n_ubatch;
    cp.n_predict    = 1;
    cp.n_gpu_layers = mp.n_gpu_layers;

    llama_context_params lcp = common_context_params_to_llama(cp);
    lcp.n_ctx           = n_ctx;
    lcp.n_batch         = n_batch;
    lcp.n_ubatch        = n_ubatch;
    return llama_init_from_model(model, lcp);
}

// model params: reuse the same device selection as the context
static void apply_devices(common_params & p, const args & a) {
    if (a.dev_list.empty()) {
        return;
    }
    p.devices.clear();
    for (const auto & name : a.dev_list) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) {
            fprintf(stderr, "bench_pp_scaling: unknown device '%s'\n", name.c_str());
            exit(1);
        }
        p.devices.push_back(dev);
    }
}

// one measured prompt phase, returns elapsed ms
double run_pp(llama_context * ctx, const std::vector<llama_token> & prompt, int n_batch) {
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
    for (size_t i = 0; i < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }

    const double t0 = now_ms();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "bench_pp_scaling: llama_decode failed\n");
        exit(1);
    }
    llama_synchronize(ctx);
    const double t1 = now_ms();
    llama_batch_free(batch);
    return t1 - t0;
}

int run_bench(const args & a, llama_model * model, const common_params & mp, int pp, int n_batch, int n_ubatch) {
    const int n_ctx = pp + 16;
    llama_context * ctx = make_ctx(model, mp, n_ctx, n_batch, n_ubatch);
    if (!ctx) {
        fprintf(stderr, "bench_pp_scaling: failed to create context (n_ctx=%d)\n", n_ctx);
        return 1;
    }
    const std::vector<llama_token> prompt = make_prompt(ctx, pp);

    for (int i = 0; i < a.warmup; i++) {
        run_pp(ctx, prompt, n_batch);
    }

    std::vector<double> ms;
    for (int i = 0; i < a.measure; i++) {
        ms.push_back(run_pp(ctx, prompt, n_batch));
    }

    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    const double tps = pp / (med / 1000.0);
    double mean = 0;
    for (double v : ms) {
        mean += v;
    }
    mean /= ms.size();
    double var = 0;
    for (double v : ms) {
        var += (v - mean) * (v - mean);
    }
    const double sd = ms.size() > 1 ? std::sqrt(var / (ms.size() - 1)) : 0;

    llama_perf_context_data perf = llama_perf_context(ctx);
    printf("RESULT,pp=%d,batch=%d,ubatch=%d,mode=%s,pp_tps=%.2f,pp_tps_sd=%.2f,"
           "pp_ms=%.3f,pp_ms_sd=%.3f,pp_tps_mean=%.2f,n_warmup=%d,n_measure=%d,"
           "peak_rss_kb=%ld,tag=%s\n",
           pp, n_batch, n_ubatch, a.mode.c_str(), tps, tps * sd / (mean > 0 ? mean : 1), med, sd,
           pp / (mean / 1000.0), a.warmup, a.measure, peak_rss_kb(), a.tag.c_str());
    fflush(stdout);

    llama_free(ctx);
    return 0;
}

int run_verify(const args & a, llama_model * model, const common_params & mp, int pp, int n_batch, int n_ubatch) {
    const int n_ctx = pp + a.n_predict + 16;
    llama_context * ctx = make_ctx(model, mp, n_ctx, n_batch, n_ubatch);
    if (!ctx) {
        fprintf(stderr, "bench_pp_scaling: failed to create context\n");
        return 1;
    }
    const std::vector<llama_token> prompt = make_prompt(ctx, pp);

    auto * sampler = llama_sampler_init_greedy();

    std::vector<llama_token> out;
    llama_batch batch = llama_batch_init(std::max((size_t) a.n_predict, (size_t) pp) + 1, 0, 1);
    for (size_t i = 0; i < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "bench_pp_scaling: llama_decode failed during verify\n");
        return 1;
    }
    llama_synchronize(ctx);

    const int n_past = (int) prompt.size();
    for (int i = 0; i < a.n_predict; i++) {
        const llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, tok);
        out.push_back(tok);
        common_batch_clear(batch);
        common_batch_add(batch, tok, (llama_pos) (n_past + i), { 0 }, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "bench_pp_scaling: llama_decode failed during verify gen\n");
            return 1;
        }
        llama_synchronize(ctx);
    }
    llama_batch_free(batch);

    printf("VERIFY,pp=%d,batch=%d,ubatch=%d,n_predict=%d,md5=%s,tokens=",
           pp, n_batch, n_ubatch, a.n_predict, md5_of(out).c_str());
    for (size_t i = 0; i < out.size(); i++) {
        printf("%s%d", i ? "," : "", out[i]);
    }
    printf("\n");
    fflush(stdout);

    llama_sampler_free(sampler);
    llama_free(ctx);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; i++) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if      (k == "--model")        a.model = next();
        else if (k == "--dev")          a.dev_list = split_commas(next());
        else if (k == "--mode")         a.mode = next();
        else if (k == "--pp")           a.pp = atoi(next().c_str());
        else if (k == "--batch")        a.batch = atoi(next().c_str());
        else if (k == "--ubatch")       a.ubatch = atoi(next().c_str());
        else if (k == "--warmup")       a.warmup = atoi(next().c_str());
        else if (k == "--measure")      a.measure = atoi(next().c_str());
        else if (k == "--n-predict")    a.n_predict = atoi(next().c_str());
        else if (k == "--threads")      a.n_threads = atoi(next().c_str());
        else if (k == "--ngl")          a.n_gpu_layers = atoi(next().c_str());
        else if (k == "--ubatch-sweep") a.ubatch_sweep = next();
        else if (k == "--tag")          a.tag = next();
        else if (k == "--ot")           a.tensor_overrides = split_commas(next());
        else if (k == "-h" || k == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", k.c_str()); usage(); return 1; }
    }
    if (a.model.empty()) {
        usage();
        return 1;
    }
    if (a.batch  == 0) a.batch  = a.pp;
    if (a.ubatch == 0) a.ubatch = a.pp;

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    common_params mp;
    mp.n_gpu_layers = a.n_gpu_layers;
    mp.n_ctx        = (uint32_t) (a.pp + a.n_predict + 16);
    mp.n_batch      = (uint32_t) a.batch;
    mp.n_ubatch     = (uint32_t) a.ubatch;
    mp.n_predict    = a.n_predict;
    if (a.n_threads > 0) {
        mp.cpuparams.n_threads = a.n_threads;
        mp.cpuparams_batch     = mp.cpuparams;
    }
    apply_devices(mp, a);

    for (const auto & spec : a.tensor_overrides) {
        const size_t eq = spec.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "bench_pp_scaling: bad --ot '%s', want pattern=Device\n", spec.c_str());
            return 1;
        }
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(spec.substr(eq + 1).c_str());
        if (!dev) {
            fprintf(stderr, "bench_pp_scaling: unknown device in --ot '%s'\n", spec.c_str());
            return 1;
        }
        llama_model_tensor_buft_override ov = { nullptr, nullptr };
        ov.pattern  = spec.substr(0, eq).c_str();
        ov.buft     = ggml_backend_dev_buffer_type(dev);
        mp.tensor_buft_overrides.push_back(ov);
    }
    // terminator required by the loader
    mp.tensor_buft_overrides.push_back(llama_model_tensor_buft_override{ nullptr, nullptr });

    llama_backend_init();
    llama_model_params lmp = common_model_params_to_llama(mp);
    llama_model * model = llama_model_load_from_file(a.model.c_str(), lmp);
    if (!model) {
        fprintf(stderr, "bench_pp_scaling: failed to load model %s\n", a.model.c_str());
        return 1;
    }
    std::string dev_str;
    for (size_t i = 0; i < mp.devices.size(); i++) {
        dev_str += (i ? "," : "");
        dev_str += ggml_backend_dev_name(mp.devices[i]);
    }
    printf("INFO,n_layer=%d,n_embd=%d,dev=%s\n",
           llama_model_n_layer(model), llama_model_n_embd(model),
           dev_str.empty() ? "auto" : dev_str.c_str());
    fflush(stdout);

    int rc = 0;
    if (a.mode == "ubatch") {
        for (int ub : parse_int_list(a.ubatch_sweep)) {
            rc |= run_bench(a, model, mp, a.pp, a.batch, ub);
        }
    } else if (a.mode == "verify") {
        rc = run_verify(a, model, mp, a.pp, a.batch, a.ubatch);
    } else {
        rc = run_bench(a, model, mp, a.pp, a.batch, a.ubatch);
    }

    llama_model_free(model);
    llama_backend_free();
    return rc;
}