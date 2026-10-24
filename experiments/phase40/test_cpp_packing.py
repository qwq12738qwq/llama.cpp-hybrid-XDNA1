#!/usr/bin/env python3
"""Phase 40M: Verify that the C++ packing logic matches the Python tile-major layout."""
import numpy as np
import sys
import os
import types

sys.path.insert(0, '/workspace/llama.cpp/experiments/phase40')

# Mock aie.iron before importing gen_q8_c4_mmul_v2
aie_mock = types.ModuleType('aie')
aie_iron_mock = types.ModuleType('aie.iron')
aie_iron_mock.CompileTime = object
aie_iron_mock.In = object
aie_iron_mock.Out = object
aie_iron_mock.ObjectFifo = object
aie_iron_mock.Runtime = object
aie_iron_mock.Worker = object
aie_iron_mock.ExternalFunction = object
aie_mock.iron = aie_iron_mock
sys.modules['aie'] = aie_mock
sys.modules['aie.iron'] = aie_iron_mock
sys.modules['aie.iron.controlflow'] = types.ModuleType('aie.iron.controlflow')
sys.modules['aie.iron.controlflow'].range_ = range
sys.modules['aie.helpers'] = types.ModuleType('aie.helpers')
sys.modules['aie.helpers.taplib'] = types.ModuleType('aie.helpers.taplib')
sys.modules['aie.helpers.taplib'].TensorAccessPattern = object
sys.modules['aie.utils'] = types.ModuleType('aie.utils')
sys.modules['aie.utils'].config = types.SimpleNamespace(cxx_header_path=lambda: '')
sys.modules['aie.utils'].DefaultNPURuntime = type('DefaultNPURuntime', (), {})
sys.modules['aie.utils'].NPUKernel = object
sys.modules['aie.utils'].tensor = np.ndarray

import importlib.util
spec = importlib.util.spec_from_file_location('gen_q8_c4_mmul_v2', '/workspace/llama.cpp/experiments/phase40/gen_q8_c4_mmul_v2.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pack_A = mod.pack_A
pack_B = mod.pack_B
ref_q8_gemm = mod.ref_q8_gemm
unpack_A = mod.unpack_A
unpack_B = mod.unpack_B
M_TILE = mod.M_TILE
N_TILE = mod.N_TILE
KC = mod.KC
NBLK = mod.NBLK


def cpp_pack_A_mmul_v2(input_i8, i_scale, M, K, m, kc, nblk, nb, n_mt, n_kc):
    a_chunk = m * kc + m * nblk * 4
    Ap = np.zeros((n_mt * n_kc * a_chunk,), dtype=np.int8)
    for mt in range(n_mt):
        for cc in range(n_kc):
            t = Ap[(mt * n_kc + cc) * a_chunk:(mt * n_kc + cc + 1) * a_chunk]
            int8_part = t[:m * kc]
            scale_part = t[m * kc:].view(np.float32).reshape(m, nblk)
            for im in range(m // 4):
                for bb in range(nblk):
                    for kk in range(4):
                        for rr in range(4):
                            for cc_col in range(8):
                                flat_idx = ((im * nblk + bb) * 4 + kk) * 32 + rr * 8 + cc_col
                                src_idx = (mt * m + im * 4 + rr) * K + cc * kc + bb * 32 + kk * 8 + cc_col
                                int8_part[flat_idx] = input_i8[src_idx]
            for i in range(m):
                for bb in range(nblk):
                    scale_part[i, bb] = i_scale[(mt * m + i) * nb + cc * nblk + bb]
    return Ap


def cpp_pack_B_mmul_v2(weight_i8, w_scale, K, N, n, kc, nblk, nb, n_cols, n_nt_c, n_kc):
    b_chunk = kc * n + nblk * n * 4
    Bp = np.zeros((n_cols * n_nt_c * n_kc * b_chunk,), dtype=np.int8)
    for col in range(n_cols):
        for nt in range(n_nt_c):
            for cc in range(n_kc):
                t = Bp[((col * n_nt_c + nt) * n_kc + cc) * b_chunk:((col * n_nt_c + nt) * n_kc + cc + 1) * b_chunk]
                int8_part = t[:kc * n]
                scale_part = t[kc * n:].view(np.float32).reshape(nblk, n)
                for jn in range(n // 8):
                    for bb in range(nblk):
                        for kk in range(4):
                            for rr in range(8):
                                for cc_col in range(8):
                                    flat_idx = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col
                                    src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col * n_nt_c + nt) * n + jn * 8 + cc_col
                                    int8_part[flat_idx] = weight_i8[src_idx]
                for bb in range(nblk):
                    for j in range(n):
                        scale_part[bb, j] = w_scale[(cc * nblk + bb) * N + (col * n_nt_c + nt) * n + j]
    return Bp


def test_mmul_v2_packing(M, K, N, n_cols):
    print(f"Testing mmul-v2 packing: M={M} K={K} N={N} cols={n_cols}")
    nb = K // 32
    n_mt = M // M_TILE
    n_nt_c = (N // n_cols) // N_TILE
    n_kc = K // KC
    m, kc, nblk = M_TILE, KC, NBLK

    rng = np.random.default_rng(42)
    A_i8 = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
    B_i8 = rng.integers(-127, 128, size=(K, N), dtype=np.int8)
    As = rng.uniform(0.001, 0.02, size=(M, nb)).astype(np.float32)
    Bs = rng.uniform(0.001, 0.02, size=(nb, N)).astype(np.float32)

    C_ref = ref_q8_gemm(A_i8, As, B_i8, Bs)

    Ap_py = pack_A(A_i8, As, M, K, M_TILE, KC)
    Bp_py = pack_B(B_i8, Bs, K, N, N_TILE, KC)

    input_i8 = A_i8.reshape(-1).copy()
    weight_i8 = B_i8.reshape(-1).copy()
    i_scale = As.reshape(-1).copy()
    w_scale = Bs.reshape(-1).copy()

    Ap_cpp = cpp_pack_A_mmul_v2(input_i8, i_scale, M, K, m, kc, nblk, nb, n_mt, n_kc)
    Bp_cpp = cpp_pack_B_mmul_v2(weight_i8, w_scale, K, N, N_TILE, KC, nblk, nb, n_cols, n_nt_c, n_kc)

    print(f"  A packing match: {np.array_equal(Ap_py, Ap_cpp)}")
    print(f"  B packing match: {np.array_equal(Bp_py, Bp_cpp)}")

    A_i8_cpp, As_cpp = unpack_A(Ap_cpp, M, K, M_TILE, KC)
    B_i8_cpp, Bs_cpp = unpack_B(Bp_cpp, K, N, N_TILE, KC)
    C_cpp = ref_q8_gemm(A_i8_cpp, As_cpp, B_i8_cpp, Bs_cpp)

    diff = np.abs(C_cpp.astype(np.float64) - C_ref.astype(np.float64))
    max_err = float(np.max(diff))
    bad = int(np.sum(diff > 1e-3 * np.maximum(np.abs(C_ref), 1.0)))
    print(f"  C++ packing vs ref: max_err={max_err:.4e} bad={bad}/{M*N}")
    return bad == 0 and np.array_equal(Ap_py, Ap_cpp) and np.array_equal(Bp_py, Bp_cpp)


if __name__ == "__main__":
    print("Phase 40M: C++ packing verification")
    print("=" * 60)
    
    results = []
    
    print("\n--- mmul-v2 packing (MT=32) ---")
    ok1 = test_mmul_v2_packing(32, 256, 64, 1)
    results.append(("mmul-v2 M32", ok1))
    ok2 = test_mmul_v2_packing(512, 1024, 3584, 4)
    results.append(("mmul-v2 M512", ok2))
    
    print("\nSummary:")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False
    
    if all_pass:
        print("\nPhase 40M packing verification: ALL PASS")
    else:
        print("\nPhase 40M packing verification: SOME FAILED")
        sys.exit(1)
