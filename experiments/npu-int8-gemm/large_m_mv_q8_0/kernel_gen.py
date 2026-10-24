#!/usr/bin/env python3
"""
Large-M Matrix-Vector Multiply Kernel for XDNA
M = 248320 (or configurable), K = 1024, N = 1
Q8_0 weights, F32 input/output
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, '/Storage/Cache/src/mlir-aie-venv/lib/python3.14/site-packages/mlir_aie/python')
import aie.iron as iron
from aie.iron.kernels import mm
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, Out

# Q8_0 constants
QK8_0 = 32
Q8_0_BLOCK_SIZE = QK8_0 + 2  # 32 int8 + 1 f16 scale (2 bytes)

def build_large_m_mv_kernel(M, K, N, num_cores, core_m_tiles, work_dir):
    """
    Build large-M matrix-vector multiply kernel.
    
    M: output dimension (e.g., 248320)
    K: input dimension (1024)
    N: output columns (1 for MV)
    num_cores: number of XDNA cores to use (6, 12, or 24)
    core_m_tiles: number of M tiles per core
    """
    print(f"\n=== Building Large-M MV Kernel: M={M}, K={K}, N={N}, cores={num_cores} ===")
    
    # Calculate M per core
    M_per_core = M // num_cores
    assert M % num_cores == 0, f"M={M} not divisible by num_cores={num_cores}"
    
    # K dimension in blocks of QK8_0
    K_blocks = K // QK8_0
    assert K % QK8_0 == 0, f"K={K} not divisible by QK8_0={QK8_0}"
    
    # Weight matrix: [M, K] in Q8_0 format
    # Stored as [M, K_blocks, Q8_0_BLOCK_SIZE] where each block has 32 int8 + 1 f16 scale
    weight_ty = np.ndarray[(M, K_blocks, Q8_0_BLOCK_SIZE), np.dtype[np.int8]]
    
    # Input vector: [K, 1] in F32
    input_ty = np.ndarray[(K, 1), np.dtype[np.float32]]
    
    # Output vector: [M, 1] in F32
    output_ty = np.ndarray[(M, 1), np.dtype[np.float32]]
    
    # Per-core FIFO types
    core_weight_ty = np.ndarray[(M_per_core, K_blocks, Q8_0_BLOCK_SIZE), np.dtype[np.int8]]
    core_input_ty = np.ndarray[(K, 1), np.dtype[np.float32]]
    core_output_ty = np.ndarray[(M_per_core, 1), np.dtype[np.float32]]
    
    # Stream FIFO types
    weight_stream_ty = np.ndarray[(M_per_core * K_blocks * Q8_0_BLOCK_SIZE,), np.dtype[np.int8]]
    input_stream_ty = np.ndarray[(K,), np.dtype[np.float32]]
    output_stream_ty = np.ndarray[(M_per_core,), np.dtype[np.float32]]
    
    # Create FIFOs for each core
    weight_fifos_L3L2 = []
    weight_fifos_L2L1 = []
    input_fifos_L3L2 = []
    input_fifos_L2L1 = []
    output_fifos_L1L2 = []
    output_fifos_L2L3 = []
    
    for c in range(num_cores):
        w_L3L2 = iron.ObjectFifo(weight_stream_ty, name=f"W_L3L2_c{c}")
        w_L2L1 = w_L3L2.cons().forward(dims_to_stream=[(M_per_core * K_blocks * Q8_0_BLOCK_SIZE, 1)], name=f"W_L2L1_c{c}")
        weight_fifos_L3L2.append(w_L3L2)
        weight_fifos_L2L1.append(w_L2L1)
        
        i_L3L2 = iron.ObjectFifo(input_stream_ty, name=f"I_L3L2_c{c}")
        i_L2L1 = i_L3L2.cons().forward(dims_to_stream=[(K, 1)], name=f"I_L2L1_c{c}")
        input_fifos_L3L2.append(i_L3L2)
        input_fifos_L2L1.append(i_L2L1)
        
        o_L1L2 = iron.ObjectFifo(output_stream_ty, name=f"O_L1L2_c{c}")
        o_L2L3 = o_L1L2.cons().forward(dims_to_stream=[(M_per_core, 1)], name=f"O_L2L3_c{c}")
        output_fifos_L1L2.append(o_L1L2)
        output_fifos_L2L3.append(o_L2L3)
    
    # Core function for matrix-vector multiply with Q8_0 weights
    def core_fn(of_w, of_i, of_o, M_local, K_local, K_blocks_local):
        # of_w: [M_local, K_blocks, 34] (Q8_0 blocks)
        # of_i: [K_local, 1] (F32)
        # of_o: [M_local, 1] (F32)
        
        for m_tile in iron.controlflow.range_(M_local):
            # Acquire output element
            elem_out = of_o.acquire(1)
            elem_out[0] = 0.0  # Initialize accumulator
            
            # Load input vector once per M_tile (broadcast)
            elem_in = of_i.acquire(1)
            # elem_in is the full input vector [K, 1]
            
            # Compute dot product with Q8_0 weight row
            acc = 0.0
            for k_block in iron.controlflow.range_(K_blocks_local):
                # Acquire weight block: 32 int8 + 1 f16 scale
                w_block = of_w.acquire(1)  # [34] int8 (32 data + 2 scale bytes)
                
                # Extract scale (first 2 bytes as f16)
                scale_bytes = w_block[:2].view(np.uint16)
                scale = np.float16(scale_bytes[0]).astype(np.float32)
                
                # Extract int8 data (32 values)
                w_data = w_block[2:34].astype(np.int32)
                
                # Load corresponding input chunk
                for k_in_block in iron.controlflow.range_(QK8_0):
                    k_idx = k_block * QK8_0 + k_in_block
                    if k_idx < K_local:
                        x_val = elem_in[k_idx]
                        w_val = w_data[k_in_block]
                        acc += x_val * (w_val * scale)
            
            elem_out[0] = acc
            of_o.release(1)
            of_i.release(1)  # Release input after each M_tile
            of_w.release(1)  # Release weight block
    
    # Create worker per core
    workers = []
    for c in range(num_cores):
        worker = iron.Worker(
            core_fn,
            [weight_fifos_L2L1[c].cons(), input_fifos_L2L1[c].cons(), output_fifos_L1L2[c].prod(), 
             CompileTime[int](M_per_core), CompileTime[int](K), CompileTime[int](K_blocks)],
        )
        workers.append(worker)
    
    # Sequence function for host orchestration
    def sequence(weight, input_vec, output, 
                 *fifo_args):
        tg = iron.TaskGroup()
        
        # Distribute weight matrix to cores
        for c in range(num_cores):
            w_prod = fifo_args[c].prod()
            w_tap = TensorAccessPattern(
                (M_per_core, K_blocks, Q8_0_BLOCK_SIZE), 
                0, 
                [M_per_core, K_blocks, Q8_0_BLOCK_SIZE], 
                [K_blocks * Q8_0_BLOCK_SIZE, Q8_0_BLOCK_SIZE, 1]
            )
            w_prod.fill(weight, tap=w_tap, group=tg, offset=(c * M_per_core, 0, 0))
        
        # Broadcast input vector to all cores
        for c in range(num_cores):
            i_prod = fifo_args[num_cores + c].prod()
            i_tap = TensorAccessPattern(
                (K, 1), 0, [K, 1], [1, 1]
            )
            i_prod.fill(input_vec, tap=i_tap, group=tg)
        
        # Collect outputs from all cores
        for c in range(num_cores):
            o_cons = fifo_args[2 * num_cores + c].cons()
            o_tap = TensorAccessPattern(
                (M_per_core, 1), 0, [M_per_core, 1], [1, 1]
            )
            o_cons.drain(output, tap=o_tap, group=tg, wait=True, offset=(c * M_per_core, 0))
        
        tg.finish()
    
    # Build program
    rt = iron.Runtime(
        sequence,
        [weight_ty, input_ty, output_ty] + 
        [w.prod() for w in weight_fifos_L3L2] +
        [i.prod() for i in input_fifos_L3L2] +
        [o.cons() for o in output_fifos_L2L3],
    )
    
    program = iron.Program(iron.get_current_device(), rt, workers=workers).resolve_program()
    
    # Compile
    xclbin_path = os.path.join(work_dir, "design.xclbin")
    insts_path = os.path.join(work_dir, "design.insts.bin")
    pdi_path = os.path.join(work_dir, "main.pdi")
    
    t0 = time.time()
    try:
        compiled = program.compile(
            xclbin_path=xclbin_path,
            inst_path=insts_path,
            pdi_path=pdi_path,
        )
    except Exception as exc:
        print(f"  COMPILE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return False, None
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")
    
    return True, (xclbin_path, insts_path, pdi_path)


def test_kernel(M, K, N, num_cores, artifacts):
    """Test the compiled kernel with random data."""
    xclbin_path, insts_path, pdi_path = artifacts
    
    print(f"\n=== Testing Kernel: M={M}, K={K}, N={N} ===")
    
    # Load compiled design
    import xrt
    device = xrt.device(0)
    xclbin = xrt.xclbin(xclbin_path)
    uuid = device.register_xclbin(xclbin)
    hw_ctx = xrt.hw_context(device, uuid)
    kernel = xrt.kernel(device, uuid, "matmul")
    
    M_per_core = M // num_cores
    K_blocks = K // QK8_0
    
    rng = np.random.default_rng(42)
    all_pass = True
    
    for pattern_name, pattern_fn in [
        ("all_ones", lambda: (
            np.ones((M, K_blocks, Q8_0_BLOCK_SIZE), dtype=np.int8),
            np.ones((K, 1), dtype=np.float32)
        )),
        ("random", lambda: (
            rng.integers(-128, 127, size=(M, K_blocks, Q8_0_BLOCK_SIZE), dtype=np.int8),
            rng.standard_normal(size=(K, 1)).astype(np.float32)
        )),
        ("pos_only", lambda: (
            rng.integers(0, 127, size=(M, K_blocks, Q8_0_BLOCK_SIZE), dtype=np.int8),
            rng.uniform(0, 1, size=(K, 1)).astype(np.float32)
        )),
    ]:
        weight_np, input_np = pattern_fn()
        
        # Compute reference
        # Dequantize Q8_0 weight
        weight_f32 = np.zeros((M, K), dtype=np.float32)
        for m in range(M):
            for k_block in range(K_blocks):
                block = weight_np[m, k_block]
                scale = np.float16(block[:2].view(np.uint16)[0]).astype(np.float32)
                w_data = block[2:34].astype(np.float32)
                weight_f32[m, k_block*QK8_0:(k_block+1)*QK8_0] = w_data * scale
        
        output_ref = weight_f32 @ input_np  # [M, 1]
        
        # Run on NPU
        weight_bo = xrt.bo(device, weight_np.nbytes, xrt.bo.flags.normal, 0)
        input_bo = xrt.bo(device, input_np.nbytes, xrt.bo.flags.normal, 0)
        output_bo = xrt.bo(device, output_ref.nbytes, xrt.bo.flags.normal, 0)
        
        weight_bo.write(weight_np.tobytes(), 0)
        input_bo.write(input_np.tobytes(), 0)
        
        run = kernel(weight_bo, input_bo, output_bo, M_per_core, K, K_blocks)
        run.wait()
        
        output_np = np.frombuffer(output_bo.read(output_ref.nbytes, 0), dtype=np.float32).reshape(M, 1)
        
        mismatches = int(np.sum(np.abs(output_np - output_ref) > 1e-3))
        max_abs_error = float(np.max(np.abs(output_np - output_ref)))
        max_rel_error = float(np.max(np.abs(output_np - output_ref) / (np.abs(output_ref) + 1e-6)))
        
        if mismatches == 0 and max_abs_error < 1e-3:
            print(f"  {pattern_name}: PASS (max_abs={max_abs_error:.6f}, max_rel={max_rel_error:.6f})")
        else:
            print(f"  {pattern_name}: FAIL - {mismatches} mismatches, max_abs={max_abs_error:.6f}, max_rel={max_rel_error:.6f}")
            all_pass = False
    
    return all_pass


def main():
    base_dir = "/workspace/llama.cpp/experiments/npu-int8-gemm/large_m_mv_q8_0"
    os.makedirs(base_dir, exist_ok=True)
    
    # Target: M=248320, K=1024, N=1
    # Test with smaller M first for faster iteration
    test_configs = [
        (64, 1024, 1, 4),   # M=64, 4 cores -> 16 per core
        (256, 1024, 1, 4),  # M=256, 4 cores -> 64 per core
        (1024, 1024, 1, 4), # M=1024, 4 cores -> 256 per core
        (4096, 1024, 1, 4), # M=4096, 4 cores -> 1024 per core
    ]
    
    results = {}
    
    for M, K, N, num_cores in test_configs:
        work_dir = os.path.join(base_dir, f"M{M}_K{K}_N{N}_c{num_cores}")
        os.makedirs(work_dir, exist_ok=True)
        
        success, artifacts = build_large_m_mv_kernel(M, K, N, num_cores, M // num_cores, work_dir)
        results[(M, K, N, num_cores)] = {"compile_success": success, "artifacts": artifacts}
        
        if success and artifacts:
            test_pass = test_kernel(M, K, N, num_cores, artifacts)
            results[(M, K, N, num_cores)]["test_pass"] = test_pass
            
            if not test_pass:
                print(f"\n*** FAILURE at M={M} ***")
                break
        else:
            print(f"\n*** COMPILE FAILURE at M={M} ***")
            break
    
    print("\n=== Large-M MV Kernel Summary ===")
    for config, result in results.items():
        status = "PASS" if result.get("test_pass", False) else ("COMPILE_FAIL" if not result["compile_success"] else "TEST_FAIL")
        print(f"  M={config[0]:6d} K={config[1]:4d} N={config[2]} cores={config[3]}: {status}")
    
    import json
    with open(os.path.join(base_dir, "large_m_mv_results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()