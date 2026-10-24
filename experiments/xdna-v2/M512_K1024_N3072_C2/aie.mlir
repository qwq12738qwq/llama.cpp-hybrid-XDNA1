// Multi-column INT8 GEMM: M=512 K=1024 N=3072 n_cols=2
// Tile: Tm=64 Tk=64 Tn=64
// n_per_col=1536 nb_m=8 nb_k=16 nb_n=48
module {
  aie.device(npu1) {
    %tile_shim_c0 = aie.tile(0, 0)
    %tile_mem_c0 = aie.tile(0, 1)
    %tile_core0 = aie.tile(0, 2)
    %tile_shim_c1 = aie.tile(1, 0)
    %tile_mem_c1 = aie.tile(1, 1)
    %tile_core1 = aie.tile(1, 2)
    %tile_shim_a = aie.tile(2, 0)
    %tile_mem_a = aie.tile(2, 1)
    %tile_shim_b = aie.tile(3, 0)
    %tile_mem_b = aie.tile(3, 1)

    aie.objectfifo @A_L3L2(%tile_shim_a, {%tile_mem_a}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @A_L2L1(%tile_mem_a, {%tile_core0, %tile_core1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [0])

    aie.objectfifo @B_L3L2(%tile_shim_b, {%tile_mem_b}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @B_L2L1(%tile_mem_b, {%tile_core0, %tile_core1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0])

    aie.objectfifo @C_L1L2_0(%tile_core0, {%tile_mem_c0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @C_L2L3_0(%tile_mem_c0, {%tile_shim_c0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@C_L1L2_0] -> [@C_L2L3_0]([] [0])
    aie.objectfifo @C_L1L2_1(%tile_core1, {%tile_mem_c1}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @C_L2L3_1(%tile_mem_c1, {%tile_shim_c1}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@C_L1L2_1] -> [@C_L2L3_1]([] [0])

    func.func private @"matmul_i8_i32"(memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) attributes {link_with = "matmul_i8_i32.o"}

    // Core 0
    %core0 = aie.core(%tile_core0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c384 = arith.constant 384 : index
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      scf.for %iter = %c0 to %c384 step %c1 {
        %a = aie.objectfifo.acquire @A_L2L1(Consume, 1) : memref<4096xi8>
        %b = aie.objectfifo.acquire @B_L2L1(Consume, 1) : memref<4096xi8>
        %c_out = aie.objectfifo.acquire @C_L1L2_0(Produce, 1) : memref<4096xi32>
        %c0_i32 = arith.constant 0 : i32
        scf.for %i = %c0 to %c4096 step %c1 {
          memref.store %c0_i32, %c_out[%i] : memref<4096xi32>
        }
        scf.for %k = %c0 to %c16 step %c1 {
          func.call @matmul_i8_i32(%a, %b, %c_out) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) -> ()
        }
        aie.objectfifo.release @A_L2L1(Consume, 1)
        aie.objectfifo.release @B_L2L1(Consume, 1)
        aie.objectfifo.release @C_L1L2_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}

    // Core 1
    %core1 = aie.core(%tile_core1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c384 = arith.constant 384 : index
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      scf.for %iter = %c0 to %c384 step %c1 {
        %a = aie.objectfifo.acquire @A_L2L1(Consume, 1) : memref<4096xi8>
        %b = aie.objectfifo.acquire @B_L2L1(Consume, 1) : memref<4096xi8>
        %c_out = aie.objectfifo.acquire @C_L1L2_1(Produce, 1) : memref<4096xi32>
        %c0_i32 = arith.constant 0 : i32
        scf.for %i = %c0 to %c4096 step %c1 {
          memref.store %c0_i32, %c_out[%i] : memref<4096xi32>
        }
        scf.for %k = %c0 to %c16 step %c1 {
          func.call @matmul_i8_i32(%a, %b, %c_out) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) -> ()
        }
        aie.objectfifo.release @A_L2L1(Consume, 1)
        aie.objectfifo.release @B_L2L1(Consume, 1)
        aie.objectfifo.release @C_L1L2_1(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}

    aie.runtime_sequence(%A: memref<512x1024xi8>, %B: memref<1024x3072xi8>, %C_out: memref<512x3072xi32>) {
      // Col 0 M-tile 0 N-tile 0
      %a_col0_m0_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n0)
      %b_col0_m0_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n0)
      %c_col0_m0_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 0 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n0)
      aiex.dma_await_task(%c_col0_m0_n0)
      aiex.dma_free_task(%a_col0_m0_n0)
      aiex.dma_free_task(%b_col0_m0_n0)
      aiex.dma_free_task(%c_col0_m0_n0)
      // Col 0 M-tile 0 N-tile 1
      %a_col0_m0_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n1)
      %b_col0_m0_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n1)
      %c_col0_m0_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 256 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n1)
      aiex.dma_await_task(%c_col0_m0_n1)
      aiex.dma_free_task(%a_col0_m0_n1)
      aiex.dma_free_task(%b_col0_m0_n1)
      aiex.dma_free_task(%c_col0_m0_n1)
      // Col 0 M-tile 0 N-tile 2
      %a_col0_m0_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n2)
      %b_col0_m0_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n2)
      %c_col0_m0_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 512 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n2)
      aiex.dma_await_task(%c_col0_m0_n2)
      aiex.dma_free_task(%a_col0_m0_n2)
      aiex.dma_free_task(%b_col0_m0_n2)
      aiex.dma_free_task(%c_col0_m0_n2)
      // Col 0 M-tile 0 N-tile 3
      %a_col0_m0_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n3)
      %b_col0_m0_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n3)
      %c_col0_m0_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 768 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n3)
      aiex.dma_await_task(%c_col0_m0_n3)
      aiex.dma_free_task(%a_col0_m0_n3)
      aiex.dma_free_task(%b_col0_m0_n3)
      aiex.dma_free_task(%c_col0_m0_n3)
      // Col 0 M-tile 0 N-tile 4
      %a_col0_m0_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n4)
      %b_col0_m0_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n4)
      %c_col0_m0_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1024 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n4)
      aiex.dma_await_task(%c_col0_m0_n4)
      aiex.dma_free_task(%a_col0_m0_n4)
      aiex.dma_free_task(%b_col0_m0_n4)
      aiex.dma_free_task(%c_col0_m0_n4)
      // Col 0 M-tile 0 N-tile 5
      %a_col0_m0_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n5)
      %b_col0_m0_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n5)
      %c_col0_m0_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1280 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n5)
      aiex.dma_await_task(%c_col0_m0_n5)
      aiex.dma_free_task(%a_col0_m0_n5)
      aiex.dma_free_task(%b_col0_m0_n5)
      aiex.dma_free_task(%c_col0_m0_n5)
      // Col 0 M-tile 0 N-tile 6
      %a_col0_m0_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n6)
      %b_col0_m0_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n6)
      %c_col0_m0_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1536 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n6)
      aiex.dma_await_task(%c_col0_m0_n6)
      aiex.dma_free_task(%a_col0_m0_n6)
      aiex.dma_free_task(%b_col0_m0_n6)
      aiex.dma_free_task(%c_col0_m0_n6)
      // Col 0 M-tile 0 N-tile 7
      %a_col0_m0_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n7)
      %b_col0_m0_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n7)
      %c_col0_m0_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1792 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n7)
      aiex.dma_await_task(%c_col0_m0_n7)
      aiex.dma_free_task(%a_col0_m0_n7)
      aiex.dma_free_task(%b_col0_m0_n7)
      aiex.dma_free_task(%c_col0_m0_n7)
      // Col 0 M-tile 0 N-tile 8
      %a_col0_m0_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n8)
      %b_col0_m0_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n8)
      %c_col0_m0_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2048 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n8)
      aiex.dma_await_task(%c_col0_m0_n8)
      aiex.dma_free_task(%a_col0_m0_n8)
      aiex.dma_free_task(%b_col0_m0_n8)
      aiex.dma_free_task(%c_col0_m0_n8)
      // Col 0 M-tile 0 N-tile 9
      %a_col0_m0_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n9)
      %b_col0_m0_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n9)
      %c_col0_m0_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2304 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n9)
      aiex.dma_await_task(%c_col0_m0_n9)
      aiex.dma_free_task(%a_col0_m0_n9)
      aiex.dma_free_task(%b_col0_m0_n9)
      aiex.dma_free_task(%c_col0_m0_n9)
      // Col 0 M-tile 0 N-tile 10
      %a_col0_m0_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n10)
      %b_col0_m0_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n10)
      %c_col0_m0_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2560 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n10)
      aiex.dma_await_task(%c_col0_m0_n10)
      aiex.dma_free_task(%a_col0_m0_n10)
      aiex.dma_free_task(%b_col0_m0_n10)
      aiex.dma_free_task(%c_col0_m0_n10)
      // Col 0 M-tile 0 N-tile 11
      %a_col0_m0_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n11)
      %b_col0_m0_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n11)
      %c_col0_m0_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2816 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n11)
      aiex.dma_await_task(%c_col0_m0_n11)
      aiex.dma_free_task(%a_col0_m0_n11)
      aiex.dma_free_task(%b_col0_m0_n11)
      aiex.dma_free_task(%c_col0_m0_n11)
      // Col 0 M-tile 0 N-tile 12
      %a_col0_m0_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n12)
      %b_col0_m0_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n12)
      %c_col0_m0_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3072 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n12)
      aiex.dma_await_task(%c_col0_m0_n12)
      aiex.dma_free_task(%a_col0_m0_n12)
      aiex.dma_free_task(%b_col0_m0_n12)
      aiex.dma_free_task(%c_col0_m0_n12)
      // Col 0 M-tile 0 N-tile 13
      %a_col0_m0_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n13)
      %b_col0_m0_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n13)
      %c_col0_m0_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3328 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n13)
      aiex.dma_await_task(%c_col0_m0_n13)
      aiex.dma_free_task(%a_col0_m0_n13)
      aiex.dma_free_task(%b_col0_m0_n13)
      aiex.dma_free_task(%c_col0_m0_n13)
      // Col 0 M-tile 0 N-tile 14
      %a_col0_m0_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n14)
      %b_col0_m0_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n14)
      %c_col0_m0_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3584 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n14)
      aiex.dma_await_task(%c_col0_m0_n14)
      aiex.dma_free_task(%a_col0_m0_n14)
      aiex.dma_free_task(%b_col0_m0_n14)
      aiex.dma_free_task(%c_col0_m0_n14)
      // Col 0 M-tile 0 N-tile 15
      %a_col0_m0_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n15)
      %b_col0_m0_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n15)
      %c_col0_m0_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3840 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n15)
      aiex.dma_await_task(%c_col0_m0_n15)
      aiex.dma_free_task(%a_col0_m0_n15)
      aiex.dma_free_task(%b_col0_m0_n15)
      aiex.dma_free_task(%c_col0_m0_n15)
      // Col 0 M-tile 0 N-tile 16
      %a_col0_m0_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n16)
      %b_col0_m0_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n16)
      %c_col0_m0_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4096 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n16)
      aiex.dma_await_task(%c_col0_m0_n16)
      aiex.dma_free_task(%a_col0_m0_n16)
      aiex.dma_free_task(%b_col0_m0_n16)
      aiex.dma_free_task(%c_col0_m0_n16)
      // Col 0 M-tile 0 N-tile 17
      %a_col0_m0_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n17)
      %b_col0_m0_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n17)
      %c_col0_m0_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4352 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n17)
      aiex.dma_await_task(%c_col0_m0_n17)
      aiex.dma_free_task(%a_col0_m0_n17)
      aiex.dma_free_task(%b_col0_m0_n17)
      aiex.dma_free_task(%c_col0_m0_n17)
      // Col 0 M-tile 0 N-tile 18
      %a_col0_m0_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n18)
      %b_col0_m0_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n18)
      %c_col0_m0_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4608 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n18)
      aiex.dma_await_task(%c_col0_m0_n18)
      aiex.dma_free_task(%a_col0_m0_n18)
      aiex.dma_free_task(%b_col0_m0_n18)
      aiex.dma_free_task(%c_col0_m0_n18)
      // Col 0 M-tile 0 N-tile 19
      %a_col0_m0_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n19)
      %b_col0_m0_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n19)
      %c_col0_m0_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4864 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n19)
      aiex.dma_await_task(%c_col0_m0_n19)
      aiex.dma_free_task(%a_col0_m0_n19)
      aiex.dma_free_task(%b_col0_m0_n19)
      aiex.dma_free_task(%c_col0_m0_n19)
      // Col 0 M-tile 0 N-tile 20
      %a_col0_m0_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n20)
      %b_col0_m0_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n20)
      %c_col0_m0_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5120 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n20)
      aiex.dma_await_task(%c_col0_m0_n20)
      aiex.dma_free_task(%a_col0_m0_n20)
      aiex.dma_free_task(%b_col0_m0_n20)
      aiex.dma_free_task(%c_col0_m0_n20)
      // Col 0 M-tile 0 N-tile 21
      %a_col0_m0_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n21)
      %b_col0_m0_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n21)
      %c_col0_m0_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5376 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n21)
      aiex.dma_await_task(%c_col0_m0_n21)
      aiex.dma_free_task(%a_col0_m0_n21)
      aiex.dma_free_task(%b_col0_m0_n21)
      aiex.dma_free_task(%c_col0_m0_n21)
      // Col 0 M-tile 0 N-tile 22
      %a_col0_m0_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n22)
      %b_col0_m0_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n22)
      %c_col0_m0_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5632 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n22)
      aiex.dma_await_task(%c_col0_m0_n22)
      aiex.dma_free_task(%a_col0_m0_n22)
      aiex.dma_free_task(%b_col0_m0_n22)
      aiex.dma_free_task(%c_col0_m0_n22)
      // Col 0 M-tile 0 N-tile 23
      %a_col0_m0_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n23)
      %b_col0_m0_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n23)
      %c_col0_m0_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5888 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n23)
      aiex.dma_await_task(%c_col0_m0_n23)
      aiex.dma_free_task(%a_col0_m0_n23)
      aiex.dma_free_task(%b_col0_m0_n23)
      aiex.dma_free_task(%c_col0_m0_n23)
      // Col 0 M-tile 1 N-tile 0
      %a_col0_m1_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n0)
      %b_col0_m1_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n0)
      %c_col0_m1_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 786432 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n0)
      aiex.dma_await_task(%c_col0_m1_n0)
      aiex.dma_free_task(%a_col0_m1_n0)
      aiex.dma_free_task(%b_col0_m1_n0)
      aiex.dma_free_task(%c_col0_m1_n0)
      // Col 0 M-tile 1 N-tile 1
      %a_col0_m1_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n1)
      %b_col0_m1_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n1)
      %c_col0_m1_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 786688 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n1)
      aiex.dma_await_task(%c_col0_m1_n1)
      aiex.dma_free_task(%a_col0_m1_n1)
      aiex.dma_free_task(%b_col0_m1_n1)
      aiex.dma_free_task(%c_col0_m1_n1)
      // Col 0 M-tile 1 N-tile 2
      %a_col0_m1_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n2)
      %b_col0_m1_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n2)
      %c_col0_m1_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 786944 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n2)
      aiex.dma_await_task(%c_col0_m1_n2)
      aiex.dma_free_task(%a_col0_m1_n2)
      aiex.dma_free_task(%b_col0_m1_n2)
      aiex.dma_free_task(%c_col0_m1_n2)
      // Col 0 M-tile 1 N-tile 3
      %a_col0_m1_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n3)
      %b_col0_m1_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n3)
      %c_col0_m1_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 787200 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n3)
      aiex.dma_await_task(%c_col0_m1_n3)
      aiex.dma_free_task(%a_col0_m1_n3)
      aiex.dma_free_task(%b_col0_m1_n3)
      aiex.dma_free_task(%c_col0_m1_n3)
      // Col 0 M-tile 1 N-tile 4
      %a_col0_m1_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n4)
      %b_col0_m1_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n4)
      %c_col0_m1_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 787456 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n4)
      aiex.dma_await_task(%c_col0_m1_n4)
      aiex.dma_free_task(%a_col0_m1_n4)
      aiex.dma_free_task(%b_col0_m1_n4)
      aiex.dma_free_task(%c_col0_m1_n4)
      // Col 0 M-tile 1 N-tile 5
      %a_col0_m1_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n5)
      %b_col0_m1_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n5)
      %c_col0_m1_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 787712 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n5)
      aiex.dma_await_task(%c_col0_m1_n5)
      aiex.dma_free_task(%a_col0_m1_n5)
      aiex.dma_free_task(%b_col0_m1_n5)
      aiex.dma_free_task(%c_col0_m1_n5)
      // Col 0 M-tile 1 N-tile 6
      %a_col0_m1_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n6)
      %b_col0_m1_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n6)
      %c_col0_m1_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 787968 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n6)
      aiex.dma_await_task(%c_col0_m1_n6)
      aiex.dma_free_task(%a_col0_m1_n6)
      aiex.dma_free_task(%b_col0_m1_n6)
      aiex.dma_free_task(%c_col0_m1_n6)
      // Col 0 M-tile 1 N-tile 7
      %a_col0_m1_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n7)
      %b_col0_m1_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n7)
      %c_col0_m1_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 788224 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n7)
      aiex.dma_await_task(%c_col0_m1_n7)
      aiex.dma_free_task(%a_col0_m1_n7)
      aiex.dma_free_task(%b_col0_m1_n7)
      aiex.dma_free_task(%c_col0_m1_n7)
      // Col 0 M-tile 1 N-tile 8
      %a_col0_m1_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n8)
      %b_col0_m1_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n8)
      %c_col0_m1_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 788480 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n8)
      aiex.dma_await_task(%c_col0_m1_n8)
      aiex.dma_free_task(%a_col0_m1_n8)
      aiex.dma_free_task(%b_col0_m1_n8)
      aiex.dma_free_task(%c_col0_m1_n8)
      // Col 0 M-tile 1 N-tile 9
      %a_col0_m1_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n9)
      %b_col0_m1_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n9)
      %c_col0_m1_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 788736 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n9)
      aiex.dma_await_task(%c_col0_m1_n9)
      aiex.dma_free_task(%a_col0_m1_n9)
      aiex.dma_free_task(%b_col0_m1_n9)
      aiex.dma_free_task(%c_col0_m1_n9)
      // Col 0 M-tile 1 N-tile 10
      %a_col0_m1_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n10)
      %b_col0_m1_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n10)
      %c_col0_m1_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 788992 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n10)
      aiex.dma_await_task(%c_col0_m1_n10)
      aiex.dma_free_task(%a_col0_m1_n10)
      aiex.dma_free_task(%b_col0_m1_n10)
      aiex.dma_free_task(%c_col0_m1_n10)
      // Col 0 M-tile 1 N-tile 11
      %a_col0_m1_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n11)
      %b_col0_m1_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n11)
      %c_col0_m1_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 789248 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n11)
      aiex.dma_await_task(%c_col0_m1_n11)
      aiex.dma_free_task(%a_col0_m1_n11)
      aiex.dma_free_task(%b_col0_m1_n11)
      aiex.dma_free_task(%c_col0_m1_n11)
      // Col 0 M-tile 1 N-tile 12
      %a_col0_m1_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n12)
      %b_col0_m1_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n12)
      %c_col0_m1_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 789504 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n12)
      aiex.dma_await_task(%c_col0_m1_n12)
      aiex.dma_free_task(%a_col0_m1_n12)
      aiex.dma_free_task(%b_col0_m1_n12)
      aiex.dma_free_task(%c_col0_m1_n12)
      // Col 0 M-tile 1 N-tile 13
      %a_col0_m1_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n13)
      %b_col0_m1_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n13)
      %c_col0_m1_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 789760 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n13)
      aiex.dma_await_task(%c_col0_m1_n13)
      aiex.dma_free_task(%a_col0_m1_n13)
      aiex.dma_free_task(%b_col0_m1_n13)
      aiex.dma_free_task(%c_col0_m1_n13)
      // Col 0 M-tile 1 N-tile 14
      %a_col0_m1_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n14)
      %b_col0_m1_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n14)
      %c_col0_m1_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 790016 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n14)
      aiex.dma_await_task(%c_col0_m1_n14)
      aiex.dma_free_task(%a_col0_m1_n14)
      aiex.dma_free_task(%b_col0_m1_n14)
      aiex.dma_free_task(%c_col0_m1_n14)
      // Col 0 M-tile 1 N-tile 15
      %a_col0_m1_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n15)
      %b_col0_m1_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n15)
      %c_col0_m1_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 790272 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n15)
      aiex.dma_await_task(%c_col0_m1_n15)
      aiex.dma_free_task(%a_col0_m1_n15)
      aiex.dma_free_task(%b_col0_m1_n15)
      aiex.dma_free_task(%c_col0_m1_n15)
      // Col 0 M-tile 1 N-tile 16
      %a_col0_m1_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n16)
      %b_col0_m1_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n16)
      %c_col0_m1_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 790528 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n16)
      aiex.dma_await_task(%c_col0_m1_n16)
      aiex.dma_free_task(%a_col0_m1_n16)
      aiex.dma_free_task(%b_col0_m1_n16)
      aiex.dma_free_task(%c_col0_m1_n16)
      // Col 0 M-tile 1 N-tile 17
      %a_col0_m1_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n17)
      %b_col0_m1_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n17)
      %c_col0_m1_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 790784 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n17)
      aiex.dma_await_task(%c_col0_m1_n17)
      aiex.dma_free_task(%a_col0_m1_n17)
      aiex.dma_free_task(%b_col0_m1_n17)
      aiex.dma_free_task(%c_col0_m1_n17)
      // Col 0 M-tile 1 N-tile 18
      %a_col0_m1_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n18)
      %b_col0_m1_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n18)
      %c_col0_m1_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 791040 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n18)
      aiex.dma_await_task(%c_col0_m1_n18)
      aiex.dma_free_task(%a_col0_m1_n18)
      aiex.dma_free_task(%b_col0_m1_n18)
      aiex.dma_free_task(%c_col0_m1_n18)
      // Col 0 M-tile 1 N-tile 19
      %a_col0_m1_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n19)
      %b_col0_m1_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n19)
      %c_col0_m1_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 791296 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n19)
      aiex.dma_await_task(%c_col0_m1_n19)
      aiex.dma_free_task(%a_col0_m1_n19)
      aiex.dma_free_task(%b_col0_m1_n19)
      aiex.dma_free_task(%c_col0_m1_n19)
      // Col 0 M-tile 1 N-tile 20
      %a_col0_m1_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n20)
      %b_col0_m1_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n20)
      %c_col0_m1_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 791552 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n20)
      aiex.dma_await_task(%c_col0_m1_n20)
      aiex.dma_free_task(%a_col0_m1_n20)
      aiex.dma_free_task(%b_col0_m1_n20)
      aiex.dma_free_task(%c_col0_m1_n20)
      // Col 0 M-tile 1 N-tile 21
      %a_col0_m1_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n21)
      %b_col0_m1_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n21)
      %c_col0_m1_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 791808 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n21)
      aiex.dma_await_task(%c_col0_m1_n21)
      aiex.dma_free_task(%a_col0_m1_n21)
      aiex.dma_free_task(%b_col0_m1_n21)
      aiex.dma_free_task(%c_col0_m1_n21)
      // Col 0 M-tile 1 N-tile 22
      %a_col0_m1_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n22)
      %b_col0_m1_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n22)
      %c_col0_m1_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 792064 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n22)
      aiex.dma_await_task(%c_col0_m1_n22)
      aiex.dma_free_task(%a_col0_m1_n22)
      aiex.dma_free_task(%b_col0_m1_n22)
      aiex.dma_free_task(%c_col0_m1_n22)
      // Col 0 M-tile 1 N-tile 23
      %a_col0_m1_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n23)
      %b_col0_m1_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n23)
      %c_col0_m1_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 792320 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n23)
      aiex.dma_await_task(%c_col0_m1_n23)
      aiex.dma_free_task(%a_col0_m1_n23)
      aiex.dma_free_task(%b_col0_m1_n23)
      aiex.dma_free_task(%c_col0_m1_n23)
      // Col 0 M-tile 2 N-tile 0
      %a_col0_m2_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n0)
      %b_col0_m2_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n0)
      %c_col0_m2_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1572864 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n0)
      aiex.dma_await_task(%c_col0_m2_n0)
      aiex.dma_free_task(%a_col0_m2_n0)
      aiex.dma_free_task(%b_col0_m2_n0)
      aiex.dma_free_task(%c_col0_m2_n0)
      // Col 0 M-tile 2 N-tile 1
      %a_col0_m2_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n1)
      %b_col0_m2_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n1)
      %c_col0_m2_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1573120 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n1)
      aiex.dma_await_task(%c_col0_m2_n1)
      aiex.dma_free_task(%a_col0_m2_n1)
      aiex.dma_free_task(%b_col0_m2_n1)
      aiex.dma_free_task(%c_col0_m2_n1)
      // Col 0 M-tile 2 N-tile 2
      %a_col0_m2_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n2)
      %b_col0_m2_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n2)
      %c_col0_m2_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1573376 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n2)
      aiex.dma_await_task(%c_col0_m2_n2)
      aiex.dma_free_task(%a_col0_m2_n2)
      aiex.dma_free_task(%b_col0_m2_n2)
      aiex.dma_free_task(%c_col0_m2_n2)
      // Col 0 M-tile 2 N-tile 3
      %a_col0_m2_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n3)
      %b_col0_m2_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n3)
      %c_col0_m2_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1573632 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n3)
      aiex.dma_await_task(%c_col0_m2_n3)
      aiex.dma_free_task(%a_col0_m2_n3)
      aiex.dma_free_task(%b_col0_m2_n3)
      aiex.dma_free_task(%c_col0_m2_n3)
      // Col 0 M-tile 2 N-tile 4
      %a_col0_m2_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n4)
      %b_col0_m2_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n4)
      %c_col0_m2_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1573888 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n4)
      aiex.dma_await_task(%c_col0_m2_n4)
      aiex.dma_free_task(%a_col0_m2_n4)
      aiex.dma_free_task(%b_col0_m2_n4)
      aiex.dma_free_task(%c_col0_m2_n4)
      // Col 0 M-tile 2 N-tile 5
      %a_col0_m2_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n5)
      %b_col0_m2_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n5)
      %c_col0_m2_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1574144 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n5)
      aiex.dma_await_task(%c_col0_m2_n5)
      aiex.dma_free_task(%a_col0_m2_n5)
      aiex.dma_free_task(%b_col0_m2_n5)
      aiex.dma_free_task(%c_col0_m2_n5)
      // Col 0 M-tile 2 N-tile 6
      %a_col0_m2_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n6)
      %b_col0_m2_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n6)
      %c_col0_m2_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1574400 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n6)
      aiex.dma_await_task(%c_col0_m2_n6)
      aiex.dma_free_task(%a_col0_m2_n6)
      aiex.dma_free_task(%b_col0_m2_n6)
      aiex.dma_free_task(%c_col0_m2_n6)
      // Col 0 M-tile 2 N-tile 7
      %a_col0_m2_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n7)
      %b_col0_m2_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n7)
      %c_col0_m2_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1574656 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n7)
      aiex.dma_await_task(%c_col0_m2_n7)
      aiex.dma_free_task(%a_col0_m2_n7)
      aiex.dma_free_task(%b_col0_m2_n7)
      aiex.dma_free_task(%c_col0_m2_n7)
      // Col 0 M-tile 2 N-tile 8
      %a_col0_m2_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n8)
      %b_col0_m2_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n8)
      %c_col0_m2_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1574912 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n8)
      aiex.dma_await_task(%c_col0_m2_n8)
      aiex.dma_free_task(%a_col0_m2_n8)
      aiex.dma_free_task(%b_col0_m2_n8)
      aiex.dma_free_task(%c_col0_m2_n8)
      // Col 0 M-tile 2 N-tile 9
      %a_col0_m2_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n9)
      %b_col0_m2_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n9)
      %c_col0_m2_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1575168 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n9)
      aiex.dma_await_task(%c_col0_m2_n9)
      aiex.dma_free_task(%a_col0_m2_n9)
      aiex.dma_free_task(%b_col0_m2_n9)
      aiex.dma_free_task(%c_col0_m2_n9)
      // Col 0 M-tile 2 N-tile 10
      %a_col0_m2_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n10)
      %b_col0_m2_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n10)
      %c_col0_m2_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1575424 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n10)
      aiex.dma_await_task(%c_col0_m2_n10)
      aiex.dma_free_task(%a_col0_m2_n10)
      aiex.dma_free_task(%b_col0_m2_n10)
      aiex.dma_free_task(%c_col0_m2_n10)
      // Col 0 M-tile 2 N-tile 11
      %a_col0_m2_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n11)
      %b_col0_m2_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n11)
      %c_col0_m2_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1575680 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n11)
      aiex.dma_await_task(%c_col0_m2_n11)
      aiex.dma_free_task(%a_col0_m2_n11)
      aiex.dma_free_task(%b_col0_m2_n11)
      aiex.dma_free_task(%c_col0_m2_n11)
      // Col 0 M-tile 2 N-tile 12
      %a_col0_m2_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n12)
      %b_col0_m2_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n12)
      %c_col0_m2_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1575936 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n12)
      aiex.dma_await_task(%c_col0_m2_n12)
      aiex.dma_free_task(%a_col0_m2_n12)
      aiex.dma_free_task(%b_col0_m2_n12)
      aiex.dma_free_task(%c_col0_m2_n12)
      // Col 0 M-tile 2 N-tile 13
      %a_col0_m2_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n13)
      %b_col0_m2_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n13)
      %c_col0_m2_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1576192 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n13)
      aiex.dma_await_task(%c_col0_m2_n13)
      aiex.dma_free_task(%a_col0_m2_n13)
      aiex.dma_free_task(%b_col0_m2_n13)
      aiex.dma_free_task(%c_col0_m2_n13)
      // Col 0 M-tile 2 N-tile 14
      %a_col0_m2_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n14)
      %b_col0_m2_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n14)
      %c_col0_m2_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1576448 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n14)
      aiex.dma_await_task(%c_col0_m2_n14)
      aiex.dma_free_task(%a_col0_m2_n14)
      aiex.dma_free_task(%b_col0_m2_n14)
      aiex.dma_free_task(%c_col0_m2_n14)
      // Col 0 M-tile 2 N-tile 15
      %a_col0_m2_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n15)
      %b_col0_m2_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n15)
      %c_col0_m2_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1576704 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n15)
      aiex.dma_await_task(%c_col0_m2_n15)
      aiex.dma_free_task(%a_col0_m2_n15)
      aiex.dma_free_task(%b_col0_m2_n15)
      aiex.dma_free_task(%c_col0_m2_n15)
      // Col 0 M-tile 2 N-tile 16
      %a_col0_m2_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n16)
      %b_col0_m2_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n16)
      %c_col0_m2_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1576960 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n16)
      aiex.dma_await_task(%c_col0_m2_n16)
      aiex.dma_free_task(%a_col0_m2_n16)
      aiex.dma_free_task(%b_col0_m2_n16)
      aiex.dma_free_task(%c_col0_m2_n16)
      // Col 0 M-tile 2 N-tile 17
      %a_col0_m2_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n17)
      %b_col0_m2_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n17)
      %c_col0_m2_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1577216 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n17)
      aiex.dma_await_task(%c_col0_m2_n17)
      aiex.dma_free_task(%a_col0_m2_n17)
      aiex.dma_free_task(%b_col0_m2_n17)
      aiex.dma_free_task(%c_col0_m2_n17)
      // Col 0 M-tile 2 N-tile 18
      %a_col0_m2_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n18)
      %b_col0_m2_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n18)
      %c_col0_m2_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1577472 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n18)
      aiex.dma_await_task(%c_col0_m2_n18)
      aiex.dma_free_task(%a_col0_m2_n18)
      aiex.dma_free_task(%b_col0_m2_n18)
      aiex.dma_free_task(%c_col0_m2_n18)
      // Col 0 M-tile 2 N-tile 19
      %a_col0_m2_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n19)
      %b_col0_m2_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n19)
      %c_col0_m2_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1577728 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n19)
      aiex.dma_await_task(%c_col0_m2_n19)
      aiex.dma_free_task(%a_col0_m2_n19)
      aiex.dma_free_task(%b_col0_m2_n19)
      aiex.dma_free_task(%c_col0_m2_n19)
      // Col 0 M-tile 2 N-tile 20
      %a_col0_m2_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n20)
      %b_col0_m2_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n20)
      %c_col0_m2_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1577984 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n20)
      aiex.dma_await_task(%c_col0_m2_n20)
      aiex.dma_free_task(%a_col0_m2_n20)
      aiex.dma_free_task(%b_col0_m2_n20)
      aiex.dma_free_task(%c_col0_m2_n20)
      // Col 0 M-tile 2 N-tile 21
      %a_col0_m2_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n21)
      %b_col0_m2_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n21)
      %c_col0_m2_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1578240 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n21)
      aiex.dma_await_task(%c_col0_m2_n21)
      aiex.dma_free_task(%a_col0_m2_n21)
      aiex.dma_free_task(%b_col0_m2_n21)
      aiex.dma_free_task(%c_col0_m2_n21)
      // Col 0 M-tile 2 N-tile 22
      %a_col0_m2_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n22)
      %b_col0_m2_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n22)
      %c_col0_m2_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1578496 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n22)
      aiex.dma_await_task(%c_col0_m2_n22)
      aiex.dma_free_task(%a_col0_m2_n22)
      aiex.dma_free_task(%b_col0_m2_n22)
      aiex.dma_free_task(%c_col0_m2_n22)
      // Col 0 M-tile 2 N-tile 23
      %a_col0_m2_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n23)
      %b_col0_m2_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n23)
      %c_col0_m2_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1578752 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n23)
      aiex.dma_await_task(%c_col0_m2_n23)
      aiex.dma_free_task(%a_col0_m2_n23)
      aiex.dma_free_task(%b_col0_m2_n23)
      aiex.dma_free_task(%c_col0_m2_n23)
      // Col 0 M-tile 3 N-tile 0
      %a_col0_m3_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n0)
      %b_col0_m3_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n0)
      %c_col0_m3_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2359296 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n0)
      aiex.dma_await_task(%c_col0_m3_n0)
      aiex.dma_free_task(%a_col0_m3_n0)
      aiex.dma_free_task(%b_col0_m3_n0)
      aiex.dma_free_task(%c_col0_m3_n0)
      // Col 0 M-tile 3 N-tile 1
      %a_col0_m3_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n1)
      %b_col0_m3_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n1)
      %c_col0_m3_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2359552 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n1)
      aiex.dma_await_task(%c_col0_m3_n1)
      aiex.dma_free_task(%a_col0_m3_n1)
      aiex.dma_free_task(%b_col0_m3_n1)
      aiex.dma_free_task(%c_col0_m3_n1)
      // Col 0 M-tile 3 N-tile 2
      %a_col0_m3_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n2)
      %b_col0_m3_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n2)
      %c_col0_m3_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2359808 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n2)
      aiex.dma_await_task(%c_col0_m3_n2)
      aiex.dma_free_task(%a_col0_m3_n2)
      aiex.dma_free_task(%b_col0_m3_n2)
      aiex.dma_free_task(%c_col0_m3_n2)
      // Col 0 M-tile 3 N-tile 3
      %a_col0_m3_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n3)
      %b_col0_m3_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n3)
      %c_col0_m3_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2360064 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n3)
      aiex.dma_await_task(%c_col0_m3_n3)
      aiex.dma_free_task(%a_col0_m3_n3)
      aiex.dma_free_task(%b_col0_m3_n3)
      aiex.dma_free_task(%c_col0_m3_n3)
      // Col 0 M-tile 3 N-tile 4
      %a_col0_m3_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n4)
      %b_col0_m3_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n4)
      %c_col0_m3_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2360320 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n4)
      aiex.dma_await_task(%c_col0_m3_n4)
      aiex.dma_free_task(%a_col0_m3_n4)
      aiex.dma_free_task(%b_col0_m3_n4)
      aiex.dma_free_task(%c_col0_m3_n4)
      // Col 0 M-tile 3 N-tile 5
      %a_col0_m3_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n5)
      %b_col0_m3_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n5)
      %c_col0_m3_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2360576 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n5)
      aiex.dma_await_task(%c_col0_m3_n5)
      aiex.dma_free_task(%a_col0_m3_n5)
      aiex.dma_free_task(%b_col0_m3_n5)
      aiex.dma_free_task(%c_col0_m3_n5)
      // Col 0 M-tile 3 N-tile 6
      %a_col0_m3_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n6)
      %b_col0_m3_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n6)
      %c_col0_m3_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2360832 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n6)
      aiex.dma_await_task(%c_col0_m3_n6)
      aiex.dma_free_task(%a_col0_m3_n6)
      aiex.dma_free_task(%b_col0_m3_n6)
      aiex.dma_free_task(%c_col0_m3_n6)
      // Col 0 M-tile 3 N-tile 7
      %a_col0_m3_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n7)
      %b_col0_m3_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n7)
      %c_col0_m3_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2361088 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n7)
      aiex.dma_await_task(%c_col0_m3_n7)
      aiex.dma_free_task(%a_col0_m3_n7)
      aiex.dma_free_task(%b_col0_m3_n7)
      aiex.dma_free_task(%c_col0_m3_n7)
      // Col 0 M-tile 3 N-tile 8
      %a_col0_m3_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n8)
      %b_col0_m3_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n8)
      %c_col0_m3_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2361344 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n8)
      aiex.dma_await_task(%c_col0_m3_n8)
      aiex.dma_free_task(%a_col0_m3_n8)
      aiex.dma_free_task(%b_col0_m3_n8)
      aiex.dma_free_task(%c_col0_m3_n8)
      // Col 0 M-tile 3 N-tile 9
      %a_col0_m3_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n9)
      %b_col0_m3_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n9)
      %c_col0_m3_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2361600 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n9)
      aiex.dma_await_task(%c_col0_m3_n9)
      aiex.dma_free_task(%a_col0_m3_n9)
      aiex.dma_free_task(%b_col0_m3_n9)
      aiex.dma_free_task(%c_col0_m3_n9)
      // Col 0 M-tile 3 N-tile 10
      %a_col0_m3_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n10)
      %b_col0_m3_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n10)
      %c_col0_m3_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2361856 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n10)
      aiex.dma_await_task(%c_col0_m3_n10)
      aiex.dma_free_task(%a_col0_m3_n10)
      aiex.dma_free_task(%b_col0_m3_n10)
      aiex.dma_free_task(%c_col0_m3_n10)
      // Col 0 M-tile 3 N-tile 11
      %a_col0_m3_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n11)
      %b_col0_m3_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n11)
      %c_col0_m3_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2362112 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n11)
      aiex.dma_await_task(%c_col0_m3_n11)
      aiex.dma_free_task(%a_col0_m3_n11)
      aiex.dma_free_task(%b_col0_m3_n11)
      aiex.dma_free_task(%c_col0_m3_n11)
      // Col 0 M-tile 3 N-tile 12
      %a_col0_m3_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n12)
      %b_col0_m3_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n12)
      %c_col0_m3_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2362368 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n12)
      aiex.dma_await_task(%c_col0_m3_n12)
      aiex.dma_free_task(%a_col0_m3_n12)
      aiex.dma_free_task(%b_col0_m3_n12)
      aiex.dma_free_task(%c_col0_m3_n12)
      // Col 0 M-tile 3 N-tile 13
      %a_col0_m3_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n13)
      %b_col0_m3_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n13)
      %c_col0_m3_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2362624 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n13)
      aiex.dma_await_task(%c_col0_m3_n13)
      aiex.dma_free_task(%a_col0_m3_n13)
      aiex.dma_free_task(%b_col0_m3_n13)
      aiex.dma_free_task(%c_col0_m3_n13)
      // Col 0 M-tile 3 N-tile 14
      %a_col0_m3_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n14)
      %b_col0_m3_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n14)
      %c_col0_m3_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2362880 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n14)
      aiex.dma_await_task(%c_col0_m3_n14)
      aiex.dma_free_task(%a_col0_m3_n14)
      aiex.dma_free_task(%b_col0_m3_n14)
      aiex.dma_free_task(%c_col0_m3_n14)
      // Col 0 M-tile 3 N-tile 15
      %a_col0_m3_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n15)
      %b_col0_m3_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n15)
      %c_col0_m3_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2363136 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n15)
      aiex.dma_await_task(%c_col0_m3_n15)
      aiex.dma_free_task(%a_col0_m3_n15)
      aiex.dma_free_task(%b_col0_m3_n15)
      aiex.dma_free_task(%c_col0_m3_n15)
      // Col 0 M-tile 3 N-tile 16
      %a_col0_m3_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n16)
      %b_col0_m3_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n16)
      %c_col0_m3_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2363392 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n16)
      aiex.dma_await_task(%c_col0_m3_n16)
      aiex.dma_free_task(%a_col0_m3_n16)
      aiex.dma_free_task(%b_col0_m3_n16)
      aiex.dma_free_task(%c_col0_m3_n16)
      // Col 0 M-tile 3 N-tile 17
      %a_col0_m3_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n17)
      %b_col0_m3_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n17)
      %c_col0_m3_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2363648 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n17)
      aiex.dma_await_task(%c_col0_m3_n17)
      aiex.dma_free_task(%a_col0_m3_n17)
      aiex.dma_free_task(%b_col0_m3_n17)
      aiex.dma_free_task(%c_col0_m3_n17)
      // Col 0 M-tile 3 N-tile 18
      %a_col0_m3_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n18)
      %b_col0_m3_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n18)
      %c_col0_m3_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2363904 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n18)
      aiex.dma_await_task(%c_col0_m3_n18)
      aiex.dma_free_task(%a_col0_m3_n18)
      aiex.dma_free_task(%b_col0_m3_n18)
      aiex.dma_free_task(%c_col0_m3_n18)
      // Col 0 M-tile 3 N-tile 19
      %a_col0_m3_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n19)
      %b_col0_m3_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n19)
      %c_col0_m3_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2364160 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n19)
      aiex.dma_await_task(%c_col0_m3_n19)
      aiex.dma_free_task(%a_col0_m3_n19)
      aiex.dma_free_task(%b_col0_m3_n19)
      aiex.dma_free_task(%c_col0_m3_n19)
      // Col 0 M-tile 3 N-tile 20
      %a_col0_m3_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n20)
      %b_col0_m3_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n20)
      %c_col0_m3_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2364416 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n20)
      aiex.dma_await_task(%c_col0_m3_n20)
      aiex.dma_free_task(%a_col0_m3_n20)
      aiex.dma_free_task(%b_col0_m3_n20)
      aiex.dma_free_task(%c_col0_m3_n20)
      // Col 0 M-tile 3 N-tile 21
      %a_col0_m3_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n21)
      %b_col0_m3_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n21)
      %c_col0_m3_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2364672 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n21)
      aiex.dma_await_task(%c_col0_m3_n21)
      aiex.dma_free_task(%a_col0_m3_n21)
      aiex.dma_free_task(%b_col0_m3_n21)
      aiex.dma_free_task(%c_col0_m3_n21)
      // Col 0 M-tile 3 N-tile 22
      %a_col0_m3_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n22)
      %b_col0_m3_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n22)
      %c_col0_m3_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2364928 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n22)
      aiex.dma_await_task(%c_col0_m3_n22)
      aiex.dma_free_task(%a_col0_m3_n22)
      aiex.dma_free_task(%b_col0_m3_n22)
      aiex.dma_free_task(%c_col0_m3_n22)
      // Col 0 M-tile 3 N-tile 23
      %a_col0_m3_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n23)
      %b_col0_m3_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n23)
      %c_col0_m3_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2365184 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n23)
      aiex.dma_await_task(%c_col0_m3_n23)
      aiex.dma_free_task(%a_col0_m3_n23)
      aiex.dma_free_task(%b_col0_m3_n23)
      aiex.dma_free_task(%c_col0_m3_n23)
      // Col 0 M-tile 4 N-tile 0
      %a_col0_m4_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n0)
      %b_col0_m4_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n0)
      %c_col0_m4_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3145728 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n0)
      aiex.dma_await_task(%c_col0_m4_n0)
      aiex.dma_free_task(%a_col0_m4_n0)
      aiex.dma_free_task(%b_col0_m4_n0)
      aiex.dma_free_task(%c_col0_m4_n0)
      // Col 0 M-tile 4 N-tile 1
      %a_col0_m4_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n1)
      %b_col0_m4_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n1)
      %c_col0_m4_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3145984 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n1)
      aiex.dma_await_task(%c_col0_m4_n1)
      aiex.dma_free_task(%a_col0_m4_n1)
      aiex.dma_free_task(%b_col0_m4_n1)
      aiex.dma_free_task(%c_col0_m4_n1)
      // Col 0 M-tile 4 N-tile 2
      %a_col0_m4_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n2)
      %b_col0_m4_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n2)
      %c_col0_m4_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3146240 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n2)
      aiex.dma_await_task(%c_col0_m4_n2)
      aiex.dma_free_task(%a_col0_m4_n2)
      aiex.dma_free_task(%b_col0_m4_n2)
      aiex.dma_free_task(%c_col0_m4_n2)
      // Col 0 M-tile 4 N-tile 3
      %a_col0_m4_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n3)
      %b_col0_m4_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n3)
      %c_col0_m4_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3146496 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n3)
      aiex.dma_await_task(%c_col0_m4_n3)
      aiex.dma_free_task(%a_col0_m4_n3)
      aiex.dma_free_task(%b_col0_m4_n3)
      aiex.dma_free_task(%c_col0_m4_n3)
      // Col 0 M-tile 4 N-tile 4
      %a_col0_m4_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n4)
      %b_col0_m4_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n4)
      %c_col0_m4_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3146752 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n4)
      aiex.dma_await_task(%c_col0_m4_n4)
      aiex.dma_free_task(%a_col0_m4_n4)
      aiex.dma_free_task(%b_col0_m4_n4)
      aiex.dma_free_task(%c_col0_m4_n4)
      // Col 0 M-tile 4 N-tile 5
      %a_col0_m4_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n5)
      %b_col0_m4_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n5)
      %c_col0_m4_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3147008 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n5)
      aiex.dma_await_task(%c_col0_m4_n5)
      aiex.dma_free_task(%a_col0_m4_n5)
      aiex.dma_free_task(%b_col0_m4_n5)
      aiex.dma_free_task(%c_col0_m4_n5)
      // Col 0 M-tile 4 N-tile 6
      %a_col0_m4_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n6)
      %b_col0_m4_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n6)
      %c_col0_m4_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3147264 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n6)
      aiex.dma_await_task(%c_col0_m4_n6)
      aiex.dma_free_task(%a_col0_m4_n6)
      aiex.dma_free_task(%b_col0_m4_n6)
      aiex.dma_free_task(%c_col0_m4_n6)
      // Col 0 M-tile 4 N-tile 7
      %a_col0_m4_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n7)
      %b_col0_m4_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n7)
      %c_col0_m4_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3147520 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n7)
      aiex.dma_await_task(%c_col0_m4_n7)
      aiex.dma_free_task(%a_col0_m4_n7)
      aiex.dma_free_task(%b_col0_m4_n7)
      aiex.dma_free_task(%c_col0_m4_n7)
      // Col 0 M-tile 4 N-tile 8
      %a_col0_m4_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n8)
      %b_col0_m4_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n8)
      %c_col0_m4_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3147776 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n8)
      aiex.dma_await_task(%c_col0_m4_n8)
      aiex.dma_free_task(%a_col0_m4_n8)
      aiex.dma_free_task(%b_col0_m4_n8)
      aiex.dma_free_task(%c_col0_m4_n8)
      // Col 0 M-tile 4 N-tile 9
      %a_col0_m4_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n9)
      %b_col0_m4_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n9)
      %c_col0_m4_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3148032 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n9)
      aiex.dma_await_task(%c_col0_m4_n9)
      aiex.dma_free_task(%a_col0_m4_n9)
      aiex.dma_free_task(%b_col0_m4_n9)
      aiex.dma_free_task(%c_col0_m4_n9)
      // Col 0 M-tile 4 N-tile 10
      %a_col0_m4_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n10)
      %b_col0_m4_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n10)
      %c_col0_m4_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3148288 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n10)
      aiex.dma_await_task(%c_col0_m4_n10)
      aiex.dma_free_task(%a_col0_m4_n10)
      aiex.dma_free_task(%b_col0_m4_n10)
      aiex.dma_free_task(%c_col0_m4_n10)
      // Col 0 M-tile 4 N-tile 11
      %a_col0_m4_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n11)
      %b_col0_m4_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n11)
      %c_col0_m4_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3148544 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n11)
      aiex.dma_await_task(%c_col0_m4_n11)
      aiex.dma_free_task(%a_col0_m4_n11)
      aiex.dma_free_task(%b_col0_m4_n11)
      aiex.dma_free_task(%c_col0_m4_n11)
      // Col 0 M-tile 4 N-tile 12
      %a_col0_m4_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n12)
      %b_col0_m4_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n12)
      %c_col0_m4_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3148800 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n12)
      aiex.dma_await_task(%c_col0_m4_n12)
      aiex.dma_free_task(%a_col0_m4_n12)
      aiex.dma_free_task(%b_col0_m4_n12)
      aiex.dma_free_task(%c_col0_m4_n12)
      // Col 0 M-tile 4 N-tile 13
      %a_col0_m4_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n13)
      %b_col0_m4_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n13)
      %c_col0_m4_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3149056 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n13)
      aiex.dma_await_task(%c_col0_m4_n13)
      aiex.dma_free_task(%a_col0_m4_n13)
      aiex.dma_free_task(%b_col0_m4_n13)
      aiex.dma_free_task(%c_col0_m4_n13)
      // Col 0 M-tile 4 N-tile 14
      %a_col0_m4_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n14)
      %b_col0_m4_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n14)
      %c_col0_m4_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3149312 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n14)
      aiex.dma_await_task(%c_col0_m4_n14)
      aiex.dma_free_task(%a_col0_m4_n14)
      aiex.dma_free_task(%b_col0_m4_n14)
      aiex.dma_free_task(%c_col0_m4_n14)
      // Col 0 M-tile 4 N-tile 15
      %a_col0_m4_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n15)
      %b_col0_m4_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n15)
      %c_col0_m4_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3149568 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n15)
      aiex.dma_await_task(%c_col0_m4_n15)
      aiex.dma_free_task(%a_col0_m4_n15)
      aiex.dma_free_task(%b_col0_m4_n15)
      aiex.dma_free_task(%c_col0_m4_n15)
      // Col 0 M-tile 4 N-tile 16
      %a_col0_m4_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n16)
      %b_col0_m4_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n16)
      %c_col0_m4_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3149824 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n16)
      aiex.dma_await_task(%c_col0_m4_n16)
      aiex.dma_free_task(%a_col0_m4_n16)
      aiex.dma_free_task(%b_col0_m4_n16)
      aiex.dma_free_task(%c_col0_m4_n16)
      // Col 0 M-tile 4 N-tile 17
      %a_col0_m4_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n17)
      %b_col0_m4_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n17)
      %c_col0_m4_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3150080 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n17)
      aiex.dma_await_task(%c_col0_m4_n17)
      aiex.dma_free_task(%a_col0_m4_n17)
      aiex.dma_free_task(%b_col0_m4_n17)
      aiex.dma_free_task(%c_col0_m4_n17)
      // Col 0 M-tile 4 N-tile 18
      %a_col0_m4_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n18)
      %b_col0_m4_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n18)
      %c_col0_m4_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3150336 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n18)
      aiex.dma_await_task(%c_col0_m4_n18)
      aiex.dma_free_task(%a_col0_m4_n18)
      aiex.dma_free_task(%b_col0_m4_n18)
      aiex.dma_free_task(%c_col0_m4_n18)
      // Col 0 M-tile 4 N-tile 19
      %a_col0_m4_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n19)
      %b_col0_m4_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n19)
      %c_col0_m4_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3150592 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n19)
      aiex.dma_await_task(%c_col0_m4_n19)
      aiex.dma_free_task(%a_col0_m4_n19)
      aiex.dma_free_task(%b_col0_m4_n19)
      aiex.dma_free_task(%c_col0_m4_n19)
      // Col 0 M-tile 4 N-tile 20
      %a_col0_m4_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n20)
      %b_col0_m4_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n20)
      %c_col0_m4_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3150848 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n20)
      aiex.dma_await_task(%c_col0_m4_n20)
      aiex.dma_free_task(%a_col0_m4_n20)
      aiex.dma_free_task(%b_col0_m4_n20)
      aiex.dma_free_task(%c_col0_m4_n20)
      // Col 0 M-tile 4 N-tile 21
      %a_col0_m4_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n21)
      %b_col0_m4_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n21)
      %c_col0_m4_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3151104 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n21)
      aiex.dma_await_task(%c_col0_m4_n21)
      aiex.dma_free_task(%a_col0_m4_n21)
      aiex.dma_free_task(%b_col0_m4_n21)
      aiex.dma_free_task(%c_col0_m4_n21)
      // Col 0 M-tile 4 N-tile 22
      %a_col0_m4_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n22)
      %b_col0_m4_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n22)
      %c_col0_m4_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3151360 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n22)
      aiex.dma_await_task(%c_col0_m4_n22)
      aiex.dma_free_task(%a_col0_m4_n22)
      aiex.dma_free_task(%b_col0_m4_n22)
      aiex.dma_free_task(%c_col0_m4_n22)
      // Col 0 M-tile 4 N-tile 23
      %a_col0_m4_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n23)
      %b_col0_m4_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n23)
      %c_col0_m4_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3151616 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n23)
      aiex.dma_await_task(%c_col0_m4_n23)
      aiex.dma_free_task(%a_col0_m4_n23)
      aiex.dma_free_task(%b_col0_m4_n23)
      aiex.dma_free_task(%c_col0_m4_n23)
      // Col 0 M-tile 5 N-tile 0
      %a_col0_m5_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n0)
      %b_col0_m5_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n0)
      %c_col0_m5_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3932160 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n0)
      aiex.dma_await_task(%c_col0_m5_n0)
      aiex.dma_free_task(%a_col0_m5_n0)
      aiex.dma_free_task(%b_col0_m5_n0)
      aiex.dma_free_task(%c_col0_m5_n0)
      // Col 0 M-tile 5 N-tile 1
      %a_col0_m5_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n1)
      %b_col0_m5_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n1)
      %c_col0_m5_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3932416 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n1)
      aiex.dma_await_task(%c_col0_m5_n1)
      aiex.dma_free_task(%a_col0_m5_n1)
      aiex.dma_free_task(%b_col0_m5_n1)
      aiex.dma_free_task(%c_col0_m5_n1)
      // Col 0 M-tile 5 N-tile 2
      %a_col0_m5_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n2)
      %b_col0_m5_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n2)
      %c_col0_m5_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3932672 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n2)
      aiex.dma_await_task(%c_col0_m5_n2)
      aiex.dma_free_task(%a_col0_m5_n2)
      aiex.dma_free_task(%b_col0_m5_n2)
      aiex.dma_free_task(%c_col0_m5_n2)
      // Col 0 M-tile 5 N-tile 3
      %a_col0_m5_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n3)
      %b_col0_m5_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n3)
      %c_col0_m5_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3932928 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n3)
      aiex.dma_await_task(%c_col0_m5_n3)
      aiex.dma_free_task(%a_col0_m5_n3)
      aiex.dma_free_task(%b_col0_m5_n3)
      aiex.dma_free_task(%c_col0_m5_n3)
      // Col 0 M-tile 5 N-tile 4
      %a_col0_m5_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n4)
      %b_col0_m5_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n4)
      %c_col0_m5_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3933184 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n4)
      aiex.dma_await_task(%c_col0_m5_n4)
      aiex.dma_free_task(%a_col0_m5_n4)
      aiex.dma_free_task(%b_col0_m5_n4)
      aiex.dma_free_task(%c_col0_m5_n4)
      // Col 0 M-tile 5 N-tile 5
      %a_col0_m5_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n5)
      %b_col0_m5_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n5)
      %c_col0_m5_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3933440 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n5)
      aiex.dma_await_task(%c_col0_m5_n5)
      aiex.dma_free_task(%a_col0_m5_n5)
      aiex.dma_free_task(%b_col0_m5_n5)
      aiex.dma_free_task(%c_col0_m5_n5)
      // Col 0 M-tile 5 N-tile 6
      %a_col0_m5_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n6)
      %b_col0_m5_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n6)
      %c_col0_m5_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3933696 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n6)
      aiex.dma_await_task(%c_col0_m5_n6)
      aiex.dma_free_task(%a_col0_m5_n6)
      aiex.dma_free_task(%b_col0_m5_n6)
      aiex.dma_free_task(%c_col0_m5_n6)
      // Col 0 M-tile 5 N-tile 7
      %a_col0_m5_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n7)
      %b_col0_m5_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n7)
      %c_col0_m5_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3933952 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n7)
      aiex.dma_await_task(%c_col0_m5_n7)
      aiex.dma_free_task(%a_col0_m5_n7)
      aiex.dma_free_task(%b_col0_m5_n7)
      aiex.dma_free_task(%c_col0_m5_n7)
      // Col 0 M-tile 5 N-tile 8
      %a_col0_m5_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n8)
      %b_col0_m5_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n8)
      %c_col0_m5_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3934208 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n8)
      aiex.dma_await_task(%c_col0_m5_n8)
      aiex.dma_free_task(%a_col0_m5_n8)
      aiex.dma_free_task(%b_col0_m5_n8)
      aiex.dma_free_task(%c_col0_m5_n8)
      // Col 0 M-tile 5 N-tile 9
      %a_col0_m5_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n9)
      %b_col0_m5_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n9)
      %c_col0_m5_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3934464 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n9)
      aiex.dma_await_task(%c_col0_m5_n9)
      aiex.dma_free_task(%a_col0_m5_n9)
      aiex.dma_free_task(%b_col0_m5_n9)
      aiex.dma_free_task(%c_col0_m5_n9)
      // Col 0 M-tile 5 N-tile 10
      %a_col0_m5_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n10)
      %b_col0_m5_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n10)
      %c_col0_m5_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3934720 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n10)
      aiex.dma_await_task(%c_col0_m5_n10)
      aiex.dma_free_task(%a_col0_m5_n10)
      aiex.dma_free_task(%b_col0_m5_n10)
      aiex.dma_free_task(%c_col0_m5_n10)
      // Col 0 M-tile 5 N-tile 11
      %a_col0_m5_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n11)
      %b_col0_m5_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n11)
      %c_col0_m5_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3934976 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n11)
      aiex.dma_await_task(%c_col0_m5_n11)
      aiex.dma_free_task(%a_col0_m5_n11)
      aiex.dma_free_task(%b_col0_m5_n11)
      aiex.dma_free_task(%c_col0_m5_n11)
      // Col 0 M-tile 5 N-tile 12
      %a_col0_m5_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n12)
      %b_col0_m5_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n12)
      %c_col0_m5_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3935232 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n12)
      aiex.dma_await_task(%c_col0_m5_n12)
      aiex.dma_free_task(%a_col0_m5_n12)
      aiex.dma_free_task(%b_col0_m5_n12)
      aiex.dma_free_task(%c_col0_m5_n12)
      // Col 0 M-tile 5 N-tile 13
      %a_col0_m5_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n13)
      %b_col0_m5_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n13)
      %c_col0_m5_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3935488 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n13)
      aiex.dma_await_task(%c_col0_m5_n13)
      aiex.dma_free_task(%a_col0_m5_n13)
      aiex.dma_free_task(%b_col0_m5_n13)
      aiex.dma_free_task(%c_col0_m5_n13)
      // Col 0 M-tile 5 N-tile 14
      %a_col0_m5_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n14)
      %b_col0_m5_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n14)
      %c_col0_m5_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3935744 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n14)
      aiex.dma_await_task(%c_col0_m5_n14)
      aiex.dma_free_task(%a_col0_m5_n14)
      aiex.dma_free_task(%b_col0_m5_n14)
      aiex.dma_free_task(%c_col0_m5_n14)
      // Col 0 M-tile 5 N-tile 15
      %a_col0_m5_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n15)
      %b_col0_m5_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n15)
      %c_col0_m5_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3936000 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n15)
      aiex.dma_await_task(%c_col0_m5_n15)
      aiex.dma_free_task(%a_col0_m5_n15)
      aiex.dma_free_task(%b_col0_m5_n15)
      aiex.dma_free_task(%c_col0_m5_n15)
      // Col 0 M-tile 5 N-tile 16
      %a_col0_m5_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n16)
      %b_col0_m5_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n16)
      %c_col0_m5_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3936256 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n16)
      aiex.dma_await_task(%c_col0_m5_n16)
      aiex.dma_free_task(%a_col0_m5_n16)
      aiex.dma_free_task(%b_col0_m5_n16)
      aiex.dma_free_task(%c_col0_m5_n16)
      // Col 0 M-tile 5 N-tile 17
      %a_col0_m5_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n17)
      %b_col0_m5_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n17)
      %c_col0_m5_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3936512 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n17)
      aiex.dma_await_task(%c_col0_m5_n17)
      aiex.dma_free_task(%a_col0_m5_n17)
      aiex.dma_free_task(%b_col0_m5_n17)
      aiex.dma_free_task(%c_col0_m5_n17)
      // Col 0 M-tile 5 N-tile 18
      %a_col0_m5_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n18)
      %b_col0_m5_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n18)
      %c_col0_m5_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3936768 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n18)
      aiex.dma_await_task(%c_col0_m5_n18)
      aiex.dma_free_task(%a_col0_m5_n18)
      aiex.dma_free_task(%b_col0_m5_n18)
      aiex.dma_free_task(%c_col0_m5_n18)
      // Col 0 M-tile 5 N-tile 19
      %a_col0_m5_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n19)
      %b_col0_m5_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n19)
      %c_col0_m5_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3937024 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n19)
      aiex.dma_await_task(%c_col0_m5_n19)
      aiex.dma_free_task(%a_col0_m5_n19)
      aiex.dma_free_task(%b_col0_m5_n19)
      aiex.dma_free_task(%c_col0_m5_n19)
      // Col 0 M-tile 5 N-tile 20
      %a_col0_m5_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n20)
      %b_col0_m5_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n20)
      %c_col0_m5_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3937280 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n20)
      aiex.dma_await_task(%c_col0_m5_n20)
      aiex.dma_free_task(%a_col0_m5_n20)
      aiex.dma_free_task(%b_col0_m5_n20)
      aiex.dma_free_task(%c_col0_m5_n20)
      // Col 0 M-tile 5 N-tile 21
      %a_col0_m5_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n21)
      %b_col0_m5_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n21)
      %c_col0_m5_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3937536 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n21)
      aiex.dma_await_task(%c_col0_m5_n21)
      aiex.dma_free_task(%a_col0_m5_n21)
      aiex.dma_free_task(%b_col0_m5_n21)
      aiex.dma_free_task(%c_col0_m5_n21)
      // Col 0 M-tile 5 N-tile 22
      %a_col0_m5_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n22)
      %b_col0_m5_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n22)
      %c_col0_m5_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3937792 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n22)
      aiex.dma_await_task(%c_col0_m5_n22)
      aiex.dma_free_task(%a_col0_m5_n22)
      aiex.dma_free_task(%b_col0_m5_n22)
      aiex.dma_free_task(%c_col0_m5_n22)
      // Col 0 M-tile 5 N-tile 23
      %a_col0_m5_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n23)
      %b_col0_m5_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n23)
      %c_col0_m5_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3938048 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n23)
      aiex.dma_await_task(%c_col0_m5_n23)
      aiex.dma_free_task(%a_col0_m5_n23)
      aiex.dma_free_task(%b_col0_m5_n23)
      aiex.dma_free_task(%c_col0_m5_n23)
      // Col 0 M-tile 6 N-tile 0
      %a_col0_m6_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n0)
      %b_col0_m6_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n0)
      %c_col0_m6_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4718592 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n0)
      aiex.dma_await_task(%c_col0_m6_n0)
      aiex.dma_free_task(%a_col0_m6_n0)
      aiex.dma_free_task(%b_col0_m6_n0)
      aiex.dma_free_task(%c_col0_m6_n0)
      // Col 0 M-tile 6 N-tile 1
      %a_col0_m6_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n1)
      %b_col0_m6_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n1)
      %c_col0_m6_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4718848 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n1)
      aiex.dma_await_task(%c_col0_m6_n1)
      aiex.dma_free_task(%a_col0_m6_n1)
      aiex.dma_free_task(%b_col0_m6_n1)
      aiex.dma_free_task(%c_col0_m6_n1)
      // Col 0 M-tile 6 N-tile 2
      %a_col0_m6_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n2)
      %b_col0_m6_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n2)
      %c_col0_m6_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4719104 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n2)
      aiex.dma_await_task(%c_col0_m6_n2)
      aiex.dma_free_task(%a_col0_m6_n2)
      aiex.dma_free_task(%b_col0_m6_n2)
      aiex.dma_free_task(%c_col0_m6_n2)
      // Col 0 M-tile 6 N-tile 3
      %a_col0_m6_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n3)
      %b_col0_m6_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n3)
      %c_col0_m6_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4719360 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n3)
      aiex.dma_await_task(%c_col0_m6_n3)
      aiex.dma_free_task(%a_col0_m6_n3)
      aiex.dma_free_task(%b_col0_m6_n3)
      aiex.dma_free_task(%c_col0_m6_n3)
      // Col 0 M-tile 6 N-tile 4
      %a_col0_m6_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n4)
      %b_col0_m6_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n4)
      %c_col0_m6_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4719616 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n4)
      aiex.dma_await_task(%c_col0_m6_n4)
      aiex.dma_free_task(%a_col0_m6_n4)
      aiex.dma_free_task(%b_col0_m6_n4)
      aiex.dma_free_task(%c_col0_m6_n4)
      // Col 0 M-tile 6 N-tile 5
      %a_col0_m6_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n5)
      %b_col0_m6_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n5)
      %c_col0_m6_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4719872 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n5)
      aiex.dma_await_task(%c_col0_m6_n5)
      aiex.dma_free_task(%a_col0_m6_n5)
      aiex.dma_free_task(%b_col0_m6_n5)
      aiex.dma_free_task(%c_col0_m6_n5)
      // Col 0 M-tile 6 N-tile 6
      %a_col0_m6_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n6)
      %b_col0_m6_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n6)
      %c_col0_m6_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4720128 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n6)
      aiex.dma_await_task(%c_col0_m6_n6)
      aiex.dma_free_task(%a_col0_m6_n6)
      aiex.dma_free_task(%b_col0_m6_n6)
      aiex.dma_free_task(%c_col0_m6_n6)
      // Col 0 M-tile 6 N-tile 7
      %a_col0_m6_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n7)
      %b_col0_m6_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n7)
      %c_col0_m6_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4720384 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n7)
      aiex.dma_await_task(%c_col0_m6_n7)
      aiex.dma_free_task(%a_col0_m6_n7)
      aiex.dma_free_task(%b_col0_m6_n7)
      aiex.dma_free_task(%c_col0_m6_n7)
      // Col 0 M-tile 6 N-tile 8
      %a_col0_m6_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n8)
      %b_col0_m6_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n8)
      %c_col0_m6_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4720640 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n8)
      aiex.dma_await_task(%c_col0_m6_n8)
      aiex.dma_free_task(%a_col0_m6_n8)
      aiex.dma_free_task(%b_col0_m6_n8)
      aiex.dma_free_task(%c_col0_m6_n8)
      // Col 0 M-tile 6 N-tile 9
      %a_col0_m6_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n9)
      %b_col0_m6_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n9)
      %c_col0_m6_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4720896 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n9)
      aiex.dma_await_task(%c_col0_m6_n9)
      aiex.dma_free_task(%a_col0_m6_n9)
      aiex.dma_free_task(%b_col0_m6_n9)
      aiex.dma_free_task(%c_col0_m6_n9)
      // Col 0 M-tile 6 N-tile 10
      %a_col0_m6_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n10)
      %b_col0_m6_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n10)
      %c_col0_m6_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4721152 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n10)
      aiex.dma_await_task(%c_col0_m6_n10)
      aiex.dma_free_task(%a_col0_m6_n10)
      aiex.dma_free_task(%b_col0_m6_n10)
      aiex.dma_free_task(%c_col0_m6_n10)
      // Col 0 M-tile 6 N-tile 11
      %a_col0_m6_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n11)
      %b_col0_m6_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n11)
      %c_col0_m6_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4721408 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n11)
      aiex.dma_await_task(%c_col0_m6_n11)
      aiex.dma_free_task(%a_col0_m6_n11)
      aiex.dma_free_task(%b_col0_m6_n11)
      aiex.dma_free_task(%c_col0_m6_n11)
      // Col 0 M-tile 6 N-tile 12
      %a_col0_m6_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n12)
      %b_col0_m6_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n12)
      %c_col0_m6_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4721664 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n12)
      aiex.dma_await_task(%c_col0_m6_n12)
      aiex.dma_free_task(%a_col0_m6_n12)
      aiex.dma_free_task(%b_col0_m6_n12)
      aiex.dma_free_task(%c_col0_m6_n12)
      // Col 0 M-tile 6 N-tile 13
      %a_col0_m6_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n13)
      %b_col0_m6_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n13)
      %c_col0_m6_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4721920 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n13)
      aiex.dma_await_task(%c_col0_m6_n13)
      aiex.dma_free_task(%a_col0_m6_n13)
      aiex.dma_free_task(%b_col0_m6_n13)
      aiex.dma_free_task(%c_col0_m6_n13)
      // Col 0 M-tile 6 N-tile 14
      %a_col0_m6_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n14)
      %b_col0_m6_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n14)
      %c_col0_m6_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4722176 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n14)
      aiex.dma_await_task(%c_col0_m6_n14)
      aiex.dma_free_task(%a_col0_m6_n14)
      aiex.dma_free_task(%b_col0_m6_n14)
      aiex.dma_free_task(%c_col0_m6_n14)
      // Col 0 M-tile 6 N-tile 15
      %a_col0_m6_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n15)
      %b_col0_m6_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n15)
      %c_col0_m6_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4722432 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n15)
      aiex.dma_await_task(%c_col0_m6_n15)
      aiex.dma_free_task(%a_col0_m6_n15)
      aiex.dma_free_task(%b_col0_m6_n15)
      aiex.dma_free_task(%c_col0_m6_n15)
      // Col 0 M-tile 6 N-tile 16
      %a_col0_m6_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n16)
      %b_col0_m6_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n16)
      %c_col0_m6_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4722688 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n16)
      aiex.dma_await_task(%c_col0_m6_n16)
      aiex.dma_free_task(%a_col0_m6_n16)
      aiex.dma_free_task(%b_col0_m6_n16)
      aiex.dma_free_task(%c_col0_m6_n16)
      // Col 0 M-tile 6 N-tile 17
      %a_col0_m6_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n17)
      %b_col0_m6_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n17)
      %c_col0_m6_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4722944 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n17)
      aiex.dma_await_task(%c_col0_m6_n17)
      aiex.dma_free_task(%a_col0_m6_n17)
      aiex.dma_free_task(%b_col0_m6_n17)
      aiex.dma_free_task(%c_col0_m6_n17)
      // Col 0 M-tile 6 N-tile 18
      %a_col0_m6_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n18)
      %b_col0_m6_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n18)
      %c_col0_m6_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4723200 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n18)
      aiex.dma_await_task(%c_col0_m6_n18)
      aiex.dma_free_task(%a_col0_m6_n18)
      aiex.dma_free_task(%b_col0_m6_n18)
      aiex.dma_free_task(%c_col0_m6_n18)
      // Col 0 M-tile 6 N-tile 19
      %a_col0_m6_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n19)
      %b_col0_m6_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n19)
      %c_col0_m6_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4723456 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n19)
      aiex.dma_await_task(%c_col0_m6_n19)
      aiex.dma_free_task(%a_col0_m6_n19)
      aiex.dma_free_task(%b_col0_m6_n19)
      aiex.dma_free_task(%c_col0_m6_n19)
      // Col 0 M-tile 6 N-tile 20
      %a_col0_m6_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n20)
      %b_col0_m6_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n20)
      %c_col0_m6_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4723712 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n20)
      aiex.dma_await_task(%c_col0_m6_n20)
      aiex.dma_free_task(%a_col0_m6_n20)
      aiex.dma_free_task(%b_col0_m6_n20)
      aiex.dma_free_task(%c_col0_m6_n20)
      // Col 0 M-tile 6 N-tile 21
      %a_col0_m6_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n21)
      %b_col0_m6_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n21)
      %c_col0_m6_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4723968 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n21)
      aiex.dma_await_task(%c_col0_m6_n21)
      aiex.dma_free_task(%a_col0_m6_n21)
      aiex.dma_free_task(%b_col0_m6_n21)
      aiex.dma_free_task(%c_col0_m6_n21)
      // Col 0 M-tile 6 N-tile 22
      %a_col0_m6_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n22)
      %b_col0_m6_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n22)
      %c_col0_m6_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4724224 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n22)
      aiex.dma_await_task(%c_col0_m6_n22)
      aiex.dma_free_task(%a_col0_m6_n22)
      aiex.dma_free_task(%b_col0_m6_n22)
      aiex.dma_free_task(%c_col0_m6_n22)
      // Col 0 M-tile 6 N-tile 23
      %a_col0_m6_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n23)
      %b_col0_m6_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n23)
      %c_col0_m6_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4724480 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n23)
      aiex.dma_await_task(%c_col0_m6_n23)
      aiex.dma_free_task(%a_col0_m6_n23)
      aiex.dma_free_task(%b_col0_m6_n23)
      aiex.dma_free_task(%c_col0_m6_n23)
      // Col 0 M-tile 7 N-tile 0
      %a_col0_m7_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n0)
      %b_col0_m7_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n0)
      %c_col0_m7_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5505024 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n0)
      aiex.dma_await_task(%c_col0_m7_n0)
      aiex.dma_free_task(%a_col0_m7_n0)
      aiex.dma_free_task(%b_col0_m7_n0)
      aiex.dma_free_task(%c_col0_m7_n0)
      // Col 0 M-tile 7 N-tile 1
      %a_col0_m7_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n1)
      %b_col0_m7_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n1)
      %c_col0_m7_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5505280 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n1)
      aiex.dma_await_task(%c_col0_m7_n1)
      aiex.dma_free_task(%a_col0_m7_n1)
      aiex.dma_free_task(%b_col0_m7_n1)
      aiex.dma_free_task(%c_col0_m7_n1)
      // Col 0 M-tile 7 N-tile 2
      %a_col0_m7_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n2)
      %b_col0_m7_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n2)
      %c_col0_m7_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5505536 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n2)
      aiex.dma_await_task(%c_col0_m7_n2)
      aiex.dma_free_task(%a_col0_m7_n2)
      aiex.dma_free_task(%b_col0_m7_n2)
      aiex.dma_free_task(%c_col0_m7_n2)
      // Col 0 M-tile 7 N-tile 3
      %a_col0_m7_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n3)
      %b_col0_m7_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n3)
      %c_col0_m7_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5505792 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n3)
      aiex.dma_await_task(%c_col0_m7_n3)
      aiex.dma_free_task(%a_col0_m7_n3)
      aiex.dma_free_task(%b_col0_m7_n3)
      aiex.dma_free_task(%c_col0_m7_n3)
      // Col 0 M-tile 7 N-tile 4
      %a_col0_m7_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n4)
      %b_col0_m7_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n4)
      %c_col0_m7_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5506048 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n4)
      aiex.dma_await_task(%c_col0_m7_n4)
      aiex.dma_free_task(%a_col0_m7_n4)
      aiex.dma_free_task(%b_col0_m7_n4)
      aiex.dma_free_task(%c_col0_m7_n4)
      // Col 0 M-tile 7 N-tile 5
      %a_col0_m7_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n5)
      %b_col0_m7_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n5)
      %c_col0_m7_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5506304 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n5)
      aiex.dma_await_task(%c_col0_m7_n5)
      aiex.dma_free_task(%a_col0_m7_n5)
      aiex.dma_free_task(%b_col0_m7_n5)
      aiex.dma_free_task(%c_col0_m7_n5)
      // Col 0 M-tile 7 N-tile 6
      %a_col0_m7_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n6)
      %b_col0_m7_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n6)
      %c_col0_m7_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5506560 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n6)
      aiex.dma_await_task(%c_col0_m7_n6)
      aiex.dma_free_task(%a_col0_m7_n6)
      aiex.dma_free_task(%b_col0_m7_n6)
      aiex.dma_free_task(%c_col0_m7_n6)
      // Col 0 M-tile 7 N-tile 7
      %a_col0_m7_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n7)
      %b_col0_m7_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n7)
      %c_col0_m7_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5506816 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n7)
      aiex.dma_await_task(%c_col0_m7_n7)
      aiex.dma_free_task(%a_col0_m7_n7)
      aiex.dma_free_task(%b_col0_m7_n7)
      aiex.dma_free_task(%c_col0_m7_n7)
      // Col 0 M-tile 7 N-tile 8
      %a_col0_m7_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n8)
      %b_col0_m7_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n8)
      %c_col0_m7_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5507072 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n8)
      aiex.dma_await_task(%c_col0_m7_n8)
      aiex.dma_free_task(%a_col0_m7_n8)
      aiex.dma_free_task(%b_col0_m7_n8)
      aiex.dma_free_task(%c_col0_m7_n8)
      // Col 0 M-tile 7 N-tile 9
      %a_col0_m7_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n9)
      %b_col0_m7_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n9)
      %c_col0_m7_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5507328 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n9)
      aiex.dma_await_task(%c_col0_m7_n9)
      aiex.dma_free_task(%a_col0_m7_n9)
      aiex.dma_free_task(%b_col0_m7_n9)
      aiex.dma_free_task(%c_col0_m7_n9)
      // Col 0 M-tile 7 N-tile 10
      %a_col0_m7_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n10)
      %b_col0_m7_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n10)
      %c_col0_m7_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5507584 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n10)
      aiex.dma_await_task(%c_col0_m7_n10)
      aiex.dma_free_task(%a_col0_m7_n10)
      aiex.dma_free_task(%b_col0_m7_n10)
      aiex.dma_free_task(%c_col0_m7_n10)
      // Col 0 M-tile 7 N-tile 11
      %a_col0_m7_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n11)
      %b_col0_m7_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n11)
      %c_col0_m7_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5507840 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n11)
      aiex.dma_await_task(%c_col0_m7_n11)
      aiex.dma_free_task(%a_col0_m7_n11)
      aiex.dma_free_task(%b_col0_m7_n11)
      aiex.dma_free_task(%c_col0_m7_n11)
      // Col 0 M-tile 7 N-tile 12
      %a_col0_m7_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n12)
      %b_col0_m7_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n12)
      %c_col0_m7_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5508096 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n12)
      aiex.dma_await_task(%c_col0_m7_n12)
      aiex.dma_free_task(%a_col0_m7_n12)
      aiex.dma_free_task(%b_col0_m7_n12)
      aiex.dma_free_task(%c_col0_m7_n12)
      // Col 0 M-tile 7 N-tile 13
      %a_col0_m7_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n13)
      %b_col0_m7_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n13)
      %c_col0_m7_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5508352 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n13)
      aiex.dma_await_task(%c_col0_m7_n13)
      aiex.dma_free_task(%a_col0_m7_n13)
      aiex.dma_free_task(%b_col0_m7_n13)
      aiex.dma_free_task(%c_col0_m7_n13)
      // Col 0 M-tile 7 N-tile 14
      %a_col0_m7_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n14)
      %b_col0_m7_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n14)
      %c_col0_m7_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5508608 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n14)
      aiex.dma_await_task(%c_col0_m7_n14)
      aiex.dma_free_task(%a_col0_m7_n14)
      aiex.dma_free_task(%b_col0_m7_n14)
      aiex.dma_free_task(%c_col0_m7_n14)
      // Col 0 M-tile 7 N-tile 15
      %a_col0_m7_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n15)
      %b_col0_m7_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n15)
      %c_col0_m7_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5508864 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n15)
      aiex.dma_await_task(%c_col0_m7_n15)
      aiex.dma_free_task(%a_col0_m7_n15)
      aiex.dma_free_task(%b_col0_m7_n15)
      aiex.dma_free_task(%c_col0_m7_n15)
      // Col 0 M-tile 7 N-tile 16
      %a_col0_m7_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n16)
      %b_col0_m7_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n16)
      %c_col0_m7_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5509120 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n16)
      aiex.dma_await_task(%c_col0_m7_n16)
      aiex.dma_free_task(%a_col0_m7_n16)
      aiex.dma_free_task(%b_col0_m7_n16)
      aiex.dma_free_task(%c_col0_m7_n16)
      // Col 0 M-tile 7 N-tile 17
      %a_col0_m7_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n17)
      %b_col0_m7_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n17)
      %c_col0_m7_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5509376 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n17)
      aiex.dma_await_task(%c_col0_m7_n17)
      aiex.dma_free_task(%a_col0_m7_n17)
      aiex.dma_free_task(%b_col0_m7_n17)
      aiex.dma_free_task(%c_col0_m7_n17)
      // Col 0 M-tile 7 N-tile 18
      %a_col0_m7_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n18)
      %b_col0_m7_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n18)
      %c_col0_m7_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5509632 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n18)
      aiex.dma_await_task(%c_col0_m7_n18)
      aiex.dma_free_task(%a_col0_m7_n18)
      aiex.dma_free_task(%b_col0_m7_n18)
      aiex.dma_free_task(%c_col0_m7_n18)
      // Col 0 M-tile 7 N-tile 19
      %a_col0_m7_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n19)
      %b_col0_m7_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n19)
      %c_col0_m7_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5509888 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n19)
      aiex.dma_await_task(%c_col0_m7_n19)
      aiex.dma_free_task(%a_col0_m7_n19)
      aiex.dma_free_task(%b_col0_m7_n19)
      aiex.dma_free_task(%c_col0_m7_n19)
      // Col 0 M-tile 7 N-tile 20
      %a_col0_m7_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n20)
      %b_col0_m7_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n20)
      %c_col0_m7_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5510144 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n20)
      aiex.dma_await_task(%c_col0_m7_n20)
      aiex.dma_free_task(%a_col0_m7_n20)
      aiex.dma_free_task(%b_col0_m7_n20)
      aiex.dma_free_task(%c_col0_m7_n20)
      // Col 0 M-tile 7 N-tile 21
      %a_col0_m7_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n21)
      %b_col0_m7_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n21)
      %c_col0_m7_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5510400 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n21)
      aiex.dma_await_task(%c_col0_m7_n21)
      aiex.dma_free_task(%a_col0_m7_n21)
      aiex.dma_free_task(%b_col0_m7_n21)
      aiex.dma_free_task(%c_col0_m7_n21)
      // Col 0 M-tile 7 N-tile 22
      %a_col0_m7_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n22)
      %b_col0_m7_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n22)
      %c_col0_m7_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5510656 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n22)
      aiex.dma_await_task(%c_col0_m7_n22)
      aiex.dma_free_task(%a_col0_m7_n22)
      aiex.dma_free_task(%b_col0_m7_n22)
      aiex.dma_free_task(%c_col0_m7_n22)
      // Col 0 M-tile 7 N-tile 23
      %a_col0_m7_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n23)
      %b_col0_m7_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n23)
      %c_col0_m7_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5510912 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n23)
      aiex.dma_await_task(%c_col0_m7_n23)
      aiex.dma_free_task(%a_col0_m7_n23)
      aiex.dma_free_task(%b_col0_m7_n23)
      aiex.dma_free_task(%c_col0_m7_n23)
      // Col 1 M-tile 0 N-tile 0
      %a_col1_m0_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n0)
      %b_col1_m0_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n0)
      %c_col1_m0_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 6144 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n0)
      aiex.dma_await_task(%c_col1_m0_n0)
      aiex.dma_free_task(%a_col1_m0_n0)
      aiex.dma_free_task(%b_col1_m0_n0)
      aiex.dma_free_task(%c_col1_m0_n0)
      // Col 1 M-tile 0 N-tile 1
      %a_col1_m0_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n1)
      %b_col1_m0_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n1)
      %c_col1_m0_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 6400 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n1)
      aiex.dma_await_task(%c_col1_m0_n1)
      aiex.dma_free_task(%a_col1_m0_n1)
      aiex.dma_free_task(%b_col1_m0_n1)
      aiex.dma_free_task(%c_col1_m0_n1)
      // Col 1 M-tile 0 N-tile 2
      %a_col1_m0_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n2)
      %b_col1_m0_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n2)
      %c_col1_m0_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 6656 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n2)
      aiex.dma_await_task(%c_col1_m0_n2)
      aiex.dma_free_task(%a_col1_m0_n2)
      aiex.dma_free_task(%b_col1_m0_n2)
      aiex.dma_free_task(%c_col1_m0_n2)
      // Col 1 M-tile 0 N-tile 3
      %a_col1_m0_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n3)
      %b_col1_m0_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n3)
      %c_col1_m0_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 6912 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n3)
      aiex.dma_await_task(%c_col1_m0_n3)
      aiex.dma_free_task(%a_col1_m0_n3)
      aiex.dma_free_task(%b_col1_m0_n3)
      aiex.dma_free_task(%c_col1_m0_n3)
      // Col 1 M-tile 0 N-tile 4
      %a_col1_m0_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n4)
      %b_col1_m0_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n4)
      %c_col1_m0_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 7168 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n4)
      aiex.dma_await_task(%c_col1_m0_n4)
      aiex.dma_free_task(%a_col1_m0_n4)
      aiex.dma_free_task(%b_col1_m0_n4)
      aiex.dma_free_task(%c_col1_m0_n4)
      // Col 1 M-tile 0 N-tile 5
      %a_col1_m0_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n5)
      %b_col1_m0_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n5)
      %c_col1_m0_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 7424 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n5)
      aiex.dma_await_task(%c_col1_m0_n5)
      aiex.dma_free_task(%a_col1_m0_n5)
      aiex.dma_free_task(%b_col1_m0_n5)
      aiex.dma_free_task(%c_col1_m0_n5)
      // Col 1 M-tile 0 N-tile 6
      %a_col1_m0_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n6)
      %b_col1_m0_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n6)
      %c_col1_m0_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 7680 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n6)
      aiex.dma_await_task(%c_col1_m0_n6)
      aiex.dma_free_task(%a_col1_m0_n6)
      aiex.dma_free_task(%b_col1_m0_n6)
      aiex.dma_free_task(%c_col1_m0_n6)
      // Col 1 M-tile 0 N-tile 7
      %a_col1_m0_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n7)
      %b_col1_m0_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n7)
      %c_col1_m0_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 7936 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n7)
      aiex.dma_await_task(%c_col1_m0_n7)
      aiex.dma_free_task(%a_col1_m0_n7)
      aiex.dma_free_task(%b_col1_m0_n7)
      aiex.dma_free_task(%c_col1_m0_n7)
      // Col 1 M-tile 0 N-tile 8
      %a_col1_m0_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n8)
      %b_col1_m0_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n8)
      %c_col1_m0_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 8192 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n8)
      aiex.dma_await_task(%c_col1_m0_n8)
      aiex.dma_free_task(%a_col1_m0_n8)
      aiex.dma_free_task(%b_col1_m0_n8)
      aiex.dma_free_task(%c_col1_m0_n8)
      // Col 1 M-tile 0 N-tile 9
      %a_col1_m0_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n9)
      %b_col1_m0_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n9)
      %c_col1_m0_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 8448 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n9)
      aiex.dma_await_task(%c_col1_m0_n9)
      aiex.dma_free_task(%a_col1_m0_n9)
      aiex.dma_free_task(%b_col1_m0_n9)
      aiex.dma_free_task(%c_col1_m0_n9)
      // Col 1 M-tile 0 N-tile 10
      %a_col1_m0_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n10)
      %b_col1_m0_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n10)
      %c_col1_m0_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 8704 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n10)
      aiex.dma_await_task(%c_col1_m0_n10)
      aiex.dma_free_task(%a_col1_m0_n10)
      aiex.dma_free_task(%b_col1_m0_n10)
      aiex.dma_free_task(%c_col1_m0_n10)
      // Col 1 M-tile 0 N-tile 11
      %a_col1_m0_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n11)
      %b_col1_m0_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n11)
      %c_col1_m0_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 8960 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n11)
      aiex.dma_await_task(%c_col1_m0_n11)
      aiex.dma_free_task(%a_col1_m0_n11)
      aiex.dma_free_task(%b_col1_m0_n11)
      aiex.dma_free_task(%c_col1_m0_n11)
      // Col 1 M-tile 0 N-tile 12
      %a_col1_m0_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n12)
      %b_col1_m0_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n12)
      %c_col1_m0_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 9216 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n12)
      aiex.dma_await_task(%c_col1_m0_n12)
      aiex.dma_free_task(%a_col1_m0_n12)
      aiex.dma_free_task(%b_col1_m0_n12)
      aiex.dma_free_task(%c_col1_m0_n12)
      // Col 1 M-tile 0 N-tile 13
      %a_col1_m0_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n13)
      %b_col1_m0_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n13)
      %c_col1_m0_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 9472 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n13)
      aiex.dma_await_task(%c_col1_m0_n13)
      aiex.dma_free_task(%a_col1_m0_n13)
      aiex.dma_free_task(%b_col1_m0_n13)
      aiex.dma_free_task(%c_col1_m0_n13)
      // Col 1 M-tile 0 N-tile 14
      %a_col1_m0_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n14)
      %b_col1_m0_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n14)
      %c_col1_m0_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 9728 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n14)
      aiex.dma_await_task(%c_col1_m0_n14)
      aiex.dma_free_task(%a_col1_m0_n14)
      aiex.dma_free_task(%b_col1_m0_n14)
      aiex.dma_free_task(%c_col1_m0_n14)
      // Col 1 M-tile 0 N-tile 15
      %a_col1_m0_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n15)
      %b_col1_m0_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n15)
      %c_col1_m0_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 9984 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n15)
      aiex.dma_await_task(%c_col1_m0_n15)
      aiex.dma_free_task(%a_col1_m0_n15)
      aiex.dma_free_task(%b_col1_m0_n15)
      aiex.dma_free_task(%c_col1_m0_n15)
      // Col 1 M-tile 0 N-tile 16
      %a_col1_m0_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n16)
      %b_col1_m0_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n16)
      %c_col1_m0_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 10240 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n16)
      aiex.dma_await_task(%c_col1_m0_n16)
      aiex.dma_free_task(%a_col1_m0_n16)
      aiex.dma_free_task(%b_col1_m0_n16)
      aiex.dma_free_task(%c_col1_m0_n16)
      // Col 1 M-tile 0 N-tile 17
      %a_col1_m0_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n17)
      %b_col1_m0_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n17)
      %c_col1_m0_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 10496 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n17)
      aiex.dma_await_task(%c_col1_m0_n17)
      aiex.dma_free_task(%a_col1_m0_n17)
      aiex.dma_free_task(%b_col1_m0_n17)
      aiex.dma_free_task(%c_col1_m0_n17)
      // Col 1 M-tile 0 N-tile 18
      %a_col1_m0_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n18)
      %b_col1_m0_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n18)
      %c_col1_m0_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 10752 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n18)
      aiex.dma_await_task(%c_col1_m0_n18)
      aiex.dma_free_task(%a_col1_m0_n18)
      aiex.dma_free_task(%b_col1_m0_n18)
      aiex.dma_free_task(%c_col1_m0_n18)
      // Col 1 M-tile 0 N-tile 19
      %a_col1_m0_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n19)
      %b_col1_m0_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n19)
      %c_col1_m0_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 11008 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n19)
      aiex.dma_await_task(%c_col1_m0_n19)
      aiex.dma_free_task(%a_col1_m0_n19)
      aiex.dma_free_task(%b_col1_m0_n19)
      aiex.dma_free_task(%c_col1_m0_n19)
      // Col 1 M-tile 0 N-tile 20
      %a_col1_m0_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n20)
      %b_col1_m0_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n20)
      %c_col1_m0_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 11264 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n20)
      aiex.dma_await_task(%c_col1_m0_n20)
      aiex.dma_free_task(%a_col1_m0_n20)
      aiex.dma_free_task(%b_col1_m0_n20)
      aiex.dma_free_task(%c_col1_m0_n20)
      // Col 1 M-tile 0 N-tile 21
      %a_col1_m0_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n21)
      %b_col1_m0_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n21)
      %c_col1_m0_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 11520 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n21)
      aiex.dma_await_task(%c_col1_m0_n21)
      aiex.dma_free_task(%a_col1_m0_n21)
      aiex.dma_free_task(%b_col1_m0_n21)
      aiex.dma_free_task(%c_col1_m0_n21)
      // Col 1 M-tile 0 N-tile 22
      %a_col1_m0_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n22)
      %b_col1_m0_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n22)
      %c_col1_m0_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 11776 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n22)
      aiex.dma_await_task(%c_col1_m0_n22)
      aiex.dma_free_task(%a_col1_m0_n22)
      aiex.dma_free_task(%b_col1_m0_n22)
      aiex.dma_free_task(%c_col1_m0_n22)
      // Col 1 M-tile 0 N-tile 23
      %a_col1_m0_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m0_n23)
      %b_col1_m0_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m0_n23)
      %c_col1_m0_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 12032 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m0_n23)
      aiex.dma_await_task(%c_col1_m0_n23)
      aiex.dma_free_task(%a_col1_m0_n23)
      aiex.dma_free_task(%b_col1_m0_n23)
      aiex.dma_free_task(%c_col1_m0_n23)
      // Col 1 M-tile 1 N-tile 0
      %a_col1_m1_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n0)
      %b_col1_m1_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n0)
      %c_col1_m1_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 792576 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n0)
      aiex.dma_await_task(%c_col1_m1_n0)
      aiex.dma_free_task(%a_col1_m1_n0)
      aiex.dma_free_task(%b_col1_m1_n0)
      aiex.dma_free_task(%c_col1_m1_n0)
      // Col 1 M-tile 1 N-tile 1
      %a_col1_m1_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n1)
      %b_col1_m1_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n1)
      %c_col1_m1_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 792832 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n1)
      aiex.dma_await_task(%c_col1_m1_n1)
      aiex.dma_free_task(%a_col1_m1_n1)
      aiex.dma_free_task(%b_col1_m1_n1)
      aiex.dma_free_task(%c_col1_m1_n1)
      // Col 1 M-tile 1 N-tile 2
      %a_col1_m1_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n2)
      %b_col1_m1_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n2)
      %c_col1_m1_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 793088 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n2)
      aiex.dma_await_task(%c_col1_m1_n2)
      aiex.dma_free_task(%a_col1_m1_n2)
      aiex.dma_free_task(%b_col1_m1_n2)
      aiex.dma_free_task(%c_col1_m1_n2)
      // Col 1 M-tile 1 N-tile 3
      %a_col1_m1_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n3)
      %b_col1_m1_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n3)
      %c_col1_m1_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 793344 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n3)
      aiex.dma_await_task(%c_col1_m1_n3)
      aiex.dma_free_task(%a_col1_m1_n3)
      aiex.dma_free_task(%b_col1_m1_n3)
      aiex.dma_free_task(%c_col1_m1_n3)
      // Col 1 M-tile 1 N-tile 4
      %a_col1_m1_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n4)
      %b_col1_m1_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n4)
      %c_col1_m1_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 793600 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n4)
      aiex.dma_await_task(%c_col1_m1_n4)
      aiex.dma_free_task(%a_col1_m1_n4)
      aiex.dma_free_task(%b_col1_m1_n4)
      aiex.dma_free_task(%c_col1_m1_n4)
      // Col 1 M-tile 1 N-tile 5
      %a_col1_m1_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n5)
      %b_col1_m1_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n5)
      %c_col1_m1_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 793856 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n5)
      aiex.dma_await_task(%c_col1_m1_n5)
      aiex.dma_free_task(%a_col1_m1_n5)
      aiex.dma_free_task(%b_col1_m1_n5)
      aiex.dma_free_task(%c_col1_m1_n5)
      // Col 1 M-tile 1 N-tile 6
      %a_col1_m1_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n6)
      %b_col1_m1_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n6)
      %c_col1_m1_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 794112 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n6)
      aiex.dma_await_task(%c_col1_m1_n6)
      aiex.dma_free_task(%a_col1_m1_n6)
      aiex.dma_free_task(%b_col1_m1_n6)
      aiex.dma_free_task(%c_col1_m1_n6)
      // Col 1 M-tile 1 N-tile 7
      %a_col1_m1_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n7)
      %b_col1_m1_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n7)
      %c_col1_m1_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 794368 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n7)
      aiex.dma_await_task(%c_col1_m1_n7)
      aiex.dma_free_task(%a_col1_m1_n7)
      aiex.dma_free_task(%b_col1_m1_n7)
      aiex.dma_free_task(%c_col1_m1_n7)
      // Col 1 M-tile 1 N-tile 8
      %a_col1_m1_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n8)
      %b_col1_m1_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n8)
      %c_col1_m1_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 794624 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n8)
      aiex.dma_await_task(%c_col1_m1_n8)
      aiex.dma_free_task(%a_col1_m1_n8)
      aiex.dma_free_task(%b_col1_m1_n8)
      aiex.dma_free_task(%c_col1_m1_n8)
      // Col 1 M-tile 1 N-tile 9
      %a_col1_m1_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n9)
      %b_col1_m1_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n9)
      %c_col1_m1_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 794880 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n9)
      aiex.dma_await_task(%c_col1_m1_n9)
      aiex.dma_free_task(%a_col1_m1_n9)
      aiex.dma_free_task(%b_col1_m1_n9)
      aiex.dma_free_task(%c_col1_m1_n9)
      // Col 1 M-tile 1 N-tile 10
      %a_col1_m1_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n10)
      %b_col1_m1_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n10)
      %c_col1_m1_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 795136 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n10)
      aiex.dma_await_task(%c_col1_m1_n10)
      aiex.dma_free_task(%a_col1_m1_n10)
      aiex.dma_free_task(%b_col1_m1_n10)
      aiex.dma_free_task(%c_col1_m1_n10)
      // Col 1 M-tile 1 N-tile 11
      %a_col1_m1_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n11)
      %b_col1_m1_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n11)
      %c_col1_m1_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 795392 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n11)
      aiex.dma_await_task(%c_col1_m1_n11)
      aiex.dma_free_task(%a_col1_m1_n11)
      aiex.dma_free_task(%b_col1_m1_n11)
      aiex.dma_free_task(%c_col1_m1_n11)
      // Col 1 M-tile 1 N-tile 12
      %a_col1_m1_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n12)
      %b_col1_m1_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n12)
      %c_col1_m1_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 795648 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n12)
      aiex.dma_await_task(%c_col1_m1_n12)
      aiex.dma_free_task(%a_col1_m1_n12)
      aiex.dma_free_task(%b_col1_m1_n12)
      aiex.dma_free_task(%c_col1_m1_n12)
      // Col 1 M-tile 1 N-tile 13
      %a_col1_m1_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n13)
      %b_col1_m1_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n13)
      %c_col1_m1_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 795904 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n13)
      aiex.dma_await_task(%c_col1_m1_n13)
      aiex.dma_free_task(%a_col1_m1_n13)
      aiex.dma_free_task(%b_col1_m1_n13)
      aiex.dma_free_task(%c_col1_m1_n13)
      // Col 1 M-tile 1 N-tile 14
      %a_col1_m1_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n14)
      %b_col1_m1_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n14)
      %c_col1_m1_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 796160 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n14)
      aiex.dma_await_task(%c_col1_m1_n14)
      aiex.dma_free_task(%a_col1_m1_n14)
      aiex.dma_free_task(%b_col1_m1_n14)
      aiex.dma_free_task(%c_col1_m1_n14)
      // Col 1 M-tile 1 N-tile 15
      %a_col1_m1_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n15)
      %b_col1_m1_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n15)
      %c_col1_m1_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 796416 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n15)
      aiex.dma_await_task(%c_col1_m1_n15)
      aiex.dma_free_task(%a_col1_m1_n15)
      aiex.dma_free_task(%b_col1_m1_n15)
      aiex.dma_free_task(%c_col1_m1_n15)
      // Col 1 M-tile 1 N-tile 16
      %a_col1_m1_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n16)
      %b_col1_m1_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n16)
      %c_col1_m1_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 796672 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n16)
      aiex.dma_await_task(%c_col1_m1_n16)
      aiex.dma_free_task(%a_col1_m1_n16)
      aiex.dma_free_task(%b_col1_m1_n16)
      aiex.dma_free_task(%c_col1_m1_n16)
      // Col 1 M-tile 1 N-tile 17
      %a_col1_m1_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n17)
      %b_col1_m1_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n17)
      %c_col1_m1_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 796928 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n17)
      aiex.dma_await_task(%c_col1_m1_n17)
      aiex.dma_free_task(%a_col1_m1_n17)
      aiex.dma_free_task(%b_col1_m1_n17)
      aiex.dma_free_task(%c_col1_m1_n17)
      // Col 1 M-tile 1 N-tile 18
      %a_col1_m1_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n18)
      %b_col1_m1_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n18)
      %c_col1_m1_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 797184 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n18)
      aiex.dma_await_task(%c_col1_m1_n18)
      aiex.dma_free_task(%a_col1_m1_n18)
      aiex.dma_free_task(%b_col1_m1_n18)
      aiex.dma_free_task(%c_col1_m1_n18)
      // Col 1 M-tile 1 N-tile 19
      %a_col1_m1_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n19)
      %b_col1_m1_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n19)
      %c_col1_m1_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 797440 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n19)
      aiex.dma_await_task(%c_col1_m1_n19)
      aiex.dma_free_task(%a_col1_m1_n19)
      aiex.dma_free_task(%b_col1_m1_n19)
      aiex.dma_free_task(%c_col1_m1_n19)
      // Col 1 M-tile 1 N-tile 20
      %a_col1_m1_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n20)
      %b_col1_m1_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n20)
      %c_col1_m1_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 797696 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n20)
      aiex.dma_await_task(%c_col1_m1_n20)
      aiex.dma_free_task(%a_col1_m1_n20)
      aiex.dma_free_task(%b_col1_m1_n20)
      aiex.dma_free_task(%c_col1_m1_n20)
      // Col 1 M-tile 1 N-tile 21
      %a_col1_m1_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n21)
      %b_col1_m1_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n21)
      %c_col1_m1_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 797952 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n21)
      aiex.dma_await_task(%c_col1_m1_n21)
      aiex.dma_free_task(%a_col1_m1_n21)
      aiex.dma_free_task(%b_col1_m1_n21)
      aiex.dma_free_task(%c_col1_m1_n21)
      // Col 1 M-tile 1 N-tile 22
      %a_col1_m1_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n22)
      %b_col1_m1_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n22)
      %c_col1_m1_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 798208 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n22)
      aiex.dma_await_task(%c_col1_m1_n22)
      aiex.dma_free_task(%a_col1_m1_n22)
      aiex.dma_free_task(%b_col1_m1_n22)
      aiex.dma_free_task(%c_col1_m1_n22)
      // Col 1 M-tile 1 N-tile 23
      %a_col1_m1_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m1_n23)
      %b_col1_m1_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m1_n23)
      %c_col1_m1_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 798464 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m1_n23)
      aiex.dma_await_task(%c_col1_m1_n23)
      aiex.dma_free_task(%a_col1_m1_n23)
      aiex.dma_free_task(%b_col1_m1_n23)
      aiex.dma_free_task(%c_col1_m1_n23)
      // Col 1 M-tile 2 N-tile 0
      %a_col1_m2_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n0)
      %b_col1_m2_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n0)
      %c_col1_m2_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1579008 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n0)
      aiex.dma_await_task(%c_col1_m2_n0)
      aiex.dma_free_task(%a_col1_m2_n0)
      aiex.dma_free_task(%b_col1_m2_n0)
      aiex.dma_free_task(%c_col1_m2_n0)
      // Col 1 M-tile 2 N-tile 1
      %a_col1_m2_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n1)
      %b_col1_m2_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n1)
      %c_col1_m2_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1579264 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n1)
      aiex.dma_await_task(%c_col1_m2_n1)
      aiex.dma_free_task(%a_col1_m2_n1)
      aiex.dma_free_task(%b_col1_m2_n1)
      aiex.dma_free_task(%c_col1_m2_n1)
      // Col 1 M-tile 2 N-tile 2
      %a_col1_m2_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n2)
      %b_col1_m2_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n2)
      %c_col1_m2_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1579520 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n2)
      aiex.dma_await_task(%c_col1_m2_n2)
      aiex.dma_free_task(%a_col1_m2_n2)
      aiex.dma_free_task(%b_col1_m2_n2)
      aiex.dma_free_task(%c_col1_m2_n2)
      // Col 1 M-tile 2 N-tile 3
      %a_col1_m2_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n3)
      %b_col1_m2_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n3)
      %c_col1_m2_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1579776 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n3)
      aiex.dma_await_task(%c_col1_m2_n3)
      aiex.dma_free_task(%a_col1_m2_n3)
      aiex.dma_free_task(%b_col1_m2_n3)
      aiex.dma_free_task(%c_col1_m2_n3)
      // Col 1 M-tile 2 N-tile 4
      %a_col1_m2_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n4)
      %b_col1_m2_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n4)
      %c_col1_m2_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1580032 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n4)
      aiex.dma_await_task(%c_col1_m2_n4)
      aiex.dma_free_task(%a_col1_m2_n4)
      aiex.dma_free_task(%b_col1_m2_n4)
      aiex.dma_free_task(%c_col1_m2_n4)
      // Col 1 M-tile 2 N-tile 5
      %a_col1_m2_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n5)
      %b_col1_m2_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n5)
      %c_col1_m2_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1580288 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n5)
      aiex.dma_await_task(%c_col1_m2_n5)
      aiex.dma_free_task(%a_col1_m2_n5)
      aiex.dma_free_task(%b_col1_m2_n5)
      aiex.dma_free_task(%c_col1_m2_n5)
      // Col 1 M-tile 2 N-tile 6
      %a_col1_m2_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n6)
      %b_col1_m2_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n6)
      %c_col1_m2_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1580544 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n6)
      aiex.dma_await_task(%c_col1_m2_n6)
      aiex.dma_free_task(%a_col1_m2_n6)
      aiex.dma_free_task(%b_col1_m2_n6)
      aiex.dma_free_task(%c_col1_m2_n6)
      // Col 1 M-tile 2 N-tile 7
      %a_col1_m2_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n7)
      %b_col1_m2_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n7)
      %c_col1_m2_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1580800 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n7)
      aiex.dma_await_task(%c_col1_m2_n7)
      aiex.dma_free_task(%a_col1_m2_n7)
      aiex.dma_free_task(%b_col1_m2_n7)
      aiex.dma_free_task(%c_col1_m2_n7)
      // Col 1 M-tile 2 N-tile 8
      %a_col1_m2_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n8)
      %b_col1_m2_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n8)
      %c_col1_m2_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1581056 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n8)
      aiex.dma_await_task(%c_col1_m2_n8)
      aiex.dma_free_task(%a_col1_m2_n8)
      aiex.dma_free_task(%b_col1_m2_n8)
      aiex.dma_free_task(%c_col1_m2_n8)
      // Col 1 M-tile 2 N-tile 9
      %a_col1_m2_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n9)
      %b_col1_m2_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n9)
      %c_col1_m2_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1581312 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n9)
      aiex.dma_await_task(%c_col1_m2_n9)
      aiex.dma_free_task(%a_col1_m2_n9)
      aiex.dma_free_task(%b_col1_m2_n9)
      aiex.dma_free_task(%c_col1_m2_n9)
      // Col 1 M-tile 2 N-tile 10
      %a_col1_m2_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n10)
      %b_col1_m2_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n10)
      %c_col1_m2_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1581568 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n10)
      aiex.dma_await_task(%c_col1_m2_n10)
      aiex.dma_free_task(%a_col1_m2_n10)
      aiex.dma_free_task(%b_col1_m2_n10)
      aiex.dma_free_task(%c_col1_m2_n10)
      // Col 1 M-tile 2 N-tile 11
      %a_col1_m2_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n11)
      %b_col1_m2_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n11)
      %c_col1_m2_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1581824 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n11)
      aiex.dma_await_task(%c_col1_m2_n11)
      aiex.dma_free_task(%a_col1_m2_n11)
      aiex.dma_free_task(%b_col1_m2_n11)
      aiex.dma_free_task(%c_col1_m2_n11)
      // Col 1 M-tile 2 N-tile 12
      %a_col1_m2_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n12)
      %b_col1_m2_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n12)
      %c_col1_m2_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1582080 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n12)
      aiex.dma_await_task(%c_col1_m2_n12)
      aiex.dma_free_task(%a_col1_m2_n12)
      aiex.dma_free_task(%b_col1_m2_n12)
      aiex.dma_free_task(%c_col1_m2_n12)
      // Col 1 M-tile 2 N-tile 13
      %a_col1_m2_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n13)
      %b_col1_m2_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n13)
      %c_col1_m2_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1582336 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n13)
      aiex.dma_await_task(%c_col1_m2_n13)
      aiex.dma_free_task(%a_col1_m2_n13)
      aiex.dma_free_task(%b_col1_m2_n13)
      aiex.dma_free_task(%c_col1_m2_n13)
      // Col 1 M-tile 2 N-tile 14
      %a_col1_m2_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n14)
      %b_col1_m2_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n14)
      %c_col1_m2_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1582592 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n14)
      aiex.dma_await_task(%c_col1_m2_n14)
      aiex.dma_free_task(%a_col1_m2_n14)
      aiex.dma_free_task(%b_col1_m2_n14)
      aiex.dma_free_task(%c_col1_m2_n14)
      // Col 1 M-tile 2 N-tile 15
      %a_col1_m2_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n15)
      %b_col1_m2_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n15)
      %c_col1_m2_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1582848 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n15)
      aiex.dma_await_task(%c_col1_m2_n15)
      aiex.dma_free_task(%a_col1_m2_n15)
      aiex.dma_free_task(%b_col1_m2_n15)
      aiex.dma_free_task(%c_col1_m2_n15)
      // Col 1 M-tile 2 N-tile 16
      %a_col1_m2_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n16)
      %b_col1_m2_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n16)
      %c_col1_m2_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1583104 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n16)
      aiex.dma_await_task(%c_col1_m2_n16)
      aiex.dma_free_task(%a_col1_m2_n16)
      aiex.dma_free_task(%b_col1_m2_n16)
      aiex.dma_free_task(%c_col1_m2_n16)
      // Col 1 M-tile 2 N-tile 17
      %a_col1_m2_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n17)
      %b_col1_m2_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n17)
      %c_col1_m2_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1583360 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n17)
      aiex.dma_await_task(%c_col1_m2_n17)
      aiex.dma_free_task(%a_col1_m2_n17)
      aiex.dma_free_task(%b_col1_m2_n17)
      aiex.dma_free_task(%c_col1_m2_n17)
      // Col 1 M-tile 2 N-tile 18
      %a_col1_m2_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n18)
      %b_col1_m2_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n18)
      %c_col1_m2_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1583616 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n18)
      aiex.dma_await_task(%c_col1_m2_n18)
      aiex.dma_free_task(%a_col1_m2_n18)
      aiex.dma_free_task(%b_col1_m2_n18)
      aiex.dma_free_task(%c_col1_m2_n18)
      // Col 1 M-tile 2 N-tile 19
      %a_col1_m2_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n19)
      %b_col1_m2_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n19)
      %c_col1_m2_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1583872 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n19)
      aiex.dma_await_task(%c_col1_m2_n19)
      aiex.dma_free_task(%a_col1_m2_n19)
      aiex.dma_free_task(%b_col1_m2_n19)
      aiex.dma_free_task(%c_col1_m2_n19)
      // Col 1 M-tile 2 N-tile 20
      %a_col1_m2_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n20)
      %b_col1_m2_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n20)
      %c_col1_m2_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1584128 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n20)
      aiex.dma_await_task(%c_col1_m2_n20)
      aiex.dma_free_task(%a_col1_m2_n20)
      aiex.dma_free_task(%b_col1_m2_n20)
      aiex.dma_free_task(%c_col1_m2_n20)
      // Col 1 M-tile 2 N-tile 21
      %a_col1_m2_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n21)
      %b_col1_m2_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n21)
      %c_col1_m2_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1584384 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n21)
      aiex.dma_await_task(%c_col1_m2_n21)
      aiex.dma_free_task(%a_col1_m2_n21)
      aiex.dma_free_task(%b_col1_m2_n21)
      aiex.dma_free_task(%c_col1_m2_n21)
      // Col 1 M-tile 2 N-tile 22
      %a_col1_m2_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n22)
      %b_col1_m2_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n22)
      %c_col1_m2_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1584640 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n22)
      aiex.dma_await_task(%c_col1_m2_n22)
      aiex.dma_free_task(%a_col1_m2_n22)
      aiex.dma_free_task(%b_col1_m2_n22)
      aiex.dma_free_task(%c_col1_m2_n22)
      // Col 1 M-tile 2 N-tile 23
      %a_col1_m2_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m2_n23)
      %b_col1_m2_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m2_n23)
      %c_col1_m2_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 1584896 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m2_n23)
      aiex.dma_await_task(%c_col1_m2_n23)
      aiex.dma_free_task(%a_col1_m2_n23)
      aiex.dma_free_task(%b_col1_m2_n23)
      aiex.dma_free_task(%c_col1_m2_n23)
      // Col 1 M-tile 3 N-tile 0
      %a_col1_m3_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n0)
      %b_col1_m3_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n0)
      %c_col1_m3_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2365440 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n0)
      aiex.dma_await_task(%c_col1_m3_n0)
      aiex.dma_free_task(%a_col1_m3_n0)
      aiex.dma_free_task(%b_col1_m3_n0)
      aiex.dma_free_task(%c_col1_m3_n0)
      // Col 1 M-tile 3 N-tile 1
      %a_col1_m3_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n1)
      %b_col1_m3_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n1)
      %c_col1_m3_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2365696 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n1)
      aiex.dma_await_task(%c_col1_m3_n1)
      aiex.dma_free_task(%a_col1_m3_n1)
      aiex.dma_free_task(%b_col1_m3_n1)
      aiex.dma_free_task(%c_col1_m3_n1)
      // Col 1 M-tile 3 N-tile 2
      %a_col1_m3_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n2)
      %b_col1_m3_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n2)
      %c_col1_m3_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2365952 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n2)
      aiex.dma_await_task(%c_col1_m3_n2)
      aiex.dma_free_task(%a_col1_m3_n2)
      aiex.dma_free_task(%b_col1_m3_n2)
      aiex.dma_free_task(%c_col1_m3_n2)
      // Col 1 M-tile 3 N-tile 3
      %a_col1_m3_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n3)
      %b_col1_m3_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n3)
      %c_col1_m3_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2366208 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n3)
      aiex.dma_await_task(%c_col1_m3_n3)
      aiex.dma_free_task(%a_col1_m3_n3)
      aiex.dma_free_task(%b_col1_m3_n3)
      aiex.dma_free_task(%c_col1_m3_n3)
      // Col 1 M-tile 3 N-tile 4
      %a_col1_m3_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n4)
      %b_col1_m3_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n4)
      %c_col1_m3_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2366464 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n4)
      aiex.dma_await_task(%c_col1_m3_n4)
      aiex.dma_free_task(%a_col1_m3_n4)
      aiex.dma_free_task(%b_col1_m3_n4)
      aiex.dma_free_task(%c_col1_m3_n4)
      // Col 1 M-tile 3 N-tile 5
      %a_col1_m3_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n5)
      %b_col1_m3_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n5)
      %c_col1_m3_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2366720 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n5)
      aiex.dma_await_task(%c_col1_m3_n5)
      aiex.dma_free_task(%a_col1_m3_n5)
      aiex.dma_free_task(%b_col1_m3_n5)
      aiex.dma_free_task(%c_col1_m3_n5)
      // Col 1 M-tile 3 N-tile 6
      %a_col1_m3_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n6)
      %b_col1_m3_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n6)
      %c_col1_m3_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2366976 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n6)
      aiex.dma_await_task(%c_col1_m3_n6)
      aiex.dma_free_task(%a_col1_m3_n6)
      aiex.dma_free_task(%b_col1_m3_n6)
      aiex.dma_free_task(%c_col1_m3_n6)
      // Col 1 M-tile 3 N-tile 7
      %a_col1_m3_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n7)
      %b_col1_m3_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n7)
      %c_col1_m3_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2367232 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n7)
      aiex.dma_await_task(%c_col1_m3_n7)
      aiex.dma_free_task(%a_col1_m3_n7)
      aiex.dma_free_task(%b_col1_m3_n7)
      aiex.dma_free_task(%c_col1_m3_n7)
      // Col 1 M-tile 3 N-tile 8
      %a_col1_m3_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n8)
      %b_col1_m3_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n8)
      %c_col1_m3_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2367488 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n8)
      aiex.dma_await_task(%c_col1_m3_n8)
      aiex.dma_free_task(%a_col1_m3_n8)
      aiex.dma_free_task(%b_col1_m3_n8)
      aiex.dma_free_task(%c_col1_m3_n8)
      // Col 1 M-tile 3 N-tile 9
      %a_col1_m3_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n9)
      %b_col1_m3_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n9)
      %c_col1_m3_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2367744 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n9)
      aiex.dma_await_task(%c_col1_m3_n9)
      aiex.dma_free_task(%a_col1_m3_n9)
      aiex.dma_free_task(%b_col1_m3_n9)
      aiex.dma_free_task(%c_col1_m3_n9)
      // Col 1 M-tile 3 N-tile 10
      %a_col1_m3_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n10)
      %b_col1_m3_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n10)
      %c_col1_m3_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2368000 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n10)
      aiex.dma_await_task(%c_col1_m3_n10)
      aiex.dma_free_task(%a_col1_m3_n10)
      aiex.dma_free_task(%b_col1_m3_n10)
      aiex.dma_free_task(%c_col1_m3_n10)
      // Col 1 M-tile 3 N-tile 11
      %a_col1_m3_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n11)
      %b_col1_m3_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n11)
      %c_col1_m3_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2368256 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n11)
      aiex.dma_await_task(%c_col1_m3_n11)
      aiex.dma_free_task(%a_col1_m3_n11)
      aiex.dma_free_task(%b_col1_m3_n11)
      aiex.dma_free_task(%c_col1_m3_n11)
      // Col 1 M-tile 3 N-tile 12
      %a_col1_m3_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n12)
      %b_col1_m3_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n12)
      %c_col1_m3_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2368512 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n12)
      aiex.dma_await_task(%c_col1_m3_n12)
      aiex.dma_free_task(%a_col1_m3_n12)
      aiex.dma_free_task(%b_col1_m3_n12)
      aiex.dma_free_task(%c_col1_m3_n12)
      // Col 1 M-tile 3 N-tile 13
      %a_col1_m3_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n13)
      %b_col1_m3_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n13)
      %c_col1_m3_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2368768 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n13)
      aiex.dma_await_task(%c_col1_m3_n13)
      aiex.dma_free_task(%a_col1_m3_n13)
      aiex.dma_free_task(%b_col1_m3_n13)
      aiex.dma_free_task(%c_col1_m3_n13)
      // Col 1 M-tile 3 N-tile 14
      %a_col1_m3_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n14)
      %b_col1_m3_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n14)
      %c_col1_m3_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2369024 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n14)
      aiex.dma_await_task(%c_col1_m3_n14)
      aiex.dma_free_task(%a_col1_m3_n14)
      aiex.dma_free_task(%b_col1_m3_n14)
      aiex.dma_free_task(%c_col1_m3_n14)
      // Col 1 M-tile 3 N-tile 15
      %a_col1_m3_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n15)
      %b_col1_m3_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n15)
      %c_col1_m3_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2369280 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n15)
      aiex.dma_await_task(%c_col1_m3_n15)
      aiex.dma_free_task(%a_col1_m3_n15)
      aiex.dma_free_task(%b_col1_m3_n15)
      aiex.dma_free_task(%c_col1_m3_n15)
      // Col 1 M-tile 3 N-tile 16
      %a_col1_m3_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n16)
      %b_col1_m3_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n16)
      %c_col1_m3_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2369536 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n16)
      aiex.dma_await_task(%c_col1_m3_n16)
      aiex.dma_free_task(%a_col1_m3_n16)
      aiex.dma_free_task(%b_col1_m3_n16)
      aiex.dma_free_task(%c_col1_m3_n16)
      // Col 1 M-tile 3 N-tile 17
      %a_col1_m3_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n17)
      %b_col1_m3_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n17)
      %c_col1_m3_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2369792 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n17)
      aiex.dma_await_task(%c_col1_m3_n17)
      aiex.dma_free_task(%a_col1_m3_n17)
      aiex.dma_free_task(%b_col1_m3_n17)
      aiex.dma_free_task(%c_col1_m3_n17)
      // Col 1 M-tile 3 N-tile 18
      %a_col1_m3_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n18)
      %b_col1_m3_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n18)
      %c_col1_m3_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2370048 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n18)
      aiex.dma_await_task(%c_col1_m3_n18)
      aiex.dma_free_task(%a_col1_m3_n18)
      aiex.dma_free_task(%b_col1_m3_n18)
      aiex.dma_free_task(%c_col1_m3_n18)
      // Col 1 M-tile 3 N-tile 19
      %a_col1_m3_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n19)
      %b_col1_m3_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n19)
      %c_col1_m3_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2370304 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n19)
      aiex.dma_await_task(%c_col1_m3_n19)
      aiex.dma_free_task(%a_col1_m3_n19)
      aiex.dma_free_task(%b_col1_m3_n19)
      aiex.dma_free_task(%c_col1_m3_n19)
      // Col 1 M-tile 3 N-tile 20
      %a_col1_m3_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n20)
      %b_col1_m3_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n20)
      %c_col1_m3_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2370560 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n20)
      aiex.dma_await_task(%c_col1_m3_n20)
      aiex.dma_free_task(%a_col1_m3_n20)
      aiex.dma_free_task(%b_col1_m3_n20)
      aiex.dma_free_task(%c_col1_m3_n20)
      // Col 1 M-tile 3 N-tile 21
      %a_col1_m3_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n21)
      %b_col1_m3_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n21)
      %c_col1_m3_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2370816 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n21)
      aiex.dma_await_task(%c_col1_m3_n21)
      aiex.dma_free_task(%a_col1_m3_n21)
      aiex.dma_free_task(%b_col1_m3_n21)
      aiex.dma_free_task(%c_col1_m3_n21)
      // Col 1 M-tile 3 N-tile 22
      %a_col1_m3_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n22)
      %b_col1_m3_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n22)
      %c_col1_m3_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2371072 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n22)
      aiex.dma_await_task(%c_col1_m3_n22)
      aiex.dma_free_task(%a_col1_m3_n22)
      aiex.dma_free_task(%b_col1_m3_n22)
      aiex.dma_free_task(%c_col1_m3_n22)
      // Col 1 M-tile 3 N-tile 23
      %a_col1_m3_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m3_n23)
      %b_col1_m3_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m3_n23)
      %c_col1_m3_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 2371328 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m3_n23)
      aiex.dma_await_task(%c_col1_m3_n23)
      aiex.dma_free_task(%a_col1_m3_n23)
      aiex.dma_free_task(%b_col1_m3_n23)
      aiex.dma_free_task(%c_col1_m3_n23)
      // Col 1 M-tile 4 N-tile 0
      %a_col1_m4_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n0)
      %b_col1_m4_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n0)
      %c_col1_m4_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3151872 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n0)
      aiex.dma_await_task(%c_col1_m4_n0)
      aiex.dma_free_task(%a_col1_m4_n0)
      aiex.dma_free_task(%b_col1_m4_n0)
      aiex.dma_free_task(%c_col1_m4_n0)
      // Col 1 M-tile 4 N-tile 1
      %a_col1_m4_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n1)
      %b_col1_m4_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n1)
      %c_col1_m4_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3152128 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n1)
      aiex.dma_await_task(%c_col1_m4_n1)
      aiex.dma_free_task(%a_col1_m4_n1)
      aiex.dma_free_task(%b_col1_m4_n1)
      aiex.dma_free_task(%c_col1_m4_n1)
      // Col 1 M-tile 4 N-tile 2
      %a_col1_m4_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n2)
      %b_col1_m4_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n2)
      %c_col1_m4_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3152384 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n2)
      aiex.dma_await_task(%c_col1_m4_n2)
      aiex.dma_free_task(%a_col1_m4_n2)
      aiex.dma_free_task(%b_col1_m4_n2)
      aiex.dma_free_task(%c_col1_m4_n2)
      // Col 1 M-tile 4 N-tile 3
      %a_col1_m4_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n3)
      %b_col1_m4_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n3)
      %c_col1_m4_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3152640 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n3)
      aiex.dma_await_task(%c_col1_m4_n3)
      aiex.dma_free_task(%a_col1_m4_n3)
      aiex.dma_free_task(%b_col1_m4_n3)
      aiex.dma_free_task(%c_col1_m4_n3)
      // Col 1 M-tile 4 N-tile 4
      %a_col1_m4_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n4)
      %b_col1_m4_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n4)
      %c_col1_m4_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3152896 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n4)
      aiex.dma_await_task(%c_col1_m4_n4)
      aiex.dma_free_task(%a_col1_m4_n4)
      aiex.dma_free_task(%b_col1_m4_n4)
      aiex.dma_free_task(%c_col1_m4_n4)
      // Col 1 M-tile 4 N-tile 5
      %a_col1_m4_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n5)
      %b_col1_m4_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n5)
      %c_col1_m4_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3153152 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n5)
      aiex.dma_await_task(%c_col1_m4_n5)
      aiex.dma_free_task(%a_col1_m4_n5)
      aiex.dma_free_task(%b_col1_m4_n5)
      aiex.dma_free_task(%c_col1_m4_n5)
      // Col 1 M-tile 4 N-tile 6
      %a_col1_m4_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n6)
      %b_col1_m4_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n6)
      %c_col1_m4_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3153408 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n6)
      aiex.dma_await_task(%c_col1_m4_n6)
      aiex.dma_free_task(%a_col1_m4_n6)
      aiex.dma_free_task(%b_col1_m4_n6)
      aiex.dma_free_task(%c_col1_m4_n6)
      // Col 1 M-tile 4 N-tile 7
      %a_col1_m4_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n7)
      %b_col1_m4_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n7)
      %c_col1_m4_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3153664 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n7)
      aiex.dma_await_task(%c_col1_m4_n7)
      aiex.dma_free_task(%a_col1_m4_n7)
      aiex.dma_free_task(%b_col1_m4_n7)
      aiex.dma_free_task(%c_col1_m4_n7)
      // Col 1 M-tile 4 N-tile 8
      %a_col1_m4_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n8)
      %b_col1_m4_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n8)
      %c_col1_m4_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3153920 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n8)
      aiex.dma_await_task(%c_col1_m4_n8)
      aiex.dma_free_task(%a_col1_m4_n8)
      aiex.dma_free_task(%b_col1_m4_n8)
      aiex.dma_free_task(%c_col1_m4_n8)
      // Col 1 M-tile 4 N-tile 9
      %a_col1_m4_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n9)
      %b_col1_m4_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n9)
      %c_col1_m4_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3154176 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n9)
      aiex.dma_await_task(%c_col1_m4_n9)
      aiex.dma_free_task(%a_col1_m4_n9)
      aiex.dma_free_task(%b_col1_m4_n9)
      aiex.dma_free_task(%c_col1_m4_n9)
      // Col 1 M-tile 4 N-tile 10
      %a_col1_m4_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n10)
      %b_col1_m4_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n10)
      %c_col1_m4_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3154432 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n10)
      aiex.dma_await_task(%c_col1_m4_n10)
      aiex.dma_free_task(%a_col1_m4_n10)
      aiex.dma_free_task(%b_col1_m4_n10)
      aiex.dma_free_task(%c_col1_m4_n10)
      // Col 1 M-tile 4 N-tile 11
      %a_col1_m4_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n11)
      %b_col1_m4_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n11)
      %c_col1_m4_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3154688 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n11)
      aiex.dma_await_task(%c_col1_m4_n11)
      aiex.dma_free_task(%a_col1_m4_n11)
      aiex.dma_free_task(%b_col1_m4_n11)
      aiex.dma_free_task(%c_col1_m4_n11)
      // Col 1 M-tile 4 N-tile 12
      %a_col1_m4_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n12)
      %b_col1_m4_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n12)
      %c_col1_m4_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3154944 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n12)
      aiex.dma_await_task(%c_col1_m4_n12)
      aiex.dma_free_task(%a_col1_m4_n12)
      aiex.dma_free_task(%b_col1_m4_n12)
      aiex.dma_free_task(%c_col1_m4_n12)
      // Col 1 M-tile 4 N-tile 13
      %a_col1_m4_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n13)
      %b_col1_m4_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n13)
      %c_col1_m4_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3155200 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n13)
      aiex.dma_await_task(%c_col1_m4_n13)
      aiex.dma_free_task(%a_col1_m4_n13)
      aiex.dma_free_task(%b_col1_m4_n13)
      aiex.dma_free_task(%c_col1_m4_n13)
      // Col 1 M-tile 4 N-tile 14
      %a_col1_m4_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n14)
      %b_col1_m4_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n14)
      %c_col1_m4_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3155456 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n14)
      aiex.dma_await_task(%c_col1_m4_n14)
      aiex.dma_free_task(%a_col1_m4_n14)
      aiex.dma_free_task(%b_col1_m4_n14)
      aiex.dma_free_task(%c_col1_m4_n14)
      // Col 1 M-tile 4 N-tile 15
      %a_col1_m4_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n15)
      %b_col1_m4_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n15)
      %c_col1_m4_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3155712 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n15)
      aiex.dma_await_task(%c_col1_m4_n15)
      aiex.dma_free_task(%a_col1_m4_n15)
      aiex.dma_free_task(%b_col1_m4_n15)
      aiex.dma_free_task(%c_col1_m4_n15)
      // Col 1 M-tile 4 N-tile 16
      %a_col1_m4_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n16)
      %b_col1_m4_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n16)
      %c_col1_m4_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3155968 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n16)
      aiex.dma_await_task(%c_col1_m4_n16)
      aiex.dma_free_task(%a_col1_m4_n16)
      aiex.dma_free_task(%b_col1_m4_n16)
      aiex.dma_free_task(%c_col1_m4_n16)
      // Col 1 M-tile 4 N-tile 17
      %a_col1_m4_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n17)
      %b_col1_m4_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n17)
      %c_col1_m4_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3156224 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n17)
      aiex.dma_await_task(%c_col1_m4_n17)
      aiex.dma_free_task(%a_col1_m4_n17)
      aiex.dma_free_task(%b_col1_m4_n17)
      aiex.dma_free_task(%c_col1_m4_n17)
      // Col 1 M-tile 4 N-tile 18
      %a_col1_m4_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n18)
      %b_col1_m4_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n18)
      %c_col1_m4_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3156480 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n18)
      aiex.dma_await_task(%c_col1_m4_n18)
      aiex.dma_free_task(%a_col1_m4_n18)
      aiex.dma_free_task(%b_col1_m4_n18)
      aiex.dma_free_task(%c_col1_m4_n18)
      // Col 1 M-tile 4 N-tile 19
      %a_col1_m4_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n19)
      %b_col1_m4_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n19)
      %c_col1_m4_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3156736 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n19)
      aiex.dma_await_task(%c_col1_m4_n19)
      aiex.dma_free_task(%a_col1_m4_n19)
      aiex.dma_free_task(%b_col1_m4_n19)
      aiex.dma_free_task(%c_col1_m4_n19)
      // Col 1 M-tile 4 N-tile 20
      %a_col1_m4_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n20)
      %b_col1_m4_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n20)
      %c_col1_m4_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3156992 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n20)
      aiex.dma_await_task(%c_col1_m4_n20)
      aiex.dma_free_task(%a_col1_m4_n20)
      aiex.dma_free_task(%b_col1_m4_n20)
      aiex.dma_free_task(%c_col1_m4_n20)
      // Col 1 M-tile 4 N-tile 21
      %a_col1_m4_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n21)
      %b_col1_m4_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n21)
      %c_col1_m4_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3157248 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n21)
      aiex.dma_await_task(%c_col1_m4_n21)
      aiex.dma_free_task(%a_col1_m4_n21)
      aiex.dma_free_task(%b_col1_m4_n21)
      aiex.dma_free_task(%c_col1_m4_n21)
      // Col 1 M-tile 4 N-tile 22
      %a_col1_m4_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n22)
      %b_col1_m4_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n22)
      %c_col1_m4_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3157504 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n22)
      aiex.dma_await_task(%c_col1_m4_n22)
      aiex.dma_free_task(%a_col1_m4_n22)
      aiex.dma_free_task(%b_col1_m4_n22)
      aiex.dma_free_task(%c_col1_m4_n22)
      // Col 1 M-tile 4 N-tile 23
      %a_col1_m4_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m4_n23)
      %b_col1_m4_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m4_n23)
      %c_col1_m4_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3157760 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m4_n23)
      aiex.dma_await_task(%c_col1_m4_n23)
      aiex.dma_free_task(%a_col1_m4_n23)
      aiex.dma_free_task(%b_col1_m4_n23)
      aiex.dma_free_task(%c_col1_m4_n23)
      // Col 1 M-tile 5 N-tile 0
      %a_col1_m5_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n0)
      %b_col1_m5_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n0)
      %c_col1_m5_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3938304 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n0)
      aiex.dma_await_task(%c_col1_m5_n0)
      aiex.dma_free_task(%a_col1_m5_n0)
      aiex.dma_free_task(%b_col1_m5_n0)
      aiex.dma_free_task(%c_col1_m5_n0)
      // Col 1 M-tile 5 N-tile 1
      %a_col1_m5_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n1)
      %b_col1_m5_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n1)
      %c_col1_m5_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3938560 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n1)
      aiex.dma_await_task(%c_col1_m5_n1)
      aiex.dma_free_task(%a_col1_m5_n1)
      aiex.dma_free_task(%b_col1_m5_n1)
      aiex.dma_free_task(%c_col1_m5_n1)
      // Col 1 M-tile 5 N-tile 2
      %a_col1_m5_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n2)
      %b_col1_m5_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n2)
      %c_col1_m5_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3938816 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n2)
      aiex.dma_await_task(%c_col1_m5_n2)
      aiex.dma_free_task(%a_col1_m5_n2)
      aiex.dma_free_task(%b_col1_m5_n2)
      aiex.dma_free_task(%c_col1_m5_n2)
      // Col 1 M-tile 5 N-tile 3
      %a_col1_m5_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n3)
      %b_col1_m5_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n3)
      %c_col1_m5_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3939072 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n3)
      aiex.dma_await_task(%c_col1_m5_n3)
      aiex.dma_free_task(%a_col1_m5_n3)
      aiex.dma_free_task(%b_col1_m5_n3)
      aiex.dma_free_task(%c_col1_m5_n3)
      // Col 1 M-tile 5 N-tile 4
      %a_col1_m5_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n4)
      %b_col1_m5_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n4)
      %c_col1_m5_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3939328 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n4)
      aiex.dma_await_task(%c_col1_m5_n4)
      aiex.dma_free_task(%a_col1_m5_n4)
      aiex.dma_free_task(%b_col1_m5_n4)
      aiex.dma_free_task(%c_col1_m5_n4)
      // Col 1 M-tile 5 N-tile 5
      %a_col1_m5_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n5)
      %b_col1_m5_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n5)
      %c_col1_m5_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3939584 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n5)
      aiex.dma_await_task(%c_col1_m5_n5)
      aiex.dma_free_task(%a_col1_m5_n5)
      aiex.dma_free_task(%b_col1_m5_n5)
      aiex.dma_free_task(%c_col1_m5_n5)
      // Col 1 M-tile 5 N-tile 6
      %a_col1_m5_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n6)
      %b_col1_m5_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n6)
      %c_col1_m5_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3939840 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n6)
      aiex.dma_await_task(%c_col1_m5_n6)
      aiex.dma_free_task(%a_col1_m5_n6)
      aiex.dma_free_task(%b_col1_m5_n6)
      aiex.dma_free_task(%c_col1_m5_n6)
      // Col 1 M-tile 5 N-tile 7
      %a_col1_m5_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n7)
      %b_col1_m5_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n7)
      %c_col1_m5_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3940096 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n7)
      aiex.dma_await_task(%c_col1_m5_n7)
      aiex.dma_free_task(%a_col1_m5_n7)
      aiex.dma_free_task(%b_col1_m5_n7)
      aiex.dma_free_task(%c_col1_m5_n7)
      // Col 1 M-tile 5 N-tile 8
      %a_col1_m5_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n8)
      %b_col1_m5_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n8)
      %c_col1_m5_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3940352 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n8)
      aiex.dma_await_task(%c_col1_m5_n8)
      aiex.dma_free_task(%a_col1_m5_n8)
      aiex.dma_free_task(%b_col1_m5_n8)
      aiex.dma_free_task(%c_col1_m5_n8)
      // Col 1 M-tile 5 N-tile 9
      %a_col1_m5_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n9)
      %b_col1_m5_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n9)
      %c_col1_m5_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3940608 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n9)
      aiex.dma_await_task(%c_col1_m5_n9)
      aiex.dma_free_task(%a_col1_m5_n9)
      aiex.dma_free_task(%b_col1_m5_n9)
      aiex.dma_free_task(%c_col1_m5_n9)
      // Col 1 M-tile 5 N-tile 10
      %a_col1_m5_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n10)
      %b_col1_m5_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n10)
      %c_col1_m5_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3940864 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n10)
      aiex.dma_await_task(%c_col1_m5_n10)
      aiex.dma_free_task(%a_col1_m5_n10)
      aiex.dma_free_task(%b_col1_m5_n10)
      aiex.dma_free_task(%c_col1_m5_n10)
      // Col 1 M-tile 5 N-tile 11
      %a_col1_m5_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n11)
      %b_col1_m5_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n11)
      %c_col1_m5_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3941120 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n11)
      aiex.dma_await_task(%c_col1_m5_n11)
      aiex.dma_free_task(%a_col1_m5_n11)
      aiex.dma_free_task(%b_col1_m5_n11)
      aiex.dma_free_task(%c_col1_m5_n11)
      // Col 1 M-tile 5 N-tile 12
      %a_col1_m5_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n12)
      %b_col1_m5_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n12)
      %c_col1_m5_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3941376 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n12)
      aiex.dma_await_task(%c_col1_m5_n12)
      aiex.dma_free_task(%a_col1_m5_n12)
      aiex.dma_free_task(%b_col1_m5_n12)
      aiex.dma_free_task(%c_col1_m5_n12)
      // Col 1 M-tile 5 N-tile 13
      %a_col1_m5_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n13)
      %b_col1_m5_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n13)
      %c_col1_m5_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3941632 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n13)
      aiex.dma_await_task(%c_col1_m5_n13)
      aiex.dma_free_task(%a_col1_m5_n13)
      aiex.dma_free_task(%b_col1_m5_n13)
      aiex.dma_free_task(%c_col1_m5_n13)
      // Col 1 M-tile 5 N-tile 14
      %a_col1_m5_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n14)
      %b_col1_m5_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n14)
      %c_col1_m5_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3941888 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n14)
      aiex.dma_await_task(%c_col1_m5_n14)
      aiex.dma_free_task(%a_col1_m5_n14)
      aiex.dma_free_task(%b_col1_m5_n14)
      aiex.dma_free_task(%c_col1_m5_n14)
      // Col 1 M-tile 5 N-tile 15
      %a_col1_m5_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n15)
      %b_col1_m5_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n15)
      %c_col1_m5_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3942144 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n15)
      aiex.dma_await_task(%c_col1_m5_n15)
      aiex.dma_free_task(%a_col1_m5_n15)
      aiex.dma_free_task(%b_col1_m5_n15)
      aiex.dma_free_task(%c_col1_m5_n15)
      // Col 1 M-tile 5 N-tile 16
      %a_col1_m5_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n16)
      %b_col1_m5_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n16)
      %c_col1_m5_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3942400 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n16)
      aiex.dma_await_task(%c_col1_m5_n16)
      aiex.dma_free_task(%a_col1_m5_n16)
      aiex.dma_free_task(%b_col1_m5_n16)
      aiex.dma_free_task(%c_col1_m5_n16)
      // Col 1 M-tile 5 N-tile 17
      %a_col1_m5_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n17)
      %b_col1_m5_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n17)
      %c_col1_m5_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3942656 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n17)
      aiex.dma_await_task(%c_col1_m5_n17)
      aiex.dma_free_task(%a_col1_m5_n17)
      aiex.dma_free_task(%b_col1_m5_n17)
      aiex.dma_free_task(%c_col1_m5_n17)
      // Col 1 M-tile 5 N-tile 18
      %a_col1_m5_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n18)
      %b_col1_m5_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n18)
      %c_col1_m5_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3942912 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n18)
      aiex.dma_await_task(%c_col1_m5_n18)
      aiex.dma_free_task(%a_col1_m5_n18)
      aiex.dma_free_task(%b_col1_m5_n18)
      aiex.dma_free_task(%c_col1_m5_n18)
      // Col 1 M-tile 5 N-tile 19
      %a_col1_m5_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n19)
      %b_col1_m5_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n19)
      %c_col1_m5_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3943168 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n19)
      aiex.dma_await_task(%c_col1_m5_n19)
      aiex.dma_free_task(%a_col1_m5_n19)
      aiex.dma_free_task(%b_col1_m5_n19)
      aiex.dma_free_task(%c_col1_m5_n19)
      // Col 1 M-tile 5 N-tile 20
      %a_col1_m5_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n20)
      %b_col1_m5_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n20)
      %c_col1_m5_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3943424 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n20)
      aiex.dma_await_task(%c_col1_m5_n20)
      aiex.dma_free_task(%a_col1_m5_n20)
      aiex.dma_free_task(%b_col1_m5_n20)
      aiex.dma_free_task(%c_col1_m5_n20)
      // Col 1 M-tile 5 N-tile 21
      %a_col1_m5_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n21)
      %b_col1_m5_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n21)
      %c_col1_m5_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3943680 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n21)
      aiex.dma_await_task(%c_col1_m5_n21)
      aiex.dma_free_task(%a_col1_m5_n21)
      aiex.dma_free_task(%b_col1_m5_n21)
      aiex.dma_free_task(%c_col1_m5_n21)
      // Col 1 M-tile 5 N-tile 22
      %a_col1_m5_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n22)
      %b_col1_m5_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n22)
      %c_col1_m5_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3943936 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n22)
      aiex.dma_await_task(%c_col1_m5_n22)
      aiex.dma_free_task(%a_col1_m5_n22)
      aiex.dma_free_task(%b_col1_m5_n22)
      aiex.dma_free_task(%c_col1_m5_n22)
      // Col 1 M-tile 5 N-tile 23
      %a_col1_m5_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m5_n23)
      %b_col1_m5_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m5_n23)
      %c_col1_m5_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 3944192 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m5_n23)
      aiex.dma_await_task(%c_col1_m5_n23)
      aiex.dma_free_task(%a_col1_m5_n23)
      aiex.dma_free_task(%b_col1_m5_n23)
      aiex.dma_free_task(%c_col1_m5_n23)
      // Col 1 M-tile 6 N-tile 0
      %a_col1_m6_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n0)
      %b_col1_m6_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n0)
      %c_col1_m6_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4724736 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n0)
      aiex.dma_await_task(%c_col1_m6_n0)
      aiex.dma_free_task(%a_col1_m6_n0)
      aiex.dma_free_task(%b_col1_m6_n0)
      aiex.dma_free_task(%c_col1_m6_n0)
      // Col 1 M-tile 6 N-tile 1
      %a_col1_m6_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n1)
      %b_col1_m6_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n1)
      %c_col1_m6_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4724992 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n1)
      aiex.dma_await_task(%c_col1_m6_n1)
      aiex.dma_free_task(%a_col1_m6_n1)
      aiex.dma_free_task(%b_col1_m6_n1)
      aiex.dma_free_task(%c_col1_m6_n1)
      // Col 1 M-tile 6 N-tile 2
      %a_col1_m6_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n2)
      %b_col1_m6_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n2)
      %c_col1_m6_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4725248 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n2)
      aiex.dma_await_task(%c_col1_m6_n2)
      aiex.dma_free_task(%a_col1_m6_n2)
      aiex.dma_free_task(%b_col1_m6_n2)
      aiex.dma_free_task(%c_col1_m6_n2)
      // Col 1 M-tile 6 N-tile 3
      %a_col1_m6_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n3)
      %b_col1_m6_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n3)
      %c_col1_m6_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4725504 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n3)
      aiex.dma_await_task(%c_col1_m6_n3)
      aiex.dma_free_task(%a_col1_m6_n3)
      aiex.dma_free_task(%b_col1_m6_n3)
      aiex.dma_free_task(%c_col1_m6_n3)
      // Col 1 M-tile 6 N-tile 4
      %a_col1_m6_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n4)
      %b_col1_m6_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n4)
      %c_col1_m6_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4725760 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n4)
      aiex.dma_await_task(%c_col1_m6_n4)
      aiex.dma_free_task(%a_col1_m6_n4)
      aiex.dma_free_task(%b_col1_m6_n4)
      aiex.dma_free_task(%c_col1_m6_n4)
      // Col 1 M-tile 6 N-tile 5
      %a_col1_m6_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n5)
      %b_col1_m6_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n5)
      %c_col1_m6_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4726016 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n5)
      aiex.dma_await_task(%c_col1_m6_n5)
      aiex.dma_free_task(%a_col1_m6_n5)
      aiex.dma_free_task(%b_col1_m6_n5)
      aiex.dma_free_task(%c_col1_m6_n5)
      // Col 1 M-tile 6 N-tile 6
      %a_col1_m6_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n6)
      %b_col1_m6_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n6)
      %c_col1_m6_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4726272 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n6)
      aiex.dma_await_task(%c_col1_m6_n6)
      aiex.dma_free_task(%a_col1_m6_n6)
      aiex.dma_free_task(%b_col1_m6_n6)
      aiex.dma_free_task(%c_col1_m6_n6)
      // Col 1 M-tile 6 N-tile 7
      %a_col1_m6_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n7)
      %b_col1_m6_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n7)
      %c_col1_m6_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4726528 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n7)
      aiex.dma_await_task(%c_col1_m6_n7)
      aiex.dma_free_task(%a_col1_m6_n7)
      aiex.dma_free_task(%b_col1_m6_n7)
      aiex.dma_free_task(%c_col1_m6_n7)
      // Col 1 M-tile 6 N-tile 8
      %a_col1_m6_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n8)
      %b_col1_m6_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n8)
      %c_col1_m6_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4726784 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n8)
      aiex.dma_await_task(%c_col1_m6_n8)
      aiex.dma_free_task(%a_col1_m6_n8)
      aiex.dma_free_task(%b_col1_m6_n8)
      aiex.dma_free_task(%c_col1_m6_n8)
      // Col 1 M-tile 6 N-tile 9
      %a_col1_m6_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n9)
      %b_col1_m6_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n9)
      %c_col1_m6_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4727040 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n9)
      aiex.dma_await_task(%c_col1_m6_n9)
      aiex.dma_free_task(%a_col1_m6_n9)
      aiex.dma_free_task(%b_col1_m6_n9)
      aiex.dma_free_task(%c_col1_m6_n9)
      // Col 1 M-tile 6 N-tile 10
      %a_col1_m6_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n10)
      %b_col1_m6_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n10)
      %c_col1_m6_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4727296 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n10)
      aiex.dma_await_task(%c_col1_m6_n10)
      aiex.dma_free_task(%a_col1_m6_n10)
      aiex.dma_free_task(%b_col1_m6_n10)
      aiex.dma_free_task(%c_col1_m6_n10)
      // Col 1 M-tile 6 N-tile 11
      %a_col1_m6_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n11)
      %b_col1_m6_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n11)
      %c_col1_m6_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4727552 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n11)
      aiex.dma_await_task(%c_col1_m6_n11)
      aiex.dma_free_task(%a_col1_m6_n11)
      aiex.dma_free_task(%b_col1_m6_n11)
      aiex.dma_free_task(%c_col1_m6_n11)
      // Col 1 M-tile 6 N-tile 12
      %a_col1_m6_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n12)
      %b_col1_m6_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n12)
      %c_col1_m6_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4727808 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n12)
      aiex.dma_await_task(%c_col1_m6_n12)
      aiex.dma_free_task(%a_col1_m6_n12)
      aiex.dma_free_task(%b_col1_m6_n12)
      aiex.dma_free_task(%c_col1_m6_n12)
      // Col 1 M-tile 6 N-tile 13
      %a_col1_m6_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n13)
      %b_col1_m6_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n13)
      %c_col1_m6_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4728064 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n13)
      aiex.dma_await_task(%c_col1_m6_n13)
      aiex.dma_free_task(%a_col1_m6_n13)
      aiex.dma_free_task(%b_col1_m6_n13)
      aiex.dma_free_task(%c_col1_m6_n13)
      // Col 1 M-tile 6 N-tile 14
      %a_col1_m6_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n14)
      %b_col1_m6_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n14)
      %c_col1_m6_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4728320 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n14)
      aiex.dma_await_task(%c_col1_m6_n14)
      aiex.dma_free_task(%a_col1_m6_n14)
      aiex.dma_free_task(%b_col1_m6_n14)
      aiex.dma_free_task(%c_col1_m6_n14)
      // Col 1 M-tile 6 N-tile 15
      %a_col1_m6_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n15)
      %b_col1_m6_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n15)
      %c_col1_m6_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4728576 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n15)
      aiex.dma_await_task(%c_col1_m6_n15)
      aiex.dma_free_task(%a_col1_m6_n15)
      aiex.dma_free_task(%b_col1_m6_n15)
      aiex.dma_free_task(%c_col1_m6_n15)
      // Col 1 M-tile 6 N-tile 16
      %a_col1_m6_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n16)
      %b_col1_m6_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n16)
      %c_col1_m6_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4728832 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n16)
      aiex.dma_await_task(%c_col1_m6_n16)
      aiex.dma_free_task(%a_col1_m6_n16)
      aiex.dma_free_task(%b_col1_m6_n16)
      aiex.dma_free_task(%c_col1_m6_n16)
      // Col 1 M-tile 6 N-tile 17
      %a_col1_m6_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n17)
      %b_col1_m6_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n17)
      %c_col1_m6_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4729088 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n17)
      aiex.dma_await_task(%c_col1_m6_n17)
      aiex.dma_free_task(%a_col1_m6_n17)
      aiex.dma_free_task(%b_col1_m6_n17)
      aiex.dma_free_task(%c_col1_m6_n17)
      // Col 1 M-tile 6 N-tile 18
      %a_col1_m6_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n18)
      %b_col1_m6_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n18)
      %c_col1_m6_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4729344 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n18)
      aiex.dma_await_task(%c_col1_m6_n18)
      aiex.dma_free_task(%a_col1_m6_n18)
      aiex.dma_free_task(%b_col1_m6_n18)
      aiex.dma_free_task(%c_col1_m6_n18)
      // Col 1 M-tile 6 N-tile 19
      %a_col1_m6_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n19)
      %b_col1_m6_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n19)
      %c_col1_m6_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4729600 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n19)
      aiex.dma_await_task(%c_col1_m6_n19)
      aiex.dma_free_task(%a_col1_m6_n19)
      aiex.dma_free_task(%b_col1_m6_n19)
      aiex.dma_free_task(%c_col1_m6_n19)
      // Col 1 M-tile 6 N-tile 20
      %a_col1_m6_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n20)
      %b_col1_m6_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n20)
      %c_col1_m6_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4729856 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n20)
      aiex.dma_await_task(%c_col1_m6_n20)
      aiex.dma_free_task(%a_col1_m6_n20)
      aiex.dma_free_task(%b_col1_m6_n20)
      aiex.dma_free_task(%c_col1_m6_n20)
      // Col 1 M-tile 6 N-tile 21
      %a_col1_m6_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n21)
      %b_col1_m6_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n21)
      %c_col1_m6_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4730112 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n21)
      aiex.dma_await_task(%c_col1_m6_n21)
      aiex.dma_free_task(%a_col1_m6_n21)
      aiex.dma_free_task(%b_col1_m6_n21)
      aiex.dma_free_task(%c_col1_m6_n21)
      // Col 1 M-tile 6 N-tile 22
      %a_col1_m6_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n22)
      %b_col1_m6_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n22)
      %c_col1_m6_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4730368 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n22)
      aiex.dma_await_task(%c_col1_m6_n22)
      aiex.dma_free_task(%a_col1_m6_n22)
      aiex.dma_free_task(%b_col1_m6_n22)
      aiex.dma_free_task(%c_col1_m6_n22)
      // Col 1 M-tile 6 N-tile 23
      %a_col1_m6_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m6_n23)
      %b_col1_m6_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m6_n23)
      %c_col1_m6_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 4730624 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m6_n23)
      aiex.dma_await_task(%c_col1_m6_n23)
      aiex.dma_free_task(%a_col1_m6_n23)
      aiex.dma_free_task(%b_col1_m6_n23)
      aiex.dma_free_task(%c_col1_m6_n23)
      // Col 1 M-tile 7 N-tile 0
      %a_col1_m7_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n0)
      %b_col1_m7_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n0)
      %c_col1_m7_n0 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5511168 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n0)
      aiex.dma_await_task(%c_col1_m7_n0)
      aiex.dma_free_task(%a_col1_m7_n0)
      aiex.dma_free_task(%b_col1_m7_n0)
      aiex.dma_free_task(%c_col1_m7_n0)
      // Col 1 M-tile 7 N-tile 1
      %a_col1_m7_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n1)
      %b_col1_m7_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n1)
      %c_col1_m7_n1 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5511424 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n1)
      aiex.dma_await_task(%c_col1_m7_n1)
      aiex.dma_free_task(%a_col1_m7_n1)
      aiex.dma_free_task(%b_col1_m7_n1)
      aiex.dma_free_task(%c_col1_m7_n1)
      // Col 1 M-tile 7 N-tile 2
      %a_col1_m7_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n2)
      %b_col1_m7_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n2)
      %c_col1_m7_n2 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5511680 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n2)
      aiex.dma_await_task(%c_col1_m7_n2)
      aiex.dma_free_task(%a_col1_m7_n2)
      aiex.dma_free_task(%b_col1_m7_n2)
      aiex.dma_free_task(%c_col1_m7_n2)
      // Col 1 M-tile 7 N-tile 3
      %a_col1_m7_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n3)
      %b_col1_m7_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n3)
      %c_col1_m7_n3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5511936 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n3)
      aiex.dma_await_task(%c_col1_m7_n3)
      aiex.dma_free_task(%a_col1_m7_n3)
      aiex.dma_free_task(%b_col1_m7_n3)
      aiex.dma_free_task(%c_col1_m7_n3)
      // Col 1 M-tile 7 N-tile 4
      %a_col1_m7_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n4)
      %b_col1_m7_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n4)
      %c_col1_m7_n4 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5512192 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n4)
      aiex.dma_await_task(%c_col1_m7_n4)
      aiex.dma_free_task(%a_col1_m7_n4)
      aiex.dma_free_task(%b_col1_m7_n4)
      aiex.dma_free_task(%c_col1_m7_n4)
      // Col 1 M-tile 7 N-tile 5
      %a_col1_m7_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n5)
      %b_col1_m7_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n5)
      %c_col1_m7_n5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5512448 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n5)
      aiex.dma_await_task(%c_col1_m7_n5)
      aiex.dma_free_task(%a_col1_m7_n5)
      aiex.dma_free_task(%b_col1_m7_n5)
      aiex.dma_free_task(%c_col1_m7_n5)
      // Col 1 M-tile 7 N-tile 6
      %a_col1_m7_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n6)
      %b_col1_m7_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n6)
      %c_col1_m7_n6 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5512704 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n6)
      aiex.dma_await_task(%c_col1_m7_n6)
      aiex.dma_free_task(%a_col1_m7_n6)
      aiex.dma_free_task(%b_col1_m7_n6)
      aiex.dma_free_task(%c_col1_m7_n6)
      // Col 1 M-tile 7 N-tile 7
      %a_col1_m7_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n7)
      %b_col1_m7_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n7)
      %c_col1_m7_n7 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5512960 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n7)
      aiex.dma_await_task(%c_col1_m7_n7)
      aiex.dma_free_task(%a_col1_m7_n7)
      aiex.dma_free_task(%b_col1_m7_n7)
      aiex.dma_free_task(%c_col1_m7_n7)
      // Col 1 M-tile 7 N-tile 8
      %a_col1_m7_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n8)
      %b_col1_m7_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n8)
      %c_col1_m7_n8 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5513216 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n8)
      aiex.dma_await_task(%c_col1_m7_n8)
      aiex.dma_free_task(%a_col1_m7_n8)
      aiex.dma_free_task(%b_col1_m7_n8)
      aiex.dma_free_task(%c_col1_m7_n8)
      // Col 1 M-tile 7 N-tile 9
      %a_col1_m7_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n9)
      %b_col1_m7_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n9)
      %c_col1_m7_n9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5513472 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n9)
      aiex.dma_await_task(%c_col1_m7_n9)
      aiex.dma_free_task(%a_col1_m7_n9)
      aiex.dma_free_task(%b_col1_m7_n9)
      aiex.dma_free_task(%c_col1_m7_n9)
      // Col 1 M-tile 7 N-tile 10
      %a_col1_m7_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n10)
      %b_col1_m7_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n10)
      %c_col1_m7_n10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5513728 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n10)
      aiex.dma_await_task(%c_col1_m7_n10)
      aiex.dma_free_task(%a_col1_m7_n10)
      aiex.dma_free_task(%b_col1_m7_n10)
      aiex.dma_free_task(%c_col1_m7_n10)
      // Col 1 M-tile 7 N-tile 11
      %a_col1_m7_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n11)
      %b_col1_m7_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n11)
      %c_col1_m7_n11 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5513984 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n11)
      aiex.dma_await_task(%c_col1_m7_n11)
      aiex.dma_free_task(%a_col1_m7_n11)
      aiex.dma_free_task(%b_col1_m7_n11)
      aiex.dma_free_task(%c_col1_m7_n11)
      // Col 1 M-tile 7 N-tile 12
      %a_col1_m7_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n12)
      %b_col1_m7_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n12)
      %c_col1_m7_n12 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5514240 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n12)
      aiex.dma_await_task(%c_col1_m7_n12)
      aiex.dma_free_task(%a_col1_m7_n12)
      aiex.dma_free_task(%b_col1_m7_n12)
      aiex.dma_free_task(%c_col1_m7_n12)
      // Col 1 M-tile 7 N-tile 13
      %a_col1_m7_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n13)
      %b_col1_m7_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n13)
      %c_col1_m7_n13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5514496 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n13)
      aiex.dma_await_task(%c_col1_m7_n13)
      aiex.dma_free_task(%a_col1_m7_n13)
      aiex.dma_free_task(%b_col1_m7_n13)
      aiex.dma_free_task(%c_col1_m7_n13)
      // Col 1 M-tile 7 N-tile 14
      %a_col1_m7_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n14)
      %b_col1_m7_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n14)
      %c_col1_m7_n14 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5514752 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n14)
      aiex.dma_await_task(%c_col1_m7_n14)
      aiex.dma_free_task(%a_col1_m7_n14)
      aiex.dma_free_task(%b_col1_m7_n14)
      aiex.dma_free_task(%c_col1_m7_n14)
      // Col 1 M-tile 7 N-tile 15
      %a_col1_m7_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n15)
      %b_col1_m7_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n15)
      %c_col1_m7_n15 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5515008 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n15)
      aiex.dma_await_task(%c_col1_m7_n15)
      aiex.dma_free_task(%a_col1_m7_n15)
      aiex.dma_free_task(%b_col1_m7_n15)
      aiex.dma_free_task(%c_col1_m7_n15)
      // Col 1 M-tile 7 N-tile 16
      %a_col1_m7_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n16)
      %b_col1_m7_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n16)
      %c_col1_m7_n16 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5515264 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n16)
      aiex.dma_await_task(%c_col1_m7_n16)
      aiex.dma_free_task(%a_col1_m7_n16)
      aiex.dma_free_task(%b_col1_m7_n16)
      aiex.dma_free_task(%c_col1_m7_n16)
      // Col 1 M-tile 7 N-tile 17
      %a_col1_m7_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n17)
      %b_col1_m7_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n17)
      %c_col1_m7_n17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5515520 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n17)
      aiex.dma_await_task(%c_col1_m7_n17)
      aiex.dma_free_task(%a_col1_m7_n17)
      aiex.dma_free_task(%b_col1_m7_n17)
      aiex.dma_free_task(%c_col1_m7_n17)
      // Col 1 M-tile 7 N-tile 18
      %a_col1_m7_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n18)
      %b_col1_m7_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n18)
      %c_col1_m7_n18 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5515776 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n18)
      aiex.dma_await_task(%c_col1_m7_n18)
      aiex.dma_free_task(%a_col1_m7_n18)
      aiex.dma_free_task(%b_col1_m7_n18)
      aiex.dma_free_task(%c_col1_m7_n18)
      // Col 1 M-tile 7 N-tile 19
      %a_col1_m7_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n19)
      %b_col1_m7_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n19)
      %c_col1_m7_n19 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5516032 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n19)
      aiex.dma_await_task(%c_col1_m7_n19)
      aiex.dma_free_task(%a_col1_m7_n19)
      aiex.dma_free_task(%b_col1_m7_n19)
      aiex.dma_free_task(%c_col1_m7_n19)
      // Col 1 M-tile 7 N-tile 20
      %a_col1_m7_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n20)
      %b_col1_m7_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n20)
      %c_col1_m7_n20 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5516288 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n20)
      aiex.dma_await_task(%c_col1_m7_n20)
      aiex.dma_free_task(%a_col1_m7_n20)
      aiex.dma_free_task(%b_col1_m7_n20)
      aiex.dma_free_task(%c_col1_m7_n20)
      // Col 1 M-tile 7 N-tile 21
      %a_col1_m7_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n21)
      %b_col1_m7_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n21)
      %c_col1_m7_n21 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5516544 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n21)
      aiex.dma_await_task(%c_col1_m7_n21)
      aiex.dma_free_task(%a_col1_m7_n21)
      aiex.dma_free_task(%b_col1_m7_n21)
      aiex.dma_free_task(%c_col1_m7_n21)
      // Col 1 M-tile 7 N-tile 22
      %a_col1_m7_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n22)
      %b_col1_m7_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n22)
      %c_col1_m7_n22 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5516800 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n22)
      aiex.dma_await_task(%c_col1_m7_n22)
      aiex.dma_free_task(%a_col1_m7_n22)
      aiex.dma_free_task(%b_col1_m7_n22)
      aiex.dma_free_task(%c_col1_m7_n22)
      // Col 1 M-tile 7 N-tile 23
      %a_col1_m7_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 4096 sizes = [1, 64, 64] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col1_m7_n23)
      %b_col1_m7_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<1024x3072xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col1_m7_n23)
      %c_col1_m7_n23 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%C_out : memref<512x3072xi32> offset = 5517056 len = 16384 sizes = [1, 64, 64] strides = [0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col1_m7_n23)
      aiex.dma_await_task(%c_col1_m7_n23)
      aiex.dma_free_task(%a_col1_m7_n23)
      aiex.dma_free_task(%b_col1_m7_n23)
      aiex.dma_free_task(%c_col1_m7_n23)
    }
  }
}