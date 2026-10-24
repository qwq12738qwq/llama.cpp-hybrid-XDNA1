// Multi-column INT8 GEMM: M=512 K=32 N=3584 n_cols=1
// Tile: Tm=64 Tk=64 Tn=64
// n_per_col=3584 nb_m=8 nb_k=0 nb_n=56
module {
  aie.device(npu1) {
    %tile_shim_c0 = aie.tile(0, 0)
    %tile_mem_c0 = aie.tile(0, 1)
    %tile_core0 = aie.tile(0, 2)
    %tile_shim_a = aie.tile(1, 0)
    %tile_mem_a = aie.tile(1, 1)
    %tile_shim_b = aie.tile(2, 0)
    %tile_mem_b = aie.tile(2, 1)

    aie.objectfifo @A_L3L2(%tile_shim_a, {%tile_mem_a}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @A_L2L1(%tile_mem_a, {%tile_core0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [0])

    aie.objectfifo @B_L3L2(%tile_shim_b, {%tile_mem_b}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @B_L2L1(%tile_mem_b, {%tile_core0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0])

    aie.objectfifo @C_L1L2_0(%tile_core0, {%tile_mem_c0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @C_L2L3_0(%tile_mem_c0, {%tile_shim_c0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@C_L1L2_0] -> [@C_L2L3_0]([] [0])

    func.func private @"matmul_i8_i32"(memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) attributes {link_with = "matmul_i8_i32.o"}

    // Core 0
    %core0 = aie.core(%tile_core0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c_mn = arith.constant 448 : index
      %c_C32 = arith.constant 4096 : index
      %c_k = arith.constant 0 : index
      scf.for %iter = %c0 to %c_mn step %c1 {
        %a = aie.objectfifo.acquire @A_L2L1(Consume, 1) : memref<4096xi8>
        %b = aie.objectfifo.acquire @B_L2L1(Consume, 1) : memref<4096xi8>
        %c_out = aie.objectfifo.acquire @C_L1L2_0(Produce, 1) : memref<4096xi32>
        %c0_i32 = arith.constant 0 : i32
        scf.for %i = %c0 to %c_C32 step %c1 {
          memref.store %c0_i32, %c_out[%i] : memref<4096xi32>
        }
        scf.for %k = %c0 to %c_k step %c1 {
          func.call @matmul_i8_i32(%a, %b, %c_out) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) -> ()
        }
        aie.objectfifo.release @A_L2L1(Consume, 1)
        aie.objectfifo.release @B_L2L1(Consume, 1)
        aie.objectfifo.release @C_L1L2_0(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}

    aie.runtime_sequence(%A: memref<512x32xi8>, %B: memref<32x3584xi8>, %C_out: memref<512x3584xi32>) {
      // Col 0 M-tile 0 N-tile 0
      %a_col0_m0_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n0)
      %b_col0_m0_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n0)
      %c_col0_m0_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 0 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n0)
      aiex.dma_await_task(%c_col0_m0_n0)
      aiex.dma_free_task(%a_col0_m0_n0)
      aiex.dma_free_task(%b_col0_m0_n0)
      aiex.dma_free_task(%c_col0_m0_n0)
      // Col 0 M-tile 0 N-tile 1
      %a_col0_m0_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n1)
      %b_col0_m0_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n1)
      %c_col0_m0_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 256 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n1)
      aiex.dma_await_task(%c_col0_m0_n1)
      aiex.dma_free_task(%a_col0_m0_n1)
      aiex.dma_free_task(%b_col0_m0_n1)
      aiex.dma_free_task(%c_col0_m0_n1)
      // Col 0 M-tile 0 N-tile 2
      %a_col0_m0_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n2)
      %b_col0_m0_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n2)
      %c_col0_m0_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 512 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n2)
      aiex.dma_await_task(%c_col0_m0_n2)
      aiex.dma_free_task(%a_col0_m0_n2)
      aiex.dma_free_task(%b_col0_m0_n2)
      aiex.dma_free_task(%c_col0_m0_n2)
      // Col 0 M-tile 0 N-tile 3
      %a_col0_m0_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n3)
      %b_col0_m0_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n3)
      %c_col0_m0_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 768 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n3)
      aiex.dma_await_task(%c_col0_m0_n3)
      aiex.dma_free_task(%a_col0_m0_n3)
      aiex.dma_free_task(%b_col0_m0_n3)
      aiex.dma_free_task(%c_col0_m0_n3)
      // Col 0 M-tile 0 N-tile 4
      %a_col0_m0_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n4)
      %b_col0_m0_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n4)
      %c_col0_m0_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1024 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n4)
      aiex.dma_await_task(%c_col0_m0_n4)
      aiex.dma_free_task(%a_col0_m0_n4)
      aiex.dma_free_task(%b_col0_m0_n4)
      aiex.dma_free_task(%c_col0_m0_n4)
      // Col 0 M-tile 0 N-tile 5
      %a_col0_m0_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n5)
      %b_col0_m0_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n5)
      %c_col0_m0_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1280 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n5)
      aiex.dma_await_task(%c_col0_m0_n5)
      aiex.dma_free_task(%a_col0_m0_n5)
      aiex.dma_free_task(%b_col0_m0_n5)
      aiex.dma_free_task(%c_col0_m0_n5)
      // Col 0 M-tile 0 N-tile 6
      %a_col0_m0_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n6)
      %b_col0_m0_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n6)
      %c_col0_m0_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1536 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n6)
      aiex.dma_await_task(%c_col0_m0_n6)
      aiex.dma_free_task(%a_col0_m0_n6)
      aiex.dma_free_task(%b_col0_m0_n6)
      aiex.dma_free_task(%c_col0_m0_n6)
      // Col 0 M-tile 0 N-tile 7
      %a_col0_m0_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n7)
      %b_col0_m0_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n7)
      %c_col0_m0_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1792 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n7)
      aiex.dma_await_task(%c_col0_m0_n7)
      aiex.dma_free_task(%a_col0_m0_n7)
      aiex.dma_free_task(%b_col0_m0_n7)
      aiex.dma_free_task(%c_col0_m0_n7)
      // Col 0 M-tile 0 N-tile 8
      %a_col0_m0_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n8)
      %b_col0_m0_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n8)
      %c_col0_m0_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2048 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n8)
      aiex.dma_await_task(%c_col0_m0_n8)
      aiex.dma_free_task(%a_col0_m0_n8)
      aiex.dma_free_task(%b_col0_m0_n8)
      aiex.dma_free_task(%c_col0_m0_n8)
      // Col 0 M-tile 0 N-tile 9
      %a_col0_m0_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n9)
      %b_col0_m0_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n9)
      %c_col0_m0_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2304 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n9)
      aiex.dma_await_task(%c_col0_m0_n9)
      aiex.dma_free_task(%a_col0_m0_n9)
      aiex.dma_free_task(%b_col0_m0_n9)
      aiex.dma_free_task(%c_col0_m0_n9)
      // Col 0 M-tile 0 N-tile 10
      %a_col0_m0_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n10)
      %b_col0_m0_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n10)
      %c_col0_m0_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2560 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n10)
      aiex.dma_await_task(%c_col0_m0_n10)
      aiex.dma_free_task(%a_col0_m0_n10)
      aiex.dma_free_task(%b_col0_m0_n10)
      aiex.dma_free_task(%c_col0_m0_n10)
      // Col 0 M-tile 0 N-tile 11
      %a_col0_m0_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n11)
      %b_col0_m0_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n11)
      %c_col0_m0_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2816 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n11)
      aiex.dma_await_task(%c_col0_m0_n11)
      aiex.dma_free_task(%a_col0_m0_n11)
      aiex.dma_free_task(%b_col0_m0_n11)
      aiex.dma_free_task(%c_col0_m0_n11)
      // Col 0 M-tile 0 N-tile 12
      %a_col0_m0_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n12)
      %b_col0_m0_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n12)
      %c_col0_m0_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3072 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n12)
      aiex.dma_await_task(%c_col0_m0_n12)
      aiex.dma_free_task(%a_col0_m0_n12)
      aiex.dma_free_task(%b_col0_m0_n12)
      aiex.dma_free_task(%c_col0_m0_n12)
      // Col 0 M-tile 0 N-tile 13
      %a_col0_m0_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n13)
      %b_col0_m0_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n13)
      %c_col0_m0_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3328 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n13)
      aiex.dma_await_task(%c_col0_m0_n13)
      aiex.dma_free_task(%a_col0_m0_n13)
      aiex.dma_free_task(%b_col0_m0_n13)
      aiex.dma_free_task(%c_col0_m0_n13)
      // Col 0 M-tile 0 N-tile 14
      %a_col0_m0_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n14)
      %b_col0_m0_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n14)
      %c_col0_m0_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3584 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n14)
      aiex.dma_await_task(%c_col0_m0_n14)
      aiex.dma_free_task(%a_col0_m0_n14)
      aiex.dma_free_task(%b_col0_m0_n14)
      aiex.dma_free_task(%c_col0_m0_n14)
      // Col 0 M-tile 0 N-tile 15
      %a_col0_m0_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n15)
      %b_col0_m0_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n15)
      %c_col0_m0_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3840 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n15)
      aiex.dma_await_task(%c_col0_m0_n15)
      aiex.dma_free_task(%a_col0_m0_n15)
      aiex.dma_free_task(%b_col0_m0_n15)
      aiex.dma_free_task(%c_col0_m0_n15)
      // Col 0 M-tile 0 N-tile 16
      %a_col0_m0_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n16)
      %b_col0_m0_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n16)
      %c_col0_m0_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4096 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n16)
      aiex.dma_await_task(%c_col0_m0_n16)
      aiex.dma_free_task(%a_col0_m0_n16)
      aiex.dma_free_task(%b_col0_m0_n16)
      aiex.dma_free_task(%c_col0_m0_n16)
      // Col 0 M-tile 0 N-tile 17
      %a_col0_m0_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n17)
      %b_col0_m0_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n17)
      %c_col0_m0_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4352 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n17)
      aiex.dma_await_task(%c_col0_m0_n17)
      aiex.dma_free_task(%a_col0_m0_n17)
      aiex.dma_free_task(%b_col0_m0_n17)
      aiex.dma_free_task(%c_col0_m0_n17)
      // Col 0 M-tile 0 N-tile 18
      %a_col0_m0_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n18)
      %b_col0_m0_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n18)
      %c_col0_m0_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4608 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n18)
      aiex.dma_await_task(%c_col0_m0_n18)
      aiex.dma_free_task(%a_col0_m0_n18)
      aiex.dma_free_task(%b_col0_m0_n18)
      aiex.dma_free_task(%c_col0_m0_n18)
      // Col 0 M-tile 0 N-tile 19
      %a_col0_m0_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n19)
      %b_col0_m0_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n19)
      %c_col0_m0_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4864 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n19)
      aiex.dma_await_task(%c_col0_m0_n19)
      aiex.dma_free_task(%a_col0_m0_n19)
      aiex.dma_free_task(%b_col0_m0_n19)
      aiex.dma_free_task(%c_col0_m0_n19)
      // Col 0 M-tile 0 N-tile 20
      %a_col0_m0_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n20)
      %b_col0_m0_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n20)
      %c_col0_m0_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5120 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n20)
      aiex.dma_await_task(%c_col0_m0_n20)
      aiex.dma_free_task(%a_col0_m0_n20)
      aiex.dma_free_task(%b_col0_m0_n20)
      aiex.dma_free_task(%c_col0_m0_n20)
      // Col 0 M-tile 0 N-tile 21
      %a_col0_m0_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n21)
      %b_col0_m0_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n21)
      %c_col0_m0_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5376 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n21)
      aiex.dma_await_task(%c_col0_m0_n21)
      aiex.dma_free_task(%a_col0_m0_n21)
      aiex.dma_free_task(%b_col0_m0_n21)
      aiex.dma_free_task(%c_col0_m0_n21)
      // Col 0 M-tile 0 N-tile 22
      %a_col0_m0_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n22)
      %b_col0_m0_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n22)
      %c_col0_m0_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5632 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n22)
      aiex.dma_await_task(%c_col0_m0_n22)
      aiex.dma_free_task(%a_col0_m0_n22)
      aiex.dma_free_task(%b_col0_m0_n22)
      aiex.dma_free_task(%c_col0_m0_n22)
      // Col 0 M-tile 0 N-tile 23
      %a_col0_m0_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n23)
      %b_col0_m0_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n23)
      %c_col0_m0_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5888 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n23)
      aiex.dma_await_task(%c_col0_m0_n23)
      aiex.dma_free_task(%a_col0_m0_n23)
      aiex.dma_free_task(%b_col0_m0_n23)
      aiex.dma_free_task(%c_col0_m0_n23)
      // Col 0 M-tile 0 N-tile 24
      %a_col0_m0_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n24)
      %b_col0_m0_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n24)
      %c_col0_m0_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6144 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n24)
      aiex.dma_await_task(%c_col0_m0_n24)
      aiex.dma_free_task(%a_col0_m0_n24)
      aiex.dma_free_task(%b_col0_m0_n24)
      aiex.dma_free_task(%c_col0_m0_n24)
      // Col 0 M-tile 0 N-tile 25
      %a_col0_m0_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n25)
      %b_col0_m0_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n25)
      %c_col0_m0_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6400 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n25)
      aiex.dma_await_task(%c_col0_m0_n25)
      aiex.dma_free_task(%a_col0_m0_n25)
      aiex.dma_free_task(%b_col0_m0_n25)
      aiex.dma_free_task(%c_col0_m0_n25)
      // Col 0 M-tile 0 N-tile 26
      %a_col0_m0_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n26)
      %b_col0_m0_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n26)
      %c_col0_m0_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6656 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n26)
      aiex.dma_await_task(%c_col0_m0_n26)
      aiex.dma_free_task(%a_col0_m0_n26)
      aiex.dma_free_task(%b_col0_m0_n26)
      aiex.dma_free_task(%c_col0_m0_n26)
      // Col 0 M-tile 0 N-tile 27
      %a_col0_m0_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n27)
      %b_col0_m0_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n27)
      %c_col0_m0_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6912 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n27)
      aiex.dma_await_task(%c_col0_m0_n27)
      aiex.dma_free_task(%a_col0_m0_n27)
      aiex.dma_free_task(%b_col0_m0_n27)
      aiex.dma_free_task(%c_col0_m0_n27)
      // Col 0 M-tile 0 N-tile 28
      %a_col0_m0_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n28)
      %b_col0_m0_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n28)
      %c_col0_m0_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 7168 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n28)
      aiex.dma_await_task(%c_col0_m0_n28)
      aiex.dma_free_task(%a_col0_m0_n28)
      aiex.dma_free_task(%b_col0_m0_n28)
      aiex.dma_free_task(%c_col0_m0_n28)
      // Col 0 M-tile 0 N-tile 29
      %a_col0_m0_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n29)
      %b_col0_m0_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n29)
      %c_col0_m0_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 7424 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n29)
      aiex.dma_await_task(%c_col0_m0_n29)
      aiex.dma_free_task(%a_col0_m0_n29)
      aiex.dma_free_task(%b_col0_m0_n29)
      aiex.dma_free_task(%c_col0_m0_n29)
      // Col 0 M-tile 0 N-tile 30
      %a_col0_m0_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n30)
      %b_col0_m0_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n30)
      %c_col0_m0_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 7680 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n30)
      aiex.dma_await_task(%c_col0_m0_n30)
      aiex.dma_free_task(%a_col0_m0_n30)
      aiex.dma_free_task(%b_col0_m0_n30)
      aiex.dma_free_task(%c_col0_m0_n30)
      // Col 0 M-tile 0 N-tile 31
      %a_col0_m0_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n31)
      %b_col0_m0_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n31)
      %c_col0_m0_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 7936 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n31)
      aiex.dma_await_task(%c_col0_m0_n31)
      aiex.dma_free_task(%a_col0_m0_n31)
      aiex.dma_free_task(%b_col0_m0_n31)
      aiex.dma_free_task(%c_col0_m0_n31)
      // Col 0 M-tile 0 N-tile 32
      %a_col0_m0_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n32)
      %b_col0_m0_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n32)
      %c_col0_m0_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 8192 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n32)
      aiex.dma_await_task(%c_col0_m0_n32)
      aiex.dma_free_task(%a_col0_m0_n32)
      aiex.dma_free_task(%b_col0_m0_n32)
      aiex.dma_free_task(%c_col0_m0_n32)
      // Col 0 M-tile 0 N-tile 33
      %a_col0_m0_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n33)
      %b_col0_m0_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n33)
      %c_col0_m0_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 8448 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n33)
      aiex.dma_await_task(%c_col0_m0_n33)
      aiex.dma_free_task(%a_col0_m0_n33)
      aiex.dma_free_task(%b_col0_m0_n33)
      aiex.dma_free_task(%c_col0_m0_n33)
      // Col 0 M-tile 0 N-tile 34
      %a_col0_m0_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n34)
      %b_col0_m0_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n34)
      %c_col0_m0_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 8704 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n34)
      aiex.dma_await_task(%c_col0_m0_n34)
      aiex.dma_free_task(%a_col0_m0_n34)
      aiex.dma_free_task(%b_col0_m0_n34)
      aiex.dma_free_task(%c_col0_m0_n34)
      // Col 0 M-tile 0 N-tile 35
      %a_col0_m0_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n35)
      %b_col0_m0_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n35)
      %c_col0_m0_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 8960 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n35)
      aiex.dma_await_task(%c_col0_m0_n35)
      aiex.dma_free_task(%a_col0_m0_n35)
      aiex.dma_free_task(%b_col0_m0_n35)
      aiex.dma_free_task(%c_col0_m0_n35)
      // Col 0 M-tile 0 N-tile 36
      %a_col0_m0_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n36)
      %b_col0_m0_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n36)
      %c_col0_m0_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 9216 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n36)
      aiex.dma_await_task(%c_col0_m0_n36)
      aiex.dma_free_task(%a_col0_m0_n36)
      aiex.dma_free_task(%b_col0_m0_n36)
      aiex.dma_free_task(%c_col0_m0_n36)
      // Col 0 M-tile 0 N-tile 37
      %a_col0_m0_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n37)
      %b_col0_m0_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n37)
      %c_col0_m0_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 9472 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n37)
      aiex.dma_await_task(%c_col0_m0_n37)
      aiex.dma_free_task(%a_col0_m0_n37)
      aiex.dma_free_task(%b_col0_m0_n37)
      aiex.dma_free_task(%c_col0_m0_n37)
      // Col 0 M-tile 0 N-tile 38
      %a_col0_m0_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n38)
      %b_col0_m0_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n38)
      %c_col0_m0_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 9728 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n38)
      aiex.dma_await_task(%c_col0_m0_n38)
      aiex.dma_free_task(%a_col0_m0_n38)
      aiex.dma_free_task(%b_col0_m0_n38)
      aiex.dma_free_task(%c_col0_m0_n38)
      // Col 0 M-tile 0 N-tile 39
      %a_col0_m0_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n39)
      %b_col0_m0_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n39)
      %c_col0_m0_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 9984 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n39)
      aiex.dma_await_task(%c_col0_m0_n39)
      aiex.dma_free_task(%a_col0_m0_n39)
      aiex.dma_free_task(%b_col0_m0_n39)
      aiex.dma_free_task(%c_col0_m0_n39)
      // Col 0 M-tile 0 N-tile 40
      %a_col0_m0_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n40)
      %b_col0_m0_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n40)
      %c_col0_m0_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 10240 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n40)
      aiex.dma_await_task(%c_col0_m0_n40)
      aiex.dma_free_task(%a_col0_m0_n40)
      aiex.dma_free_task(%b_col0_m0_n40)
      aiex.dma_free_task(%c_col0_m0_n40)
      // Col 0 M-tile 0 N-tile 41
      %a_col0_m0_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n41)
      %b_col0_m0_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n41)
      %c_col0_m0_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 10496 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n41)
      aiex.dma_await_task(%c_col0_m0_n41)
      aiex.dma_free_task(%a_col0_m0_n41)
      aiex.dma_free_task(%b_col0_m0_n41)
      aiex.dma_free_task(%c_col0_m0_n41)
      // Col 0 M-tile 0 N-tile 42
      %a_col0_m0_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n42)
      %b_col0_m0_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n42)
      %c_col0_m0_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 10752 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n42)
      aiex.dma_await_task(%c_col0_m0_n42)
      aiex.dma_free_task(%a_col0_m0_n42)
      aiex.dma_free_task(%b_col0_m0_n42)
      aiex.dma_free_task(%c_col0_m0_n42)
      // Col 0 M-tile 0 N-tile 43
      %a_col0_m0_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n43)
      %b_col0_m0_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n43)
      %c_col0_m0_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 11008 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n43)
      aiex.dma_await_task(%c_col0_m0_n43)
      aiex.dma_free_task(%a_col0_m0_n43)
      aiex.dma_free_task(%b_col0_m0_n43)
      aiex.dma_free_task(%c_col0_m0_n43)
      // Col 0 M-tile 0 N-tile 44
      %a_col0_m0_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n44)
      %b_col0_m0_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n44)
      %c_col0_m0_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 11264 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n44)
      aiex.dma_await_task(%c_col0_m0_n44)
      aiex.dma_free_task(%a_col0_m0_n44)
      aiex.dma_free_task(%b_col0_m0_n44)
      aiex.dma_free_task(%c_col0_m0_n44)
      // Col 0 M-tile 0 N-tile 45
      %a_col0_m0_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n45)
      %b_col0_m0_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n45)
      %c_col0_m0_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 11520 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n45)
      aiex.dma_await_task(%c_col0_m0_n45)
      aiex.dma_free_task(%a_col0_m0_n45)
      aiex.dma_free_task(%b_col0_m0_n45)
      aiex.dma_free_task(%c_col0_m0_n45)
      // Col 0 M-tile 0 N-tile 46
      %a_col0_m0_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n46)
      %b_col0_m0_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n46)
      %c_col0_m0_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 11776 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n46)
      aiex.dma_await_task(%c_col0_m0_n46)
      aiex.dma_free_task(%a_col0_m0_n46)
      aiex.dma_free_task(%b_col0_m0_n46)
      aiex.dma_free_task(%c_col0_m0_n46)
      // Col 0 M-tile 0 N-tile 47
      %a_col0_m0_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n47)
      %b_col0_m0_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n47)
      %c_col0_m0_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 12032 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n47)
      aiex.dma_await_task(%c_col0_m0_n47)
      aiex.dma_free_task(%a_col0_m0_n47)
      aiex.dma_free_task(%b_col0_m0_n47)
      aiex.dma_free_task(%c_col0_m0_n47)
      // Col 0 M-tile 0 N-tile 48
      %a_col0_m0_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n48)
      %b_col0_m0_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n48)
      %c_col0_m0_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 12288 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n48)
      aiex.dma_await_task(%c_col0_m0_n48)
      aiex.dma_free_task(%a_col0_m0_n48)
      aiex.dma_free_task(%b_col0_m0_n48)
      aiex.dma_free_task(%c_col0_m0_n48)
      // Col 0 M-tile 0 N-tile 49
      %a_col0_m0_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n49)
      %b_col0_m0_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n49)
      %c_col0_m0_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 12544 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n49)
      aiex.dma_await_task(%c_col0_m0_n49)
      aiex.dma_free_task(%a_col0_m0_n49)
      aiex.dma_free_task(%b_col0_m0_n49)
      aiex.dma_free_task(%c_col0_m0_n49)
      // Col 0 M-tile 0 N-tile 50
      %a_col0_m0_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n50)
      %b_col0_m0_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n50)
      %c_col0_m0_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 12800 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n50)
      aiex.dma_await_task(%c_col0_m0_n50)
      aiex.dma_free_task(%a_col0_m0_n50)
      aiex.dma_free_task(%b_col0_m0_n50)
      aiex.dma_free_task(%c_col0_m0_n50)
      // Col 0 M-tile 0 N-tile 51
      %a_col0_m0_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n51)
      %b_col0_m0_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n51)
      %c_col0_m0_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 13056 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n51)
      aiex.dma_await_task(%c_col0_m0_n51)
      aiex.dma_free_task(%a_col0_m0_n51)
      aiex.dma_free_task(%b_col0_m0_n51)
      aiex.dma_free_task(%c_col0_m0_n51)
      // Col 0 M-tile 0 N-tile 52
      %a_col0_m0_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n52)
      %b_col0_m0_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n52)
      %c_col0_m0_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 13312 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n52)
      aiex.dma_await_task(%c_col0_m0_n52)
      aiex.dma_free_task(%a_col0_m0_n52)
      aiex.dma_free_task(%b_col0_m0_n52)
      aiex.dma_free_task(%c_col0_m0_n52)
      // Col 0 M-tile 0 N-tile 53
      %a_col0_m0_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n53)
      %b_col0_m0_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n53)
      %c_col0_m0_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 13568 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n53)
      aiex.dma_await_task(%c_col0_m0_n53)
      aiex.dma_free_task(%a_col0_m0_n53)
      aiex.dma_free_task(%b_col0_m0_n53)
      aiex.dma_free_task(%c_col0_m0_n53)
      // Col 0 M-tile 0 N-tile 54
      %a_col0_m0_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n54)
      %b_col0_m0_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n54)
      %c_col0_m0_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 13824 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n54)
      aiex.dma_await_task(%c_col0_m0_n54)
      aiex.dma_free_task(%a_col0_m0_n54)
      aiex.dma_free_task(%b_col0_m0_n54)
      aiex.dma_free_task(%c_col0_m0_n54)
      // Col 0 M-tile 0 N-tile 55
      %a_col0_m0_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n55)
      %b_col0_m0_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n55)
      %c_col0_m0_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 14080 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n55)
      aiex.dma_await_task(%c_col0_m0_n55)
      aiex.dma_free_task(%a_col0_m0_n55)
      aiex.dma_free_task(%b_col0_m0_n55)
      aiex.dma_free_task(%c_col0_m0_n55)
      // Col 0 M-tile 1 N-tile 0
      %a_col0_m1_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n0)
      %b_col0_m1_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n0)
      %c_col0_m1_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 917504 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n0)
      aiex.dma_await_task(%c_col0_m1_n0)
      aiex.dma_free_task(%a_col0_m1_n0)
      aiex.dma_free_task(%b_col0_m1_n0)
      aiex.dma_free_task(%c_col0_m1_n0)
      // Col 0 M-tile 1 N-tile 1
      %a_col0_m1_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n1)
      %b_col0_m1_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n1)
      %c_col0_m1_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 917760 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n1)
      aiex.dma_await_task(%c_col0_m1_n1)
      aiex.dma_free_task(%a_col0_m1_n1)
      aiex.dma_free_task(%b_col0_m1_n1)
      aiex.dma_free_task(%c_col0_m1_n1)
      // Col 0 M-tile 1 N-tile 2
      %a_col0_m1_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n2)
      %b_col0_m1_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n2)
      %c_col0_m1_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 918016 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n2)
      aiex.dma_await_task(%c_col0_m1_n2)
      aiex.dma_free_task(%a_col0_m1_n2)
      aiex.dma_free_task(%b_col0_m1_n2)
      aiex.dma_free_task(%c_col0_m1_n2)
      // Col 0 M-tile 1 N-tile 3
      %a_col0_m1_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n3)
      %b_col0_m1_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n3)
      %c_col0_m1_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 918272 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n3)
      aiex.dma_await_task(%c_col0_m1_n3)
      aiex.dma_free_task(%a_col0_m1_n3)
      aiex.dma_free_task(%b_col0_m1_n3)
      aiex.dma_free_task(%c_col0_m1_n3)
      // Col 0 M-tile 1 N-tile 4
      %a_col0_m1_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n4)
      %b_col0_m1_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n4)
      %c_col0_m1_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 918528 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n4)
      aiex.dma_await_task(%c_col0_m1_n4)
      aiex.dma_free_task(%a_col0_m1_n4)
      aiex.dma_free_task(%b_col0_m1_n4)
      aiex.dma_free_task(%c_col0_m1_n4)
      // Col 0 M-tile 1 N-tile 5
      %a_col0_m1_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n5)
      %b_col0_m1_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n5)
      %c_col0_m1_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 918784 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n5)
      aiex.dma_await_task(%c_col0_m1_n5)
      aiex.dma_free_task(%a_col0_m1_n5)
      aiex.dma_free_task(%b_col0_m1_n5)
      aiex.dma_free_task(%c_col0_m1_n5)
      // Col 0 M-tile 1 N-tile 6
      %a_col0_m1_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n6)
      %b_col0_m1_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n6)
      %c_col0_m1_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 919040 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n6)
      aiex.dma_await_task(%c_col0_m1_n6)
      aiex.dma_free_task(%a_col0_m1_n6)
      aiex.dma_free_task(%b_col0_m1_n6)
      aiex.dma_free_task(%c_col0_m1_n6)
      // Col 0 M-tile 1 N-tile 7
      %a_col0_m1_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n7)
      %b_col0_m1_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n7)
      %c_col0_m1_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 919296 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n7)
      aiex.dma_await_task(%c_col0_m1_n7)
      aiex.dma_free_task(%a_col0_m1_n7)
      aiex.dma_free_task(%b_col0_m1_n7)
      aiex.dma_free_task(%c_col0_m1_n7)
      // Col 0 M-tile 1 N-tile 8
      %a_col0_m1_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n8)
      %b_col0_m1_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n8)
      %c_col0_m1_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 919552 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n8)
      aiex.dma_await_task(%c_col0_m1_n8)
      aiex.dma_free_task(%a_col0_m1_n8)
      aiex.dma_free_task(%b_col0_m1_n8)
      aiex.dma_free_task(%c_col0_m1_n8)
      // Col 0 M-tile 1 N-tile 9
      %a_col0_m1_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n9)
      %b_col0_m1_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n9)
      %c_col0_m1_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 919808 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n9)
      aiex.dma_await_task(%c_col0_m1_n9)
      aiex.dma_free_task(%a_col0_m1_n9)
      aiex.dma_free_task(%b_col0_m1_n9)
      aiex.dma_free_task(%c_col0_m1_n9)
      // Col 0 M-tile 1 N-tile 10
      %a_col0_m1_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n10)
      %b_col0_m1_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n10)
      %c_col0_m1_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 920064 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n10)
      aiex.dma_await_task(%c_col0_m1_n10)
      aiex.dma_free_task(%a_col0_m1_n10)
      aiex.dma_free_task(%b_col0_m1_n10)
      aiex.dma_free_task(%c_col0_m1_n10)
      // Col 0 M-tile 1 N-tile 11
      %a_col0_m1_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n11)
      %b_col0_m1_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n11)
      %c_col0_m1_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 920320 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n11)
      aiex.dma_await_task(%c_col0_m1_n11)
      aiex.dma_free_task(%a_col0_m1_n11)
      aiex.dma_free_task(%b_col0_m1_n11)
      aiex.dma_free_task(%c_col0_m1_n11)
      // Col 0 M-tile 1 N-tile 12
      %a_col0_m1_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n12)
      %b_col0_m1_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n12)
      %c_col0_m1_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 920576 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n12)
      aiex.dma_await_task(%c_col0_m1_n12)
      aiex.dma_free_task(%a_col0_m1_n12)
      aiex.dma_free_task(%b_col0_m1_n12)
      aiex.dma_free_task(%c_col0_m1_n12)
      // Col 0 M-tile 1 N-tile 13
      %a_col0_m1_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n13)
      %b_col0_m1_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n13)
      %c_col0_m1_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 920832 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n13)
      aiex.dma_await_task(%c_col0_m1_n13)
      aiex.dma_free_task(%a_col0_m1_n13)
      aiex.dma_free_task(%b_col0_m1_n13)
      aiex.dma_free_task(%c_col0_m1_n13)
      // Col 0 M-tile 1 N-tile 14
      %a_col0_m1_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n14)
      %b_col0_m1_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n14)
      %c_col0_m1_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 921088 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n14)
      aiex.dma_await_task(%c_col0_m1_n14)
      aiex.dma_free_task(%a_col0_m1_n14)
      aiex.dma_free_task(%b_col0_m1_n14)
      aiex.dma_free_task(%c_col0_m1_n14)
      // Col 0 M-tile 1 N-tile 15
      %a_col0_m1_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n15)
      %b_col0_m1_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n15)
      %c_col0_m1_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 921344 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n15)
      aiex.dma_await_task(%c_col0_m1_n15)
      aiex.dma_free_task(%a_col0_m1_n15)
      aiex.dma_free_task(%b_col0_m1_n15)
      aiex.dma_free_task(%c_col0_m1_n15)
      // Col 0 M-tile 1 N-tile 16
      %a_col0_m1_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n16)
      %b_col0_m1_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n16)
      %c_col0_m1_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 921600 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n16)
      aiex.dma_await_task(%c_col0_m1_n16)
      aiex.dma_free_task(%a_col0_m1_n16)
      aiex.dma_free_task(%b_col0_m1_n16)
      aiex.dma_free_task(%c_col0_m1_n16)
      // Col 0 M-tile 1 N-tile 17
      %a_col0_m1_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n17)
      %b_col0_m1_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n17)
      %c_col0_m1_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 921856 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n17)
      aiex.dma_await_task(%c_col0_m1_n17)
      aiex.dma_free_task(%a_col0_m1_n17)
      aiex.dma_free_task(%b_col0_m1_n17)
      aiex.dma_free_task(%c_col0_m1_n17)
      // Col 0 M-tile 1 N-tile 18
      %a_col0_m1_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n18)
      %b_col0_m1_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n18)
      %c_col0_m1_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 922112 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n18)
      aiex.dma_await_task(%c_col0_m1_n18)
      aiex.dma_free_task(%a_col0_m1_n18)
      aiex.dma_free_task(%b_col0_m1_n18)
      aiex.dma_free_task(%c_col0_m1_n18)
      // Col 0 M-tile 1 N-tile 19
      %a_col0_m1_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n19)
      %b_col0_m1_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n19)
      %c_col0_m1_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 922368 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n19)
      aiex.dma_await_task(%c_col0_m1_n19)
      aiex.dma_free_task(%a_col0_m1_n19)
      aiex.dma_free_task(%b_col0_m1_n19)
      aiex.dma_free_task(%c_col0_m1_n19)
      // Col 0 M-tile 1 N-tile 20
      %a_col0_m1_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n20)
      %b_col0_m1_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n20)
      %c_col0_m1_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 922624 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n20)
      aiex.dma_await_task(%c_col0_m1_n20)
      aiex.dma_free_task(%a_col0_m1_n20)
      aiex.dma_free_task(%b_col0_m1_n20)
      aiex.dma_free_task(%c_col0_m1_n20)
      // Col 0 M-tile 1 N-tile 21
      %a_col0_m1_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n21)
      %b_col0_m1_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n21)
      %c_col0_m1_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 922880 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n21)
      aiex.dma_await_task(%c_col0_m1_n21)
      aiex.dma_free_task(%a_col0_m1_n21)
      aiex.dma_free_task(%b_col0_m1_n21)
      aiex.dma_free_task(%c_col0_m1_n21)
      // Col 0 M-tile 1 N-tile 22
      %a_col0_m1_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n22)
      %b_col0_m1_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n22)
      %c_col0_m1_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 923136 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n22)
      aiex.dma_await_task(%c_col0_m1_n22)
      aiex.dma_free_task(%a_col0_m1_n22)
      aiex.dma_free_task(%b_col0_m1_n22)
      aiex.dma_free_task(%c_col0_m1_n22)
      // Col 0 M-tile 1 N-tile 23
      %a_col0_m1_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n23)
      %b_col0_m1_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n23)
      %c_col0_m1_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 923392 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n23)
      aiex.dma_await_task(%c_col0_m1_n23)
      aiex.dma_free_task(%a_col0_m1_n23)
      aiex.dma_free_task(%b_col0_m1_n23)
      aiex.dma_free_task(%c_col0_m1_n23)
      // Col 0 M-tile 1 N-tile 24
      %a_col0_m1_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n24)
      %b_col0_m1_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n24)
      %c_col0_m1_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 923648 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n24)
      aiex.dma_await_task(%c_col0_m1_n24)
      aiex.dma_free_task(%a_col0_m1_n24)
      aiex.dma_free_task(%b_col0_m1_n24)
      aiex.dma_free_task(%c_col0_m1_n24)
      // Col 0 M-tile 1 N-tile 25
      %a_col0_m1_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n25)
      %b_col0_m1_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n25)
      %c_col0_m1_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 923904 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n25)
      aiex.dma_await_task(%c_col0_m1_n25)
      aiex.dma_free_task(%a_col0_m1_n25)
      aiex.dma_free_task(%b_col0_m1_n25)
      aiex.dma_free_task(%c_col0_m1_n25)
      // Col 0 M-tile 1 N-tile 26
      %a_col0_m1_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n26)
      %b_col0_m1_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n26)
      %c_col0_m1_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 924160 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n26)
      aiex.dma_await_task(%c_col0_m1_n26)
      aiex.dma_free_task(%a_col0_m1_n26)
      aiex.dma_free_task(%b_col0_m1_n26)
      aiex.dma_free_task(%c_col0_m1_n26)
      // Col 0 M-tile 1 N-tile 27
      %a_col0_m1_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n27)
      %b_col0_m1_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n27)
      %c_col0_m1_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 924416 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n27)
      aiex.dma_await_task(%c_col0_m1_n27)
      aiex.dma_free_task(%a_col0_m1_n27)
      aiex.dma_free_task(%b_col0_m1_n27)
      aiex.dma_free_task(%c_col0_m1_n27)
      // Col 0 M-tile 1 N-tile 28
      %a_col0_m1_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n28)
      %b_col0_m1_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n28)
      %c_col0_m1_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 924672 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n28)
      aiex.dma_await_task(%c_col0_m1_n28)
      aiex.dma_free_task(%a_col0_m1_n28)
      aiex.dma_free_task(%b_col0_m1_n28)
      aiex.dma_free_task(%c_col0_m1_n28)
      // Col 0 M-tile 1 N-tile 29
      %a_col0_m1_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n29)
      %b_col0_m1_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n29)
      %c_col0_m1_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 924928 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n29)
      aiex.dma_await_task(%c_col0_m1_n29)
      aiex.dma_free_task(%a_col0_m1_n29)
      aiex.dma_free_task(%b_col0_m1_n29)
      aiex.dma_free_task(%c_col0_m1_n29)
      // Col 0 M-tile 1 N-tile 30
      %a_col0_m1_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n30)
      %b_col0_m1_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n30)
      %c_col0_m1_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 925184 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n30)
      aiex.dma_await_task(%c_col0_m1_n30)
      aiex.dma_free_task(%a_col0_m1_n30)
      aiex.dma_free_task(%b_col0_m1_n30)
      aiex.dma_free_task(%c_col0_m1_n30)
      // Col 0 M-tile 1 N-tile 31
      %a_col0_m1_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n31)
      %b_col0_m1_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n31)
      %c_col0_m1_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 925440 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n31)
      aiex.dma_await_task(%c_col0_m1_n31)
      aiex.dma_free_task(%a_col0_m1_n31)
      aiex.dma_free_task(%b_col0_m1_n31)
      aiex.dma_free_task(%c_col0_m1_n31)
      // Col 0 M-tile 1 N-tile 32
      %a_col0_m1_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n32)
      %b_col0_m1_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n32)
      %c_col0_m1_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 925696 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n32)
      aiex.dma_await_task(%c_col0_m1_n32)
      aiex.dma_free_task(%a_col0_m1_n32)
      aiex.dma_free_task(%b_col0_m1_n32)
      aiex.dma_free_task(%c_col0_m1_n32)
      // Col 0 M-tile 1 N-tile 33
      %a_col0_m1_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n33)
      %b_col0_m1_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n33)
      %c_col0_m1_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 925952 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n33)
      aiex.dma_await_task(%c_col0_m1_n33)
      aiex.dma_free_task(%a_col0_m1_n33)
      aiex.dma_free_task(%b_col0_m1_n33)
      aiex.dma_free_task(%c_col0_m1_n33)
      // Col 0 M-tile 1 N-tile 34
      %a_col0_m1_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n34)
      %b_col0_m1_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n34)
      %c_col0_m1_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 926208 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n34)
      aiex.dma_await_task(%c_col0_m1_n34)
      aiex.dma_free_task(%a_col0_m1_n34)
      aiex.dma_free_task(%b_col0_m1_n34)
      aiex.dma_free_task(%c_col0_m1_n34)
      // Col 0 M-tile 1 N-tile 35
      %a_col0_m1_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n35)
      %b_col0_m1_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n35)
      %c_col0_m1_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 926464 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n35)
      aiex.dma_await_task(%c_col0_m1_n35)
      aiex.dma_free_task(%a_col0_m1_n35)
      aiex.dma_free_task(%b_col0_m1_n35)
      aiex.dma_free_task(%c_col0_m1_n35)
      // Col 0 M-tile 1 N-tile 36
      %a_col0_m1_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n36)
      %b_col0_m1_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n36)
      %c_col0_m1_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 926720 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n36)
      aiex.dma_await_task(%c_col0_m1_n36)
      aiex.dma_free_task(%a_col0_m1_n36)
      aiex.dma_free_task(%b_col0_m1_n36)
      aiex.dma_free_task(%c_col0_m1_n36)
      // Col 0 M-tile 1 N-tile 37
      %a_col0_m1_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n37)
      %b_col0_m1_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n37)
      %c_col0_m1_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 926976 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n37)
      aiex.dma_await_task(%c_col0_m1_n37)
      aiex.dma_free_task(%a_col0_m1_n37)
      aiex.dma_free_task(%b_col0_m1_n37)
      aiex.dma_free_task(%c_col0_m1_n37)
      // Col 0 M-tile 1 N-tile 38
      %a_col0_m1_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n38)
      %b_col0_m1_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n38)
      %c_col0_m1_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 927232 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n38)
      aiex.dma_await_task(%c_col0_m1_n38)
      aiex.dma_free_task(%a_col0_m1_n38)
      aiex.dma_free_task(%b_col0_m1_n38)
      aiex.dma_free_task(%c_col0_m1_n38)
      // Col 0 M-tile 1 N-tile 39
      %a_col0_m1_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n39)
      %b_col0_m1_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n39)
      %c_col0_m1_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 927488 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n39)
      aiex.dma_await_task(%c_col0_m1_n39)
      aiex.dma_free_task(%a_col0_m1_n39)
      aiex.dma_free_task(%b_col0_m1_n39)
      aiex.dma_free_task(%c_col0_m1_n39)
      // Col 0 M-tile 1 N-tile 40
      %a_col0_m1_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n40)
      %b_col0_m1_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n40)
      %c_col0_m1_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 927744 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n40)
      aiex.dma_await_task(%c_col0_m1_n40)
      aiex.dma_free_task(%a_col0_m1_n40)
      aiex.dma_free_task(%b_col0_m1_n40)
      aiex.dma_free_task(%c_col0_m1_n40)
      // Col 0 M-tile 1 N-tile 41
      %a_col0_m1_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n41)
      %b_col0_m1_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n41)
      %c_col0_m1_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 928000 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n41)
      aiex.dma_await_task(%c_col0_m1_n41)
      aiex.dma_free_task(%a_col0_m1_n41)
      aiex.dma_free_task(%b_col0_m1_n41)
      aiex.dma_free_task(%c_col0_m1_n41)
      // Col 0 M-tile 1 N-tile 42
      %a_col0_m1_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n42)
      %b_col0_m1_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n42)
      %c_col0_m1_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 928256 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n42)
      aiex.dma_await_task(%c_col0_m1_n42)
      aiex.dma_free_task(%a_col0_m1_n42)
      aiex.dma_free_task(%b_col0_m1_n42)
      aiex.dma_free_task(%c_col0_m1_n42)
      // Col 0 M-tile 1 N-tile 43
      %a_col0_m1_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n43)
      %b_col0_m1_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n43)
      %c_col0_m1_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 928512 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n43)
      aiex.dma_await_task(%c_col0_m1_n43)
      aiex.dma_free_task(%a_col0_m1_n43)
      aiex.dma_free_task(%b_col0_m1_n43)
      aiex.dma_free_task(%c_col0_m1_n43)
      // Col 0 M-tile 1 N-tile 44
      %a_col0_m1_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n44)
      %b_col0_m1_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n44)
      %c_col0_m1_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 928768 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n44)
      aiex.dma_await_task(%c_col0_m1_n44)
      aiex.dma_free_task(%a_col0_m1_n44)
      aiex.dma_free_task(%b_col0_m1_n44)
      aiex.dma_free_task(%c_col0_m1_n44)
      // Col 0 M-tile 1 N-tile 45
      %a_col0_m1_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n45)
      %b_col0_m1_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n45)
      %c_col0_m1_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 929024 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n45)
      aiex.dma_await_task(%c_col0_m1_n45)
      aiex.dma_free_task(%a_col0_m1_n45)
      aiex.dma_free_task(%b_col0_m1_n45)
      aiex.dma_free_task(%c_col0_m1_n45)
      // Col 0 M-tile 1 N-tile 46
      %a_col0_m1_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n46)
      %b_col0_m1_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n46)
      %c_col0_m1_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 929280 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n46)
      aiex.dma_await_task(%c_col0_m1_n46)
      aiex.dma_free_task(%a_col0_m1_n46)
      aiex.dma_free_task(%b_col0_m1_n46)
      aiex.dma_free_task(%c_col0_m1_n46)
      // Col 0 M-tile 1 N-tile 47
      %a_col0_m1_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n47)
      %b_col0_m1_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n47)
      %c_col0_m1_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 929536 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n47)
      aiex.dma_await_task(%c_col0_m1_n47)
      aiex.dma_free_task(%a_col0_m1_n47)
      aiex.dma_free_task(%b_col0_m1_n47)
      aiex.dma_free_task(%c_col0_m1_n47)
      // Col 0 M-tile 1 N-tile 48
      %a_col0_m1_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n48)
      %b_col0_m1_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n48)
      %c_col0_m1_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 929792 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n48)
      aiex.dma_await_task(%c_col0_m1_n48)
      aiex.dma_free_task(%a_col0_m1_n48)
      aiex.dma_free_task(%b_col0_m1_n48)
      aiex.dma_free_task(%c_col0_m1_n48)
      // Col 0 M-tile 1 N-tile 49
      %a_col0_m1_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n49)
      %b_col0_m1_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n49)
      %c_col0_m1_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 930048 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n49)
      aiex.dma_await_task(%c_col0_m1_n49)
      aiex.dma_free_task(%a_col0_m1_n49)
      aiex.dma_free_task(%b_col0_m1_n49)
      aiex.dma_free_task(%c_col0_m1_n49)
      // Col 0 M-tile 1 N-tile 50
      %a_col0_m1_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n50)
      %b_col0_m1_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n50)
      %c_col0_m1_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 930304 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n50)
      aiex.dma_await_task(%c_col0_m1_n50)
      aiex.dma_free_task(%a_col0_m1_n50)
      aiex.dma_free_task(%b_col0_m1_n50)
      aiex.dma_free_task(%c_col0_m1_n50)
      // Col 0 M-tile 1 N-tile 51
      %a_col0_m1_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n51)
      %b_col0_m1_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n51)
      %c_col0_m1_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 930560 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n51)
      aiex.dma_await_task(%c_col0_m1_n51)
      aiex.dma_free_task(%a_col0_m1_n51)
      aiex.dma_free_task(%b_col0_m1_n51)
      aiex.dma_free_task(%c_col0_m1_n51)
      // Col 0 M-tile 1 N-tile 52
      %a_col0_m1_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n52)
      %b_col0_m1_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n52)
      %c_col0_m1_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 930816 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n52)
      aiex.dma_await_task(%c_col0_m1_n52)
      aiex.dma_free_task(%a_col0_m1_n52)
      aiex.dma_free_task(%b_col0_m1_n52)
      aiex.dma_free_task(%c_col0_m1_n52)
      // Col 0 M-tile 1 N-tile 53
      %a_col0_m1_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n53)
      %b_col0_m1_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n53)
      %c_col0_m1_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 931072 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n53)
      aiex.dma_await_task(%c_col0_m1_n53)
      aiex.dma_free_task(%a_col0_m1_n53)
      aiex.dma_free_task(%b_col0_m1_n53)
      aiex.dma_free_task(%c_col0_m1_n53)
      // Col 0 M-tile 1 N-tile 54
      %a_col0_m1_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n54)
      %b_col0_m1_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n54)
      %c_col0_m1_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 931328 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n54)
      aiex.dma_await_task(%c_col0_m1_n54)
      aiex.dma_free_task(%a_col0_m1_n54)
      aiex.dma_free_task(%b_col0_m1_n54)
      aiex.dma_free_task(%c_col0_m1_n54)
      // Col 0 M-tile 1 N-tile 55
      %a_col0_m1_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n55)
      %b_col0_m1_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n55)
      %c_col0_m1_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 931584 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n55)
      aiex.dma_await_task(%c_col0_m1_n55)
      aiex.dma_free_task(%a_col0_m1_n55)
      aiex.dma_free_task(%b_col0_m1_n55)
      aiex.dma_free_task(%c_col0_m1_n55)
      // Col 0 M-tile 2 N-tile 0
      %a_col0_m2_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n0)
      %b_col0_m2_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n0)
      %c_col0_m2_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1835008 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n0)
      aiex.dma_await_task(%c_col0_m2_n0)
      aiex.dma_free_task(%a_col0_m2_n0)
      aiex.dma_free_task(%b_col0_m2_n0)
      aiex.dma_free_task(%c_col0_m2_n0)
      // Col 0 M-tile 2 N-tile 1
      %a_col0_m2_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n1)
      %b_col0_m2_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n1)
      %c_col0_m2_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1835264 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n1)
      aiex.dma_await_task(%c_col0_m2_n1)
      aiex.dma_free_task(%a_col0_m2_n1)
      aiex.dma_free_task(%b_col0_m2_n1)
      aiex.dma_free_task(%c_col0_m2_n1)
      // Col 0 M-tile 2 N-tile 2
      %a_col0_m2_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n2)
      %b_col0_m2_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n2)
      %c_col0_m2_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1835520 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n2)
      aiex.dma_await_task(%c_col0_m2_n2)
      aiex.dma_free_task(%a_col0_m2_n2)
      aiex.dma_free_task(%b_col0_m2_n2)
      aiex.dma_free_task(%c_col0_m2_n2)
      // Col 0 M-tile 2 N-tile 3
      %a_col0_m2_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n3)
      %b_col0_m2_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n3)
      %c_col0_m2_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1835776 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n3)
      aiex.dma_await_task(%c_col0_m2_n3)
      aiex.dma_free_task(%a_col0_m2_n3)
      aiex.dma_free_task(%b_col0_m2_n3)
      aiex.dma_free_task(%c_col0_m2_n3)
      // Col 0 M-tile 2 N-tile 4
      %a_col0_m2_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n4)
      %b_col0_m2_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n4)
      %c_col0_m2_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1836032 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n4)
      aiex.dma_await_task(%c_col0_m2_n4)
      aiex.dma_free_task(%a_col0_m2_n4)
      aiex.dma_free_task(%b_col0_m2_n4)
      aiex.dma_free_task(%c_col0_m2_n4)
      // Col 0 M-tile 2 N-tile 5
      %a_col0_m2_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n5)
      %b_col0_m2_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n5)
      %c_col0_m2_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1836288 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n5)
      aiex.dma_await_task(%c_col0_m2_n5)
      aiex.dma_free_task(%a_col0_m2_n5)
      aiex.dma_free_task(%b_col0_m2_n5)
      aiex.dma_free_task(%c_col0_m2_n5)
      // Col 0 M-tile 2 N-tile 6
      %a_col0_m2_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n6)
      %b_col0_m2_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n6)
      %c_col0_m2_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1836544 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n6)
      aiex.dma_await_task(%c_col0_m2_n6)
      aiex.dma_free_task(%a_col0_m2_n6)
      aiex.dma_free_task(%b_col0_m2_n6)
      aiex.dma_free_task(%c_col0_m2_n6)
      // Col 0 M-tile 2 N-tile 7
      %a_col0_m2_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n7)
      %b_col0_m2_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n7)
      %c_col0_m2_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1836800 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n7)
      aiex.dma_await_task(%c_col0_m2_n7)
      aiex.dma_free_task(%a_col0_m2_n7)
      aiex.dma_free_task(%b_col0_m2_n7)
      aiex.dma_free_task(%c_col0_m2_n7)
      // Col 0 M-tile 2 N-tile 8
      %a_col0_m2_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n8)
      %b_col0_m2_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n8)
      %c_col0_m2_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1837056 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n8)
      aiex.dma_await_task(%c_col0_m2_n8)
      aiex.dma_free_task(%a_col0_m2_n8)
      aiex.dma_free_task(%b_col0_m2_n8)
      aiex.dma_free_task(%c_col0_m2_n8)
      // Col 0 M-tile 2 N-tile 9
      %a_col0_m2_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n9)
      %b_col0_m2_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n9)
      %c_col0_m2_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1837312 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n9)
      aiex.dma_await_task(%c_col0_m2_n9)
      aiex.dma_free_task(%a_col0_m2_n9)
      aiex.dma_free_task(%b_col0_m2_n9)
      aiex.dma_free_task(%c_col0_m2_n9)
      // Col 0 M-tile 2 N-tile 10
      %a_col0_m2_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n10)
      %b_col0_m2_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n10)
      %c_col0_m2_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1837568 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n10)
      aiex.dma_await_task(%c_col0_m2_n10)
      aiex.dma_free_task(%a_col0_m2_n10)
      aiex.dma_free_task(%b_col0_m2_n10)
      aiex.dma_free_task(%c_col0_m2_n10)
      // Col 0 M-tile 2 N-tile 11
      %a_col0_m2_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n11)
      %b_col0_m2_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n11)
      %c_col0_m2_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1837824 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n11)
      aiex.dma_await_task(%c_col0_m2_n11)
      aiex.dma_free_task(%a_col0_m2_n11)
      aiex.dma_free_task(%b_col0_m2_n11)
      aiex.dma_free_task(%c_col0_m2_n11)
      // Col 0 M-tile 2 N-tile 12
      %a_col0_m2_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n12)
      %b_col0_m2_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n12)
      %c_col0_m2_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1838080 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n12)
      aiex.dma_await_task(%c_col0_m2_n12)
      aiex.dma_free_task(%a_col0_m2_n12)
      aiex.dma_free_task(%b_col0_m2_n12)
      aiex.dma_free_task(%c_col0_m2_n12)
      // Col 0 M-tile 2 N-tile 13
      %a_col0_m2_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n13)
      %b_col0_m2_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n13)
      %c_col0_m2_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1838336 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n13)
      aiex.dma_await_task(%c_col0_m2_n13)
      aiex.dma_free_task(%a_col0_m2_n13)
      aiex.dma_free_task(%b_col0_m2_n13)
      aiex.dma_free_task(%c_col0_m2_n13)
      // Col 0 M-tile 2 N-tile 14
      %a_col0_m2_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n14)
      %b_col0_m2_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n14)
      %c_col0_m2_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1838592 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n14)
      aiex.dma_await_task(%c_col0_m2_n14)
      aiex.dma_free_task(%a_col0_m2_n14)
      aiex.dma_free_task(%b_col0_m2_n14)
      aiex.dma_free_task(%c_col0_m2_n14)
      // Col 0 M-tile 2 N-tile 15
      %a_col0_m2_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n15)
      %b_col0_m2_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n15)
      %c_col0_m2_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1838848 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n15)
      aiex.dma_await_task(%c_col0_m2_n15)
      aiex.dma_free_task(%a_col0_m2_n15)
      aiex.dma_free_task(%b_col0_m2_n15)
      aiex.dma_free_task(%c_col0_m2_n15)
      // Col 0 M-tile 2 N-tile 16
      %a_col0_m2_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n16)
      %b_col0_m2_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n16)
      %c_col0_m2_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1839104 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n16)
      aiex.dma_await_task(%c_col0_m2_n16)
      aiex.dma_free_task(%a_col0_m2_n16)
      aiex.dma_free_task(%b_col0_m2_n16)
      aiex.dma_free_task(%c_col0_m2_n16)
      // Col 0 M-tile 2 N-tile 17
      %a_col0_m2_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n17)
      %b_col0_m2_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n17)
      %c_col0_m2_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1839360 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n17)
      aiex.dma_await_task(%c_col0_m2_n17)
      aiex.dma_free_task(%a_col0_m2_n17)
      aiex.dma_free_task(%b_col0_m2_n17)
      aiex.dma_free_task(%c_col0_m2_n17)
      // Col 0 M-tile 2 N-tile 18
      %a_col0_m2_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n18)
      %b_col0_m2_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n18)
      %c_col0_m2_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1839616 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n18)
      aiex.dma_await_task(%c_col0_m2_n18)
      aiex.dma_free_task(%a_col0_m2_n18)
      aiex.dma_free_task(%b_col0_m2_n18)
      aiex.dma_free_task(%c_col0_m2_n18)
      // Col 0 M-tile 2 N-tile 19
      %a_col0_m2_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n19)
      %b_col0_m2_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n19)
      %c_col0_m2_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1839872 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n19)
      aiex.dma_await_task(%c_col0_m2_n19)
      aiex.dma_free_task(%a_col0_m2_n19)
      aiex.dma_free_task(%b_col0_m2_n19)
      aiex.dma_free_task(%c_col0_m2_n19)
      // Col 0 M-tile 2 N-tile 20
      %a_col0_m2_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n20)
      %b_col0_m2_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n20)
      %c_col0_m2_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1840128 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n20)
      aiex.dma_await_task(%c_col0_m2_n20)
      aiex.dma_free_task(%a_col0_m2_n20)
      aiex.dma_free_task(%b_col0_m2_n20)
      aiex.dma_free_task(%c_col0_m2_n20)
      // Col 0 M-tile 2 N-tile 21
      %a_col0_m2_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n21)
      %b_col0_m2_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n21)
      %c_col0_m2_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1840384 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n21)
      aiex.dma_await_task(%c_col0_m2_n21)
      aiex.dma_free_task(%a_col0_m2_n21)
      aiex.dma_free_task(%b_col0_m2_n21)
      aiex.dma_free_task(%c_col0_m2_n21)
      // Col 0 M-tile 2 N-tile 22
      %a_col0_m2_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n22)
      %b_col0_m2_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n22)
      %c_col0_m2_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1840640 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n22)
      aiex.dma_await_task(%c_col0_m2_n22)
      aiex.dma_free_task(%a_col0_m2_n22)
      aiex.dma_free_task(%b_col0_m2_n22)
      aiex.dma_free_task(%c_col0_m2_n22)
      // Col 0 M-tile 2 N-tile 23
      %a_col0_m2_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n23)
      %b_col0_m2_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n23)
      %c_col0_m2_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1840896 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n23)
      aiex.dma_await_task(%c_col0_m2_n23)
      aiex.dma_free_task(%a_col0_m2_n23)
      aiex.dma_free_task(%b_col0_m2_n23)
      aiex.dma_free_task(%c_col0_m2_n23)
      // Col 0 M-tile 2 N-tile 24
      %a_col0_m2_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n24)
      %b_col0_m2_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n24)
      %c_col0_m2_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1841152 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n24)
      aiex.dma_await_task(%c_col0_m2_n24)
      aiex.dma_free_task(%a_col0_m2_n24)
      aiex.dma_free_task(%b_col0_m2_n24)
      aiex.dma_free_task(%c_col0_m2_n24)
      // Col 0 M-tile 2 N-tile 25
      %a_col0_m2_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n25)
      %b_col0_m2_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n25)
      %c_col0_m2_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1841408 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n25)
      aiex.dma_await_task(%c_col0_m2_n25)
      aiex.dma_free_task(%a_col0_m2_n25)
      aiex.dma_free_task(%b_col0_m2_n25)
      aiex.dma_free_task(%c_col0_m2_n25)
      // Col 0 M-tile 2 N-tile 26
      %a_col0_m2_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n26)
      %b_col0_m2_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n26)
      %c_col0_m2_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1841664 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n26)
      aiex.dma_await_task(%c_col0_m2_n26)
      aiex.dma_free_task(%a_col0_m2_n26)
      aiex.dma_free_task(%b_col0_m2_n26)
      aiex.dma_free_task(%c_col0_m2_n26)
      // Col 0 M-tile 2 N-tile 27
      %a_col0_m2_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n27)
      %b_col0_m2_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n27)
      %c_col0_m2_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1841920 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n27)
      aiex.dma_await_task(%c_col0_m2_n27)
      aiex.dma_free_task(%a_col0_m2_n27)
      aiex.dma_free_task(%b_col0_m2_n27)
      aiex.dma_free_task(%c_col0_m2_n27)
      // Col 0 M-tile 2 N-tile 28
      %a_col0_m2_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n28)
      %b_col0_m2_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n28)
      %c_col0_m2_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1842176 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n28)
      aiex.dma_await_task(%c_col0_m2_n28)
      aiex.dma_free_task(%a_col0_m2_n28)
      aiex.dma_free_task(%b_col0_m2_n28)
      aiex.dma_free_task(%c_col0_m2_n28)
      // Col 0 M-tile 2 N-tile 29
      %a_col0_m2_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n29)
      %b_col0_m2_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n29)
      %c_col0_m2_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1842432 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n29)
      aiex.dma_await_task(%c_col0_m2_n29)
      aiex.dma_free_task(%a_col0_m2_n29)
      aiex.dma_free_task(%b_col0_m2_n29)
      aiex.dma_free_task(%c_col0_m2_n29)
      // Col 0 M-tile 2 N-tile 30
      %a_col0_m2_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n30)
      %b_col0_m2_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n30)
      %c_col0_m2_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1842688 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n30)
      aiex.dma_await_task(%c_col0_m2_n30)
      aiex.dma_free_task(%a_col0_m2_n30)
      aiex.dma_free_task(%b_col0_m2_n30)
      aiex.dma_free_task(%c_col0_m2_n30)
      // Col 0 M-tile 2 N-tile 31
      %a_col0_m2_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n31)
      %b_col0_m2_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n31)
      %c_col0_m2_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1842944 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n31)
      aiex.dma_await_task(%c_col0_m2_n31)
      aiex.dma_free_task(%a_col0_m2_n31)
      aiex.dma_free_task(%b_col0_m2_n31)
      aiex.dma_free_task(%c_col0_m2_n31)
      // Col 0 M-tile 2 N-tile 32
      %a_col0_m2_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n32)
      %b_col0_m2_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n32)
      %c_col0_m2_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1843200 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n32)
      aiex.dma_await_task(%c_col0_m2_n32)
      aiex.dma_free_task(%a_col0_m2_n32)
      aiex.dma_free_task(%b_col0_m2_n32)
      aiex.dma_free_task(%c_col0_m2_n32)
      // Col 0 M-tile 2 N-tile 33
      %a_col0_m2_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n33)
      %b_col0_m2_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n33)
      %c_col0_m2_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1843456 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n33)
      aiex.dma_await_task(%c_col0_m2_n33)
      aiex.dma_free_task(%a_col0_m2_n33)
      aiex.dma_free_task(%b_col0_m2_n33)
      aiex.dma_free_task(%c_col0_m2_n33)
      // Col 0 M-tile 2 N-tile 34
      %a_col0_m2_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n34)
      %b_col0_m2_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n34)
      %c_col0_m2_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1843712 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n34)
      aiex.dma_await_task(%c_col0_m2_n34)
      aiex.dma_free_task(%a_col0_m2_n34)
      aiex.dma_free_task(%b_col0_m2_n34)
      aiex.dma_free_task(%c_col0_m2_n34)
      // Col 0 M-tile 2 N-tile 35
      %a_col0_m2_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n35)
      %b_col0_m2_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n35)
      %c_col0_m2_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1843968 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n35)
      aiex.dma_await_task(%c_col0_m2_n35)
      aiex.dma_free_task(%a_col0_m2_n35)
      aiex.dma_free_task(%b_col0_m2_n35)
      aiex.dma_free_task(%c_col0_m2_n35)
      // Col 0 M-tile 2 N-tile 36
      %a_col0_m2_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n36)
      %b_col0_m2_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n36)
      %c_col0_m2_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1844224 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n36)
      aiex.dma_await_task(%c_col0_m2_n36)
      aiex.dma_free_task(%a_col0_m2_n36)
      aiex.dma_free_task(%b_col0_m2_n36)
      aiex.dma_free_task(%c_col0_m2_n36)
      // Col 0 M-tile 2 N-tile 37
      %a_col0_m2_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n37)
      %b_col0_m2_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n37)
      %c_col0_m2_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1844480 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n37)
      aiex.dma_await_task(%c_col0_m2_n37)
      aiex.dma_free_task(%a_col0_m2_n37)
      aiex.dma_free_task(%b_col0_m2_n37)
      aiex.dma_free_task(%c_col0_m2_n37)
      // Col 0 M-tile 2 N-tile 38
      %a_col0_m2_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n38)
      %b_col0_m2_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n38)
      %c_col0_m2_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1844736 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n38)
      aiex.dma_await_task(%c_col0_m2_n38)
      aiex.dma_free_task(%a_col0_m2_n38)
      aiex.dma_free_task(%b_col0_m2_n38)
      aiex.dma_free_task(%c_col0_m2_n38)
      // Col 0 M-tile 2 N-tile 39
      %a_col0_m2_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n39)
      %b_col0_m2_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n39)
      %c_col0_m2_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1844992 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n39)
      aiex.dma_await_task(%c_col0_m2_n39)
      aiex.dma_free_task(%a_col0_m2_n39)
      aiex.dma_free_task(%b_col0_m2_n39)
      aiex.dma_free_task(%c_col0_m2_n39)
      // Col 0 M-tile 2 N-tile 40
      %a_col0_m2_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n40)
      %b_col0_m2_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n40)
      %c_col0_m2_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1845248 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n40)
      aiex.dma_await_task(%c_col0_m2_n40)
      aiex.dma_free_task(%a_col0_m2_n40)
      aiex.dma_free_task(%b_col0_m2_n40)
      aiex.dma_free_task(%c_col0_m2_n40)
      // Col 0 M-tile 2 N-tile 41
      %a_col0_m2_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n41)
      %b_col0_m2_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n41)
      %c_col0_m2_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1845504 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n41)
      aiex.dma_await_task(%c_col0_m2_n41)
      aiex.dma_free_task(%a_col0_m2_n41)
      aiex.dma_free_task(%b_col0_m2_n41)
      aiex.dma_free_task(%c_col0_m2_n41)
      // Col 0 M-tile 2 N-tile 42
      %a_col0_m2_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n42)
      %b_col0_m2_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n42)
      %c_col0_m2_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1845760 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n42)
      aiex.dma_await_task(%c_col0_m2_n42)
      aiex.dma_free_task(%a_col0_m2_n42)
      aiex.dma_free_task(%b_col0_m2_n42)
      aiex.dma_free_task(%c_col0_m2_n42)
      // Col 0 M-tile 2 N-tile 43
      %a_col0_m2_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n43)
      %b_col0_m2_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n43)
      %c_col0_m2_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1846016 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n43)
      aiex.dma_await_task(%c_col0_m2_n43)
      aiex.dma_free_task(%a_col0_m2_n43)
      aiex.dma_free_task(%b_col0_m2_n43)
      aiex.dma_free_task(%c_col0_m2_n43)
      // Col 0 M-tile 2 N-tile 44
      %a_col0_m2_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n44)
      %b_col0_m2_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n44)
      %c_col0_m2_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1846272 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n44)
      aiex.dma_await_task(%c_col0_m2_n44)
      aiex.dma_free_task(%a_col0_m2_n44)
      aiex.dma_free_task(%b_col0_m2_n44)
      aiex.dma_free_task(%c_col0_m2_n44)
      // Col 0 M-tile 2 N-tile 45
      %a_col0_m2_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n45)
      %b_col0_m2_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n45)
      %c_col0_m2_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1846528 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n45)
      aiex.dma_await_task(%c_col0_m2_n45)
      aiex.dma_free_task(%a_col0_m2_n45)
      aiex.dma_free_task(%b_col0_m2_n45)
      aiex.dma_free_task(%c_col0_m2_n45)
      // Col 0 M-tile 2 N-tile 46
      %a_col0_m2_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n46)
      %b_col0_m2_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n46)
      %c_col0_m2_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1846784 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n46)
      aiex.dma_await_task(%c_col0_m2_n46)
      aiex.dma_free_task(%a_col0_m2_n46)
      aiex.dma_free_task(%b_col0_m2_n46)
      aiex.dma_free_task(%c_col0_m2_n46)
      // Col 0 M-tile 2 N-tile 47
      %a_col0_m2_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n47)
      %b_col0_m2_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n47)
      %c_col0_m2_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1847040 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n47)
      aiex.dma_await_task(%c_col0_m2_n47)
      aiex.dma_free_task(%a_col0_m2_n47)
      aiex.dma_free_task(%b_col0_m2_n47)
      aiex.dma_free_task(%c_col0_m2_n47)
      // Col 0 M-tile 2 N-tile 48
      %a_col0_m2_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n48)
      %b_col0_m2_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n48)
      %c_col0_m2_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1847296 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n48)
      aiex.dma_await_task(%c_col0_m2_n48)
      aiex.dma_free_task(%a_col0_m2_n48)
      aiex.dma_free_task(%b_col0_m2_n48)
      aiex.dma_free_task(%c_col0_m2_n48)
      // Col 0 M-tile 2 N-tile 49
      %a_col0_m2_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n49)
      %b_col0_m2_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n49)
      %c_col0_m2_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1847552 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n49)
      aiex.dma_await_task(%c_col0_m2_n49)
      aiex.dma_free_task(%a_col0_m2_n49)
      aiex.dma_free_task(%b_col0_m2_n49)
      aiex.dma_free_task(%c_col0_m2_n49)
      // Col 0 M-tile 2 N-tile 50
      %a_col0_m2_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n50)
      %b_col0_m2_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n50)
      %c_col0_m2_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1847808 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n50)
      aiex.dma_await_task(%c_col0_m2_n50)
      aiex.dma_free_task(%a_col0_m2_n50)
      aiex.dma_free_task(%b_col0_m2_n50)
      aiex.dma_free_task(%c_col0_m2_n50)
      // Col 0 M-tile 2 N-tile 51
      %a_col0_m2_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n51)
      %b_col0_m2_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n51)
      %c_col0_m2_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1848064 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n51)
      aiex.dma_await_task(%c_col0_m2_n51)
      aiex.dma_free_task(%a_col0_m2_n51)
      aiex.dma_free_task(%b_col0_m2_n51)
      aiex.dma_free_task(%c_col0_m2_n51)
      // Col 0 M-tile 2 N-tile 52
      %a_col0_m2_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n52)
      %b_col0_m2_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n52)
      %c_col0_m2_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1848320 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n52)
      aiex.dma_await_task(%c_col0_m2_n52)
      aiex.dma_free_task(%a_col0_m2_n52)
      aiex.dma_free_task(%b_col0_m2_n52)
      aiex.dma_free_task(%c_col0_m2_n52)
      // Col 0 M-tile 2 N-tile 53
      %a_col0_m2_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n53)
      %b_col0_m2_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n53)
      %c_col0_m2_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1848576 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n53)
      aiex.dma_await_task(%c_col0_m2_n53)
      aiex.dma_free_task(%a_col0_m2_n53)
      aiex.dma_free_task(%b_col0_m2_n53)
      aiex.dma_free_task(%c_col0_m2_n53)
      // Col 0 M-tile 2 N-tile 54
      %a_col0_m2_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n54)
      %b_col0_m2_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n54)
      %c_col0_m2_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1848832 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n54)
      aiex.dma_await_task(%c_col0_m2_n54)
      aiex.dma_free_task(%a_col0_m2_n54)
      aiex.dma_free_task(%b_col0_m2_n54)
      aiex.dma_free_task(%c_col0_m2_n54)
      // Col 0 M-tile 2 N-tile 55
      %a_col0_m2_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n55)
      %b_col0_m2_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n55)
      %c_col0_m2_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 1849088 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n55)
      aiex.dma_await_task(%c_col0_m2_n55)
      aiex.dma_free_task(%a_col0_m2_n55)
      aiex.dma_free_task(%b_col0_m2_n55)
      aiex.dma_free_task(%c_col0_m2_n55)
      // Col 0 M-tile 3 N-tile 0
      %a_col0_m3_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n0)
      %b_col0_m3_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n0)
      %c_col0_m3_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2752512 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n0)
      aiex.dma_await_task(%c_col0_m3_n0)
      aiex.dma_free_task(%a_col0_m3_n0)
      aiex.dma_free_task(%b_col0_m3_n0)
      aiex.dma_free_task(%c_col0_m3_n0)
      // Col 0 M-tile 3 N-tile 1
      %a_col0_m3_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n1)
      %b_col0_m3_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n1)
      %c_col0_m3_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2752768 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n1)
      aiex.dma_await_task(%c_col0_m3_n1)
      aiex.dma_free_task(%a_col0_m3_n1)
      aiex.dma_free_task(%b_col0_m3_n1)
      aiex.dma_free_task(%c_col0_m3_n1)
      // Col 0 M-tile 3 N-tile 2
      %a_col0_m3_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n2)
      %b_col0_m3_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n2)
      %c_col0_m3_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2753024 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n2)
      aiex.dma_await_task(%c_col0_m3_n2)
      aiex.dma_free_task(%a_col0_m3_n2)
      aiex.dma_free_task(%b_col0_m3_n2)
      aiex.dma_free_task(%c_col0_m3_n2)
      // Col 0 M-tile 3 N-tile 3
      %a_col0_m3_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n3)
      %b_col0_m3_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n3)
      %c_col0_m3_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2753280 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n3)
      aiex.dma_await_task(%c_col0_m3_n3)
      aiex.dma_free_task(%a_col0_m3_n3)
      aiex.dma_free_task(%b_col0_m3_n3)
      aiex.dma_free_task(%c_col0_m3_n3)
      // Col 0 M-tile 3 N-tile 4
      %a_col0_m3_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n4)
      %b_col0_m3_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n4)
      %c_col0_m3_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2753536 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n4)
      aiex.dma_await_task(%c_col0_m3_n4)
      aiex.dma_free_task(%a_col0_m3_n4)
      aiex.dma_free_task(%b_col0_m3_n4)
      aiex.dma_free_task(%c_col0_m3_n4)
      // Col 0 M-tile 3 N-tile 5
      %a_col0_m3_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n5)
      %b_col0_m3_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n5)
      %c_col0_m3_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2753792 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n5)
      aiex.dma_await_task(%c_col0_m3_n5)
      aiex.dma_free_task(%a_col0_m3_n5)
      aiex.dma_free_task(%b_col0_m3_n5)
      aiex.dma_free_task(%c_col0_m3_n5)
      // Col 0 M-tile 3 N-tile 6
      %a_col0_m3_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n6)
      %b_col0_m3_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n6)
      %c_col0_m3_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2754048 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n6)
      aiex.dma_await_task(%c_col0_m3_n6)
      aiex.dma_free_task(%a_col0_m3_n6)
      aiex.dma_free_task(%b_col0_m3_n6)
      aiex.dma_free_task(%c_col0_m3_n6)
      // Col 0 M-tile 3 N-tile 7
      %a_col0_m3_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n7)
      %b_col0_m3_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n7)
      %c_col0_m3_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2754304 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n7)
      aiex.dma_await_task(%c_col0_m3_n7)
      aiex.dma_free_task(%a_col0_m3_n7)
      aiex.dma_free_task(%b_col0_m3_n7)
      aiex.dma_free_task(%c_col0_m3_n7)
      // Col 0 M-tile 3 N-tile 8
      %a_col0_m3_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n8)
      %b_col0_m3_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n8)
      %c_col0_m3_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2754560 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n8)
      aiex.dma_await_task(%c_col0_m3_n8)
      aiex.dma_free_task(%a_col0_m3_n8)
      aiex.dma_free_task(%b_col0_m3_n8)
      aiex.dma_free_task(%c_col0_m3_n8)
      // Col 0 M-tile 3 N-tile 9
      %a_col0_m3_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n9)
      %b_col0_m3_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n9)
      %c_col0_m3_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2754816 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n9)
      aiex.dma_await_task(%c_col0_m3_n9)
      aiex.dma_free_task(%a_col0_m3_n9)
      aiex.dma_free_task(%b_col0_m3_n9)
      aiex.dma_free_task(%c_col0_m3_n9)
      // Col 0 M-tile 3 N-tile 10
      %a_col0_m3_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n10)
      %b_col0_m3_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n10)
      %c_col0_m3_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2755072 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n10)
      aiex.dma_await_task(%c_col0_m3_n10)
      aiex.dma_free_task(%a_col0_m3_n10)
      aiex.dma_free_task(%b_col0_m3_n10)
      aiex.dma_free_task(%c_col0_m3_n10)
      // Col 0 M-tile 3 N-tile 11
      %a_col0_m3_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n11)
      %b_col0_m3_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n11)
      %c_col0_m3_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2755328 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n11)
      aiex.dma_await_task(%c_col0_m3_n11)
      aiex.dma_free_task(%a_col0_m3_n11)
      aiex.dma_free_task(%b_col0_m3_n11)
      aiex.dma_free_task(%c_col0_m3_n11)
      // Col 0 M-tile 3 N-tile 12
      %a_col0_m3_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n12)
      %b_col0_m3_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n12)
      %c_col0_m3_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2755584 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n12)
      aiex.dma_await_task(%c_col0_m3_n12)
      aiex.dma_free_task(%a_col0_m3_n12)
      aiex.dma_free_task(%b_col0_m3_n12)
      aiex.dma_free_task(%c_col0_m3_n12)
      // Col 0 M-tile 3 N-tile 13
      %a_col0_m3_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n13)
      %b_col0_m3_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n13)
      %c_col0_m3_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2755840 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n13)
      aiex.dma_await_task(%c_col0_m3_n13)
      aiex.dma_free_task(%a_col0_m3_n13)
      aiex.dma_free_task(%b_col0_m3_n13)
      aiex.dma_free_task(%c_col0_m3_n13)
      // Col 0 M-tile 3 N-tile 14
      %a_col0_m3_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n14)
      %b_col0_m3_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n14)
      %c_col0_m3_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2756096 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n14)
      aiex.dma_await_task(%c_col0_m3_n14)
      aiex.dma_free_task(%a_col0_m3_n14)
      aiex.dma_free_task(%b_col0_m3_n14)
      aiex.dma_free_task(%c_col0_m3_n14)
      // Col 0 M-tile 3 N-tile 15
      %a_col0_m3_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n15)
      %b_col0_m3_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n15)
      %c_col0_m3_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2756352 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n15)
      aiex.dma_await_task(%c_col0_m3_n15)
      aiex.dma_free_task(%a_col0_m3_n15)
      aiex.dma_free_task(%b_col0_m3_n15)
      aiex.dma_free_task(%c_col0_m3_n15)
      // Col 0 M-tile 3 N-tile 16
      %a_col0_m3_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n16)
      %b_col0_m3_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n16)
      %c_col0_m3_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2756608 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n16)
      aiex.dma_await_task(%c_col0_m3_n16)
      aiex.dma_free_task(%a_col0_m3_n16)
      aiex.dma_free_task(%b_col0_m3_n16)
      aiex.dma_free_task(%c_col0_m3_n16)
      // Col 0 M-tile 3 N-tile 17
      %a_col0_m3_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n17)
      %b_col0_m3_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n17)
      %c_col0_m3_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2756864 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n17)
      aiex.dma_await_task(%c_col0_m3_n17)
      aiex.dma_free_task(%a_col0_m3_n17)
      aiex.dma_free_task(%b_col0_m3_n17)
      aiex.dma_free_task(%c_col0_m3_n17)
      // Col 0 M-tile 3 N-tile 18
      %a_col0_m3_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n18)
      %b_col0_m3_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n18)
      %c_col0_m3_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2757120 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n18)
      aiex.dma_await_task(%c_col0_m3_n18)
      aiex.dma_free_task(%a_col0_m3_n18)
      aiex.dma_free_task(%b_col0_m3_n18)
      aiex.dma_free_task(%c_col0_m3_n18)
      // Col 0 M-tile 3 N-tile 19
      %a_col0_m3_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n19)
      %b_col0_m3_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n19)
      %c_col0_m3_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2757376 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n19)
      aiex.dma_await_task(%c_col0_m3_n19)
      aiex.dma_free_task(%a_col0_m3_n19)
      aiex.dma_free_task(%b_col0_m3_n19)
      aiex.dma_free_task(%c_col0_m3_n19)
      // Col 0 M-tile 3 N-tile 20
      %a_col0_m3_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n20)
      %b_col0_m3_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n20)
      %c_col0_m3_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2757632 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n20)
      aiex.dma_await_task(%c_col0_m3_n20)
      aiex.dma_free_task(%a_col0_m3_n20)
      aiex.dma_free_task(%b_col0_m3_n20)
      aiex.dma_free_task(%c_col0_m3_n20)
      // Col 0 M-tile 3 N-tile 21
      %a_col0_m3_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n21)
      %b_col0_m3_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n21)
      %c_col0_m3_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2757888 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n21)
      aiex.dma_await_task(%c_col0_m3_n21)
      aiex.dma_free_task(%a_col0_m3_n21)
      aiex.dma_free_task(%b_col0_m3_n21)
      aiex.dma_free_task(%c_col0_m3_n21)
      // Col 0 M-tile 3 N-tile 22
      %a_col0_m3_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n22)
      %b_col0_m3_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n22)
      %c_col0_m3_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2758144 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n22)
      aiex.dma_await_task(%c_col0_m3_n22)
      aiex.dma_free_task(%a_col0_m3_n22)
      aiex.dma_free_task(%b_col0_m3_n22)
      aiex.dma_free_task(%c_col0_m3_n22)
      // Col 0 M-tile 3 N-tile 23
      %a_col0_m3_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n23)
      %b_col0_m3_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n23)
      %c_col0_m3_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2758400 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n23)
      aiex.dma_await_task(%c_col0_m3_n23)
      aiex.dma_free_task(%a_col0_m3_n23)
      aiex.dma_free_task(%b_col0_m3_n23)
      aiex.dma_free_task(%c_col0_m3_n23)
      // Col 0 M-tile 3 N-tile 24
      %a_col0_m3_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n24)
      %b_col0_m3_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n24)
      %c_col0_m3_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2758656 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n24)
      aiex.dma_await_task(%c_col0_m3_n24)
      aiex.dma_free_task(%a_col0_m3_n24)
      aiex.dma_free_task(%b_col0_m3_n24)
      aiex.dma_free_task(%c_col0_m3_n24)
      // Col 0 M-tile 3 N-tile 25
      %a_col0_m3_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n25)
      %b_col0_m3_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n25)
      %c_col0_m3_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2758912 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n25)
      aiex.dma_await_task(%c_col0_m3_n25)
      aiex.dma_free_task(%a_col0_m3_n25)
      aiex.dma_free_task(%b_col0_m3_n25)
      aiex.dma_free_task(%c_col0_m3_n25)
      // Col 0 M-tile 3 N-tile 26
      %a_col0_m3_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n26)
      %b_col0_m3_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n26)
      %c_col0_m3_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2759168 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n26)
      aiex.dma_await_task(%c_col0_m3_n26)
      aiex.dma_free_task(%a_col0_m3_n26)
      aiex.dma_free_task(%b_col0_m3_n26)
      aiex.dma_free_task(%c_col0_m3_n26)
      // Col 0 M-tile 3 N-tile 27
      %a_col0_m3_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n27)
      %b_col0_m3_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n27)
      %c_col0_m3_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2759424 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n27)
      aiex.dma_await_task(%c_col0_m3_n27)
      aiex.dma_free_task(%a_col0_m3_n27)
      aiex.dma_free_task(%b_col0_m3_n27)
      aiex.dma_free_task(%c_col0_m3_n27)
      // Col 0 M-tile 3 N-tile 28
      %a_col0_m3_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n28)
      %b_col0_m3_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n28)
      %c_col0_m3_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2759680 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n28)
      aiex.dma_await_task(%c_col0_m3_n28)
      aiex.dma_free_task(%a_col0_m3_n28)
      aiex.dma_free_task(%b_col0_m3_n28)
      aiex.dma_free_task(%c_col0_m3_n28)
      // Col 0 M-tile 3 N-tile 29
      %a_col0_m3_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n29)
      %b_col0_m3_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n29)
      %c_col0_m3_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2759936 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n29)
      aiex.dma_await_task(%c_col0_m3_n29)
      aiex.dma_free_task(%a_col0_m3_n29)
      aiex.dma_free_task(%b_col0_m3_n29)
      aiex.dma_free_task(%c_col0_m3_n29)
      // Col 0 M-tile 3 N-tile 30
      %a_col0_m3_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n30)
      %b_col0_m3_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n30)
      %c_col0_m3_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2760192 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n30)
      aiex.dma_await_task(%c_col0_m3_n30)
      aiex.dma_free_task(%a_col0_m3_n30)
      aiex.dma_free_task(%b_col0_m3_n30)
      aiex.dma_free_task(%c_col0_m3_n30)
      // Col 0 M-tile 3 N-tile 31
      %a_col0_m3_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n31)
      %b_col0_m3_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n31)
      %c_col0_m3_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2760448 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n31)
      aiex.dma_await_task(%c_col0_m3_n31)
      aiex.dma_free_task(%a_col0_m3_n31)
      aiex.dma_free_task(%b_col0_m3_n31)
      aiex.dma_free_task(%c_col0_m3_n31)
      // Col 0 M-tile 3 N-tile 32
      %a_col0_m3_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n32)
      %b_col0_m3_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n32)
      %c_col0_m3_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2760704 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n32)
      aiex.dma_await_task(%c_col0_m3_n32)
      aiex.dma_free_task(%a_col0_m3_n32)
      aiex.dma_free_task(%b_col0_m3_n32)
      aiex.dma_free_task(%c_col0_m3_n32)
      // Col 0 M-tile 3 N-tile 33
      %a_col0_m3_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n33)
      %b_col0_m3_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n33)
      %c_col0_m3_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2760960 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n33)
      aiex.dma_await_task(%c_col0_m3_n33)
      aiex.dma_free_task(%a_col0_m3_n33)
      aiex.dma_free_task(%b_col0_m3_n33)
      aiex.dma_free_task(%c_col0_m3_n33)
      // Col 0 M-tile 3 N-tile 34
      %a_col0_m3_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n34)
      %b_col0_m3_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n34)
      %c_col0_m3_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2761216 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n34)
      aiex.dma_await_task(%c_col0_m3_n34)
      aiex.dma_free_task(%a_col0_m3_n34)
      aiex.dma_free_task(%b_col0_m3_n34)
      aiex.dma_free_task(%c_col0_m3_n34)
      // Col 0 M-tile 3 N-tile 35
      %a_col0_m3_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n35)
      %b_col0_m3_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n35)
      %c_col0_m3_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2761472 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n35)
      aiex.dma_await_task(%c_col0_m3_n35)
      aiex.dma_free_task(%a_col0_m3_n35)
      aiex.dma_free_task(%b_col0_m3_n35)
      aiex.dma_free_task(%c_col0_m3_n35)
      // Col 0 M-tile 3 N-tile 36
      %a_col0_m3_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n36)
      %b_col0_m3_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n36)
      %c_col0_m3_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2761728 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n36)
      aiex.dma_await_task(%c_col0_m3_n36)
      aiex.dma_free_task(%a_col0_m3_n36)
      aiex.dma_free_task(%b_col0_m3_n36)
      aiex.dma_free_task(%c_col0_m3_n36)
      // Col 0 M-tile 3 N-tile 37
      %a_col0_m3_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n37)
      %b_col0_m3_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n37)
      %c_col0_m3_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2761984 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n37)
      aiex.dma_await_task(%c_col0_m3_n37)
      aiex.dma_free_task(%a_col0_m3_n37)
      aiex.dma_free_task(%b_col0_m3_n37)
      aiex.dma_free_task(%c_col0_m3_n37)
      // Col 0 M-tile 3 N-tile 38
      %a_col0_m3_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n38)
      %b_col0_m3_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n38)
      %c_col0_m3_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2762240 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n38)
      aiex.dma_await_task(%c_col0_m3_n38)
      aiex.dma_free_task(%a_col0_m3_n38)
      aiex.dma_free_task(%b_col0_m3_n38)
      aiex.dma_free_task(%c_col0_m3_n38)
      // Col 0 M-tile 3 N-tile 39
      %a_col0_m3_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n39)
      %b_col0_m3_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n39)
      %c_col0_m3_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2762496 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n39)
      aiex.dma_await_task(%c_col0_m3_n39)
      aiex.dma_free_task(%a_col0_m3_n39)
      aiex.dma_free_task(%b_col0_m3_n39)
      aiex.dma_free_task(%c_col0_m3_n39)
      // Col 0 M-tile 3 N-tile 40
      %a_col0_m3_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n40)
      %b_col0_m3_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n40)
      %c_col0_m3_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2762752 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n40)
      aiex.dma_await_task(%c_col0_m3_n40)
      aiex.dma_free_task(%a_col0_m3_n40)
      aiex.dma_free_task(%b_col0_m3_n40)
      aiex.dma_free_task(%c_col0_m3_n40)
      // Col 0 M-tile 3 N-tile 41
      %a_col0_m3_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n41)
      %b_col0_m3_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n41)
      %c_col0_m3_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2763008 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n41)
      aiex.dma_await_task(%c_col0_m3_n41)
      aiex.dma_free_task(%a_col0_m3_n41)
      aiex.dma_free_task(%b_col0_m3_n41)
      aiex.dma_free_task(%c_col0_m3_n41)
      // Col 0 M-tile 3 N-tile 42
      %a_col0_m3_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n42)
      %b_col0_m3_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n42)
      %c_col0_m3_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2763264 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n42)
      aiex.dma_await_task(%c_col0_m3_n42)
      aiex.dma_free_task(%a_col0_m3_n42)
      aiex.dma_free_task(%b_col0_m3_n42)
      aiex.dma_free_task(%c_col0_m3_n42)
      // Col 0 M-tile 3 N-tile 43
      %a_col0_m3_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n43)
      %b_col0_m3_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n43)
      %c_col0_m3_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2763520 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n43)
      aiex.dma_await_task(%c_col0_m3_n43)
      aiex.dma_free_task(%a_col0_m3_n43)
      aiex.dma_free_task(%b_col0_m3_n43)
      aiex.dma_free_task(%c_col0_m3_n43)
      // Col 0 M-tile 3 N-tile 44
      %a_col0_m3_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n44)
      %b_col0_m3_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n44)
      %c_col0_m3_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2763776 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n44)
      aiex.dma_await_task(%c_col0_m3_n44)
      aiex.dma_free_task(%a_col0_m3_n44)
      aiex.dma_free_task(%b_col0_m3_n44)
      aiex.dma_free_task(%c_col0_m3_n44)
      // Col 0 M-tile 3 N-tile 45
      %a_col0_m3_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n45)
      %b_col0_m3_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n45)
      %c_col0_m3_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2764032 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n45)
      aiex.dma_await_task(%c_col0_m3_n45)
      aiex.dma_free_task(%a_col0_m3_n45)
      aiex.dma_free_task(%b_col0_m3_n45)
      aiex.dma_free_task(%c_col0_m3_n45)
      // Col 0 M-tile 3 N-tile 46
      %a_col0_m3_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n46)
      %b_col0_m3_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n46)
      %c_col0_m3_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2764288 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n46)
      aiex.dma_await_task(%c_col0_m3_n46)
      aiex.dma_free_task(%a_col0_m3_n46)
      aiex.dma_free_task(%b_col0_m3_n46)
      aiex.dma_free_task(%c_col0_m3_n46)
      // Col 0 M-tile 3 N-tile 47
      %a_col0_m3_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n47)
      %b_col0_m3_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n47)
      %c_col0_m3_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2764544 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n47)
      aiex.dma_await_task(%c_col0_m3_n47)
      aiex.dma_free_task(%a_col0_m3_n47)
      aiex.dma_free_task(%b_col0_m3_n47)
      aiex.dma_free_task(%c_col0_m3_n47)
      // Col 0 M-tile 3 N-tile 48
      %a_col0_m3_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n48)
      %b_col0_m3_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n48)
      %c_col0_m3_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2764800 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n48)
      aiex.dma_await_task(%c_col0_m3_n48)
      aiex.dma_free_task(%a_col0_m3_n48)
      aiex.dma_free_task(%b_col0_m3_n48)
      aiex.dma_free_task(%c_col0_m3_n48)
      // Col 0 M-tile 3 N-tile 49
      %a_col0_m3_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n49)
      %b_col0_m3_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n49)
      %c_col0_m3_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2765056 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n49)
      aiex.dma_await_task(%c_col0_m3_n49)
      aiex.dma_free_task(%a_col0_m3_n49)
      aiex.dma_free_task(%b_col0_m3_n49)
      aiex.dma_free_task(%c_col0_m3_n49)
      // Col 0 M-tile 3 N-tile 50
      %a_col0_m3_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n50)
      %b_col0_m3_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n50)
      %c_col0_m3_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2765312 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n50)
      aiex.dma_await_task(%c_col0_m3_n50)
      aiex.dma_free_task(%a_col0_m3_n50)
      aiex.dma_free_task(%b_col0_m3_n50)
      aiex.dma_free_task(%c_col0_m3_n50)
      // Col 0 M-tile 3 N-tile 51
      %a_col0_m3_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n51)
      %b_col0_m3_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n51)
      %c_col0_m3_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2765568 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n51)
      aiex.dma_await_task(%c_col0_m3_n51)
      aiex.dma_free_task(%a_col0_m3_n51)
      aiex.dma_free_task(%b_col0_m3_n51)
      aiex.dma_free_task(%c_col0_m3_n51)
      // Col 0 M-tile 3 N-tile 52
      %a_col0_m3_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n52)
      %b_col0_m3_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n52)
      %c_col0_m3_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2765824 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n52)
      aiex.dma_await_task(%c_col0_m3_n52)
      aiex.dma_free_task(%a_col0_m3_n52)
      aiex.dma_free_task(%b_col0_m3_n52)
      aiex.dma_free_task(%c_col0_m3_n52)
      // Col 0 M-tile 3 N-tile 53
      %a_col0_m3_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n53)
      %b_col0_m3_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n53)
      %c_col0_m3_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2766080 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n53)
      aiex.dma_await_task(%c_col0_m3_n53)
      aiex.dma_free_task(%a_col0_m3_n53)
      aiex.dma_free_task(%b_col0_m3_n53)
      aiex.dma_free_task(%c_col0_m3_n53)
      // Col 0 M-tile 3 N-tile 54
      %a_col0_m3_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n54)
      %b_col0_m3_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n54)
      %c_col0_m3_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2766336 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n54)
      aiex.dma_await_task(%c_col0_m3_n54)
      aiex.dma_free_task(%a_col0_m3_n54)
      aiex.dma_free_task(%b_col0_m3_n54)
      aiex.dma_free_task(%c_col0_m3_n54)
      // Col 0 M-tile 3 N-tile 55
      %a_col0_m3_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n55)
      %b_col0_m3_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n55)
      %c_col0_m3_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 2766592 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n55)
      aiex.dma_await_task(%c_col0_m3_n55)
      aiex.dma_free_task(%a_col0_m3_n55)
      aiex.dma_free_task(%b_col0_m3_n55)
      aiex.dma_free_task(%c_col0_m3_n55)
      // Col 0 M-tile 4 N-tile 0
      %a_col0_m4_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n0)
      %b_col0_m4_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n0)
      %c_col0_m4_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3670016 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n0)
      aiex.dma_await_task(%c_col0_m4_n0)
      aiex.dma_free_task(%a_col0_m4_n0)
      aiex.dma_free_task(%b_col0_m4_n0)
      aiex.dma_free_task(%c_col0_m4_n0)
      // Col 0 M-tile 4 N-tile 1
      %a_col0_m4_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n1)
      %b_col0_m4_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n1)
      %c_col0_m4_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3670272 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n1)
      aiex.dma_await_task(%c_col0_m4_n1)
      aiex.dma_free_task(%a_col0_m4_n1)
      aiex.dma_free_task(%b_col0_m4_n1)
      aiex.dma_free_task(%c_col0_m4_n1)
      // Col 0 M-tile 4 N-tile 2
      %a_col0_m4_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n2)
      %b_col0_m4_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n2)
      %c_col0_m4_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3670528 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n2)
      aiex.dma_await_task(%c_col0_m4_n2)
      aiex.dma_free_task(%a_col0_m4_n2)
      aiex.dma_free_task(%b_col0_m4_n2)
      aiex.dma_free_task(%c_col0_m4_n2)
      // Col 0 M-tile 4 N-tile 3
      %a_col0_m4_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n3)
      %b_col0_m4_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n3)
      %c_col0_m4_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3670784 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n3)
      aiex.dma_await_task(%c_col0_m4_n3)
      aiex.dma_free_task(%a_col0_m4_n3)
      aiex.dma_free_task(%b_col0_m4_n3)
      aiex.dma_free_task(%c_col0_m4_n3)
      // Col 0 M-tile 4 N-tile 4
      %a_col0_m4_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n4)
      %b_col0_m4_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n4)
      %c_col0_m4_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3671040 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n4)
      aiex.dma_await_task(%c_col0_m4_n4)
      aiex.dma_free_task(%a_col0_m4_n4)
      aiex.dma_free_task(%b_col0_m4_n4)
      aiex.dma_free_task(%c_col0_m4_n4)
      // Col 0 M-tile 4 N-tile 5
      %a_col0_m4_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n5)
      %b_col0_m4_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n5)
      %c_col0_m4_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3671296 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n5)
      aiex.dma_await_task(%c_col0_m4_n5)
      aiex.dma_free_task(%a_col0_m4_n5)
      aiex.dma_free_task(%b_col0_m4_n5)
      aiex.dma_free_task(%c_col0_m4_n5)
      // Col 0 M-tile 4 N-tile 6
      %a_col0_m4_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n6)
      %b_col0_m4_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n6)
      %c_col0_m4_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3671552 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n6)
      aiex.dma_await_task(%c_col0_m4_n6)
      aiex.dma_free_task(%a_col0_m4_n6)
      aiex.dma_free_task(%b_col0_m4_n6)
      aiex.dma_free_task(%c_col0_m4_n6)
      // Col 0 M-tile 4 N-tile 7
      %a_col0_m4_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n7)
      %b_col0_m4_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n7)
      %c_col0_m4_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3671808 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n7)
      aiex.dma_await_task(%c_col0_m4_n7)
      aiex.dma_free_task(%a_col0_m4_n7)
      aiex.dma_free_task(%b_col0_m4_n7)
      aiex.dma_free_task(%c_col0_m4_n7)
      // Col 0 M-tile 4 N-tile 8
      %a_col0_m4_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n8)
      %b_col0_m4_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n8)
      %c_col0_m4_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3672064 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n8)
      aiex.dma_await_task(%c_col0_m4_n8)
      aiex.dma_free_task(%a_col0_m4_n8)
      aiex.dma_free_task(%b_col0_m4_n8)
      aiex.dma_free_task(%c_col0_m4_n8)
      // Col 0 M-tile 4 N-tile 9
      %a_col0_m4_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n9)
      %b_col0_m4_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n9)
      %c_col0_m4_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3672320 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n9)
      aiex.dma_await_task(%c_col0_m4_n9)
      aiex.dma_free_task(%a_col0_m4_n9)
      aiex.dma_free_task(%b_col0_m4_n9)
      aiex.dma_free_task(%c_col0_m4_n9)
      // Col 0 M-tile 4 N-tile 10
      %a_col0_m4_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n10)
      %b_col0_m4_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n10)
      %c_col0_m4_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3672576 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n10)
      aiex.dma_await_task(%c_col0_m4_n10)
      aiex.dma_free_task(%a_col0_m4_n10)
      aiex.dma_free_task(%b_col0_m4_n10)
      aiex.dma_free_task(%c_col0_m4_n10)
      // Col 0 M-tile 4 N-tile 11
      %a_col0_m4_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n11)
      %b_col0_m4_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n11)
      %c_col0_m4_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3672832 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n11)
      aiex.dma_await_task(%c_col0_m4_n11)
      aiex.dma_free_task(%a_col0_m4_n11)
      aiex.dma_free_task(%b_col0_m4_n11)
      aiex.dma_free_task(%c_col0_m4_n11)
      // Col 0 M-tile 4 N-tile 12
      %a_col0_m4_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n12)
      %b_col0_m4_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n12)
      %c_col0_m4_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3673088 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n12)
      aiex.dma_await_task(%c_col0_m4_n12)
      aiex.dma_free_task(%a_col0_m4_n12)
      aiex.dma_free_task(%b_col0_m4_n12)
      aiex.dma_free_task(%c_col0_m4_n12)
      // Col 0 M-tile 4 N-tile 13
      %a_col0_m4_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n13)
      %b_col0_m4_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n13)
      %c_col0_m4_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3673344 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n13)
      aiex.dma_await_task(%c_col0_m4_n13)
      aiex.dma_free_task(%a_col0_m4_n13)
      aiex.dma_free_task(%b_col0_m4_n13)
      aiex.dma_free_task(%c_col0_m4_n13)
      // Col 0 M-tile 4 N-tile 14
      %a_col0_m4_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n14)
      %b_col0_m4_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n14)
      %c_col0_m4_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3673600 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n14)
      aiex.dma_await_task(%c_col0_m4_n14)
      aiex.dma_free_task(%a_col0_m4_n14)
      aiex.dma_free_task(%b_col0_m4_n14)
      aiex.dma_free_task(%c_col0_m4_n14)
      // Col 0 M-tile 4 N-tile 15
      %a_col0_m4_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n15)
      %b_col0_m4_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n15)
      %c_col0_m4_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3673856 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n15)
      aiex.dma_await_task(%c_col0_m4_n15)
      aiex.dma_free_task(%a_col0_m4_n15)
      aiex.dma_free_task(%b_col0_m4_n15)
      aiex.dma_free_task(%c_col0_m4_n15)
      // Col 0 M-tile 4 N-tile 16
      %a_col0_m4_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n16)
      %b_col0_m4_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n16)
      %c_col0_m4_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3674112 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n16)
      aiex.dma_await_task(%c_col0_m4_n16)
      aiex.dma_free_task(%a_col0_m4_n16)
      aiex.dma_free_task(%b_col0_m4_n16)
      aiex.dma_free_task(%c_col0_m4_n16)
      // Col 0 M-tile 4 N-tile 17
      %a_col0_m4_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n17)
      %b_col0_m4_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n17)
      %c_col0_m4_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3674368 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n17)
      aiex.dma_await_task(%c_col0_m4_n17)
      aiex.dma_free_task(%a_col0_m4_n17)
      aiex.dma_free_task(%b_col0_m4_n17)
      aiex.dma_free_task(%c_col0_m4_n17)
      // Col 0 M-tile 4 N-tile 18
      %a_col0_m4_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n18)
      %b_col0_m4_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n18)
      %c_col0_m4_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3674624 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n18)
      aiex.dma_await_task(%c_col0_m4_n18)
      aiex.dma_free_task(%a_col0_m4_n18)
      aiex.dma_free_task(%b_col0_m4_n18)
      aiex.dma_free_task(%c_col0_m4_n18)
      // Col 0 M-tile 4 N-tile 19
      %a_col0_m4_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n19)
      %b_col0_m4_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n19)
      %c_col0_m4_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3674880 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n19)
      aiex.dma_await_task(%c_col0_m4_n19)
      aiex.dma_free_task(%a_col0_m4_n19)
      aiex.dma_free_task(%b_col0_m4_n19)
      aiex.dma_free_task(%c_col0_m4_n19)
      // Col 0 M-tile 4 N-tile 20
      %a_col0_m4_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n20)
      %b_col0_m4_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n20)
      %c_col0_m4_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3675136 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n20)
      aiex.dma_await_task(%c_col0_m4_n20)
      aiex.dma_free_task(%a_col0_m4_n20)
      aiex.dma_free_task(%b_col0_m4_n20)
      aiex.dma_free_task(%c_col0_m4_n20)
      // Col 0 M-tile 4 N-tile 21
      %a_col0_m4_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n21)
      %b_col0_m4_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n21)
      %c_col0_m4_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3675392 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n21)
      aiex.dma_await_task(%c_col0_m4_n21)
      aiex.dma_free_task(%a_col0_m4_n21)
      aiex.dma_free_task(%b_col0_m4_n21)
      aiex.dma_free_task(%c_col0_m4_n21)
      // Col 0 M-tile 4 N-tile 22
      %a_col0_m4_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n22)
      %b_col0_m4_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n22)
      %c_col0_m4_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3675648 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n22)
      aiex.dma_await_task(%c_col0_m4_n22)
      aiex.dma_free_task(%a_col0_m4_n22)
      aiex.dma_free_task(%b_col0_m4_n22)
      aiex.dma_free_task(%c_col0_m4_n22)
      // Col 0 M-tile 4 N-tile 23
      %a_col0_m4_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n23)
      %b_col0_m4_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n23)
      %c_col0_m4_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3675904 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n23)
      aiex.dma_await_task(%c_col0_m4_n23)
      aiex.dma_free_task(%a_col0_m4_n23)
      aiex.dma_free_task(%b_col0_m4_n23)
      aiex.dma_free_task(%c_col0_m4_n23)
      // Col 0 M-tile 4 N-tile 24
      %a_col0_m4_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n24)
      %b_col0_m4_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n24)
      %c_col0_m4_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3676160 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n24)
      aiex.dma_await_task(%c_col0_m4_n24)
      aiex.dma_free_task(%a_col0_m4_n24)
      aiex.dma_free_task(%b_col0_m4_n24)
      aiex.dma_free_task(%c_col0_m4_n24)
      // Col 0 M-tile 4 N-tile 25
      %a_col0_m4_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n25)
      %b_col0_m4_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n25)
      %c_col0_m4_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3676416 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n25)
      aiex.dma_await_task(%c_col0_m4_n25)
      aiex.dma_free_task(%a_col0_m4_n25)
      aiex.dma_free_task(%b_col0_m4_n25)
      aiex.dma_free_task(%c_col0_m4_n25)
      // Col 0 M-tile 4 N-tile 26
      %a_col0_m4_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n26)
      %b_col0_m4_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n26)
      %c_col0_m4_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3676672 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n26)
      aiex.dma_await_task(%c_col0_m4_n26)
      aiex.dma_free_task(%a_col0_m4_n26)
      aiex.dma_free_task(%b_col0_m4_n26)
      aiex.dma_free_task(%c_col0_m4_n26)
      // Col 0 M-tile 4 N-tile 27
      %a_col0_m4_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n27)
      %b_col0_m4_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n27)
      %c_col0_m4_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3676928 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n27)
      aiex.dma_await_task(%c_col0_m4_n27)
      aiex.dma_free_task(%a_col0_m4_n27)
      aiex.dma_free_task(%b_col0_m4_n27)
      aiex.dma_free_task(%c_col0_m4_n27)
      // Col 0 M-tile 4 N-tile 28
      %a_col0_m4_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n28)
      %b_col0_m4_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n28)
      %c_col0_m4_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3677184 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n28)
      aiex.dma_await_task(%c_col0_m4_n28)
      aiex.dma_free_task(%a_col0_m4_n28)
      aiex.dma_free_task(%b_col0_m4_n28)
      aiex.dma_free_task(%c_col0_m4_n28)
      // Col 0 M-tile 4 N-tile 29
      %a_col0_m4_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n29)
      %b_col0_m4_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n29)
      %c_col0_m4_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3677440 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n29)
      aiex.dma_await_task(%c_col0_m4_n29)
      aiex.dma_free_task(%a_col0_m4_n29)
      aiex.dma_free_task(%b_col0_m4_n29)
      aiex.dma_free_task(%c_col0_m4_n29)
      // Col 0 M-tile 4 N-tile 30
      %a_col0_m4_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n30)
      %b_col0_m4_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n30)
      %c_col0_m4_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3677696 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n30)
      aiex.dma_await_task(%c_col0_m4_n30)
      aiex.dma_free_task(%a_col0_m4_n30)
      aiex.dma_free_task(%b_col0_m4_n30)
      aiex.dma_free_task(%c_col0_m4_n30)
      // Col 0 M-tile 4 N-tile 31
      %a_col0_m4_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n31)
      %b_col0_m4_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n31)
      %c_col0_m4_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3677952 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n31)
      aiex.dma_await_task(%c_col0_m4_n31)
      aiex.dma_free_task(%a_col0_m4_n31)
      aiex.dma_free_task(%b_col0_m4_n31)
      aiex.dma_free_task(%c_col0_m4_n31)
      // Col 0 M-tile 4 N-tile 32
      %a_col0_m4_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n32)
      %b_col0_m4_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n32)
      %c_col0_m4_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3678208 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n32)
      aiex.dma_await_task(%c_col0_m4_n32)
      aiex.dma_free_task(%a_col0_m4_n32)
      aiex.dma_free_task(%b_col0_m4_n32)
      aiex.dma_free_task(%c_col0_m4_n32)
      // Col 0 M-tile 4 N-tile 33
      %a_col0_m4_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n33)
      %b_col0_m4_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n33)
      %c_col0_m4_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3678464 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n33)
      aiex.dma_await_task(%c_col0_m4_n33)
      aiex.dma_free_task(%a_col0_m4_n33)
      aiex.dma_free_task(%b_col0_m4_n33)
      aiex.dma_free_task(%c_col0_m4_n33)
      // Col 0 M-tile 4 N-tile 34
      %a_col0_m4_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n34)
      %b_col0_m4_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n34)
      %c_col0_m4_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3678720 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n34)
      aiex.dma_await_task(%c_col0_m4_n34)
      aiex.dma_free_task(%a_col0_m4_n34)
      aiex.dma_free_task(%b_col0_m4_n34)
      aiex.dma_free_task(%c_col0_m4_n34)
      // Col 0 M-tile 4 N-tile 35
      %a_col0_m4_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n35)
      %b_col0_m4_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n35)
      %c_col0_m4_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3678976 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n35)
      aiex.dma_await_task(%c_col0_m4_n35)
      aiex.dma_free_task(%a_col0_m4_n35)
      aiex.dma_free_task(%b_col0_m4_n35)
      aiex.dma_free_task(%c_col0_m4_n35)
      // Col 0 M-tile 4 N-tile 36
      %a_col0_m4_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n36)
      %b_col0_m4_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n36)
      %c_col0_m4_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3679232 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n36)
      aiex.dma_await_task(%c_col0_m4_n36)
      aiex.dma_free_task(%a_col0_m4_n36)
      aiex.dma_free_task(%b_col0_m4_n36)
      aiex.dma_free_task(%c_col0_m4_n36)
      // Col 0 M-tile 4 N-tile 37
      %a_col0_m4_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n37)
      %b_col0_m4_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n37)
      %c_col0_m4_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3679488 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n37)
      aiex.dma_await_task(%c_col0_m4_n37)
      aiex.dma_free_task(%a_col0_m4_n37)
      aiex.dma_free_task(%b_col0_m4_n37)
      aiex.dma_free_task(%c_col0_m4_n37)
      // Col 0 M-tile 4 N-tile 38
      %a_col0_m4_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n38)
      %b_col0_m4_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n38)
      %c_col0_m4_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3679744 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n38)
      aiex.dma_await_task(%c_col0_m4_n38)
      aiex.dma_free_task(%a_col0_m4_n38)
      aiex.dma_free_task(%b_col0_m4_n38)
      aiex.dma_free_task(%c_col0_m4_n38)
      // Col 0 M-tile 4 N-tile 39
      %a_col0_m4_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n39)
      %b_col0_m4_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n39)
      %c_col0_m4_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3680000 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n39)
      aiex.dma_await_task(%c_col0_m4_n39)
      aiex.dma_free_task(%a_col0_m4_n39)
      aiex.dma_free_task(%b_col0_m4_n39)
      aiex.dma_free_task(%c_col0_m4_n39)
      // Col 0 M-tile 4 N-tile 40
      %a_col0_m4_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n40)
      %b_col0_m4_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n40)
      %c_col0_m4_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3680256 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n40)
      aiex.dma_await_task(%c_col0_m4_n40)
      aiex.dma_free_task(%a_col0_m4_n40)
      aiex.dma_free_task(%b_col0_m4_n40)
      aiex.dma_free_task(%c_col0_m4_n40)
      // Col 0 M-tile 4 N-tile 41
      %a_col0_m4_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n41)
      %b_col0_m4_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n41)
      %c_col0_m4_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3680512 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n41)
      aiex.dma_await_task(%c_col0_m4_n41)
      aiex.dma_free_task(%a_col0_m4_n41)
      aiex.dma_free_task(%b_col0_m4_n41)
      aiex.dma_free_task(%c_col0_m4_n41)
      // Col 0 M-tile 4 N-tile 42
      %a_col0_m4_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n42)
      %b_col0_m4_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n42)
      %c_col0_m4_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3680768 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n42)
      aiex.dma_await_task(%c_col0_m4_n42)
      aiex.dma_free_task(%a_col0_m4_n42)
      aiex.dma_free_task(%b_col0_m4_n42)
      aiex.dma_free_task(%c_col0_m4_n42)
      // Col 0 M-tile 4 N-tile 43
      %a_col0_m4_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n43)
      %b_col0_m4_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n43)
      %c_col0_m4_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3681024 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n43)
      aiex.dma_await_task(%c_col0_m4_n43)
      aiex.dma_free_task(%a_col0_m4_n43)
      aiex.dma_free_task(%b_col0_m4_n43)
      aiex.dma_free_task(%c_col0_m4_n43)
      // Col 0 M-tile 4 N-tile 44
      %a_col0_m4_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n44)
      %b_col0_m4_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n44)
      %c_col0_m4_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3681280 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n44)
      aiex.dma_await_task(%c_col0_m4_n44)
      aiex.dma_free_task(%a_col0_m4_n44)
      aiex.dma_free_task(%b_col0_m4_n44)
      aiex.dma_free_task(%c_col0_m4_n44)
      // Col 0 M-tile 4 N-tile 45
      %a_col0_m4_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n45)
      %b_col0_m4_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n45)
      %c_col0_m4_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3681536 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n45)
      aiex.dma_await_task(%c_col0_m4_n45)
      aiex.dma_free_task(%a_col0_m4_n45)
      aiex.dma_free_task(%b_col0_m4_n45)
      aiex.dma_free_task(%c_col0_m4_n45)
      // Col 0 M-tile 4 N-tile 46
      %a_col0_m4_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n46)
      %b_col0_m4_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n46)
      %c_col0_m4_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3681792 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n46)
      aiex.dma_await_task(%c_col0_m4_n46)
      aiex.dma_free_task(%a_col0_m4_n46)
      aiex.dma_free_task(%b_col0_m4_n46)
      aiex.dma_free_task(%c_col0_m4_n46)
      // Col 0 M-tile 4 N-tile 47
      %a_col0_m4_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n47)
      %b_col0_m4_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n47)
      %c_col0_m4_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3682048 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n47)
      aiex.dma_await_task(%c_col0_m4_n47)
      aiex.dma_free_task(%a_col0_m4_n47)
      aiex.dma_free_task(%b_col0_m4_n47)
      aiex.dma_free_task(%c_col0_m4_n47)
      // Col 0 M-tile 4 N-tile 48
      %a_col0_m4_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n48)
      %b_col0_m4_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n48)
      %c_col0_m4_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3682304 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n48)
      aiex.dma_await_task(%c_col0_m4_n48)
      aiex.dma_free_task(%a_col0_m4_n48)
      aiex.dma_free_task(%b_col0_m4_n48)
      aiex.dma_free_task(%c_col0_m4_n48)
      // Col 0 M-tile 4 N-tile 49
      %a_col0_m4_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n49)
      %b_col0_m4_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n49)
      %c_col0_m4_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3682560 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n49)
      aiex.dma_await_task(%c_col0_m4_n49)
      aiex.dma_free_task(%a_col0_m4_n49)
      aiex.dma_free_task(%b_col0_m4_n49)
      aiex.dma_free_task(%c_col0_m4_n49)
      // Col 0 M-tile 4 N-tile 50
      %a_col0_m4_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n50)
      %b_col0_m4_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n50)
      %c_col0_m4_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3682816 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n50)
      aiex.dma_await_task(%c_col0_m4_n50)
      aiex.dma_free_task(%a_col0_m4_n50)
      aiex.dma_free_task(%b_col0_m4_n50)
      aiex.dma_free_task(%c_col0_m4_n50)
      // Col 0 M-tile 4 N-tile 51
      %a_col0_m4_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n51)
      %b_col0_m4_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n51)
      %c_col0_m4_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3683072 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n51)
      aiex.dma_await_task(%c_col0_m4_n51)
      aiex.dma_free_task(%a_col0_m4_n51)
      aiex.dma_free_task(%b_col0_m4_n51)
      aiex.dma_free_task(%c_col0_m4_n51)
      // Col 0 M-tile 4 N-tile 52
      %a_col0_m4_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n52)
      %b_col0_m4_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n52)
      %c_col0_m4_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3683328 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n52)
      aiex.dma_await_task(%c_col0_m4_n52)
      aiex.dma_free_task(%a_col0_m4_n52)
      aiex.dma_free_task(%b_col0_m4_n52)
      aiex.dma_free_task(%c_col0_m4_n52)
      // Col 0 M-tile 4 N-tile 53
      %a_col0_m4_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n53)
      %b_col0_m4_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n53)
      %c_col0_m4_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3683584 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n53)
      aiex.dma_await_task(%c_col0_m4_n53)
      aiex.dma_free_task(%a_col0_m4_n53)
      aiex.dma_free_task(%b_col0_m4_n53)
      aiex.dma_free_task(%c_col0_m4_n53)
      // Col 0 M-tile 4 N-tile 54
      %a_col0_m4_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n54)
      %b_col0_m4_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n54)
      %c_col0_m4_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3683840 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n54)
      aiex.dma_await_task(%c_col0_m4_n54)
      aiex.dma_free_task(%a_col0_m4_n54)
      aiex.dma_free_task(%b_col0_m4_n54)
      aiex.dma_free_task(%c_col0_m4_n54)
      // Col 0 M-tile 4 N-tile 55
      %a_col0_m4_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n55)
      %b_col0_m4_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n55)
      %c_col0_m4_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 3684096 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n55)
      aiex.dma_await_task(%c_col0_m4_n55)
      aiex.dma_free_task(%a_col0_m4_n55)
      aiex.dma_free_task(%b_col0_m4_n55)
      aiex.dma_free_task(%c_col0_m4_n55)
      // Col 0 M-tile 5 N-tile 0
      %a_col0_m5_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n0)
      %b_col0_m5_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n0)
      %c_col0_m5_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4587520 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n0)
      aiex.dma_await_task(%c_col0_m5_n0)
      aiex.dma_free_task(%a_col0_m5_n0)
      aiex.dma_free_task(%b_col0_m5_n0)
      aiex.dma_free_task(%c_col0_m5_n0)
      // Col 0 M-tile 5 N-tile 1
      %a_col0_m5_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n1)
      %b_col0_m5_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n1)
      %c_col0_m5_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4587776 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n1)
      aiex.dma_await_task(%c_col0_m5_n1)
      aiex.dma_free_task(%a_col0_m5_n1)
      aiex.dma_free_task(%b_col0_m5_n1)
      aiex.dma_free_task(%c_col0_m5_n1)
      // Col 0 M-tile 5 N-tile 2
      %a_col0_m5_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n2)
      %b_col0_m5_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n2)
      %c_col0_m5_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4588032 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n2)
      aiex.dma_await_task(%c_col0_m5_n2)
      aiex.dma_free_task(%a_col0_m5_n2)
      aiex.dma_free_task(%b_col0_m5_n2)
      aiex.dma_free_task(%c_col0_m5_n2)
      // Col 0 M-tile 5 N-tile 3
      %a_col0_m5_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n3)
      %b_col0_m5_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n3)
      %c_col0_m5_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4588288 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n3)
      aiex.dma_await_task(%c_col0_m5_n3)
      aiex.dma_free_task(%a_col0_m5_n3)
      aiex.dma_free_task(%b_col0_m5_n3)
      aiex.dma_free_task(%c_col0_m5_n3)
      // Col 0 M-tile 5 N-tile 4
      %a_col0_m5_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n4)
      %b_col0_m5_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n4)
      %c_col0_m5_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4588544 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n4)
      aiex.dma_await_task(%c_col0_m5_n4)
      aiex.dma_free_task(%a_col0_m5_n4)
      aiex.dma_free_task(%b_col0_m5_n4)
      aiex.dma_free_task(%c_col0_m5_n4)
      // Col 0 M-tile 5 N-tile 5
      %a_col0_m5_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n5)
      %b_col0_m5_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n5)
      %c_col0_m5_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4588800 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n5)
      aiex.dma_await_task(%c_col0_m5_n5)
      aiex.dma_free_task(%a_col0_m5_n5)
      aiex.dma_free_task(%b_col0_m5_n5)
      aiex.dma_free_task(%c_col0_m5_n5)
      // Col 0 M-tile 5 N-tile 6
      %a_col0_m5_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n6)
      %b_col0_m5_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n6)
      %c_col0_m5_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4589056 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n6)
      aiex.dma_await_task(%c_col0_m5_n6)
      aiex.dma_free_task(%a_col0_m5_n6)
      aiex.dma_free_task(%b_col0_m5_n6)
      aiex.dma_free_task(%c_col0_m5_n6)
      // Col 0 M-tile 5 N-tile 7
      %a_col0_m5_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n7)
      %b_col0_m5_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n7)
      %c_col0_m5_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4589312 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n7)
      aiex.dma_await_task(%c_col0_m5_n7)
      aiex.dma_free_task(%a_col0_m5_n7)
      aiex.dma_free_task(%b_col0_m5_n7)
      aiex.dma_free_task(%c_col0_m5_n7)
      // Col 0 M-tile 5 N-tile 8
      %a_col0_m5_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n8)
      %b_col0_m5_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n8)
      %c_col0_m5_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4589568 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n8)
      aiex.dma_await_task(%c_col0_m5_n8)
      aiex.dma_free_task(%a_col0_m5_n8)
      aiex.dma_free_task(%b_col0_m5_n8)
      aiex.dma_free_task(%c_col0_m5_n8)
      // Col 0 M-tile 5 N-tile 9
      %a_col0_m5_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n9)
      %b_col0_m5_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n9)
      %c_col0_m5_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4589824 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n9)
      aiex.dma_await_task(%c_col0_m5_n9)
      aiex.dma_free_task(%a_col0_m5_n9)
      aiex.dma_free_task(%b_col0_m5_n9)
      aiex.dma_free_task(%c_col0_m5_n9)
      // Col 0 M-tile 5 N-tile 10
      %a_col0_m5_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n10)
      %b_col0_m5_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n10)
      %c_col0_m5_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4590080 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n10)
      aiex.dma_await_task(%c_col0_m5_n10)
      aiex.dma_free_task(%a_col0_m5_n10)
      aiex.dma_free_task(%b_col0_m5_n10)
      aiex.dma_free_task(%c_col0_m5_n10)
      // Col 0 M-tile 5 N-tile 11
      %a_col0_m5_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n11)
      %b_col0_m5_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n11)
      %c_col0_m5_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4590336 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n11)
      aiex.dma_await_task(%c_col0_m5_n11)
      aiex.dma_free_task(%a_col0_m5_n11)
      aiex.dma_free_task(%b_col0_m5_n11)
      aiex.dma_free_task(%c_col0_m5_n11)
      // Col 0 M-tile 5 N-tile 12
      %a_col0_m5_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n12)
      %b_col0_m5_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n12)
      %c_col0_m5_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4590592 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n12)
      aiex.dma_await_task(%c_col0_m5_n12)
      aiex.dma_free_task(%a_col0_m5_n12)
      aiex.dma_free_task(%b_col0_m5_n12)
      aiex.dma_free_task(%c_col0_m5_n12)
      // Col 0 M-tile 5 N-tile 13
      %a_col0_m5_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n13)
      %b_col0_m5_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n13)
      %c_col0_m5_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4590848 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n13)
      aiex.dma_await_task(%c_col0_m5_n13)
      aiex.dma_free_task(%a_col0_m5_n13)
      aiex.dma_free_task(%b_col0_m5_n13)
      aiex.dma_free_task(%c_col0_m5_n13)
      // Col 0 M-tile 5 N-tile 14
      %a_col0_m5_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n14)
      %b_col0_m5_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n14)
      %c_col0_m5_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4591104 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n14)
      aiex.dma_await_task(%c_col0_m5_n14)
      aiex.dma_free_task(%a_col0_m5_n14)
      aiex.dma_free_task(%b_col0_m5_n14)
      aiex.dma_free_task(%c_col0_m5_n14)
      // Col 0 M-tile 5 N-tile 15
      %a_col0_m5_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n15)
      %b_col0_m5_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n15)
      %c_col0_m5_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4591360 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n15)
      aiex.dma_await_task(%c_col0_m5_n15)
      aiex.dma_free_task(%a_col0_m5_n15)
      aiex.dma_free_task(%b_col0_m5_n15)
      aiex.dma_free_task(%c_col0_m5_n15)
      // Col 0 M-tile 5 N-tile 16
      %a_col0_m5_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n16)
      %b_col0_m5_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n16)
      %c_col0_m5_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4591616 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n16)
      aiex.dma_await_task(%c_col0_m5_n16)
      aiex.dma_free_task(%a_col0_m5_n16)
      aiex.dma_free_task(%b_col0_m5_n16)
      aiex.dma_free_task(%c_col0_m5_n16)
      // Col 0 M-tile 5 N-tile 17
      %a_col0_m5_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n17)
      %b_col0_m5_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n17)
      %c_col0_m5_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4591872 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n17)
      aiex.dma_await_task(%c_col0_m5_n17)
      aiex.dma_free_task(%a_col0_m5_n17)
      aiex.dma_free_task(%b_col0_m5_n17)
      aiex.dma_free_task(%c_col0_m5_n17)
      // Col 0 M-tile 5 N-tile 18
      %a_col0_m5_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n18)
      %b_col0_m5_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n18)
      %c_col0_m5_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4592128 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n18)
      aiex.dma_await_task(%c_col0_m5_n18)
      aiex.dma_free_task(%a_col0_m5_n18)
      aiex.dma_free_task(%b_col0_m5_n18)
      aiex.dma_free_task(%c_col0_m5_n18)
      // Col 0 M-tile 5 N-tile 19
      %a_col0_m5_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n19)
      %b_col0_m5_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n19)
      %c_col0_m5_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4592384 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n19)
      aiex.dma_await_task(%c_col0_m5_n19)
      aiex.dma_free_task(%a_col0_m5_n19)
      aiex.dma_free_task(%b_col0_m5_n19)
      aiex.dma_free_task(%c_col0_m5_n19)
      // Col 0 M-tile 5 N-tile 20
      %a_col0_m5_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n20)
      %b_col0_m5_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n20)
      %c_col0_m5_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4592640 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n20)
      aiex.dma_await_task(%c_col0_m5_n20)
      aiex.dma_free_task(%a_col0_m5_n20)
      aiex.dma_free_task(%b_col0_m5_n20)
      aiex.dma_free_task(%c_col0_m5_n20)
      // Col 0 M-tile 5 N-tile 21
      %a_col0_m5_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n21)
      %b_col0_m5_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n21)
      %c_col0_m5_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4592896 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n21)
      aiex.dma_await_task(%c_col0_m5_n21)
      aiex.dma_free_task(%a_col0_m5_n21)
      aiex.dma_free_task(%b_col0_m5_n21)
      aiex.dma_free_task(%c_col0_m5_n21)
      // Col 0 M-tile 5 N-tile 22
      %a_col0_m5_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n22)
      %b_col0_m5_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n22)
      %c_col0_m5_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4593152 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n22)
      aiex.dma_await_task(%c_col0_m5_n22)
      aiex.dma_free_task(%a_col0_m5_n22)
      aiex.dma_free_task(%b_col0_m5_n22)
      aiex.dma_free_task(%c_col0_m5_n22)
      // Col 0 M-tile 5 N-tile 23
      %a_col0_m5_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n23)
      %b_col0_m5_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n23)
      %c_col0_m5_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4593408 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n23)
      aiex.dma_await_task(%c_col0_m5_n23)
      aiex.dma_free_task(%a_col0_m5_n23)
      aiex.dma_free_task(%b_col0_m5_n23)
      aiex.dma_free_task(%c_col0_m5_n23)
      // Col 0 M-tile 5 N-tile 24
      %a_col0_m5_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n24)
      %b_col0_m5_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n24)
      %c_col0_m5_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4593664 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n24)
      aiex.dma_await_task(%c_col0_m5_n24)
      aiex.dma_free_task(%a_col0_m5_n24)
      aiex.dma_free_task(%b_col0_m5_n24)
      aiex.dma_free_task(%c_col0_m5_n24)
      // Col 0 M-tile 5 N-tile 25
      %a_col0_m5_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n25)
      %b_col0_m5_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n25)
      %c_col0_m5_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4593920 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n25)
      aiex.dma_await_task(%c_col0_m5_n25)
      aiex.dma_free_task(%a_col0_m5_n25)
      aiex.dma_free_task(%b_col0_m5_n25)
      aiex.dma_free_task(%c_col0_m5_n25)
      // Col 0 M-tile 5 N-tile 26
      %a_col0_m5_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n26)
      %b_col0_m5_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n26)
      %c_col0_m5_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4594176 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n26)
      aiex.dma_await_task(%c_col0_m5_n26)
      aiex.dma_free_task(%a_col0_m5_n26)
      aiex.dma_free_task(%b_col0_m5_n26)
      aiex.dma_free_task(%c_col0_m5_n26)
      // Col 0 M-tile 5 N-tile 27
      %a_col0_m5_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n27)
      %b_col0_m5_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n27)
      %c_col0_m5_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4594432 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n27)
      aiex.dma_await_task(%c_col0_m5_n27)
      aiex.dma_free_task(%a_col0_m5_n27)
      aiex.dma_free_task(%b_col0_m5_n27)
      aiex.dma_free_task(%c_col0_m5_n27)
      // Col 0 M-tile 5 N-tile 28
      %a_col0_m5_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n28)
      %b_col0_m5_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n28)
      %c_col0_m5_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4594688 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n28)
      aiex.dma_await_task(%c_col0_m5_n28)
      aiex.dma_free_task(%a_col0_m5_n28)
      aiex.dma_free_task(%b_col0_m5_n28)
      aiex.dma_free_task(%c_col0_m5_n28)
      // Col 0 M-tile 5 N-tile 29
      %a_col0_m5_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n29)
      %b_col0_m5_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n29)
      %c_col0_m5_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4594944 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n29)
      aiex.dma_await_task(%c_col0_m5_n29)
      aiex.dma_free_task(%a_col0_m5_n29)
      aiex.dma_free_task(%b_col0_m5_n29)
      aiex.dma_free_task(%c_col0_m5_n29)
      // Col 0 M-tile 5 N-tile 30
      %a_col0_m5_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n30)
      %b_col0_m5_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n30)
      %c_col0_m5_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4595200 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n30)
      aiex.dma_await_task(%c_col0_m5_n30)
      aiex.dma_free_task(%a_col0_m5_n30)
      aiex.dma_free_task(%b_col0_m5_n30)
      aiex.dma_free_task(%c_col0_m5_n30)
      // Col 0 M-tile 5 N-tile 31
      %a_col0_m5_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n31)
      %b_col0_m5_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n31)
      %c_col0_m5_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4595456 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n31)
      aiex.dma_await_task(%c_col0_m5_n31)
      aiex.dma_free_task(%a_col0_m5_n31)
      aiex.dma_free_task(%b_col0_m5_n31)
      aiex.dma_free_task(%c_col0_m5_n31)
      // Col 0 M-tile 5 N-tile 32
      %a_col0_m5_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n32)
      %b_col0_m5_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n32)
      %c_col0_m5_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4595712 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n32)
      aiex.dma_await_task(%c_col0_m5_n32)
      aiex.dma_free_task(%a_col0_m5_n32)
      aiex.dma_free_task(%b_col0_m5_n32)
      aiex.dma_free_task(%c_col0_m5_n32)
      // Col 0 M-tile 5 N-tile 33
      %a_col0_m5_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n33)
      %b_col0_m5_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n33)
      %c_col0_m5_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4595968 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n33)
      aiex.dma_await_task(%c_col0_m5_n33)
      aiex.dma_free_task(%a_col0_m5_n33)
      aiex.dma_free_task(%b_col0_m5_n33)
      aiex.dma_free_task(%c_col0_m5_n33)
      // Col 0 M-tile 5 N-tile 34
      %a_col0_m5_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n34)
      %b_col0_m5_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n34)
      %c_col0_m5_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4596224 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n34)
      aiex.dma_await_task(%c_col0_m5_n34)
      aiex.dma_free_task(%a_col0_m5_n34)
      aiex.dma_free_task(%b_col0_m5_n34)
      aiex.dma_free_task(%c_col0_m5_n34)
      // Col 0 M-tile 5 N-tile 35
      %a_col0_m5_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n35)
      %b_col0_m5_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n35)
      %c_col0_m5_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4596480 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n35)
      aiex.dma_await_task(%c_col0_m5_n35)
      aiex.dma_free_task(%a_col0_m5_n35)
      aiex.dma_free_task(%b_col0_m5_n35)
      aiex.dma_free_task(%c_col0_m5_n35)
      // Col 0 M-tile 5 N-tile 36
      %a_col0_m5_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n36)
      %b_col0_m5_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n36)
      %c_col0_m5_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4596736 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n36)
      aiex.dma_await_task(%c_col0_m5_n36)
      aiex.dma_free_task(%a_col0_m5_n36)
      aiex.dma_free_task(%b_col0_m5_n36)
      aiex.dma_free_task(%c_col0_m5_n36)
      // Col 0 M-tile 5 N-tile 37
      %a_col0_m5_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n37)
      %b_col0_m5_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n37)
      %c_col0_m5_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4596992 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n37)
      aiex.dma_await_task(%c_col0_m5_n37)
      aiex.dma_free_task(%a_col0_m5_n37)
      aiex.dma_free_task(%b_col0_m5_n37)
      aiex.dma_free_task(%c_col0_m5_n37)
      // Col 0 M-tile 5 N-tile 38
      %a_col0_m5_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n38)
      %b_col0_m5_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n38)
      %c_col0_m5_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4597248 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n38)
      aiex.dma_await_task(%c_col0_m5_n38)
      aiex.dma_free_task(%a_col0_m5_n38)
      aiex.dma_free_task(%b_col0_m5_n38)
      aiex.dma_free_task(%c_col0_m5_n38)
      // Col 0 M-tile 5 N-tile 39
      %a_col0_m5_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n39)
      %b_col0_m5_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n39)
      %c_col0_m5_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4597504 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n39)
      aiex.dma_await_task(%c_col0_m5_n39)
      aiex.dma_free_task(%a_col0_m5_n39)
      aiex.dma_free_task(%b_col0_m5_n39)
      aiex.dma_free_task(%c_col0_m5_n39)
      // Col 0 M-tile 5 N-tile 40
      %a_col0_m5_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n40)
      %b_col0_m5_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n40)
      %c_col0_m5_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4597760 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n40)
      aiex.dma_await_task(%c_col0_m5_n40)
      aiex.dma_free_task(%a_col0_m5_n40)
      aiex.dma_free_task(%b_col0_m5_n40)
      aiex.dma_free_task(%c_col0_m5_n40)
      // Col 0 M-tile 5 N-tile 41
      %a_col0_m5_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n41)
      %b_col0_m5_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n41)
      %c_col0_m5_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4598016 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n41)
      aiex.dma_await_task(%c_col0_m5_n41)
      aiex.dma_free_task(%a_col0_m5_n41)
      aiex.dma_free_task(%b_col0_m5_n41)
      aiex.dma_free_task(%c_col0_m5_n41)
      // Col 0 M-tile 5 N-tile 42
      %a_col0_m5_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n42)
      %b_col0_m5_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n42)
      %c_col0_m5_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4598272 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n42)
      aiex.dma_await_task(%c_col0_m5_n42)
      aiex.dma_free_task(%a_col0_m5_n42)
      aiex.dma_free_task(%b_col0_m5_n42)
      aiex.dma_free_task(%c_col0_m5_n42)
      // Col 0 M-tile 5 N-tile 43
      %a_col0_m5_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n43)
      %b_col0_m5_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n43)
      %c_col0_m5_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4598528 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n43)
      aiex.dma_await_task(%c_col0_m5_n43)
      aiex.dma_free_task(%a_col0_m5_n43)
      aiex.dma_free_task(%b_col0_m5_n43)
      aiex.dma_free_task(%c_col0_m5_n43)
      // Col 0 M-tile 5 N-tile 44
      %a_col0_m5_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n44)
      %b_col0_m5_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n44)
      %c_col0_m5_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4598784 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n44)
      aiex.dma_await_task(%c_col0_m5_n44)
      aiex.dma_free_task(%a_col0_m5_n44)
      aiex.dma_free_task(%b_col0_m5_n44)
      aiex.dma_free_task(%c_col0_m5_n44)
      // Col 0 M-tile 5 N-tile 45
      %a_col0_m5_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n45)
      %b_col0_m5_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n45)
      %c_col0_m5_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4599040 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n45)
      aiex.dma_await_task(%c_col0_m5_n45)
      aiex.dma_free_task(%a_col0_m5_n45)
      aiex.dma_free_task(%b_col0_m5_n45)
      aiex.dma_free_task(%c_col0_m5_n45)
      // Col 0 M-tile 5 N-tile 46
      %a_col0_m5_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n46)
      %b_col0_m5_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n46)
      %c_col0_m5_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4599296 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n46)
      aiex.dma_await_task(%c_col0_m5_n46)
      aiex.dma_free_task(%a_col0_m5_n46)
      aiex.dma_free_task(%b_col0_m5_n46)
      aiex.dma_free_task(%c_col0_m5_n46)
      // Col 0 M-tile 5 N-tile 47
      %a_col0_m5_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n47)
      %b_col0_m5_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n47)
      %c_col0_m5_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4599552 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n47)
      aiex.dma_await_task(%c_col0_m5_n47)
      aiex.dma_free_task(%a_col0_m5_n47)
      aiex.dma_free_task(%b_col0_m5_n47)
      aiex.dma_free_task(%c_col0_m5_n47)
      // Col 0 M-tile 5 N-tile 48
      %a_col0_m5_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n48)
      %b_col0_m5_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n48)
      %c_col0_m5_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4599808 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n48)
      aiex.dma_await_task(%c_col0_m5_n48)
      aiex.dma_free_task(%a_col0_m5_n48)
      aiex.dma_free_task(%b_col0_m5_n48)
      aiex.dma_free_task(%c_col0_m5_n48)
      // Col 0 M-tile 5 N-tile 49
      %a_col0_m5_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n49)
      %b_col0_m5_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n49)
      %c_col0_m5_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4600064 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n49)
      aiex.dma_await_task(%c_col0_m5_n49)
      aiex.dma_free_task(%a_col0_m5_n49)
      aiex.dma_free_task(%b_col0_m5_n49)
      aiex.dma_free_task(%c_col0_m5_n49)
      // Col 0 M-tile 5 N-tile 50
      %a_col0_m5_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n50)
      %b_col0_m5_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n50)
      %c_col0_m5_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4600320 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n50)
      aiex.dma_await_task(%c_col0_m5_n50)
      aiex.dma_free_task(%a_col0_m5_n50)
      aiex.dma_free_task(%b_col0_m5_n50)
      aiex.dma_free_task(%c_col0_m5_n50)
      // Col 0 M-tile 5 N-tile 51
      %a_col0_m5_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n51)
      %b_col0_m5_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n51)
      %c_col0_m5_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4600576 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n51)
      aiex.dma_await_task(%c_col0_m5_n51)
      aiex.dma_free_task(%a_col0_m5_n51)
      aiex.dma_free_task(%b_col0_m5_n51)
      aiex.dma_free_task(%c_col0_m5_n51)
      // Col 0 M-tile 5 N-tile 52
      %a_col0_m5_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n52)
      %b_col0_m5_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n52)
      %c_col0_m5_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4600832 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n52)
      aiex.dma_await_task(%c_col0_m5_n52)
      aiex.dma_free_task(%a_col0_m5_n52)
      aiex.dma_free_task(%b_col0_m5_n52)
      aiex.dma_free_task(%c_col0_m5_n52)
      // Col 0 M-tile 5 N-tile 53
      %a_col0_m5_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n53)
      %b_col0_m5_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n53)
      %c_col0_m5_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4601088 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n53)
      aiex.dma_await_task(%c_col0_m5_n53)
      aiex.dma_free_task(%a_col0_m5_n53)
      aiex.dma_free_task(%b_col0_m5_n53)
      aiex.dma_free_task(%c_col0_m5_n53)
      // Col 0 M-tile 5 N-tile 54
      %a_col0_m5_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n54)
      %b_col0_m5_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n54)
      %c_col0_m5_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4601344 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n54)
      aiex.dma_await_task(%c_col0_m5_n54)
      aiex.dma_free_task(%a_col0_m5_n54)
      aiex.dma_free_task(%b_col0_m5_n54)
      aiex.dma_free_task(%c_col0_m5_n54)
      // Col 0 M-tile 5 N-tile 55
      %a_col0_m5_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n55)
      %b_col0_m5_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n55)
      %c_col0_m5_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 4601600 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n55)
      aiex.dma_await_task(%c_col0_m5_n55)
      aiex.dma_free_task(%a_col0_m5_n55)
      aiex.dma_free_task(%b_col0_m5_n55)
      aiex.dma_free_task(%c_col0_m5_n55)
      // Col 0 M-tile 6 N-tile 0
      %a_col0_m6_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n0)
      %b_col0_m6_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n0)
      %c_col0_m6_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5505024 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n0)
      aiex.dma_await_task(%c_col0_m6_n0)
      aiex.dma_free_task(%a_col0_m6_n0)
      aiex.dma_free_task(%b_col0_m6_n0)
      aiex.dma_free_task(%c_col0_m6_n0)
      // Col 0 M-tile 6 N-tile 1
      %a_col0_m6_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n1)
      %b_col0_m6_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n1)
      %c_col0_m6_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5505280 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n1)
      aiex.dma_await_task(%c_col0_m6_n1)
      aiex.dma_free_task(%a_col0_m6_n1)
      aiex.dma_free_task(%b_col0_m6_n1)
      aiex.dma_free_task(%c_col0_m6_n1)
      // Col 0 M-tile 6 N-tile 2
      %a_col0_m6_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n2)
      %b_col0_m6_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n2)
      %c_col0_m6_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5505536 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n2)
      aiex.dma_await_task(%c_col0_m6_n2)
      aiex.dma_free_task(%a_col0_m6_n2)
      aiex.dma_free_task(%b_col0_m6_n2)
      aiex.dma_free_task(%c_col0_m6_n2)
      // Col 0 M-tile 6 N-tile 3
      %a_col0_m6_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n3)
      %b_col0_m6_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n3)
      %c_col0_m6_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5505792 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n3)
      aiex.dma_await_task(%c_col0_m6_n3)
      aiex.dma_free_task(%a_col0_m6_n3)
      aiex.dma_free_task(%b_col0_m6_n3)
      aiex.dma_free_task(%c_col0_m6_n3)
      // Col 0 M-tile 6 N-tile 4
      %a_col0_m6_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n4)
      %b_col0_m6_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n4)
      %c_col0_m6_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5506048 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n4)
      aiex.dma_await_task(%c_col0_m6_n4)
      aiex.dma_free_task(%a_col0_m6_n4)
      aiex.dma_free_task(%b_col0_m6_n4)
      aiex.dma_free_task(%c_col0_m6_n4)
      // Col 0 M-tile 6 N-tile 5
      %a_col0_m6_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n5)
      %b_col0_m6_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n5)
      %c_col0_m6_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5506304 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n5)
      aiex.dma_await_task(%c_col0_m6_n5)
      aiex.dma_free_task(%a_col0_m6_n5)
      aiex.dma_free_task(%b_col0_m6_n5)
      aiex.dma_free_task(%c_col0_m6_n5)
      // Col 0 M-tile 6 N-tile 6
      %a_col0_m6_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n6)
      %b_col0_m6_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n6)
      %c_col0_m6_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5506560 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n6)
      aiex.dma_await_task(%c_col0_m6_n6)
      aiex.dma_free_task(%a_col0_m6_n6)
      aiex.dma_free_task(%b_col0_m6_n6)
      aiex.dma_free_task(%c_col0_m6_n6)
      // Col 0 M-tile 6 N-tile 7
      %a_col0_m6_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n7)
      %b_col0_m6_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n7)
      %c_col0_m6_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5506816 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n7)
      aiex.dma_await_task(%c_col0_m6_n7)
      aiex.dma_free_task(%a_col0_m6_n7)
      aiex.dma_free_task(%b_col0_m6_n7)
      aiex.dma_free_task(%c_col0_m6_n7)
      // Col 0 M-tile 6 N-tile 8
      %a_col0_m6_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n8)
      %b_col0_m6_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n8)
      %c_col0_m6_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5507072 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n8)
      aiex.dma_await_task(%c_col0_m6_n8)
      aiex.dma_free_task(%a_col0_m6_n8)
      aiex.dma_free_task(%b_col0_m6_n8)
      aiex.dma_free_task(%c_col0_m6_n8)
      // Col 0 M-tile 6 N-tile 9
      %a_col0_m6_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n9)
      %b_col0_m6_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n9)
      %c_col0_m6_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5507328 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n9)
      aiex.dma_await_task(%c_col0_m6_n9)
      aiex.dma_free_task(%a_col0_m6_n9)
      aiex.dma_free_task(%b_col0_m6_n9)
      aiex.dma_free_task(%c_col0_m6_n9)
      // Col 0 M-tile 6 N-tile 10
      %a_col0_m6_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n10)
      %b_col0_m6_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n10)
      %c_col0_m6_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5507584 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n10)
      aiex.dma_await_task(%c_col0_m6_n10)
      aiex.dma_free_task(%a_col0_m6_n10)
      aiex.dma_free_task(%b_col0_m6_n10)
      aiex.dma_free_task(%c_col0_m6_n10)
      // Col 0 M-tile 6 N-tile 11
      %a_col0_m6_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n11)
      %b_col0_m6_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n11)
      %c_col0_m6_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5507840 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n11)
      aiex.dma_await_task(%c_col0_m6_n11)
      aiex.dma_free_task(%a_col0_m6_n11)
      aiex.dma_free_task(%b_col0_m6_n11)
      aiex.dma_free_task(%c_col0_m6_n11)
      // Col 0 M-tile 6 N-tile 12
      %a_col0_m6_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n12)
      %b_col0_m6_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n12)
      %c_col0_m6_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5508096 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n12)
      aiex.dma_await_task(%c_col0_m6_n12)
      aiex.dma_free_task(%a_col0_m6_n12)
      aiex.dma_free_task(%b_col0_m6_n12)
      aiex.dma_free_task(%c_col0_m6_n12)
      // Col 0 M-tile 6 N-tile 13
      %a_col0_m6_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n13)
      %b_col0_m6_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n13)
      %c_col0_m6_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5508352 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n13)
      aiex.dma_await_task(%c_col0_m6_n13)
      aiex.dma_free_task(%a_col0_m6_n13)
      aiex.dma_free_task(%b_col0_m6_n13)
      aiex.dma_free_task(%c_col0_m6_n13)
      // Col 0 M-tile 6 N-tile 14
      %a_col0_m6_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n14)
      %b_col0_m6_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n14)
      %c_col0_m6_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5508608 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n14)
      aiex.dma_await_task(%c_col0_m6_n14)
      aiex.dma_free_task(%a_col0_m6_n14)
      aiex.dma_free_task(%b_col0_m6_n14)
      aiex.dma_free_task(%c_col0_m6_n14)
      // Col 0 M-tile 6 N-tile 15
      %a_col0_m6_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n15)
      %b_col0_m6_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n15)
      %c_col0_m6_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5508864 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n15)
      aiex.dma_await_task(%c_col0_m6_n15)
      aiex.dma_free_task(%a_col0_m6_n15)
      aiex.dma_free_task(%b_col0_m6_n15)
      aiex.dma_free_task(%c_col0_m6_n15)
      // Col 0 M-tile 6 N-tile 16
      %a_col0_m6_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n16)
      %b_col0_m6_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n16)
      %c_col0_m6_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5509120 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n16)
      aiex.dma_await_task(%c_col0_m6_n16)
      aiex.dma_free_task(%a_col0_m6_n16)
      aiex.dma_free_task(%b_col0_m6_n16)
      aiex.dma_free_task(%c_col0_m6_n16)
      // Col 0 M-tile 6 N-tile 17
      %a_col0_m6_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n17)
      %b_col0_m6_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n17)
      %c_col0_m6_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5509376 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n17)
      aiex.dma_await_task(%c_col0_m6_n17)
      aiex.dma_free_task(%a_col0_m6_n17)
      aiex.dma_free_task(%b_col0_m6_n17)
      aiex.dma_free_task(%c_col0_m6_n17)
      // Col 0 M-tile 6 N-tile 18
      %a_col0_m6_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n18)
      %b_col0_m6_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n18)
      %c_col0_m6_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5509632 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n18)
      aiex.dma_await_task(%c_col0_m6_n18)
      aiex.dma_free_task(%a_col0_m6_n18)
      aiex.dma_free_task(%b_col0_m6_n18)
      aiex.dma_free_task(%c_col0_m6_n18)
      // Col 0 M-tile 6 N-tile 19
      %a_col0_m6_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n19)
      %b_col0_m6_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n19)
      %c_col0_m6_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5509888 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n19)
      aiex.dma_await_task(%c_col0_m6_n19)
      aiex.dma_free_task(%a_col0_m6_n19)
      aiex.dma_free_task(%b_col0_m6_n19)
      aiex.dma_free_task(%c_col0_m6_n19)
      // Col 0 M-tile 6 N-tile 20
      %a_col0_m6_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n20)
      %b_col0_m6_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n20)
      %c_col0_m6_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5510144 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n20)
      aiex.dma_await_task(%c_col0_m6_n20)
      aiex.dma_free_task(%a_col0_m6_n20)
      aiex.dma_free_task(%b_col0_m6_n20)
      aiex.dma_free_task(%c_col0_m6_n20)
      // Col 0 M-tile 6 N-tile 21
      %a_col0_m6_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n21)
      %b_col0_m6_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n21)
      %c_col0_m6_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5510400 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n21)
      aiex.dma_await_task(%c_col0_m6_n21)
      aiex.dma_free_task(%a_col0_m6_n21)
      aiex.dma_free_task(%b_col0_m6_n21)
      aiex.dma_free_task(%c_col0_m6_n21)
      // Col 0 M-tile 6 N-tile 22
      %a_col0_m6_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n22)
      %b_col0_m6_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n22)
      %c_col0_m6_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5510656 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n22)
      aiex.dma_await_task(%c_col0_m6_n22)
      aiex.dma_free_task(%a_col0_m6_n22)
      aiex.dma_free_task(%b_col0_m6_n22)
      aiex.dma_free_task(%c_col0_m6_n22)
      // Col 0 M-tile 6 N-tile 23
      %a_col0_m6_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n23)
      %b_col0_m6_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n23)
      %c_col0_m6_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5510912 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n23)
      aiex.dma_await_task(%c_col0_m6_n23)
      aiex.dma_free_task(%a_col0_m6_n23)
      aiex.dma_free_task(%b_col0_m6_n23)
      aiex.dma_free_task(%c_col0_m6_n23)
      // Col 0 M-tile 6 N-tile 24
      %a_col0_m6_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n24)
      %b_col0_m6_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n24)
      %c_col0_m6_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5511168 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n24)
      aiex.dma_await_task(%c_col0_m6_n24)
      aiex.dma_free_task(%a_col0_m6_n24)
      aiex.dma_free_task(%b_col0_m6_n24)
      aiex.dma_free_task(%c_col0_m6_n24)
      // Col 0 M-tile 6 N-tile 25
      %a_col0_m6_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n25)
      %b_col0_m6_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n25)
      %c_col0_m6_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5511424 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n25)
      aiex.dma_await_task(%c_col0_m6_n25)
      aiex.dma_free_task(%a_col0_m6_n25)
      aiex.dma_free_task(%b_col0_m6_n25)
      aiex.dma_free_task(%c_col0_m6_n25)
      // Col 0 M-tile 6 N-tile 26
      %a_col0_m6_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n26)
      %b_col0_m6_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n26)
      %c_col0_m6_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5511680 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n26)
      aiex.dma_await_task(%c_col0_m6_n26)
      aiex.dma_free_task(%a_col0_m6_n26)
      aiex.dma_free_task(%b_col0_m6_n26)
      aiex.dma_free_task(%c_col0_m6_n26)
      // Col 0 M-tile 6 N-tile 27
      %a_col0_m6_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n27)
      %b_col0_m6_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n27)
      %c_col0_m6_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5511936 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n27)
      aiex.dma_await_task(%c_col0_m6_n27)
      aiex.dma_free_task(%a_col0_m6_n27)
      aiex.dma_free_task(%b_col0_m6_n27)
      aiex.dma_free_task(%c_col0_m6_n27)
      // Col 0 M-tile 6 N-tile 28
      %a_col0_m6_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n28)
      %b_col0_m6_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n28)
      %c_col0_m6_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5512192 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n28)
      aiex.dma_await_task(%c_col0_m6_n28)
      aiex.dma_free_task(%a_col0_m6_n28)
      aiex.dma_free_task(%b_col0_m6_n28)
      aiex.dma_free_task(%c_col0_m6_n28)
      // Col 0 M-tile 6 N-tile 29
      %a_col0_m6_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n29)
      %b_col0_m6_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n29)
      %c_col0_m6_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5512448 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n29)
      aiex.dma_await_task(%c_col0_m6_n29)
      aiex.dma_free_task(%a_col0_m6_n29)
      aiex.dma_free_task(%b_col0_m6_n29)
      aiex.dma_free_task(%c_col0_m6_n29)
      // Col 0 M-tile 6 N-tile 30
      %a_col0_m6_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n30)
      %b_col0_m6_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n30)
      %c_col0_m6_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5512704 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n30)
      aiex.dma_await_task(%c_col0_m6_n30)
      aiex.dma_free_task(%a_col0_m6_n30)
      aiex.dma_free_task(%b_col0_m6_n30)
      aiex.dma_free_task(%c_col0_m6_n30)
      // Col 0 M-tile 6 N-tile 31
      %a_col0_m6_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n31)
      %b_col0_m6_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n31)
      %c_col0_m6_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5512960 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n31)
      aiex.dma_await_task(%c_col0_m6_n31)
      aiex.dma_free_task(%a_col0_m6_n31)
      aiex.dma_free_task(%b_col0_m6_n31)
      aiex.dma_free_task(%c_col0_m6_n31)
      // Col 0 M-tile 6 N-tile 32
      %a_col0_m6_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n32)
      %b_col0_m6_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n32)
      %c_col0_m6_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5513216 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n32)
      aiex.dma_await_task(%c_col0_m6_n32)
      aiex.dma_free_task(%a_col0_m6_n32)
      aiex.dma_free_task(%b_col0_m6_n32)
      aiex.dma_free_task(%c_col0_m6_n32)
      // Col 0 M-tile 6 N-tile 33
      %a_col0_m6_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n33)
      %b_col0_m6_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n33)
      %c_col0_m6_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5513472 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n33)
      aiex.dma_await_task(%c_col0_m6_n33)
      aiex.dma_free_task(%a_col0_m6_n33)
      aiex.dma_free_task(%b_col0_m6_n33)
      aiex.dma_free_task(%c_col0_m6_n33)
      // Col 0 M-tile 6 N-tile 34
      %a_col0_m6_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n34)
      %b_col0_m6_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n34)
      %c_col0_m6_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5513728 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n34)
      aiex.dma_await_task(%c_col0_m6_n34)
      aiex.dma_free_task(%a_col0_m6_n34)
      aiex.dma_free_task(%b_col0_m6_n34)
      aiex.dma_free_task(%c_col0_m6_n34)
      // Col 0 M-tile 6 N-tile 35
      %a_col0_m6_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n35)
      %b_col0_m6_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n35)
      %c_col0_m6_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5513984 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n35)
      aiex.dma_await_task(%c_col0_m6_n35)
      aiex.dma_free_task(%a_col0_m6_n35)
      aiex.dma_free_task(%b_col0_m6_n35)
      aiex.dma_free_task(%c_col0_m6_n35)
      // Col 0 M-tile 6 N-tile 36
      %a_col0_m6_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n36)
      %b_col0_m6_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n36)
      %c_col0_m6_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5514240 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n36)
      aiex.dma_await_task(%c_col0_m6_n36)
      aiex.dma_free_task(%a_col0_m6_n36)
      aiex.dma_free_task(%b_col0_m6_n36)
      aiex.dma_free_task(%c_col0_m6_n36)
      // Col 0 M-tile 6 N-tile 37
      %a_col0_m6_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n37)
      %b_col0_m6_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n37)
      %c_col0_m6_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5514496 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n37)
      aiex.dma_await_task(%c_col0_m6_n37)
      aiex.dma_free_task(%a_col0_m6_n37)
      aiex.dma_free_task(%b_col0_m6_n37)
      aiex.dma_free_task(%c_col0_m6_n37)
      // Col 0 M-tile 6 N-tile 38
      %a_col0_m6_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n38)
      %b_col0_m6_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n38)
      %c_col0_m6_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5514752 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n38)
      aiex.dma_await_task(%c_col0_m6_n38)
      aiex.dma_free_task(%a_col0_m6_n38)
      aiex.dma_free_task(%b_col0_m6_n38)
      aiex.dma_free_task(%c_col0_m6_n38)
      // Col 0 M-tile 6 N-tile 39
      %a_col0_m6_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n39)
      %b_col0_m6_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n39)
      %c_col0_m6_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5515008 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n39)
      aiex.dma_await_task(%c_col0_m6_n39)
      aiex.dma_free_task(%a_col0_m6_n39)
      aiex.dma_free_task(%b_col0_m6_n39)
      aiex.dma_free_task(%c_col0_m6_n39)
      // Col 0 M-tile 6 N-tile 40
      %a_col0_m6_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n40)
      %b_col0_m6_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n40)
      %c_col0_m6_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5515264 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n40)
      aiex.dma_await_task(%c_col0_m6_n40)
      aiex.dma_free_task(%a_col0_m6_n40)
      aiex.dma_free_task(%b_col0_m6_n40)
      aiex.dma_free_task(%c_col0_m6_n40)
      // Col 0 M-tile 6 N-tile 41
      %a_col0_m6_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n41)
      %b_col0_m6_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n41)
      %c_col0_m6_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5515520 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n41)
      aiex.dma_await_task(%c_col0_m6_n41)
      aiex.dma_free_task(%a_col0_m6_n41)
      aiex.dma_free_task(%b_col0_m6_n41)
      aiex.dma_free_task(%c_col0_m6_n41)
      // Col 0 M-tile 6 N-tile 42
      %a_col0_m6_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n42)
      %b_col0_m6_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n42)
      %c_col0_m6_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5515776 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n42)
      aiex.dma_await_task(%c_col0_m6_n42)
      aiex.dma_free_task(%a_col0_m6_n42)
      aiex.dma_free_task(%b_col0_m6_n42)
      aiex.dma_free_task(%c_col0_m6_n42)
      // Col 0 M-tile 6 N-tile 43
      %a_col0_m6_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n43)
      %b_col0_m6_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n43)
      %c_col0_m6_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5516032 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n43)
      aiex.dma_await_task(%c_col0_m6_n43)
      aiex.dma_free_task(%a_col0_m6_n43)
      aiex.dma_free_task(%b_col0_m6_n43)
      aiex.dma_free_task(%c_col0_m6_n43)
      // Col 0 M-tile 6 N-tile 44
      %a_col0_m6_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n44)
      %b_col0_m6_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n44)
      %c_col0_m6_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5516288 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n44)
      aiex.dma_await_task(%c_col0_m6_n44)
      aiex.dma_free_task(%a_col0_m6_n44)
      aiex.dma_free_task(%b_col0_m6_n44)
      aiex.dma_free_task(%c_col0_m6_n44)
      // Col 0 M-tile 6 N-tile 45
      %a_col0_m6_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n45)
      %b_col0_m6_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n45)
      %c_col0_m6_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5516544 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n45)
      aiex.dma_await_task(%c_col0_m6_n45)
      aiex.dma_free_task(%a_col0_m6_n45)
      aiex.dma_free_task(%b_col0_m6_n45)
      aiex.dma_free_task(%c_col0_m6_n45)
      // Col 0 M-tile 6 N-tile 46
      %a_col0_m6_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n46)
      %b_col0_m6_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n46)
      %c_col0_m6_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5516800 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n46)
      aiex.dma_await_task(%c_col0_m6_n46)
      aiex.dma_free_task(%a_col0_m6_n46)
      aiex.dma_free_task(%b_col0_m6_n46)
      aiex.dma_free_task(%c_col0_m6_n46)
      // Col 0 M-tile 6 N-tile 47
      %a_col0_m6_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n47)
      %b_col0_m6_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n47)
      %c_col0_m6_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5517056 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n47)
      aiex.dma_await_task(%c_col0_m6_n47)
      aiex.dma_free_task(%a_col0_m6_n47)
      aiex.dma_free_task(%b_col0_m6_n47)
      aiex.dma_free_task(%c_col0_m6_n47)
      // Col 0 M-tile 6 N-tile 48
      %a_col0_m6_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n48)
      %b_col0_m6_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n48)
      %c_col0_m6_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5517312 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n48)
      aiex.dma_await_task(%c_col0_m6_n48)
      aiex.dma_free_task(%a_col0_m6_n48)
      aiex.dma_free_task(%b_col0_m6_n48)
      aiex.dma_free_task(%c_col0_m6_n48)
      // Col 0 M-tile 6 N-tile 49
      %a_col0_m6_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n49)
      %b_col0_m6_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n49)
      %c_col0_m6_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5517568 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n49)
      aiex.dma_await_task(%c_col0_m6_n49)
      aiex.dma_free_task(%a_col0_m6_n49)
      aiex.dma_free_task(%b_col0_m6_n49)
      aiex.dma_free_task(%c_col0_m6_n49)
      // Col 0 M-tile 6 N-tile 50
      %a_col0_m6_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n50)
      %b_col0_m6_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n50)
      %c_col0_m6_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5517824 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n50)
      aiex.dma_await_task(%c_col0_m6_n50)
      aiex.dma_free_task(%a_col0_m6_n50)
      aiex.dma_free_task(%b_col0_m6_n50)
      aiex.dma_free_task(%c_col0_m6_n50)
      // Col 0 M-tile 6 N-tile 51
      %a_col0_m6_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n51)
      %b_col0_m6_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n51)
      %c_col0_m6_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5518080 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n51)
      aiex.dma_await_task(%c_col0_m6_n51)
      aiex.dma_free_task(%a_col0_m6_n51)
      aiex.dma_free_task(%b_col0_m6_n51)
      aiex.dma_free_task(%c_col0_m6_n51)
      // Col 0 M-tile 6 N-tile 52
      %a_col0_m6_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n52)
      %b_col0_m6_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n52)
      %c_col0_m6_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5518336 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n52)
      aiex.dma_await_task(%c_col0_m6_n52)
      aiex.dma_free_task(%a_col0_m6_n52)
      aiex.dma_free_task(%b_col0_m6_n52)
      aiex.dma_free_task(%c_col0_m6_n52)
      // Col 0 M-tile 6 N-tile 53
      %a_col0_m6_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n53)
      %b_col0_m6_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n53)
      %c_col0_m6_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5518592 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n53)
      aiex.dma_await_task(%c_col0_m6_n53)
      aiex.dma_free_task(%a_col0_m6_n53)
      aiex.dma_free_task(%b_col0_m6_n53)
      aiex.dma_free_task(%c_col0_m6_n53)
      // Col 0 M-tile 6 N-tile 54
      %a_col0_m6_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n54)
      %b_col0_m6_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n54)
      %c_col0_m6_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5518848 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n54)
      aiex.dma_await_task(%c_col0_m6_n54)
      aiex.dma_free_task(%a_col0_m6_n54)
      aiex.dma_free_task(%b_col0_m6_n54)
      aiex.dma_free_task(%c_col0_m6_n54)
      // Col 0 M-tile 6 N-tile 55
      %a_col0_m6_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n55)
      %b_col0_m6_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n55)
      %c_col0_m6_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 5519104 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n55)
      aiex.dma_await_task(%c_col0_m6_n55)
      aiex.dma_free_task(%a_col0_m6_n55)
      aiex.dma_free_task(%b_col0_m6_n55)
      aiex.dma_free_task(%c_col0_m6_n55)
      // Col 0 M-tile 7 N-tile 0
      %a_col0_m7_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n0)
      %b_col0_m7_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n0)
      %c_col0_m7_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6422528 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n0)
      aiex.dma_await_task(%c_col0_m7_n0)
      aiex.dma_free_task(%a_col0_m7_n0)
      aiex.dma_free_task(%b_col0_m7_n0)
      aiex.dma_free_task(%c_col0_m7_n0)
      // Col 0 M-tile 7 N-tile 1
      %a_col0_m7_n1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n1)
      %b_col0_m7_n1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n1)
      %c_col0_m7_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6422784 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n1)
      aiex.dma_await_task(%c_col0_m7_n1)
      aiex.dma_free_task(%a_col0_m7_n1)
      aiex.dma_free_task(%b_col0_m7_n1)
      aiex.dma_free_task(%c_col0_m7_n1)
      // Col 0 M-tile 7 N-tile 2
      %a_col0_m7_n2 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n2)
      %b_col0_m7_n2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n2)
      %c_col0_m7_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6423040 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n2)
      aiex.dma_await_task(%c_col0_m7_n2)
      aiex.dma_free_task(%a_col0_m7_n2)
      aiex.dma_free_task(%b_col0_m7_n2)
      aiex.dma_free_task(%c_col0_m7_n2)
      // Col 0 M-tile 7 N-tile 3
      %a_col0_m7_n3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n3)
      %b_col0_m7_n3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n3)
      %c_col0_m7_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6423296 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n3)
      aiex.dma_await_task(%c_col0_m7_n3)
      aiex.dma_free_task(%a_col0_m7_n3)
      aiex.dma_free_task(%b_col0_m7_n3)
      aiex.dma_free_task(%c_col0_m7_n3)
      // Col 0 M-tile 7 N-tile 4
      %a_col0_m7_n4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n4)
      %b_col0_m7_n4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n4)
      %c_col0_m7_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6423552 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n4)
      aiex.dma_await_task(%c_col0_m7_n4)
      aiex.dma_free_task(%a_col0_m7_n4)
      aiex.dma_free_task(%b_col0_m7_n4)
      aiex.dma_free_task(%c_col0_m7_n4)
      // Col 0 M-tile 7 N-tile 5
      %a_col0_m7_n5 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n5)
      %b_col0_m7_n5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n5)
      %c_col0_m7_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6423808 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n5)
      aiex.dma_await_task(%c_col0_m7_n5)
      aiex.dma_free_task(%a_col0_m7_n5)
      aiex.dma_free_task(%b_col0_m7_n5)
      aiex.dma_free_task(%c_col0_m7_n5)
      // Col 0 M-tile 7 N-tile 6
      %a_col0_m7_n6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n6)
      %b_col0_m7_n6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n6)
      %c_col0_m7_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6424064 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n6)
      aiex.dma_await_task(%c_col0_m7_n6)
      aiex.dma_free_task(%a_col0_m7_n6)
      aiex.dma_free_task(%b_col0_m7_n6)
      aiex.dma_free_task(%c_col0_m7_n6)
      // Col 0 M-tile 7 N-tile 7
      %a_col0_m7_n7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n7)
      %b_col0_m7_n7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n7)
      %c_col0_m7_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6424320 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n7)
      aiex.dma_await_task(%c_col0_m7_n7)
      aiex.dma_free_task(%a_col0_m7_n7)
      aiex.dma_free_task(%b_col0_m7_n7)
      aiex.dma_free_task(%c_col0_m7_n7)
      // Col 0 M-tile 7 N-tile 8
      %a_col0_m7_n8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n8)
      %b_col0_m7_n8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n8)
      %c_col0_m7_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6424576 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n8)
      aiex.dma_await_task(%c_col0_m7_n8)
      aiex.dma_free_task(%a_col0_m7_n8)
      aiex.dma_free_task(%b_col0_m7_n8)
      aiex.dma_free_task(%c_col0_m7_n8)
      // Col 0 M-tile 7 N-tile 9
      %a_col0_m7_n9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n9)
      %b_col0_m7_n9 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n9)
      %c_col0_m7_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6424832 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n9)
      aiex.dma_await_task(%c_col0_m7_n9)
      aiex.dma_free_task(%a_col0_m7_n9)
      aiex.dma_free_task(%b_col0_m7_n9)
      aiex.dma_free_task(%c_col0_m7_n9)
      // Col 0 M-tile 7 N-tile 10
      %a_col0_m7_n10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n10)
      %b_col0_m7_n10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n10)
      %c_col0_m7_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6425088 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n10)
      aiex.dma_await_task(%c_col0_m7_n10)
      aiex.dma_free_task(%a_col0_m7_n10)
      aiex.dma_free_task(%b_col0_m7_n10)
      aiex.dma_free_task(%c_col0_m7_n10)
      // Col 0 M-tile 7 N-tile 11
      %a_col0_m7_n11 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n11)
      %b_col0_m7_n11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n11)
      %c_col0_m7_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6425344 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n11)
      aiex.dma_await_task(%c_col0_m7_n11)
      aiex.dma_free_task(%a_col0_m7_n11)
      aiex.dma_free_task(%b_col0_m7_n11)
      aiex.dma_free_task(%c_col0_m7_n11)
      // Col 0 M-tile 7 N-tile 12
      %a_col0_m7_n12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n12)
      %b_col0_m7_n12 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n12)
      %c_col0_m7_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6425600 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n12)
      aiex.dma_await_task(%c_col0_m7_n12)
      aiex.dma_free_task(%a_col0_m7_n12)
      aiex.dma_free_task(%b_col0_m7_n12)
      aiex.dma_free_task(%c_col0_m7_n12)
      // Col 0 M-tile 7 N-tile 13
      %a_col0_m7_n13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n13)
      %b_col0_m7_n13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n13)
      %c_col0_m7_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6425856 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n13)
      aiex.dma_await_task(%c_col0_m7_n13)
      aiex.dma_free_task(%a_col0_m7_n13)
      aiex.dma_free_task(%b_col0_m7_n13)
      aiex.dma_free_task(%c_col0_m7_n13)
      // Col 0 M-tile 7 N-tile 14
      %a_col0_m7_n14 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n14)
      %b_col0_m7_n14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n14)
      %c_col0_m7_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6426112 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n14)
      aiex.dma_await_task(%c_col0_m7_n14)
      aiex.dma_free_task(%a_col0_m7_n14)
      aiex.dma_free_task(%b_col0_m7_n14)
      aiex.dma_free_task(%c_col0_m7_n14)
      // Col 0 M-tile 7 N-tile 15
      %a_col0_m7_n15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n15)
      %b_col0_m7_n15 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n15)
      %c_col0_m7_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6426368 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n15)
      aiex.dma_await_task(%c_col0_m7_n15)
      aiex.dma_free_task(%a_col0_m7_n15)
      aiex.dma_free_task(%b_col0_m7_n15)
      aiex.dma_free_task(%c_col0_m7_n15)
      // Col 0 M-tile 7 N-tile 16
      %a_col0_m7_n16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n16)
      %b_col0_m7_n16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n16)
      %c_col0_m7_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6426624 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n16)
      aiex.dma_await_task(%c_col0_m7_n16)
      aiex.dma_free_task(%a_col0_m7_n16)
      aiex.dma_free_task(%b_col0_m7_n16)
      aiex.dma_free_task(%c_col0_m7_n16)
      // Col 0 M-tile 7 N-tile 17
      %a_col0_m7_n17 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n17)
      %b_col0_m7_n17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n17)
      %c_col0_m7_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6426880 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n17)
      aiex.dma_await_task(%c_col0_m7_n17)
      aiex.dma_free_task(%a_col0_m7_n17)
      aiex.dma_free_task(%b_col0_m7_n17)
      aiex.dma_free_task(%c_col0_m7_n17)
      // Col 0 M-tile 7 N-tile 18
      %a_col0_m7_n18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n18)
      %b_col0_m7_n18 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n18)
      %c_col0_m7_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6427136 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n18)
      aiex.dma_await_task(%c_col0_m7_n18)
      aiex.dma_free_task(%a_col0_m7_n18)
      aiex.dma_free_task(%b_col0_m7_n18)
      aiex.dma_free_task(%c_col0_m7_n18)
      // Col 0 M-tile 7 N-tile 19
      %a_col0_m7_n19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n19)
      %b_col0_m7_n19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n19)
      %c_col0_m7_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6427392 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n19)
      aiex.dma_await_task(%c_col0_m7_n19)
      aiex.dma_free_task(%a_col0_m7_n19)
      aiex.dma_free_task(%b_col0_m7_n19)
      aiex.dma_free_task(%c_col0_m7_n19)
      // Col 0 M-tile 7 N-tile 20
      %a_col0_m7_n20 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n20)
      %b_col0_m7_n20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n20)
      %c_col0_m7_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6427648 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n20)
      aiex.dma_await_task(%c_col0_m7_n20)
      aiex.dma_free_task(%a_col0_m7_n20)
      aiex.dma_free_task(%b_col0_m7_n20)
      aiex.dma_free_task(%c_col0_m7_n20)
      // Col 0 M-tile 7 N-tile 21
      %a_col0_m7_n21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n21)
      %b_col0_m7_n21 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n21)
      %c_col0_m7_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6427904 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n21)
      aiex.dma_await_task(%c_col0_m7_n21)
      aiex.dma_free_task(%a_col0_m7_n21)
      aiex.dma_free_task(%b_col0_m7_n21)
      aiex.dma_free_task(%c_col0_m7_n21)
      // Col 0 M-tile 7 N-tile 22
      %a_col0_m7_n22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n22)
      %b_col0_m7_n22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n22)
      %c_col0_m7_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6428160 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n22)
      aiex.dma_await_task(%c_col0_m7_n22)
      aiex.dma_free_task(%a_col0_m7_n22)
      aiex.dma_free_task(%b_col0_m7_n22)
      aiex.dma_free_task(%c_col0_m7_n22)
      // Col 0 M-tile 7 N-tile 23
      %a_col0_m7_n23 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n23)
      %b_col0_m7_n23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n23)
      %c_col0_m7_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6428416 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n23)
      aiex.dma_await_task(%c_col0_m7_n23)
      aiex.dma_free_task(%a_col0_m7_n23)
      aiex.dma_free_task(%b_col0_m7_n23)
      aiex.dma_free_task(%c_col0_m7_n23)
      // Col 0 M-tile 7 N-tile 24
      %a_col0_m7_n24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n24)
      %b_col0_m7_n24 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n24)
      %c_col0_m7_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6428672 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n24)
      aiex.dma_await_task(%c_col0_m7_n24)
      aiex.dma_free_task(%a_col0_m7_n24)
      aiex.dma_free_task(%b_col0_m7_n24)
      aiex.dma_free_task(%c_col0_m7_n24)
      // Col 0 M-tile 7 N-tile 25
      %a_col0_m7_n25 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n25)
      %b_col0_m7_n25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n25)
      %c_col0_m7_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6428928 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n25)
      aiex.dma_await_task(%c_col0_m7_n25)
      aiex.dma_free_task(%a_col0_m7_n25)
      aiex.dma_free_task(%b_col0_m7_n25)
      aiex.dma_free_task(%c_col0_m7_n25)
      // Col 0 M-tile 7 N-tile 26
      %a_col0_m7_n26 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n26)
      %b_col0_m7_n26 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n26)
      %c_col0_m7_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6429184 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n26)
      aiex.dma_await_task(%c_col0_m7_n26)
      aiex.dma_free_task(%a_col0_m7_n26)
      aiex.dma_free_task(%b_col0_m7_n26)
      aiex.dma_free_task(%c_col0_m7_n26)
      // Col 0 M-tile 7 N-tile 27
      %a_col0_m7_n27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n27)
      %b_col0_m7_n27 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n27)
      %c_col0_m7_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6429440 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n27)
      aiex.dma_await_task(%c_col0_m7_n27)
      aiex.dma_free_task(%a_col0_m7_n27)
      aiex.dma_free_task(%b_col0_m7_n27)
      aiex.dma_free_task(%c_col0_m7_n27)
      // Col 0 M-tile 7 N-tile 28
      %a_col0_m7_n28 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n28)
      %b_col0_m7_n28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n28)
      %c_col0_m7_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6429696 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n28)
      aiex.dma_await_task(%c_col0_m7_n28)
      aiex.dma_free_task(%a_col0_m7_n28)
      aiex.dma_free_task(%b_col0_m7_n28)
      aiex.dma_free_task(%c_col0_m7_n28)
      // Col 0 M-tile 7 N-tile 29
      %a_col0_m7_n29 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n29)
      %b_col0_m7_n29 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n29)
      %c_col0_m7_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6429952 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n29)
      aiex.dma_await_task(%c_col0_m7_n29)
      aiex.dma_free_task(%a_col0_m7_n29)
      aiex.dma_free_task(%b_col0_m7_n29)
      aiex.dma_free_task(%c_col0_m7_n29)
      // Col 0 M-tile 7 N-tile 30
      %a_col0_m7_n30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n30)
      %b_col0_m7_n30 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n30)
      %c_col0_m7_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6430208 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n30)
      aiex.dma_await_task(%c_col0_m7_n30)
      aiex.dma_free_task(%a_col0_m7_n30)
      aiex.dma_free_task(%b_col0_m7_n30)
      aiex.dma_free_task(%c_col0_m7_n30)
      // Col 0 M-tile 7 N-tile 31
      %a_col0_m7_n31 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n31)
      %b_col0_m7_n31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n31)
      %c_col0_m7_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6430464 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n31)
      aiex.dma_await_task(%c_col0_m7_n31)
      aiex.dma_free_task(%a_col0_m7_n31)
      aiex.dma_free_task(%b_col0_m7_n31)
      aiex.dma_free_task(%c_col0_m7_n31)
      // Col 0 M-tile 7 N-tile 32
      %a_col0_m7_n32 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n32)
      %b_col0_m7_n32 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n32)
      %c_col0_m7_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6430720 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n32)
      aiex.dma_await_task(%c_col0_m7_n32)
      aiex.dma_free_task(%a_col0_m7_n32)
      aiex.dma_free_task(%b_col0_m7_n32)
      aiex.dma_free_task(%c_col0_m7_n32)
      // Col 0 M-tile 7 N-tile 33
      %a_col0_m7_n33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n33)
      %b_col0_m7_n33 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n33)
      %c_col0_m7_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6430976 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n33)
      aiex.dma_await_task(%c_col0_m7_n33)
      aiex.dma_free_task(%a_col0_m7_n33)
      aiex.dma_free_task(%b_col0_m7_n33)
      aiex.dma_free_task(%c_col0_m7_n33)
      // Col 0 M-tile 7 N-tile 34
      %a_col0_m7_n34 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n34)
      %b_col0_m7_n34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n34)
      %c_col0_m7_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6431232 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n34)
      aiex.dma_await_task(%c_col0_m7_n34)
      aiex.dma_free_task(%a_col0_m7_n34)
      aiex.dma_free_task(%b_col0_m7_n34)
      aiex.dma_free_task(%c_col0_m7_n34)
      // Col 0 M-tile 7 N-tile 35
      %a_col0_m7_n35 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n35)
      %b_col0_m7_n35 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n35)
      %c_col0_m7_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6431488 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n35)
      aiex.dma_await_task(%c_col0_m7_n35)
      aiex.dma_free_task(%a_col0_m7_n35)
      aiex.dma_free_task(%b_col0_m7_n35)
      aiex.dma_free_task(%c_col0_m7_n35)
      // Col 0 M-tile 7 N-tile 36
      %a_col0_m7_n36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n36)
      %b_col0_m7_n36 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2304 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n36)
      %c_col0_m7_n36 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6431744 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n36)
      aiex.dma_await_task(%c_col0_m7_n36)
      aiex.dma_free_task(%a_col0_m7_n36)
      aiex.dma_free_task(%b_col0_m7_n36)
      aiex.dma_free_task(%c_col0_m7_n36)
      // Col 0 M-tile 7 N-tile 37
      %a_col0_m7_n37 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n37)
      %b_col0_m7_n37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2368 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n37)
      %c_col0_m7_n37 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6432000 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n37)
      aiex.dma_await_task(%c_col0_m7_n37)
      aiex.dma_free_task(%a_col0_m7_n37)
      aiex.dma_free_task(%b_col0_m7_n37)
      aiex.dma_free_task(%c_col0_m7_n37)
      // Col 0 M-tile 7 N-tile 38
      %a_col0_m7_n38 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n38)
      %b_col0_m7_n38 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2432 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n38)
      %c_col0_m7_n38 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6432256 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n38)
      aiex.dma_await_task(%c_col0_m7_n38)
      aiex.dma_free_task(%a_col0_m7_n38)
      aiex.dma_free_task(%b_col0_m7_n38)
      aiex.dma_free_task(%c_col0_m7_n38)
      // Col 0 M-tile 7 N-tile 39
      %a_col0_m7_n39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n39)
      %b_col0_m7_n39 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2496 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n39)
      %c_col0_m7_n39 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6432512 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n39)
      aiex.dma_await_task(%c_col0_m7_n39)
      aiex.dma_free_task(%a_col0_m7_n39)
      aiex.dma_free_task(%b_col0_m7_n39)
      aiex.dma_free_task(%c_col0_m7_n39)
      // Col 0 M-tile 7 N-tile 40
      %a_col0_m7_n40 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n40)
      %b_col0_m7_n40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2560 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n40)
      %c_col0_m7_n40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6432768 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n40)
      aiex.dma_await_task(%c_col0_m7_n40)
      aiex.dma_free_task(%a_col0_m7_n40)
      aiex.dma_free_task(%b_col0_m7_n40)
      aiex.dma_free_task(%c_col0_m7_n40)
      // Col 0 M-tile 7 N-tile 41
      %a_col0_m7_n41 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n41)
      %b_col0_m7_n41 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2624 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n41)
      %c_col0_m7_n41 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6433024 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n41)
      aiex.dma_await_task(%c_col0_m7_n41)
      aiex.dma_free_task(%a_col0_m7_n41)
      aiex.dma_free_task(%b_col0_m7_n41)
      aiex.dma_free_task(%c_col0_m7_n41)
      // Col 0 M-tile 7 N-tile 42
      %a_col0_m7_n42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n42)
      %b_col0_m7_n42 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2688 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n42)
      %c_col0_m7_n42 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6433280 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n42)
      aiex.dma_await_task(%c_col0_m7_n42)
      aiex.dma_free_task(%a_col0_m7_n42)
      aiex.dma_free_task(%b_col0_m7_n42)
      aiex.dma_free_task(%c_col0_m7_n42)
      // Col 0 M-tile 7 N-tile 43
      %a_col0_m7_n43 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n43)
      %b_col0_m7_n43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2752 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n43)
      %c_col0_m7_n43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6433536 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n43)
      aiex.dma_await_task(%c_col0_m7_n43)
      aiex.dma_free_task(%a_col0_m7_n43)
      aiex.dma_free_task(%b_col0_m7_n43)
      aiex.dma_free_task(%c_col0_m7_n43)
      // Col 0 M-tile 7 N-tile 44
      %a_col0_m7_n44 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n44)
      %b_col0_m7_n44 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2816 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n44)
      %c_col0_m7_n44 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6433792 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n44)
      aiex.dma_await_task(%c_col0_m7_n44)
      aiex.dma_free_task(%a_col0_m7_n44)
      aiex.dma_free_task(%b_col0_m7_n44)
      aiex.dma_free_task(%c_col0_m7_n44)
      // Col 0 M-tile 7 N-tile 45
      %a_col0_m7_n45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n45)
      %b_col0_m7_n45 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2880 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n45)
      %c_col0_m7_n45 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6434048 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n45)
      aiex.dma_await_task(%c_col0_m7_n45)
      aiex.dma_free_task(%a_col0_m7_n45)
      aiex.dma_free_task(%b_col0_m7_n45)
      aiex.dma_free_task(%c_col0_m7_n45)
      // Col 0 M-tile 7 N-tile 46
      %a_col0_m7_n46 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n46)
      %b_col0_m7_n46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 2944 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n46)
      %c_col0_m7_n46 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6434304 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n46)
      aiex.dma_await_task(%c_col0_m7_n46)
      aiex.dma_free_task(%a_col0_m7_n46)
      aiex.dma_free_task(%b_col0_m7_n46)
      aiex.dma_free_task(%c_col0_m7_n46)
      // Col 0 M-tile 7 N-tile 47
      %a_col0_m7_n47 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n47)
      %b_col0_m7_n47 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3008 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n47)
      %c_col0_m7_n47 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6434560 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n47)
      aiex.dma_await_task(%c_col0_m7_n47)
      aiex.dma_free_task(%a_col0_m7_n47)
      aiex.dma_free_task(%b_col0_m7_n47)
      aiex.dma_free_task(%c_col0_m7_n47)
      // Col 0 M-tile 7 N-tile 48
      %a_col0_m7_n48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n48)
      %b_col0_m7_n48 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3072 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n48)
      %c_col0_m7_n48 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6434816 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n48)
      aiex.dma_await_task(%c_col0_m7_n48)
      aiex.dma_free_task(%a_col0_m7_n48)
      aiex.dma_free_task(%b_col0_m7_n48)
      aiex.dma_free_task(%c_col0_m7_n48)
      // Col 0 M-tile 7 N-tile 49
      %a_col0_m7_n49 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n49)
      %b_col0_m7_n49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3136 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n49)
      %c_col0_m7_n49 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6435072 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n49)
      aiex.dma_await_task(%c_col0_m7_n49)
      aiex.dma_free_task(%a_col0_m7_n49)
      aiex.dma_free_task(%b_col0_m7_n49)
      aiex.dma_free_task(%c_col0_m7_n49)
      // Col 0 M-tile 7 N-tile 50
      %a_col0_m7_n50 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n50)
      %b_col0_m7_n50 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3200 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n50)
      %c_col0_m7_n50 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6435328 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n50)
      aiex.dma_await_task(%c_col0_m7_n50)
      aiex.dma_free_task(%a_col0_m7_n50)
      aiex.dma_free_task(%b_col0_m7_n50)
      aiex.dma_free_task(%c_col0_m7_n50)
      // Col 0 M-tile 7 N-tile 51
      %a_col0_m7_n51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n51)
      %b_col0_m7_n51 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3264 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n51)
      %c_col0_m7_n51 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6435584 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n51)
      aiex.dma_await_task(%c_col0_m7_n51)
      aiex.dma_free_task(%a_col0_m7_n51)
      aiex.dma_free_task(%b_col0_m7_n51)
      aiex.dma_free_task(%c_col0_m7_n51)
      // Col 0 M-tile 7 N-tile 52
      %a_col0_m7_n52 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n52)
      %b_col0_m7_n52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3328 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n52)
      %c_col0_m7_n52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6435840 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n52)
      aiex.dma_await_task(%c_col0_m7_n52)
      aiex.dma_free_task(%a_col0_m7_n52)
      aiex.dma_free_task(%b_col0_m7_n52)
      aiex.dma_free_task(%c_col0_m7_n52)
      // Col 0 M-tile 7 N-tile 53
      %a_col0_m7_n53 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n53)
      %b_col0_m7_n53 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3392 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n53)
      %c_col0_m7_n53 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6436096 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n53)
      aiex.dma_await_task(%c_col0_m7_n53)
      aiex.dma_free_task(%a_col0_m7_n53)
      aiex.dma_free_task(%b_col0_m7_n53)
      aiex.dma_free_task(%c_col0_m7_n53)
      // Col 0 M-tile 7 N-tile 54
      %a_col0_m7_n54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n54)
      %b_col0_m7_n54 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3456 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n54)
      %c_col0_m7_n54 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6436352 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n54)
      aiex.dma_await_task(%c_col0_m7_n54)
      aiex.dma_free_task(%a_col0_m7_n54)
      aiex.dma_free_task(%b_col0_m7_n54)
      aiex.dma_free_task(%c_col0_m7_n54)
      // Col 0 M-tile 7 N-tile 55
      %a_col0_m7_n55 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n55)
      %b_col0_m7_n55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x3584xi8> offset = 3520 len = 4096 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n55)
      %c_col0_m7_n55 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x3584xi32> offset = 6436608 len = 16384 sizes = [1, 64, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n55)
      aiex.dma_await_task(%c_col0_m7_n55)
      aiex.dma_free_task(%a_col0_m7_n55)
      aiex.dma_free_task(%b_col0_m7_n55)
      aiex.dma_free_task(%c_col0_m7_n55)
    }
  }
}