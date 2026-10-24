// Multi-column INT8 GEMM: M=512 K=32 N=768 n_cols=1
// Tile: Tm=64 Tk=64 Tn=64
// n_per_col=768 nb_m=8 nb_k=0 nb_n=12
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
      %c_mn = arith.constant 96 : index
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

    aie.runtime_sequence(%A: memref<512x32xi8>, %B: memref<32x768xi8>, %C_out: memref<512x768xi32>) {
      // Col 0 M-tile 0 N-tile 0
      %a_col0_m0_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n0)
      %b_col0_m0_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n0)
      %c_col0_m0_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 0 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n1)
      %c_col0_m0_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 256 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n2)
      %c_col0_m0_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 512 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n3)
      %c_col0_m0_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 768 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n4)
      %c_col0_m0_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1024 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n5)
      %c_col0_m0_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1280 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n6)
      %c_col0_m0_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1536 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n7)
      %c_col0_m0_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1792 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n8)
      %c_col0_m0_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 2048 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n9)
      %c_col0_m0_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 2304 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n10)
      %c_col0_m0_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 2560 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n11)
      %c_col0_m0_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 2816 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n11)
      aiex.dma_await_task(%c_col0_m0_n11)
      aiex.dma_free_task(%a_col0_m0_n11)
      aiex.dma_free_task(%b_col0_m0_n11)
      aiex.dma_free_task(%c_col0_m0_n11)
      // Col 0 M-tile 1 N-tile 0
      %a_col0_m1_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n0)
      %b_col0_m1_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n0)
      %c_col0_m1_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 196608 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n1)
      %c_col0_m1_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 196864 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n2)
      %c_col0_m1_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 197120 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n3)
      %c_col0_m1_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 197376 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n4)
      %c_col0_m1_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 197632 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n5)
      %c_col0_m1_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 197888 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n6)
      %c_col0_m1_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 198144 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n7)
      %c_col0_m1_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 198400 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n8)
      %c_col0_m1_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 198656 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n9)
      %c_col0_m1_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 198912 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n10)
      %c_col0_m1_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 199168 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n11)
      %c_col0_m1_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 199424 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n11)
      aiex.dma_await_task(%c_col0_m1_n11)
      aiex.dma_free_task(%a_col0_m1_n11)
      aiex.dma_free_task(%b_col0_m1_n11)
      aiex.dma_free_task(%c_col0_m1_n11)
      // Col 0 M-tile 2 N-tile 0
      %a_col0_m2_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n0)
      %b_col0_m2_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n0)
      %c_col0_m2_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 393216 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n1)
      %c_col0_m2_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 393472 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n2)
      %c_col0_m2_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 393728 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n3)
      %c_col0_m2_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 393984 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n4)
      %c_col0_m2_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 394240 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n5)
      %c_col0_m2_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 394496 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n6)
      %c_col0_m2_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 394752 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n7)
      %c_col0_m2_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 395008 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n8)
      %c_col0_m2_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 395264 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n9)
      %c_col0_m2_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 395520 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n10)
      %c_col0_m2_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 395776 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n11)
      %c_col0_m2_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 396032 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n11)
      aiex.dma_await_task(%c_col0_m2_n11)
      aiex.dma_free_task(%a_col0_m2_n11)
      aiex.dma_free_task(%b_col0_m2_n11)
      aiex.dma_free_task(%c_col0_m2_n11)
      // Col 0 M-tile 3 N-tile 0
      %a_col0_m3_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n0)
      %b_col0_m3_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n0)
      %c_col0_m3_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 589824 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n1)
      %c_col0_m3_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 590080 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n2)
      %c_col0_m3_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 590336 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n3)
      %c_col0_m3_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 590592 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n4)
      %c_col0_m3_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 590848 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n5)
      %c_col0_m3_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 591104 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n6)
      %c_col0_m3_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 591360 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n7)
      %c_col0_m3_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 591616 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n8)
      %c_col0_m3_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 591872 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n9)
      %c_col0_m3_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 592128 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n10)
      %c_col0_m3_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 592384 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n11)
      %c_col0_m3_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 592640 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n11)
      aiex.dma_await_task(%c_col0_m3_n11)
      aiex.dma_free_task(%a_col0_m3_n11)
      aiex.dma_free_task(%b_col0_m3_n11)
      aiex.dma_free_task(%c_col0_m3_n11)
      // Col 0 M-tile 4 N-tile 0
      %a_col0_m4_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n0)
      %b_col0_m4_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n0)
      %c_col0_m4_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 786432 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n1)
      %c_col0_m4_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 786688 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n2)
      %c_col0_m4_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 786944 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n3)
      %c_col0_m4_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 787200 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n4)
      %c_col0_m4_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 787456 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n5)
      %c_col0_m4_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 787712 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n6)
      %c_col0_m4_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 787968 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n7)
      %c_col0_m4_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 788224 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n8)
      %c_col0_m4_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 788480 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n9)
      %c_col0_m4_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 788736 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n10)
      %c_col0_m4_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 788992 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n11)
      %c_col0_m4_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 789248 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n11)
      aiex.dma_await_task(%c_col0_m4_n11)
      aiex.dma_free_task(%a_col0_m4_n11)
      aiex.dma_free_task(%b_col0_m4_n11)
      aiex.dma_free_task(%c_col0_m4_n11)
      // Col 0 M-tile 5 N-tile 0
      %a_col0_m5_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n0)
      %b_col0_m5_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n0)
      %c_col0_m5_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 983040 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n1)
      %c_col0_m5_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 983296 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n2)
      %c_col0_m5_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 983552 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n3)
      %c_col0_m5_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 983808 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n4)
      %c_col0_m5_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 984064 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n5)
      %c_col0_m5_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 984320 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n6)
      %c_col0_m5_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 984576 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n7)
      %c_col0_m5_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 984832 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n8)
      %c_col0_m5_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 985088 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n9)
      %c_col0_m5_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 985344 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n10)
      %c_col0_m5_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 985600 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n11)
      %c_col0_m5_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 985856 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n11)
      aiex.dma_await_task(%c_col0_m5_n11)
      aiex.dma_free_task(%a_col0_m5_n11)
      aiex.dma_free_task(%b_col0_m5_n11)
      aiex.dma_free_task(%c_col0_m5_n11)
      // Col 0 M-tile 6 N-tile 0
      %a_col0_m6_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n0)
      %b_col0_m6_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n0)
      %c_col0_m6_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1179648 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n1)
      %c_col0_m6_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1179904 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n2)
      %c_col0_m6_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1180160 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n3)
      %c_col0_m6_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1180416 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n4)
      %c_col0_m6_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1180672 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n5)
      %c_col0_m6_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1180928 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n6)
      %c_col0_m6_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1181184 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n7)
      %c_col0_m6_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1181440 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n8)
      %c_col0_m6_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1181696 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n9)
      %c_col0_m6_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1181952 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n10)
      %c_col0_m6_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1182208 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n11)
      %c_col0_m6_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1182464 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n11)
      aiex.dma_await_task(%c_col0_m6_n11)
      aiex.dma_free_task(%a_col0_m6_n11)
      aiex.dma_free_task(%b_col0_m6_n11)
      aiex.dma_free_task(%c_col0_m6_n11)
      // Col 0 M-tile 7 N-tile 0
      %a_col0_m7_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n0)
      %b_col0_m7_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x768xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n0)
      %c_col0_m7_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1376256 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n1)
      %c_col0_m7_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1376512 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n2)
      %c_col0_m7_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1376768 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n3)
      %c_col0_m7_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1377024 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n4)
      %c_col0_m7_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1377280 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n5)
      %c_col0_m7_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1377536 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n6)
      %c_col0_m7_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1377792 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n7)
      %c_col0_m7_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1378048 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n8)
      %c_col0_m7_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1378304 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n9)
      %c_col0_m7_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1378560 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n10)
      %c_col0_m7_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1378816 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
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
        aie.dma_bd(%B : memref<32x768xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n11)
      %c_col0_m7_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x768xi32> offset = 1379072 len = 16384 sizes = [1, 64, 64] strides = [0, 768, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n11)
      aiex.dma_await_task(%c_col0_m7_n11)
      aiex.dma_free_task(%a_col0_m7_n11)
      aiex.dma_free_task(%b_col0_m7_n11)
      aiex.dma_free_task(%c_col0_m7_n11)
    }
  }
}