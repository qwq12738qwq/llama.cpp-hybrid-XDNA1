// Multi-column INT8 GEMM: M=512 K=32 N=2304 n_cols=1
// Tile: Tm=64 Tk=64 Tn=64
// n_per_col=2304 nb_m=8 nb_k=0 nb_n=36
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
      %c_mn = arith.constant 288 : index
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

    aie.runtime_sequence(%A: memref<512x32xi8>, %B: memref<32x2304xi8>, %C_out: memref<512x2304xi32>) {
      // Col 0 M-tile 0 N-tile 0
      %a_col0_m0_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m0_n0)
      %b_col0_m0_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n0)
      %c_col0_m0_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 0 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n1)
      %c_col0_m0_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 256 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n2)
      %c_col0_m0_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 512 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n3)
      %c_col0_m0_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 768 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n4)
      %c_col0_m0_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1024 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n5)
      %c_col0_m0_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1280 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n6)
      %c_col0_m0_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1536 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n7)
      %c_col0_m0_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1792 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n8)
      %c_col0_m0_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2048 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n9)
      %c_col0_m0_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2304 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n10)
      %c_col0_m0_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2560 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n11)
      %c_col0_m0_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2816 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n12)
      %c_col0_m0_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3072 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n13)
      %c_col0_m0_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3328 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n14)
      %c_col0_m0_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3584 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n15)
      %c_col0_m0_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3840 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n16)
      %c_col0_m0_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4096 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n17)
      %c_col0_m0_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4352 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n18)
      %c_col0_m0_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4608 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n19)
      %c_col0_m0_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4864 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n20)
      %c_col0_m0_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 5120 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n21)
      %c_col0_m0_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 5376 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n22)
      %c_col0_m0_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 5632 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n23)
      %c_col0_m0_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 5888 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n24)
      %c_col0_m0_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 6144 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n25)
      %c_col0_m0_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 6400 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n26)
      %c_col0_m0_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 6656 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n27)
      %c_col0_m0_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 6912 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n28)
      %c_col0_m0_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 7168 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n29)
      %c_col0_m0_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 7424 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n30)
      %c_col0_m0_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 7680 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n31)
      %c_col0_m0_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 7936 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n32)
      %c_col0_m0_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 8192 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n33)
      %c_col0_m0_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 8448 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n34)
      %c_col0_m0_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 8704 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m0_n35)
      %c_col0_m0_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 8960 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m0_n35)
      aiex.dma_await_task(%c_col0_m0_n35)
      aiex.dma_free_task(%a_col0_m0_n35)
      aiex.dma_free_task(%b_col0_m0_n35)
      aiex.dma_free_task(%c_col0_m0_n35)
      // Col 0 M-tile 1 N-tile 0
      %a_col0_m1_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m1_n0)
      %b_col0_m1_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n0)
      %c_col0_m1_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 589824 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n1)
      %c_col0_m1_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 590080 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n2)
      %c_col0_m1_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 590336 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n3)
      %c_col0_m1_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 590592 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n4)
      %c_col0_m1_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 590848 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n5)
      %c_col0_m1_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 591104 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n6)
      %c_col0_m1_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 591360 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n7)
      %c_col0_m1_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 591616 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n8)
      %c_col0_m1_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 591872 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n9)
      %c_col0_m1_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 592128 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n10)
      %c_col0_m1_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 592384 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n11)
      %c_col0_m1_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 592640 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n12)
      %c_col0_m1_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 592896 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n13)
      %c_col0_m1_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 593152 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n14)
      %c_col0_m1_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 593408 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n15)
      %c_col0_m1_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 593664 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n16)
      %c_col0_m1_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 593920 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n17)
      %c_col0_m1_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 594176 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n18)
      %c_col0_m1_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 594432 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n19)
      %c_col0_m1_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 594688 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n20)
      %c_col0_m1_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 594944 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n21)
      %c_col0_m1_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 595200 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n22)
      %c_col0_m1_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 595456 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n23)
      %c_col0_m1_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 595712 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n24)
      %c_col0_m1_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 595968 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n25)
      %c_col0_m1_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 596224 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n26)
      %c_col0_m1_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 596480 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n27)
      %c_col0_m1_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 596736 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n28)
      %c_col0_m1_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 596992 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n29)
      %c_col0_m1_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 597248 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n30)
      %c_col0_m1_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 597504 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n31)
      %c_col0_m1_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 597760 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n32)
      %c_col0_m1_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 598016 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n33)
      %c_col0_m1_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 598272 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n34)
      %c_col0_m1_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 598528 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m1_n35)
      %c_col0_m1_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 598784 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m1_n35)
      aiex.dma_await_task(%c_col0_m1_n35)
      aiex.dma_free_task(%a_col0_m1_n35)
      aiex.dma_free_task(%b_col0_m1_n35)
      aiex.dma_free_task(%c_col0_m1_n35)
      // Col 0 M-tile 2 N-tile 0
      %a_col0_m2_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 4096 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m2_n0)
      %b_col0_m2_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n0)
      %c_col0_m2_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1179648 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n1)
      %c_col0_m2_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1179904 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n2)
      %c_col0_m2_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1180160 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n3)
      %c_col0_m2_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1180416 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n4)
      %c_col0_m2_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1180672 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n5)
      %c_col0_m2_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1180928 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n6)
      %c_col0_m2_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1181184 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n7)
      %c_col0_m2_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1181440 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n8)
      %c_col0_m2_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1181696 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n9)
      %c_col0_m2_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1181952 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n10)
      %c_col0_m2_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1182208 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n11)
      %c_col0_m2_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1182464 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n12)
      %c_col0_m2_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1182720 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n13)
      %c_col0_m2_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1182976 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n14)
      %c_col0_m2_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1183232 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n15)
      %c_col0_m2_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1183488 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n16)
      %c_col0_m2_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1183744 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n17)
      %c_col0_m2_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1184000 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n18)
      %c_col0_m2_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1184256 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n19)
      %c_col0_m2_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1184512 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n20)
      %c_col0_m2_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1184768 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n21)
      %c_col0_m2_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1185024 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n22)
      %c_col0_m2_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1185280 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n23)
      %c_col0_m2_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1185536 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n24)
      %c_col0_m2_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1185792 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n25)
      %c_col0_m2_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1186048 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n26)
      %c_col0_m2_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1186304 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n27)
      %c_col0_m2_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1186560 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n28)
      %c_col0_m2_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1186816 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n29)
      %c_col0_m2_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1187072 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n30)
      %c_col0_m2_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1187328 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n31)
      %c_col0_m2_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1187584 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n32)
      %c_col0_m2_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1187840 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n33)
      %c_col0_m2_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1188096 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n34)
      %c_col0_m2_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1188352 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m2_n35)
      %c_col0_m2_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1188608 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m2_n35)
      aiex.dma_await_task(%c_col0_m2_n35)
      aiex.dma_free_task(%a_col0_m2_n35)
      aiex.dma_free_task(%b_col0_m2_n35)
      aiex.dma_free_task(%c_col0_m2_n35)
      // Col 0 M-tile 3 N-tile 0
      %a_col0_m3_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 6144 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m3_n0)
      %b_col0_m3_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n0)
      %c_col0_m3_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1769472 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n1)
      %c_col0_m3_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1769728 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n2)
      %c_col0_m3_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1769984 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n3)
      %c_col0_m3_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1770240 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n4)
      %c_col0_m3_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1770496 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n5)
      %c_col0_m3_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1770752 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n6)
      %c_col0_m3_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1771008 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n7)
      %c_col0_m3_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1771264 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n8)
      %c_col0_m3_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1771520 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n9)
      %c_col0_m3_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1771776 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n10)
      %c_col0_m3_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1772032 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n11)
      %c_col0_m3_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1772288 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n12)
      %c_col0_m3_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1772544 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n13)
      %c_col0_m3_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1772800 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n14)
      %c_col0_m3_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1773056 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n15)
      %c_col0_m3_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1773312 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n16)
      %c_col0_m3_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1773568 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n17)
      %c_col0_m3_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1773824 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n18)
      %c_col0_m3_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1774080 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n19)
      %c_col0_m3_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1774336 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n20)
      %c_col0_m3_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1774592 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n21)
      %c_col0_m3_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1774848 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n22)
      %c_col0_m3_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1775104 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n23)
      %c_col0_m3_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1775360 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n24)
      %c_col0_m3_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1775616 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n25)
      %c_col0_m3_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1775872 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n26)
      %c_col0_m3_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1776128 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n27)
      %c_col0_m3_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1776384 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n28)
      %c_col0_m3_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1776640 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n29)
      %c_col0_m3_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1776896 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n30)
      %c_col0_m3_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1777152 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n31)
      %c_col0_m3_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1777408 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n32)
      %c_col0_m3_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1777664 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n33)
      %c_col0_m3_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1777920 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n34)
      %c_col0_m3_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1778176 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m3_n35)
      %c_col0_m3_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 1778432 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m3_n35)
      aiex.dma_await_task(%c_col0_m3_n35)
      aiex.dma_free_task(%a_col0_m3_n35)
      aiex.dma_free_task(%b_col0_m3_n35)
      aiex.dma_free_task(%c_col0_m3_n35)
      // Col 0 M-tile 4 N-tile 0
      %a_col0_m4_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 8192 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m4_n0)
      %b_col0_m4_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n0)
      %c_col0_m4_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2359296 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n1)
      %c_col0_m4_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2359552 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n2)
      %c_col0_m4_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2359808 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n3)
      %c_col0_m4_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2360064 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n4)
      %c_col0_m4_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2360320 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n5)
      %c_col0_m4_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2360576 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n6)
      %c_col0_m4_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2360832 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n7)
      %c_col0_m4_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2361088 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n8)
      %c_col0_m4_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2361344 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n9)
      %c_col0_m4_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2361600 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n10)
      %c_col0_m4_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2361856 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n11)
      %c_col0_m4_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2362112 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n12)
      %c_col0_m4_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2362368 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n13)
      %c_col0_m4_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2362624 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n14)
      %c_col0_m4_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2362880 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n15)
      %c_col0_m4_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2363136 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n16)
      %c_col0_m4_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2363392 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n17)
      %c_col0_m4_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2363648 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n18)
      %c_col0_m4_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2363904 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n19)
      %c_col0_m4_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2364160 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n20)
      %c_col0_m4_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2364416 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n21)
      %c_col0_m4_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2364672 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n22)
      %c_col0_m4_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2364928 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n23)
      %c_col0_m4_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2365184 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n24)
      %c_col0_m4_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2365440 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n25)
      %c_col0_m4_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2365696 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n26)
      %c_col0_m4_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2365952 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n27)
      %c_col0_m4_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2366208 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n28)
      %c_col0_m4_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2366464 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n29)
      %c_col0_m4_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2366720 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n30)
      %c_col0_m4_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2366976 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n31)
      %c_col0_m4_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2367232 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n32)
      %c_col0_m4_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2367488 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n33)
      %c_col0_m4_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2367744 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n34)
      %c_col0_m4_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2368000 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m4_n35)
      %c_col0_m4_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2368256 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m4_n35)
      aiex.dma_await_task(%c_col0_m4_n35)
      aiex.dma_free_task(%a_col0_m4_n35)
      aiex.dma_free_task(%b_col0_m4_n35)
      aiex.dma_free_task(%c_col0_m4_n35)
      // Col 0 M-tile 5 N-tile 0
      %a_col0_m5_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 10240 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m5_n0)
      %b_col0_m5_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n0)
      %c_col0_m5_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2949120 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n1)
      %c_col0_m5_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2949376 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n2)
      %c_col0_m5_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2949632 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n3)
      %c_col0_m5_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2949888 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n4)
      %c_col0_m5_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2950144 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n5)
      %c_col0_m5_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2950400 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n6)
      %c_col0_m5_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2950656 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n7)
      %c_col0_m5_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2950912 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n8)
      %c_col0_m5_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2951168 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n9)
      %c_col0_m5_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2951424 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n10)
      %c_col0_m5_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2951680 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n11)
      %c_col0_m5_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2951936 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n12)
      %c_col0_m5_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2952192 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n13)
      %c_col0_m5_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2952448 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n14)
      %c_col0_m5_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2952704 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n15)
      %c_col0_m5_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2952960 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n16)
      %c_col0_m5_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2953216 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n17)
      %c_col0_m5_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2953472 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n18)
      %c_col0_m5_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2953728 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n19)
      %c_col0_m5_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2953984 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n20)
      %c_col0_m5_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2954240 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n21)
      %c_col0_m5_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2954496 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n22)
      %c_col0_m5_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2954752 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n23)
      %c_col0_m5_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2955008 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n24)
      %c_col0_m5_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2955264 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n25)
      %c_col0_m5_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2955520 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n26)
      %c_col0_m5_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2955776 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n27)
      %c_col0_m5_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2956032 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n28)
      %c_col0_m5_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2956288 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n29)
      %c_col0_m5_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2956544 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n30)
      %c_col0_m5_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2956800 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n31)
      %c_col0_m5_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2957056 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n32)
      %c_col0_m5_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2957312 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n33)
      %c_col0_m5_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2957568 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n34)
      %c_col0_m5_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2957824 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m5_n35)
      %c_col0_m5_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 2958080 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m5_n35)
      aiex.dma_await_task(%c_col0_m5_n35)
      aiex.dma_free_task(%a_col0_m5_n35)
      aiex.dma_free_task(%b_col0_m5_n35)
      aiex.dma_free_task(%c_col0_m5_n35)
      // Col 0 M-tile 6 N-tile 0
      %a_col0_m6_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 12288 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m6_n0)
      %b_col0_m6_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n0)
      %c_col0_m6_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3538944 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n1)
      %c_col0_m6_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3539200 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n2)
      %c_col0_m6_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3539456 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n3)
      %c_col0_m6_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3539712 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n4)
      %c_col0_m6_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3539968 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n5)
      %c_col0_m6_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3540224 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n6)
      %c_col0_m6_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3540480 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n7)
      %c_col0_m6_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3540736 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n8)
      %c_col0_m6_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3540992 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n9)
      %c_col0_m6_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3541248 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n10)
      %c_col0_m6_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3541504 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n11)
      %c_col0_m6_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3541760 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n12)
      %c_col0_m6_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3542016 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n13)
      %c_col0_m6_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3542272 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n14)
      %c_col0_m6_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3542528 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n15)
      %c_col0_m6_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3542784 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n16)
      %c_col0_m6_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3543040 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n17)
      %c_col0_m6_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3543296 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n18)
      %c_col0_m6_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3543552 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n19)
      %c_col0_m6_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3543808 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n20)
      %c_col0_m6_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3544064 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n21)
      %c_col0_m6_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3544320 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n22)
      %c_col0_m6_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3544576 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n23)
      %c_col0_m6_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3544832 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n24)
      %c_col0_m6_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3545088 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n25)
      %c_col0_m6_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3545344 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n26)
      %c_col0_m6_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3545600 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n27)
      %c_col0_m6_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3545856 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n28)
      %c_col0_m6_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3546112 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n29)
      %c_col0_m6_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3546368 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n30)
      %c_col0_m6_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3546624 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n31)
      %c_col0_m6_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3546880 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n32)
      %c_col0_m6_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3547136 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n33)
      %c_col0_m6_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3547392 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n34)
      %c_col0_m6_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3547648 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m6_n35)
      %c_col0_m6_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 3547904 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m6_n35)
      aiex.dma_await_task(%c_col0_m6_n35)
      aiex.dma_free_task(%a_col0_m6_n35)
      aiex.dma_free_task(%b_col0_m6_n35)
      aiex.dma_free_task(%c_col0_m6_n35)
      // Col 0 M-tile 7 N-tile 0
      %a_col0_m7_n0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%A : memref<512x32xi8> offset = 14336 len = 4096 sizes = [1, 64, 64] strides = [0, 32, 1])
        aie.end
      }
      aiex.dma_start_task(%a_col0_m7_n0)
      %b_col0_m7_n0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%B : memref<32x2304xi8> offset = 0 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n0)
      %c_col0_m7_n0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4128768 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 64 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n1)
      %c_col0_m7_n1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4129024 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 128 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n2)
      %c_col0_m7_n2 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4129280 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 192 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n3)
      %c_col0_m7_n3 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4129536 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 256 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n4)
      %c_col0_m7_n4 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4129792 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 320 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n5)
      %c_col0_m7_n5 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4130048 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 384 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n6)
      %c_col0_m7_n6 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4130304 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 448 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n7)
      %c_col0_m7_n7 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4130560 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 512 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n8)
      %c_col0_m7_n8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4130816 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 576 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n9)
      %c_col0_m7_n9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4131072 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 640 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n10)
      %c_col0_m7_n10 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4131328 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 704 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n11)
      %c_col0_m7_n11 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4131584 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 768 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n12)
      %c_col0_m7_n12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4131840 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 832 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n13)
      %c_col0_m7_n13 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4132096 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 896 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n14)
      %c_col0_m7_n14 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4132352 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 960 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n15)
      %c_col0_m7_n15 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4132608 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1024 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n16)
      %c_col0_m7_n16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4132864 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1088 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n17)
      %c_col0_m7_n17 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4133120 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1152 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n18)
      %c_col0_m7_n18 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4133376 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1216 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n19)
      %c_col0_m7_n19 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4133632 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1280 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n20)
      %c_col0_m7_n20 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4133888 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1344 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n21)
      %c_col0_m7_n21 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4134144 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1408 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n22)
      %c_col0_m7_n22 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4134400 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1472 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n23)
      %c_col0_m7_n23 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4134656 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1536 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n24)
      %c_col0_m7_n24 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4134912 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1600 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n25)
      %c_col0_m7_n25 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4135168 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1664 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n26)
      %c_col0_m7_n26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4135424 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1728 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n27)
      %c_col0_m7_n27 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4135680 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1792 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n28)
      %c_col0_m7_n28 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4135936 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1856 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n29)
      %c_col0_m7_n29 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4136192 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1920 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n30)
      %c_col0_m7_n30 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4136448 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 1984 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n31)
      %c_col0_m7_n31 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4136704 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2048 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n32)
      %c_col0_m7_n32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4136960 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2112 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n33)
      %c_col0_m7_n33 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4137216 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2176 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n34)
      %c_col0_m7_n34 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4137472 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
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
        aie.dma_bd(%B : memref<32x2304xi8> offset = 2240 len = 4096 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      }
      aiex.dma_start_task(%b_col0_m7_n35)
      %c_col0_m7_n35 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%C_out : memref<512x2304xi32> offset = 4137728 len = 16384 sizes = [1, 64, 64] strides = [0, 2304, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_col0_m7_n35)
      aiex.dma_await_task(%c_col0_m7_n35)
      aiex.dma_free_task(%a_col0_m7_n35)
      aiex.dma_free_task(%b_col0_m7_n35)
      aiex.dma_free_task(%c_col0_m7_n35)
    }
  }
}