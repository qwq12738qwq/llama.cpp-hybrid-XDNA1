module {
  aie.device(npu1) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_0 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_2 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_3 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @A_L2L1(%logical_mem dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_core}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>  
    aie.objectfifo @A_L3L2(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>  
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [0])
    aie.objectfifo @B_L2L1(%logical_mem_0 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%logical_core}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>  
    aie.objectfifo @B_L3L2(%logical_shim_noc_1, {%logical_mem_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>  
    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0])
    aie.objectfifo @C_L1L2(%logical_core, {%logical_mem_2}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>  
    aie.objectfifo @C_L2L3(%logical_mem_2 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_3}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>  
    aie.objectfifo.link [@C_L1L2] -> [@C_L2L3]([] [0])
    func.func private @"91e3a8c6_matmul_i8_i32"(memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) attributes {link_with = "matmul_i8_i32_91e3a8c6.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c384 = arith.constant 384 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c384 step %c1_5 {
          %1 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<4096xi32>>
          %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<4096xi32>> -> memref<4096xi32>
          %c0_6 = arith.constant 0 : index
          %c4096 = arith.constant 4096 : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c4096 step %c1_7 {
            %c0_i32 = arith.constant 0 : i32
            memref.store %c0_i32, %2[%arg2] : memref<4096xi32>
          }
          %c0_8 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c16 step %c1_9 {
            %3 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
            %5 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
            func.call @"91e3a8c6_matmul_i8_i32"(%4, %6, %2) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
            aie.objectfifo.release @B_L2L1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<512x1024xi8>, %arg1: memref<1024x3072xi8>, %arg2: memref<512x3072xi32>) {
      %1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 0 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 0 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 65536 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 196608 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%6)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      %7 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 131072 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 393216 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      aiex.dma_await_task(%9)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      %10 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 196608 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 589824 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      aiex.dma_await_task(%12)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      %13 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 262144 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 786432 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      aiex.dma_await_task(%15)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
      %16 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 327680 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 983040 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      aiex.dma_await_task(%18)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      %19 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 393216 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 1179648 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      aiex.dma_await_task(%21)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      %22 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<512x1024xi8> offset = 458752 len = 65536 sizes = [48, 16, 64, 64] strides = [0, 64, 1024, 1])
        aie.end
      } {repeat_count = 47 : i32}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1024x3072xi8> offset = 0 len = 3145728 sizes = [1, 48, 1024, 64] strides = [0, 64, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<512x3072xi32> offset = 1376256 len = 196608 sizes = [1, 48, 64, 64] strides = [0, 64, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%24)
      aiex.dma_await_task(%24)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
    }
  }
}
