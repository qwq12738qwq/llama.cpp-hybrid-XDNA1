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
    func.func private @f240505e_matmul_i8_i32(memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) attributes {link_with = "matmul_i8_i32_f240505e.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_4 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c128 step %c1_5 {
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
          %c8 = arith.constant 8 : index
          %c1_9 = arith.constant 1 : index
          scf.for %arg2 = %c0_8 to %c8 step %c1_9 {
            %3 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
            %5 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
            func.call @f240505e_matmul_i8_i32(%4, %6, %2) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi32>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
            aie.objectfifo.release @B_L2L1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<128x512xi8>, %arg1: memref<512x4096xi8>, %arg2: memref<128x4096xi32>) {
      %1 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<128x512xi8> offset = 0 len = 32768 sizes = [64, 8, 64, 64] strides = [0, 64, 512, 1])
        aie.end
      } {repeat_count = 63 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<512x4096xi8> offset = 0 len = 2097152 sizes = [1, 64, 512, 64] strides = [0, 64, 4096, 1])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<128x4096xi32> offset = 0 len = 262144 sizes = [1, 64, 64, 64] strides = [0, 64, 4096, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<128x512xi8> offset = 32768 len = 32768 sizes = [64, 8, 64, 64] strides = [0, 64, 512, 1])
        aie.end
      } {repeat_count = 63 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<512x4096xi8> offset = 0 len = 2097152 sizes = [1, 64, 512, 64] strides = [0, 64, 4096, 1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<128x4096xi32> offset = 262144 len = 262144 sizes = [1, 64, 64, 64] strides = [0, 64, 4096, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%6)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
    }
  }
}
