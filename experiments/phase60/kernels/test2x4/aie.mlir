// Phase60 multi-tile: M=512 K=1024 N=3584 rows=2 cols=4
module {
  aie.device(npu1) {
    %tile_core_r0_c0 = aie.tile(0, 2)
    %tile_core_r0_c1 = aie.tile(1, 2)
    %tile_core_r0_c2 = aie.tile(2, 2)
    %tile_core_r0_c3 = aie.tile(3, 2)
    %tile_core_r1_c0 = aie.tile(0, 3)
    %tile_core_r1_c1 = aie.tile(1, 3)
    %tile_core_r1_c2 = aie.tile(2, 3)
    %tile_core_r1_c3 = aie.tile(3, 3)
    %shim_a = aie.tile(0, 0)
    %mem_a = aie.tile(0, 1)
    %shim_col0 = aie.tile(0, 0)
    %mem_col0 = aie.tile(0, 1)
    %shim_col1 = aie.tile(1, 0)
    %mem_col1 = aie.tile(1, 1)
    %shim_col2 = aie.tile(2, 0)
    %mem_col2 = aie.tile(2, 1)
    %shim_col3 = aie.tile(3, 0)
    %mem_col3 = aie.tile(3, 1)
    aie.objectfifo @A_r0(%shim_a, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @A_r1(%shim_a, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @B_c0(%shim_col0, {%mem_col0}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @B_c1(%shim_col1, {%mem_col1}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @B_c2(%shim_col2, {%mem_col2}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @B_c3(%shim_col3, {%mem_col3}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @A2_r0(%mem_a, {%tile_core_r0_c0, %tile_core_r0_c1, %tile_core_r0_c2, %tile_core_r0_c3}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo.link [@A_r0] -> [@A2_r0]([] [0])
    aie.objectfifo @A2_r1(%mem_a, {%tile_core_r1_c0, %tile_core_r1_c1, %tile_core_r1_c2, %tile_core_r1_c3}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo.link [@A_r1] -> [@A2_r1]([] [0])
    aie.objectfifo @B2_c0(%mem_col0, {%tile_core_r0_c0, %tile_core_r1_c0}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo.link [@B_c0] -> [@B2_c0]([] [0])
    aie.objectfifo @B2_c1(%mem_col1, {%tile_core_r0_c1, %tile_core_r1_c1}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo.link [@B_c1] -> [@B2_c1]([] [0])
    aie.objectfifo @B2_c2(%mem_col2, {%tile_core_r0_c2, %tile_core_r1_c2}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo.link [@B_c2] -> [@B2_c2]([] [0])
    aie.objectfifo @B2_c3(%mem_col3, {%tile_core_r0_c3, %tile_core_r1_c3}, 2 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo.link [@B_c3] -> [@B2_c3]([] [0])
    aie.objectfifo @C_r0_c0(%tile_core_r0_c0, {%mem_col0}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r0_c1(%tile_core_r0_c1, {%mem_col1}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r0_c2(%tile_core_r0_c2, {%mem_col2}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r0_c3(%tile_core_r0_c3, {%mem_col3}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r1_c0(%tile_core_r1_c0, {%mem_col0}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r1_c1(%tile_core_r1_c1, {%mem_col1}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r1_c2(%tile_core_r1_c2, {%mem_col2}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C_r1_c3(%tile_core_r1_c3, {%mem_col3}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @C2_r0_c0(%mem_col0, {%shim_col0}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r0_c0] -> [@C2_r0_c0]([] [0])
    aie.objectfifo @C2_r0_c1(%mem_col1, {%shim_col1}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r0_c1] -> [@C2_r0_c1]([] [0])
    aie.objectfifo @C2_r0_c2(%mem_col2, {%shim_col2}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r0_c2] -> [@C2_r0_c2]([] [0])
    aie.objectfifo @C2_r0_c3(%mem_col3, {%shim_col3}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r0_c3] -> [@C2_r0_c3]([] [0])
    aie.objectfifo @C2_r1_c0(%mem_col0, {%shim_col0}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r1_c0] -> [@C2_r1_c0]([] [0])
    aie.objectfifo @C2_r1_c1(%mem_col1, {%shim_col1}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r1_c1] -> [@C2_r1_c1]([] [0])
    aie.objectfifo @C2_r1_c2(%mem_col2, {%shim_col2}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r1_c2] -> [@C2_r1_c2]([] [0])
    aie.objectfifo @C2_r1_c3(%mem_col3, {%shim_col3}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@C_r1_c3] -> [@C2_r1_c3]([] [0])
    func.func private @"int8_pure_mmul_v2"(memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) attributes {link_with = "int8_pure_mmul_v2.o"}
    %core_r0_c0 = aie.core(%tile_core_r0_c0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r0_c0(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r0(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c0(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r0(Consume, 1)
          aie.objectfifo.release @B2_c0(Consume, 1)
        }
        aie.objectfifo.release @C_r0_c0(Produce, 1)
      }
      aie.end
    }
    %core_r0_c1 = aie.core(%tile_core_r0_c1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r0_c1(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r0(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c1(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r0(Consume, 1)
          aie.objectfifo.release @B2_c1(Consume, 1)
        }
        aie.objectfifo.release @C_r0_c1(Produce, 1)
      }
      aie.end
    }
    %core_r0_c2 = aie.core(%tile_core_r0_c2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r0_c2(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r0(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c2(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r0(Consume, 1)
          aie.objectfifo.release @B2_c2(Consume, 1)
        }
        aie.objectfifo.release @C_r0_c2(Produce, 1)
      }
      aie.end
    }
    %core_r0_c3 = aie.core(%tile_core_r0_c3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r0_c3(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r0(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c3(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r0(Consume, 1)
          aie.objectfifo.release @B2_c3(Consume, 1)
        }
        aie.objectfifo.release @C_r0_c3(Produce, 1)
      }
      aie.end
    }
    %core_r1_c0 = aie.core(%tile_core_r1_c0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r1_c0(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r1(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c0(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r1(Consume, 1)
          aie.objectfifo.release @B2_c0(Consume, 1)
        }
        aie.objectfifo.release @C_r1_c0(Produce, 1)
      }
      aie.end
    }
    %core_r1_c1 = aie.core(%tile_core_r1_c1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r1_c1(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r1(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c1(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r1(Consume, 1)
          aie.objectfifo.release @B2_c1(Consume, 1)
        }
        aie.objectfifo.release @C_r1_c1(Produce, 1)
      }
      aie.end
    }
    %core_r1_c2 = aie.core(%tile_core_r1_c2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r1_c2(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r1(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c2(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r1(Consume, 1)
          aie.objectfifo.release @B2_c2(Consume, 1)
        }
        aie.objectfifo.release @C_r1_c2(Produce, 1)
      }
      aie.end
    }
    %core_r1_c3 = aie.core(%tile_core_r1_c3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cnk = arith.constant 4 : index
      %ctiles = arith.constant 112 : index
      scf.for %t = %c0 to %ctiles step %c1 {
        %cout = aie.objectfifo.acquire @C_r1_c3(Produce, 1) : memref<2048xi32>
        %zc = arith.constant 0 : i32
        %csz = arith.constant 2048 : index
        scf.for %i = %c0 to %csz step %c1 { memref.store %zc, %cout[%i] : memref<2048xi32> }
        %ck = arith.constant 0 : index
        scf.for %k = %ck to %cnk step %c1 {
          %ain = aie.objectfifo.acquire @A2_r1(Consume, 1) : memref<8192xi8>
          %bin = aie.objectfifo.acquire @B2_c3(Consume, 1) : memref<16384xi8>
          func.call @int8_pure_mmul_v2(%ain, %bin, %cout) : (memref<8192xi8>, memref<16384xi8>, memref<2048xi32>) -> ()
          aie.objectfifo.release @A2_r1(Consume, 1)
          aie.objectfifo.release @B2_c3(Consume, 1)
        }
        aie.objectfifo.release @C_r1_c3(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%A: memref<512x1024xi8>, %B: memref<1024x3584xi8>, %Cout: memref<512x3584xi32>) {
      // core r0c0 m0 n0
      %a_r0c0_m0_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n0)
      %b_r0c0_m0_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n0)
      %c_r0c0_m0_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 0 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n0)
      aiex.dma_await_task(%c_r0c0_m0_n0)
      aiex.dma_free_task(%a_r0c0_m0_n0)
      aiex.dma_free_task(%b_r0c0_m0_n0)
      aiex.dma_free_task(%c_r0c0_m0_n0)
      // core r0c0 m0 n1
      %a_r0c0_m0_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n1)
      %b_r0c0_m0_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n1)
      %c_r0c0_m0_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 64 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n1)
      aiex.dma_await_task(%c_r0c0_m0_n1)
      aiex.dma_free_task(%a_r0c0_m0_n1)
      aiex.dma_free_task(%b_r0c0_m0_n1)
      aiex.dma_free_task(%c_r0c0_m0_n1)
      // core r0c0 m0 n2
      %a_r0c0_m0_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n2)
      %b_r0c0_m0_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n2)
      %c_r0c0_m0_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n2)
      aiex.dma_await_task(%c_r0c0_m0_n2)
      aiex.dma_free_task(%a_r0c0_m0_n2)
      aiex.dma_free_task(%b_r0c0_m0_n2)
      aiex.dma_free_task(%c_r0c0_m0_n2)
      // core r0c0 m0 n3
      %a_r0c0_m0_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n3)
      %b_r0c0_m0_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n3)
      %c_r0c0_m0_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n3)
      aiex.dma_await_task(%c_r0c0_m0_n3)
      aiex.dma_free_task(%a_r0c0_m0_n3)
      aiex.dma_free_task(%b_r0c0_m0_n3)
      aiex.dma_free_task(%c_r0c0_m0_n3)
      // core r0c0 m0 n4
      %a_r0c0_m0_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n4)
      %b_r0c0_m0_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n4)
      %c_r0c0_m0_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n4)
      aiex.dma_await_task(%c_r0c0_m0_n4)
      aiex.dma_free_task(%a_r0c0_m0_n4)
      aiex.dma_free_task(%b_r0c0_m0_n4)
      aiex.dma_free_task(%c_r0c0_m0_n4)
      // core r0c0 m0 n5
      %a_r0c0_m0_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n5)
      %b_r0c0_m0_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n5)
      %c_r0c0_m0_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n5)
      aiex.dma_await_task(%c_r0c0_m0_n5)
      aiex.dma_free_task(%a_r0c0_m0_n5)
      aiex.dma_free_task(%b_r0c0_m0_n5)
      aiex.dma_free_task(%c_r0c0_m0_n5)
      // core r0c0 m0 n6
      %a_r0c0_m0_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n6)
      %b_r0c0_m0_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n6)
      %c_r0c0_m0_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n6)
      aiex.dma_await_task(%c_r0c0_m0_n6)
      aiex.dma_free_task(%a_r0c0_m0_n6)
      aiex.dma_free_task(%b_r0c0_m0_n6)
      aiex.dma_free_task(%c_r0c0_m0_n6)
      // core r0c0 m0 n7
      %a_r0c0_m0_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n7)
      %b_r0c0_m0_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n7)
      %c_r0c0_m0_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n7)
      aiex.dma_await_task(%c_r0c0_m0_n7)
      aiex.dma_free_task(%a_r0c0_m0_n7)
      aiex.dma_free_task(%b_r0c0_m0_n7)
      aiex.dma_free_task(%c_r0c0_m0_n7)
      // core r0c0 m0 n8
      %a_r0c0_m0_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n8)
      %b_r0c0_m0_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n8)
      %c_r0c0_m0_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n8)
      aiex.dma_await_task(%c_r0c0_m0_n8)
      aiex.dma_free_task(%a_r0c0_m0_n8)
      aiex.dma_free_task(%b_r0c0_m0_n8)
      aiex.dma_free_task(%c_r0c0_m0_n8)
      // core r0c0 m0 n9
      %a_r0c0_m0_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n9)
      %b_r0c0_m0_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n9)
      %c_r0c0_m0_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n9)
      aiex.dma_await_task(%c_r0c0_m0_n9)
      aiex.dma_free_task(%a_r0c0_m0_n9)
      aiex.dma_free_task(%b_r0c0_m0_n9)
      aiex.dma_free_task(%c_r0c0_m0_n9)
      // core r0c0 m0 n10
      %a_r0c0_m0_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n10)
      %b_r0c0_m0_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n10)
      %c_r0c0_m0_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n10)
      aiex.dma_await_task(%c_r0c0_m0_n10)
      aiex.dma_free_task(%a_r0c0_m0_n10)
      aiex.dma_free_task(%b_r0c0_m0_n10)
      aiex.dma_free_task(%c_r0c0_m0_n10)
      // core r0c0 m0 n11
      %a_r0c0_m0_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n11)
      %b_r0c0_m0_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n11)
      %c_r0c0_m0_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n11)
      aiex.dma_await_task(%c_r0c0_m0_n11)
      aiex.dma_free_task(%a_r0c0_m0_n11)
      aiex.dma_free_task(%b_r0c0_m0_n11)
      aiex.dma_free_task(%c_r0c0_m0_n11)
      // core r0c0 m0 n12
      %a_r0c0_m0_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n12)
      %b_r0c0_m0_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n12)
      %c_r0c0_m0_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n12)
      aiex.dma_await_task(%c_r0c0_m0_n12)
      aiex.dma_free_task(%a_r0c0_m0_n12)
      aiex.dma_free_task(%b_r0c0_m0_n12)
      aiex.dma_free_task(%c_r0c0_m0_n12)
      // core r0c0 m0 n13
      %a_r0c0_m0_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m0_n13)
      %b_r0c0_m0_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m0_n13)
      %c_r0c0_m0_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m0_n13)
      aiex.dma_await_task(%c_r0c0_m0_n13)
      aiex.dma_free_task(%a_r0c0_m0_n13)
      aiex.dma_free_task(%b_r0c0_m0_n13)
      aiex.dma_free_task(%c_r0c0_m0_n13)
      // core r0c0 m1 n0
      %a_r0c0_m1_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n0)
      %b_r0c0_m1_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n0)
      %c_r0c0_m1_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 114688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n0)
      aiex.dma_await_task(%c_r0c0_m1_n0)
      aiex.dma_free_task(%a_r0c0_m1_n0)
      aiex.dma_free_task(%b_r0c0_m1_n0)
      aiex.dma_free_task(%c_r0c0_m1_n0)
      // core r0c0 m1 n1
      %a_r0c0_m1_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n1)
      %b_r0c0_m1_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n1)
      %c_r0c0_m1_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 114752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n1)
      aiex.dma_await_task(%c_r0c0_m1_n1)
      aiex.dma_free_task(%a_r0c0_m1_n1)
      aiex.dma_free_task(%b_r0c0_m1_n1)
      aiex.dma_free_task(%c_r0c0_m1_n1)
      // core r0c0 m1 n2
      %a_r0c0_m1_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n2)
      %b_r0c0_m1_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n2)
      %c_r0c0_m1_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 114816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n2)
      aiex.dma_await_task(%c_r0c0_m1_n2)
      aiex.dma_free_task(%a_r0c0_m1_n2)
      aiex.dma_free_task(%b_r0c0_m1_n2)
      aiex.dma_free_task(%c_r0c0_m1_n2)
      // core r0c0 m1 n3
      %a_r0c0_m1_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n3)
      %b_r0c0_m1_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n3)
      %c_r0c0_m1_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 114880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n3)
      aiex.dma_await_task(%c_r0c0_m1_n3)
      aiex.dma_free_task(%a_r0c0_m1_n3)
      aiex.dma_free_task(%b_r0c0_m1_n3)
      aiex.dma_free_task(%c_r0c0_m1_n3)
      // core r0c0 m1 n4
      %a_r0c0_m1_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n4)
      %b_r0c0_m1_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n4)
      %c_r0c0_m1_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 114944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n4)
      aiex.dma_await_task(%c_r0c0_m1_n4)
      aiex.dma_free_task(%a_r0c0_m1_n4)
      aiex.dma_free_task(%b_r0c0_m1_n4)
      aiex.dma_free_task(%c_r0c0_m1_n4)
      // core r0c0 m1 n5
      %a_r0c0_m1_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n5)
      %b_r0c0_m1_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n5)
      %c_r0c0_m1_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n5)
      aiex.dma_await_task(%c_r0c0_m1_n5)
      aiex.dma_free_task(%a_r0c0_m1_n5)
      aiex.dma_free_task(%b_r0c0_m1_n5)
      aiex.dma_free_task(%c_r0c0_m1_n5)
      // core r0c0 m1 n6
      %a_r0c0_m1_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n6)
      %b_r0c0_m1_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n6)
      %c_r0c0_m1_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n6)
      aiex.dma_await_task(%c_r0c0_m1_n6)
      aiex.dma_free_task(%a_r0c0_m1_n6)
      aiex.dma_free_task(%b_r0c0_m1_n6)
      aiex.dma_free_task(%c_r0c0_m1_n6)
      // core r0c0 m1 n7
      %a_r0c0_m1_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n7)
      %b_r0c0_m1_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n7)
      %c_r0c0_m1_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n7)
      aiex.dma_await_task(%c_r0c0_m1_n7)
      aiex.dma_free_task(%a_r0c0_m1_n7)
      aiex.dma_free_task(%b_r0c0_m1_n7)
      aiex.dma_free_task(%c_r0c0_m1_n7)
      // core r0c0 m1 n8
      %a_r0c0_m1_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n8)
      %b_r0c0_m1_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n8)
      %c_r0c0_m1_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n8)
      aiex.dma_await_task(%c_r0c0_m1_n8)
      aiex.dma_free_task(%a_r0c0_m1_n8)
      aiex.dma_free_task(%b_r0c0_m1_n8)
      aiex.dma_free_task(%c_r0c0_m1_n8)
      // core r0c0 m1 n9
      %a_r0c0_m1_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n9)
      %b_r0c0_m1_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n9)
      %c_r0c0_m1_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n9)
      aiex.dma_await_task(%c_r0c0_m1_n9)
      aiex.dma_free_task(%a_r0c0_m1_n9)
      aiex.dma_free_task(%b_r0c0_m1_n9)
      aiex.dma_free_task(%c_r0c0_m1_n9)
      // core r0c0 m1 n10
      %a_r0c0_m1_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n10)
      %b_r0c0_m1_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n10)
      %c_r0c0_m1_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n10)
      aiex.dma_await_task(%c_r0c0_m1_n10)
      aiex.dma_free_task(%a_r0c0_m1_n10)
      aiex.dma_free_task(%b_r0c0_m1_n10)
      aiex.dma_free_task(%c_r0c0_m1_n10)
      // core r0c0 m1 n11
      %a_r0c0_m1_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n11)
      %b_r0c0_m1_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n11)
      %c_r0c0_m1_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n11)
      aiex.dma_await_task(%c_r0c0_m1_n11)
      aiex.dma_free_task(%a_r0c0_m1_n11)
      aiex.dma_free_task(%b_r0c0_m1_n11)
      aiex.dma_free_task(%c_r0c0_m1_n11)
      // core r0c0 m1 n12
      %a_r0c0_m1_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n12)
      %b_r0c0_m1_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n12)
      %c_r0c0_m1_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n12)
      aiex.dma_await_task(%c_r0c0_m1_n12)
      aiex.dma_free_task(%a_r0c0_m1_n12)
      aiex.dma_free_task(%b_r0c0_m1_n12)
      aiex.dma_free_task(%c_r0c0_m1_n12)
      // core r0c0 m1 n13
      %a_r0c0_m1_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m1_n13)
      %b_r0c0_m1_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m1_n13)
      %c_r0c0_m1_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m1_n13)
      aiex.dma_await_task(%c_r0c0_m1_n13)
      aiex.dma_free_task(%a_r0c0_m1_n13)
      aiex.dma_free_task(%b_r0c0_m1_n13)
      aiex.dma_free_task(%c_r0c0_m1_n13)
      // core r0c0 m2 n0
      %a_r0c0_m2_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n0)
      %b_r0c0_m2_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n0)
      %c_r0c0_m2_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229376 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n0)
      aiex.dma_await_task(%c_r0c0_m2_n0)
      aiex.dma_free_task(%a_r0c0_m2_n0)
      aiex.dma_free_task(%b_r0c0_m2_n0)
      aiex.dma_free_task(%c_r0c0_m2_n0)
      // core r0c0 m2 n1
      %a_r0c0_m2_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n1)
      %b_r0c0_m2_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n1)
      %c_r0c0_m2_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n1)
      aiex.dma_await_task(%c_r0c0_m2_n1)
      aiex.dma_free_task(%a_r0c0_m2_n1)
      aiex.dma_free_task(%b_r0c0_m2_n1)
      aiex.dma_free_task(%c_r0c0_m2_n1)
      // core r0c0 m2 n2
      %a_r0c0_m2_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n2)
      %b_r0c0_m2_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n2)
      %c_r0c0_m2_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n2)
      aiex.dma_await_task(%c_r0c0_m2_n2)
      aiex.dma_free_task(%a_r0c0_m2_n2)
      aiex.dma_free_task(%b_r0c0_m2_n2)
      aiex.dma_free_task(%c_r0c0_m2_n2)
      // core r0c0 m2 n3
      %a_r0c0_m2_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n3)
      %b_r0c0_m2_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n3)
      %c_r0c0_m2_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n3)
      aiex.dma_await_task(%c_r0c0_m2_n3)
      aiex.dma_free_task(%a_r0c0_m2_n3)
      aiex.dma_free_task(%b_r0c0_m2_n3)
      aiex.dma_free_task(%c_r0c0_m2_n3)
      // core r0c0 m2 n4
      %a_r0c0_m2_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n4)
      %b_r0c0_m2_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n4)
      %c_r0c0_m2_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n4)
      aiex.dma_await_task(%c_r0c0_m2_n4)
      aiex.dma_free_task(%a_r0c0_m2_n4)
      aiex.dma_free_task(%b_r0c0_m2_n4)
      aiex.dma_free_task(%c_r0c0_m2_n4)
      // core r0c0 m2 n5
      %a_r0c0_m2_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n5)
      %b_r0c0_m2_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n5)
      %c_r0c0_m2_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n5)
      aiex.dma_await_task(%c_r0c0_m2_n5)
      aiex.dma_free_task(%a_r0c0_m2_n5)
      aiex.dma_free_task(%b_r0c0_m2_n5)
      aiex.dma_free_task(%c_r0c0_m2_n5)
      // core r0c0 m2 n6
      %a_r0c0_m2_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n6)
      %b_r0c0_m2_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n6)
      %c_r0c0_m2_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n6)
      aiex.dma_await_task(%c_r0c0_m2_n6)
      aiex.dma_free_task(%a_r0c0_m2_n6)
      aiex.dma_free_task(%b_r0c0_m2_n6)
      aiex.dma_free_task(%c_r0c0_m2_n6)
      // core r0c0 m2 n7
      %a_r0c0_m2_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n7)
      %b_r0c0_m2_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n7)
      %c_r0c0_m2_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n7)
      aiex.dma_await_task(%c_r0c0_m2_n7)
      aiex.dma_free_task(%a_r0c0_m2_n7)
      aiex.dma_free_task(%b_r0c0_m2_n7)
      aiex.dma_free_task(%c_r0c0_m2_n7)
      // core r0c0 m2 n8
      %a_r0c0_m2_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n8)
      %b_r0c0_m2_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n8)
      %c_r0c0_m2_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n8)
      aiex.dma_await_task(%c_r0c0_m2_n8)
      aiex.dma_free_task(%a_r0c0_m2_n8)
      aiex.dma_free_task(%b_r0c0_m2_n8)
      aiex.dma_free_task(%c_r0c0_m2_n8)
      // core r0c0 m2 n9
      %a_r0c0_m2_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n9)
      %b_r0c0_m2_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n9)
      %c_r0c0_m2_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 229952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n9)
      aiex.dma_await_task(%c_r0c0_m2_n9)
      aiex.dma_free_task(%a_r0c0_m2_n9)
      aiex.dma_free_task(%b_r0c0_m2_n9)
      aiex.dma_free_task(%c_r0c0_m2_n9)
      // core r0c0 m2 n10
      %a_r0c0_m2_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n10)
      %b_r0c0_m2_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n10)
      %c_r0c0_m2_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n10)
      aiex.dma_await_task(%c_r0c0_m2_n10)
      aiex.dma_free_task(%a_r0c0_m2_n10)
      aiex.dma_free_task(%b_r0c0_m2_n10)
      aiex.dma_free_task(%c_r0c0_m2_n10)
      // core r0c0 m2 n11
      %a_r0c0_m2_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n11)
      %b_r0c0_m2_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n11)
      %c_r0c0_m2_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n11)
      aiex.dma_await_task(%c_r0c0_m2_n11)
      aiex.dma_free_task(%a_r0c0_m2_n11)
      aiex.dma_free_task(%b_r0c0_m2_n11)
      aiex.dma_free_task(%c_r0c0_m2_n11)
      // core r0c0 m2 n12
      %a_r0c0_m2_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n12)
      %b_r0c0_m2_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n12)
      %c_r0c0_m2_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n12)
      aiex.dma_await_task(%c_r0c0_m2_n12)
      aiex.dma_free_task(%a_r0c0_m2_n12)
      aiex.dma_free_task(%b_r0c0_m2_n12)
      aiex.dma_free_task(%c_r0c0_m2_n12)
      // core r0c0 m2 n13
      %a_r0c0_m2_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m2_n13)
      %b_r0c0_m2_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m2_n13)
      %c_r0c0_m2_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m2_n13)
      aiex.dma_await_task(%c_r0c0_m2_n13)
      aiex.dma_free_task(%a_r0c0_m2_n13)
      aiex.dma_free_task(%b_r0c0_m2_n13)
      aiex.dma_free_task(%c_r0c0_m2_n13)
      // core r0c0 m3 n0
      %a_r0c0_m3_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n0)
      %b_r0c0_m3_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n0)
      %c_r0c0_m3_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344064 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n0)
      aiex.dma_await_task(%c_r0c0_m3_n0)
      aiex.dma_free_task(%a_r0c0_m3_n0)
      aiex.dma_free_task(%b_r0c0_m3_n0)
      aiex.dma_free_task(%c_r0c0_m3_n0)
      // core r0c0 m3 n1
      %a_r0c0_m3_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n1)
      %b_r0c0_m3_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n1)
      %c_r0c0_m3_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n1)
      aiex.dma_await_task(%c_r0c0_m3_n1)
      aiex.dma_free_task(%a_r0c0_m3_n1)
      aiex.dma_free_task(%b_r0c0_m3_n1)
      aiex.dma_free_task(%c_r0c0_m3_n1)
      // core r0c0 m3 n2
      %a_r0c0_m3_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n2)
      %b_r0c0_m3_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n2)
      %c_r0c0_m3_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n2)
      aiex.dma_await_task(%c_r0c0_m3_n2)
      aiex.dma_free_task(%a_r0c0_m3_n2)
      aiex.dma_free_task(%b_r0c0_m3_n2)
      aiex.dma_free_task(%c_r0c0_m3_n2)
      // core r0c0 m3 n3
      %a_r0c0_m3_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n3)
      %b_r0c0_m3_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n3)
      %c_r0c0_m3_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n3)
      aiex.dma_await_task(%c_r0c0_m3_n3)
      aiex.dma_free_task(%a_r0c0_m3_n3)
      aiex.dma_free_task(%b_r0c0_m3_n3)
      aiex.dma_free_task(%c_r0c0_m3_n3)
      // core r0c0 m3 n4
      %a_r0c0_m3_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n4)
      %b_r0c0_m3_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n4)
      %c_r0c0_m3_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n4)
      aiex.dma_await_task(%c_r0c0_m3_n4)
      aiex.dma_free_task(%a_r0c0_m3_n4)
      aiex.dma_free_task(%b_r0c0_m3_n4)
      aiex.dma_free_task(%c_r0c0_m3_n4)
      // core r0c0 m3 n5
      %a_r0c0_m3_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n5)
      %b_r0c0_m3_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n5)
      %c_r0c0_m3_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n5)
      aiex.dma_await_task(%c_r0c0_m3_n5)
      aiex.dma_free_task(%a_r0c0_m3_n5)
      aiex.dma_free_task(%b_r0c0_m3_n5)
      aiex.dma_free_task(%c_r0c0_m3_n5)
      // core r0c0 m3 n6
      %a_r0c0_m3_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n6)
      %b_r0c0_m3_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n6)
      %c_r0c0_m3_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n6)
      aiex.dma_await_task(%c_r0c0_m3_n6)
      aiex.dma_free_task(%a_r0c0_m3_n6)
      aiex.dma_free_task(%b_r0c0_m3_n6)
      aiex.dma_free_task(%c_r0c0_m3_n6)
      // core r0c0 m3 n7
      %a_r0c0_m3_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n7)
      %b_r0c0_m3_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n7)
      %c_r0c0_m3_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n7)
      aiex.dma_await_task(%c_r0c0_m3_n7)
      aiex.dma_free_task(%a_r0c0_m3_n7)
      aiex.dma_free_task(%b_r0c0_m3_n7)
      aiex.dma_free_task(%c_r0c0_m3_n7)
      // core r0c0 m3 n8
      %a_r0c0_m3_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n8)
      %b_r0c0_m3_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n8)
      %c_r0c0_m3_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n8)
      aiex.dma_await_task(%c_r0c0_m3_n8)
      aiex.dma_free_task(%a_r0c0_m3_n8)
      aiex.dma_free_task(%b_r0c0_m3_n8)
      aiex.dma_free_task(%c_r0c0_m3_n8)
      // core r0c0 m3 n9
      %a_r0c0_m3_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n9)
      %b_r0c0_m3_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n9)
      %c_r0c0_m3_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n9)
      aiex.dma_await_task(%c_r0c0_m3_n9)
      aiex.dma_free_task(%a_r0c0_m3_n9)
      aiex.dma_free_task(%b_r0c0_m3_n9)
      aiex.dma_free_task(%c_r0c0_m3_n9)
      // core r0c0 m3 n10
      %a_r0c0_m3_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n10)
      %b_r0c0_m3_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n10)
      %c_r0c0_m3_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n10)
      aiex.dma_await_task(%c_r0c0_m3_n10)
      aiex.dma_free_task(%a_r0c0_m3_n10)
      aiex.dma_free_task(%b_r0c0_m3_n10)
      aiex.dma_free_task(%c_r0c0_m3_n10)
      // core r0c0 m3 n11
      %a_r0c0_m3_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n11)
      %b_r0c0_m3_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n11)
      %c_r0c0_m3_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n11)
      aiex.dma_await_task(%c_r0c0_m3_n11)
      aiex.dma_free_task(%a_r0c0_m3_n11)
      aiex.dma_free_task(%b_r0c0_m3_n11)
      aiex.dma_free_task(%c_r0c0_m3_n11)
      // core r0c0 m3 n12
      %a_r0c0_m3_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n12)
      %b_r0c0_m3_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n12)
      %c_r0c0_m3_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n12)
      aiex.dma_await_task(%c_r0c0_m3_n12)
      aiex.dma_free_task(%a_r0c0_m3_n12)
      aiex.dma_free_task(%b_r0c0_m3_n12)
      aiex.dma_free_task(%c_r0c0_m3_n12)
      // core r0c0 m3 n13
      %a_r0c0_m3_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m3_n13)
      %b_r0c0_m3_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m3_n13)
      %c_r0c0_m3_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m3_n13)
      aiex.dma_await_task(%c_r0c0_m3_n13)
      aiex.dma_free_task(%a_r0c0_m3_n13)
      aiex.dma_free_task(%b_r0c0_m3_n13)
      aiex.dma_free_task(%c_r0c0_m3_n13)
      // core r0c0 m4 n0
      %a_r0c0_m4_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n0)
      %b_r0c0_m4_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n0)
      %c_r0c0_m4_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 458752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n0)
      aiex.dma_await_task(%c_r0c0_m4_n0)
      aiex.dma_free_task(%a_r0c0_m4_n0)
      aiex.dma_free_task(%b_r0c0_m4_n0)
      aiex.dma_free_task(%c_r0c0_m4_n0)
      // core r0c0 m4 n1
      %a_r0c0_m4_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n1)
      %b_r0c0_m4_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n1)
      %c_r0c0_m4_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 458816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n1)
      aiex.dma_await_task(%c_r0c0_m4_n1)
      aiex.dma_free_task(%a_r0c0_m4_n1)
      aiex.dma_free_task(%b_r0c0_m4_n1)
      aiex.dma_free_task(%c_r0c0_m4_n1)
      // core r0c0 m4 n2
      %a_r0c0_m4_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n2)
      %b_r0c0_m4_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n2)
      %c_r0c0_m4_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 458880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n2)
      aiex.dma_await_task(%c_r0c0_m4_n2)
      aiex.dma_free_task(%a_r0c0_m4_n2)
      aiex.dma_free_task(%b_r0c0_m4_n2)
      aiex.dma_free_task(%c_r0c0_m4_n2)
      // core r0c0 m4 n3
      %a_r0c0_m4_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n3)
      %b_r0c0_m4_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n3)
      %c_r0c0_m4_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 458944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n3)
      aiex.dma_await_task(%c_r0c0_m4_n3)
      aiex.dma_free_task(%a_r0c0_m4_n3)
      aiex.dma_free_task(%b_r0c0_m4_n3)
      aiex.dma_free_task(%c_r0c0_m4_n3)
      // core r0c0 m4 n4
      %a_r0c0_m4_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n4)
      %b_r0c0_m4_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n4)
      %c_r0c0_m4_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n4)
      aiex.dma_await_task(%c_r0c0_m4_n4)
      aiex.dma_free_task(%a_r0c0_m4_n4)
      aiex.dma_free_task(%b_r0c0_m4_n4)
      aiex.dma_free_task(%c_r0c0_m4_n4)
      // core r0c0 m4 n5
      %a_r0c0_m4_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n5)
      %b_r0c0_m4_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n5)
      %c_r0c0_m4_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n5)
      aiex.dma_await_task(%c_r0c0_m4_n5)
      aiex.dma_free_task(%a_r0c0_m4_n5)
      aiex.dma_free_task(%b_r0c0_m4_n5)
      aiex.dma_free_task(%c_r0c0_m4_n5)
      // core r0c0 m4 n6
      %a_r0c0_m4_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n6)
      %b_r0c0_m4_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n6)
      %c_r0c0_m4_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n6)
      aiex.dma_await_task(%c_r0c0_m4_n6)
      aiex.dma_free_task(%a_r0c0_m4_n6)
      aiex.dma_free_task(%b_r0c0_m4_n6)
      aiex.dma_free_task(%c_r0c0_m4_n6)
      // core r0c0 m4 n7
      %a_r0c0_m4_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n7)
      %b_r0c0_m4_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n7)
      %c_r0c0_m4_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n7)
      aiex.dma_await_task(%c_r0c0_m4_n7)
      aiex.dma_free_task(%a_r0c0_m4_n7)
      aiex.dma_free_task(%b_r0c0_m4_n7)
      aiex.dma_free_task(%c_r0c0_m4_n7)
      // core r0c0 m4 n8
      %a_r0c0_m4_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n8)
      %b_r0c0_m4_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n8)
      %c_r0c0_m4_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n8)
      aiex.dma_await_task(%c_r0c0_m4_n8)
      aiex.dma_free_task(%a_r0c0_m4_n8)
      aiex.dma_free_task(%b_r0c0_m4_n8)
      aiex.dma_free_task(%c_r0c0_m4_n8)
      // core r0c0 m4 n9
      %a_r0c0_m4_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n9)
      %b_r0c0_m4_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n9)
      %c_r0c0_m4_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n9)
      aiex.dma_await_task(%c_r0c0_m4_n9)
      aiex.dma_free_task(%a_r0c0_m4_n9)
      aiex.dma_free_task(%b_r0c0_m4_n9)
      aiex.dma_free_task(%c_r0c0_m4_n9)
      // core r0c0 m4 n10
      %a_r0c0_m4_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n10)
      %b_r0c0_m4_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n10)
      %c_r0c0_m4_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n10)
      aiex.dma_await_task(%c_r0c0_m4_n10)
      aiex.dma_free_task(%a_r0c0_m4_n10)
      aiex.dma_free_task(%b_r0c0_m4_n10)
      aiex.dma_free_task(%c_r0c0_m4_n10)
      // core r0c0 m4 n11
      %a_r0c0_m4_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n11)
      %b_r0c0_m4_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n11)
      %c_r0c0_m4_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n11)
      aiex.dma_await_task(%c_r0c0_m4_n11)
      aiex.dma_free_task(%a_r0c0_m4_n11)
      aiex.dma_free_task(%b_r0c0_m4_n11)
      aiex.dma_free_task(%c_r0c0_m4_n11)
      // core r0c0 m4 n12
      %a_r0c0_m4_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n12)
      %b_r0c0_m4_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n12)
      %c_r0c0_m4_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n12)
      aiex.dma_await_task(%c_r0c0_m4_n12)
      aiex.dma_free_task(%a_r0c0_m4_n12)
      aiex.dma_free_task(%b_r0c0_m4_n12)
      aiex.dma_free_task(%c_r0c0_m4_n12)
      // core r0c0 m4 n13
      %a_r0c0_m4_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m4_n13)
      %b_r0c0_m4_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m4_n13)
      %c_r0c0_m4_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m4_n13)
      aiex.dma_await_task(%c_r0c0_m4_n13)
      aiex.dma_free_task(%a_r0c0_m4_n13)
      aiex.dma_free_task(%b_r0c0_m4_n13)
      aiex.dma_free_task(%c_r0c0_m4_n13)
      // core r0c0 m5 n0
      %a_r0c0_m5_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n0)
      %b_r0c0_m5_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n0)
      %c_r0c0_m5_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n0)
      aiex.dma_await_task(%c_r0c0_m5_n0)
      aiex.dma_free_task(%a_r0c0_m5_n0)
      aiex.dma_free_task(%b_r0c0_m5_n0)
      aiex.dma_free_task(%c_r0c0_m5_n0)
      // core r0c0 m5 n1
      %a_r0c0_m5_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n1)
      %b_r0c0_m5_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n1)
      %c_r0c0_m5_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n1)
      aiex.dma_await_task(%c_r0c0_m5_n1)
      aiex.dma_free_task(%a_r0c0_m5_n1)
      aiex.dma_free_task(%b_r0c0_m5_n1)
      aiex.dma_free_task(%c_r0c0_m5_n1)
      // core r0c0 m5 n2
      %a_r0c0_m5_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n2)
      %b_r0c0_m5_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n2)
      %c_r0c0_m5_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n2)
      aiex.dma_await_task(%c_r0c0_m5_n2)
      aiex.dma_free_task(%a_r0c0_m5_n2)
      aiex.dma_free_task(%b_r0c0_m5_n2)
      aiex.dma_free_task(%c_r0c0_m5_n2)
      // core r0c0 m5 n3
      %a_r0c0_m5_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n3)
      %b_r0c0_m5_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n3)
      %c_r0c0_m5_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n3)
      aiex.dma_await_task(%c_r0c0_m5_n3)
      aiex.dma_free_task(%a_r0c0_m5_n3)
      aiex.dma_free_task(%b_r0c0_m5_n3)
      aiex.dma_free_task(%c_r0c0_m5_n3)
      // core r0c0 m5 n4
      %a_r0c0_m5_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n4)
      %b_r0c0_m5_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n4)
      %c_r0c0_m5_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n4)
      aiex.dma_await_task(%c_r0c0_m5_n4)
      aiex.dma_free_task(%a_r0c0_m5_n4)
      aiex.dma_free_task(%b_r0c0_m5_n4)
      aiex.dma_free_task(%c_r0c0_m5_n4)
      // core r0c0 m5 n5
      %a_r0c0_m5_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n5)
      %b_r0c0_m5_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n5)
      %c_r0c0_m5_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n5)
      aiex.dma_await_task(%c_r0c0_m5_n5)
      aiex.dma_free_task(%a_r0c0_m5_n5)
      aiex.dma_free_task(%b_r0c0_m5_n5)
      aiex.dma_free_task(%c_r0c0_m5_n5)
      // core r0c0 m5 n6
      %a_r0c0_m5_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n6)
      %b_r0c0_m5_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n6)
      %c_r0c0_m5_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n6)
      aiex.dma_await_task(%c_r0c0_m5_n6)
      aiex.dma_free_task(%a_r0c0_m5_n6)
      aiex.dma_free_task(%b_r0c0_m5_n6)
      aiex.dma_free_task(%c_r0c0_m5_n6)
      // core r0c0 m5 n7
      %a_r0c0_m5_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n7)
      %b_r0c0_m5_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n7)
      %c_r0c0_m5_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n7)
      aiex.dma_await_task(%c_r0c0_m5_n7)
      aiex.dma_free_task(%a_r0c0_m5_n7)
      aiex.dma_free_task(%b_r0c0_m5_n7)
      aiex.dma_free_task(%c_r0c0_m5_n7)
      // core r0c0 m5 n8
      %a_r0c0_m5_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n8)
      %b_r0c0_m5_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n8)
      %c_r0c0_m5_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 573952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n8)
      aiex.dma_await_task(%c_r0c0_m5_n8)
      aiex.dma_free_task(%a_r0c0_m5_n8)
      aiex.dma_free_task(%b_r0c0_m5_n8)
      aiex.dma_free_task(%c_r0c0_m5_n8)
      // core r0c0 m5 n9
      %a_r0c0_m5_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n9)
      %b_r0c0_m5_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n9)
      %c_r0c0_m5_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n9)
      aiex.dma_await_task(%c_r0c0_m5_n9)
      aiex.dma_free_task(%a_r0c0_m5_n9)
      aiex.dma_free_task(%b_r0c0_m5_n9)
      aiex.dma_free_task(%c_r0c0_m5_n9)
      // core r0c0 m5 n10
      %a_r0c0_m5_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n10)
      %b_r0c0_m5_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n10)
      %c_r0c0_m5_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n10)
      aiex.dma_await_task(%c_r0c0_m5_n10)
      aiex.dma_free_task(%a_r0c0_m5_n10)
      aiex.dma_free_task(%b_r0c0_m5_n10)
      aiex.dma_free_task(%c_r0c0_m5_n10)
      // core r0c0 m5 n11
      %a_r0c0_m5_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n11)
      %b_r0c0_m5_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n11)
      %c_r0c0_m5_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n11)
      aiex.dma_await_task(%c_r0c0_m5_n11)
      aiex.dma_free_task(%a_r0c0_m5_n11)
      aiex.dma_free_task(%b_r0c0_m5_n11)
      aiex.dma_free_task(%c_r0c0_m5_n11)
      // core r0c0 m5 n12
      %a_r0c0_m5_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n12)
      %b_r0c0_m5_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n12)
      %c_r0c0_m5_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n12)
      aiex.dma_await_task(%c_r0c0_m5_n12)
      aiex.dma_free_task(%a_r0c0_m5_n12)
      aiex.dma_free_task(%b_r0c0_m5_n12)
      aiex.dma_free_task(%c_r0c0_m5_n12)
      // core r0c0 m5 n13
      %a_r0c0_m5_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m5_n13)
      %b_r0c0_m5_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m5_n13)
      %c_r0c0_m5_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m5_n13)
      aiex.dma_await_task(%c_r0c0_m5_n13)
      aiex.dma_free_task(%a_r0c0_m5_n13)
      aiex.dma_free_task(%b_r0c0_m5_n13)
      aiex.dma_free_task(%c_r0c0_m5_n13)
      // core r0c0 m6 n0
      %a_r0c0_m6_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n0)
      %b_r0c0_m6_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n0)
      %c_r0c0_m6_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n0)
      aiex.dma_await_task(%c_r0c0_m6_n0)
      aiex.dma_free_task(%a_r0c0_m6_n0)
      aiex.dma_free_task(%b_r0c0_m6_n0)
      aiex.dma_free_task(%c_r0c0_m6_n0)
      // core r0c0 m6 n1
      %a_r0c0_m6_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n1)
      %b_r0c0_m6_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n1)
      %c_r0c0_m6_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n1)
      aiex.dma_await_task(%c_r0c0_m6_n1)
      aiex.dma_free_task(%a_r0c0_m6_n1)
      aiex.dma_free_task(%b_r0c0_m6_n1)
      aiex.dma_free_task(%c_r0c0_m6_n1)
      // core r0c0 m6 n2
      %a_r0c0_m6_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n2)
      %b_r0c0_m6_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n2)
      %c_r0c0_m6_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n2)
      aiex.dma_await_task(%c_r0c0_m6_n2)
      aiex.dma_free_task(%a_r0c0_m6_n2)
      aiex.dma_free_task(%b_r0c0_m6_n2)
      aiex.dma_free_task(%c_r0c0_m6_n2)
      // core r0c0 m6 n3
      %a_r0c0_m6_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n3)
      %b_r0c0_m6_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n3)
      %c_r0c0_m6_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n3)
      aiex.dma_await_task(%c_r0c0_m6_n3)
      aiex.dma_free_task(%a_r0c0_m6_n3)
      aiex.dma_free_task(%b_r0c0_m6_n3)
      aiex.dma_free_task(%c_r0c0_m6_n3)
      // core r0c0 m6 n4
      %a_r0c0_m6_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n4)
      %b_r0c0_m6_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n4)
      %c_r0c0_m6_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n4)
      aiex.dma_await_task(%c_r0c0_m6_n4)
      aiex.dma_free_task(%a_r0c0_m6_n4)
      aiex.dma_free_task(%b_r0c0_m6_n4)
      aiex.dma_free_task(%c_r0c0_m6_n4)
      // core r0c0 m6 n5
      %a_r0c0_m6_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n5)
      %b_r0c0_m6_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n5)
      %c_r0c0_m6_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n5)
      aiex.dma_await_task(%c_r0c0_m6_n5)
      aiex.dma_free_task(%a_r0c0_m6_n5)
      aiex.dma_free_task(%b_r0c0_m6_n5)
      aiex.dma_free_task(%c_r0c0_m6_n5)
      // core r0c0 m6 n6
      %a_r0c0_m6_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n6)
      %b_r0c0_m6_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n6)
      %c_r0c0_m6_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n6)
      aiex.dma_await_task(%c_r0c0_m6_n6)
      aiex.dma_free_task(%a_r0c0_m6_n6)
      aiex.dma_free_task(%b_r0c0_m6_n6)
      aiex.dma_free_task(%c_r0c0_m6_n6)
      // core r0c0 m6 n7
      %a_r0c0_m6_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n7)
      %b_r0c0_m6_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n7)
      %c_r0c0_m6_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n7)
      aiex.dma_await_task(%c_r0c0_m6_n7)
      aiex.dma_free_task(%a_r0c0_m6_n7)
      aiex.dma_free_task(%b_r0c0_m6_n7)
      aiex.dma_free_task(%c_r0c0_m6_n7)
      // core r0c0 m6 n8
      %a_r0c0_m6_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n8)
      %b_r0c0_m6_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n8)
      %c_r0c0_m6_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n8)
      aiex.dma_await_task(%c_r0c0_m6_n8)
      aiex.dma_free_task(%a_r0c0_m6_n8)
      aiex.dma_free_task(%b_r0c0_m6_n8)
      aiex.dma_free_task(%c_r0c0_m6_n8)
      // core r0c0 m6 n9
      %a_r0c0_m6_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n9)
      %b_r0c0_m6_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n9)
      %c_r0c0_m6_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n9)
      aiex.dma_await_task(%c_r0c0_m6_n9)
      aiex.dma_free_task(%a_r0c0_m6_n9)
      aiex.dma_free_task(%b_r0c0_m6_n9)
      aiex.dma_free_task(%c_r0c0_m6_n9)
      // core r0c0 m6 n10
      %a_r0c0_m6_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n10)
      %b_r0c0_m6_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n10)
      %c_r0c0_m6_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n10)
      aiex.dma_await_task(%c_r0c0_m6_n10)
      aiex.dma_free_task(%a_r0c0_m6_n10)
      aiex.dma_free_task(%b_r0c0_m6_n10)
      aiex.dma_free_task(%c_r0c0_m6_n10)
      // core r0c0 m6 n11
      %a_r0c0_m6_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n11)
      %b_r0c0_m6_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n11)
      %c_r0c0_m6_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n11)
      aiex.dma_await_task(%c_r0c0_m6_n11)
      aiex.dma_free_task(%a_r0c0_m6_n11)
      aiex.dma_free_task(%b_r0c0_m6_n11)
      aiex.dma_free_task(%c_r0c0_m6_n11)
      // core r0c0 m6 n12
      %a_r0c0_m6_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n12)
      %b_r0c0_m6_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n12)
      %c_r0c0_m6_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n12)
      aiex.dma_await_task(%c_r0c0_m6_n12)
      aiex.dma_free_task(%a_r0c0_m6_n12)
      aiex.dma_free_task(%b_r0c0_m6_n12)
      aiex.dma_free_task(%c_r0c0_m6_n12)
      // core r0c0 m6 n13
      %a_r0c0_m6_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m6_n13)
      %b_r0c0_m6_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m6_n13)
      %c_r0c0_m6_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 688960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m6_n13)
      aiex.dma_await_task(%c_r0c0_m6_n13)
      aiex.dma_free_task(%a_r0c0_m6_n13)
      aiex.dma_free_task(%b_r0c0_m6_n13)
      aiex.dma_free_task(%c_r0c0_m6_n13)
      // core r0c0 m7 n0
      %a_r0c0_m7_n0 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n0)
      %b_r0c0_m7_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n0)
      %c_r0c0_m7_n0 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 802816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n0)
      aiex.dma_await_task(%c_r0c0_m7_n0)
      aiex.dma_free_task(%a_r0c0_m7_n0)
      aiex.dma_free_task(%b_r0c0_m7_n0)
      aiex.dma_free_task(%c_r0c0_m7_n0)
      // core r0c0 m7 n1
      %a_r0c0_m7_n1 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n1)
      %b_r0c0_m7_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n1)
      %c_r0c0_m7_n1 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 802880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n1)
      aiex.dma_await_task(%c_r0c0_m7_n1)
      aiex.dma_free_task(%a_r0c0_m7_n1)
      aiex.dma_free_task(%b_r0c0_m7_n1)
      aiex.dma_free_task(%c_r0c0_m7_n1)
      // core r0c0 m7 n2
      %a_r0c0_m7_n2 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n2)
      %b_r0c0_m7_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n2)
      %c_r0c0_m7_n2 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 802944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n2)
      aiex.dma_await_task(%c_r0c0_m7_n2)
      aiex.dma_free_task(%a_r0c0_m7_n2)
      aiex.dma_free_task(%b_r0c0_m7_n2)
      aiex.dma_free_task(%c_r0c0_m7_n2)
      // core r0c0 m7 n3
      %a_r0c0_m7_n3 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n3)
      %b_r0c0_m7_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n3)
      %c_r0c0_m7_n3 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n3)
      aiex.dma_await_task(%c_r0c0_m7_n3)
      aiex.dma_free_task(%a_r0c0_m7_n3)
      aiex.dma_free_task(%b_r0c0_m7_n3)
      aiex.dma_free_task(%c_r0c0_m7_n3)
      // core r0c0 m7 n4
      %a_r0c0_m7_n4 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n4)
      %b_r0c0_m7_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n4)
      %c_r0c0_m7_n4 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n4)
      aiex.dma_await_task(%c_r0c0_m7_n4)
      aiex.dma_free_task(%a_r0c0_m7_n4)
      aiex.dma_free_task(%b_r0c0_m7_n4)
      aiex.dma_free_task(%c_r0c0_m7_n4)
      // core r0c0 m7 n5
      %a_r0c0_m7_n5 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n5)
      %b_r0c0_m7_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n5)
      %c_r0c0_m7_n5 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n5)
      aiex.dma_await_task(%c_r0c0_m7_n5)
      aiex.dma_free_task(%a_r0c0_m7_n5)
      aiex.dma_free_task(%b_r0c0_m7_n5)
      aiex.dma_free_task(%c_r0c0_m7_n5)
      // core r0c0 m7 n6
      %a_r0c0_m7_n6 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n6)
      %b_r0c0_m7_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n6)
      %c_r0c0_m7_n6 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n6)
      aiex.dma_await_task(%c_r0c0_m7_n6)
      aiex.dma_free_task(%a_r0c0_m7_n6)
      aiex.dma_free_task(%b_r0c0_m7_n6)
      aiex.dma_free_task(%c_r0c0_m7_n6)
      // core r0c0 m7 n7
      %a_r0c0_m7_n7 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n7)
      %b_r0c0_m7_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n7)
      %c_r0c0_m7_n7 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n7)
      aiex.dma_await_task(%c_r0c0_m7_n7)
      aiex.dma_free_task(%a_r0c0_m7_n7)
      aiex.dma_free_task(%b_r0c0_m7_n7)
      aiex.dma_free_task(%c_r0c0_m7_n7)
      // core r0c0 m7 n8
      %a_r0c0_m7_n8 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n8)
      %b_r0c0_m7_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n8)
      %c_r0c0_m7_n8 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n8)
      aiex.dma_await_task(%c_r0c0_m7_n8)
      aiex.dma_free_task(%a_r0c0_m7_n8)
      aiex.dma_free_task(%b_r0c0_m7_n8)
      aiex.dma_free_task(%c_r0c0_m7_n8)
      // core r0c0 m7 n9
      %a_r0c0_m7_n9 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n9)
      %b_r0c0_m7_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n9)
      %c_r0c0_m7_n9 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n9)
      aiex.dma_await_task(%c_r0c0_m7_n9)
      aiex.dma_free_task(%a_r0c0_m7_n9)
      aiex.dma_free_task(%b_r0c0_m7_n9)
      aiex.dma_free_task(%c_r0c0_m7_n9)
      // core r0c0 m7 n10
      %a_r0c0_m7_n10 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n10)
      %b_r0c0_m7_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n10)
      %c_r0c0_m7_n10 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n10)
      aiex.dma_await_task(%c_r0c0_m7_n10)
      aiex.dma_free_task(%a_r0c0_m7_n10)
      aiex.dma_free_task(%b_r0c0_m7_n10)
      aiex.dma_free_task(%c_r0c0_m7_n10)
      // core r0c0 m7 n11
      %a_r0c0_m7_n11 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n11)
      %b_r0c0_m7_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n11)
      %c_r0c0_m7_n11 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n11)
      aiex.dma_await_task(%c_r0c0_m7_n11)
      aiex.dma_free_task(%a_r0c0_m7_n11)
      aiex.dma_free_task(%b_r0c0_m7_n11)
      aiex.dma_free_task(%c_r0c0_m7_n11)
      // core r0c0 m7 n12
      %a_r0c0_m7_n12 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n12)
      %b_r0c0_m7_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n12)
      %c_r0c0_m7_n12 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n12)
      aiex.dma_await_task(%c_r0c0_m7_n12)
      aiex.dma_free_task(%a_r0c0_m7_n12)
      aiex.dma_free_task(%b_r0c0_m7_n12)
      aiex.dma_free_task(%c_r0c0_m7_n12)
      // core r0c0 m7 n13
      %a_r0c0_m7_n13 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c0_m7_n13)
      %b_r0c0_m7_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c0_m7_n13)
      %c_r0c0_m7_n13 = aiex.dma_configure_task_for @C2_r0_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c0_m7_n13)
      aiex.dma_await_task(%c_r0c0_m7_n13)
      aiex.dma_free_task(%a_r0c0_m7_n13)
      aiex.dma_free_task(%b_r0c0_m7_n13)
      aiex.dma_free_task(%c_r0c0_m7_n13)
      // core r0c1 m0 n14
      %a_r0c1_m0_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n14)
      %b_r0c1_m0_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n14)
      %c_r0c1_m0_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n14)
      aiex.dma_await_task(%c_r0c1_m0_n14)
      aiex.dma_free_task(%a_r0c1_m0_n14)
      aiex.dma_free_task(%b_r0c1_m0_n14)
      aiex.dma_free_task(%c_r0c1_m0_n14)
      // core r0c1 m0 n15
      %a_r0c1_m0_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n15)
      %b_r0c1_m0_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n15)
      %c_r0c1_m0_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n15)
      aiex.dma_await_task(%c_r0c1_m0_n15)
      aiex.dma_free_task(%a_r0c1_m0_n15)
      aiex.dma_free_task(%b_r0c1_m0_n15)
      aiex.dma_free_task(%c_r0c1_m0_n15)
      // core r0c1 m0 n16
      %a_r0c1_m0_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n16)
      %b_r0c1_m0_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n16)
      %c_r0c1_m0_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n16)
      aiex.dma_await_task(%c_r0c1_m0_n16)
      aiex.dma_free_task(%a_r0c1_m0_n16)
      aiex.dma_free_task(%b_r0c1_m0_n16)
      aiex.dma_free_task(%c_r0c1_m0_n16)
      // core r0c1 m0 n17
      %a_r0c1_m0_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n17)
      %b_r0c1_m0_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n17)
      %c_r0c1_m0_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n17)
      aiex.dma_await_task(%c_r0c1_m0_n17)
      aiex.dma_free_task(%a_r0c1_m0_n17)
      aiex.dma_free_task(%b_r0c1_m0_n17)
      aiex.dma_free_task(%c_r0c1_m0_n17)
      // core r0c1 m0 n18
      %a_r0c1_m0_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n18)
      %b_r0c1_m0_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n18)
      %c_r0c1_m0_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n18)
      aiex.dma_await_task(%c_r0c1_m0_n18)
      aiex.dma_free_task(%a_r0c1_m0_n18)
      aiex.dma_free_task(%b_r0c1_m0_n18)
      aiex.dma_free_task(%c_r0c1_m0_n18)
      // core r0c1 m0 n19
      %a_r0c1_m0_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n19)
      %b_r0c1_m0_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n19)
      %c_r0c1_m0_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1216 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n19)
      aiex.dma_await_task(%c_r0c1_m0_n19)
      aiex.dma_free_task(%a_r0c1_m0_n19)
      aiex.dma_free_task(%b_r0c1_m0_n19)
      aiex.dma_free_task(%c_r0c1_m0_n19)
      // core r0c1 m0 n20
      %a_r0c1_m0_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n20)
      %b_r0c1_m0_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n20)
      %c_r0c1_m0_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1280 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n20)
      aiex.dma_await_task(%c_r0c1_m0_n20)
      aiex.dma_free_task(%a_r0c1_m0_n20)
      aiex.dma_free_task(%b_r0c1_m0_n20)
      aiex.dma_free_task(%c_r0c1_m0_n20)
      // core r0c1 m0 n21
      %a_r0c1_m0_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n21)
      %b_r0c1_m0_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n21)
      %c_r0c1_m0_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1344 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n21)
      aiex.dma_await_task(%c_r0c1_m0_n21)
      aiex.dma_free_task(%a_r0c1_m0_n21)
      aiex.dma_free_task(%b_r0c1_m0_n21)
      aiex.dma_free_task(%c_r0c1_m0_n21)
      // core r0c1 m0 n22
      %a_r0c1_m0_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n22)
      %b_r0c1_m0_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n22)
      %c_r0c1_m0_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1408 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n22)
      aiex.dma_await_task(%c_r0c1_m0_n22)
      aiex.dma_free_task(%a_r0c1_m0_n22)
      aiex.dma_free_task(%b_r0c1_m0_n22)
      aiex.dma_free_task(%c_r0c1_m0_n22)
      // core r0c1 m0 n23
      %a_r0c1_m0_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n23)
      %b_r0c1_m0_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n23)
      %c_r0c1_m0_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1472 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n23)
      aiex.dma_await_task(%c_r0c1_m0_n23)
      aiex.dma_free_task(%a_r0c1_m0_n23)
      aiex.dma_free_task(%b_r0c1_m0_n23)
      aiex.dma_free_task(%c_r0c1_m0_n23)
      // core r0c1 m0 n24
      %a_r0c1_m0_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n24)
      %b_r0c1_m0_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n24)
      %c_r0c1_m0_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1536 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n24)
      aiex.dma_await_task(%c_r0c1_m0_n24)
      aiex.dma_free_task(%a_r0c1_m0_n24)
      aiex.dma_free_task(%b_r0c1_m0_n24)
      aiex.dma_free_task(%c_r0c1_m0_n24)
      // core r0c1 m0 n25
      %a_r0c1_m0_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n25)
      %b_r0c1_m0_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n25)
      %c_r0c1_m0_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1600 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n25)
      aiex.dma_await_task(%c_r0c1_m0_n25)
      aiex.dma_free_task(%a_r0c1_m0_n25)
      aiex.dma_free_task(%b_r0c1_m0_n25)
      aiex.dma_free_task(%c_r0c1_m0_n25)
      // core r0c1 m0 n26
      %a_r0c1_m0_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n26)
      %b_r0c1_m0_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n26)
      %c_r0c1_m0_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1664 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n26)
      aiex.dma_await_task(%c_r0c1_m0_n26)
      aiex.dma_free_task(%a_r0c1_m0_n26)
      aiex.dma_free_task(%b_r0c1_m0_n26)
      aiex.dma_free_task(%c_r0c1_m0_n26)
      // core r0c1 m0 n27
      %a_r0c1_m0_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m0_n27)
      %b_r0c1_m0_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m0_n27)
      %c_r0c1_m0_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1728 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m0_n27)
      aiex.dma_await_task(%c_r0c1_m0_n27)
      aiex.dma_free_task(%a_r0c1_m0_n27)
      aiex.dma_free_task(%b_r0c1_m0_n27)
      aiex.dma_free_task(%c_r0c1_m0_n27)
      // core r0c1 m1 n14
      %a_r0c1_m1_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n14)
      %b_r0c1_m1_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n14)
      %c_r0c1_m1_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n14)
      aiex.dma_await_task(%c_r0c1_m1_n14)
      aiex.dma_free_task(%a_r0c1_m1_n14)
      aiex.dma_free_task(%b_r0c1_m1_n14)
      aiex.dma_free_task(%c_r0c1_m1_n14)
      // core r0c1 m1 n15
      %a_r0c1_m1_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n15)
      %b_r0c1_m1_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n15)
      %c_r0c1_m1_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n15)
      aiex.dma_await_task(%c_r0c1_m1_n15)
      aiex.dma_free_task(%a_r0c1_m1_n15)
      aiex.dma_free_task(%b_r0c1_m1_n15)
      aiex.dma_free_task(%c_r0c1_m1_n15)
      // core r0c1 m1 n16
      %a_r0c1_m1_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n16)
      %b_r0c1_m1_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n16)
      %c_r0c1_m1_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n16)
      aiex.dma_await_task(%c_r0c1_m1_n16)
      aiex.dma_free_task(%a_r0c1_m1_n16)
      aiex.dma_free_task(%b_r0c1_m1_n16)
      aiex.dma_free_task(%c_r0c1_m1_n16)
      // core r0c1 m1 n17
      %a_r0c1_m1_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n17)
      %b_r0c1_m1_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n17)
      %c_r0c1_m1_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n17)
      aiex.dma_await_task(%c_r0c1_m1_n17)
      aiex.dma_free_task(%a_r0c1_m1_n17)
      aiex.dma_free_task(%b_r0c1_m1_n17)
      aiex.dma_free_task(%c_r0c1_m1_n17)
      // core r0c1 m1 n18
      %a_r0c1_m1_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n18)
      %b_r0c1_m1_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n18)
      %c_r0c1_m1_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115840 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n18)
      aiex.dma_await_task(%c_r0c1_m1_n18)
      aiex.dma_free_task(%a_r0c1_m1_n18)
      aiex.dma_free_task(%b_r0c1_m1_n18)
      aiex.dma_free_task(%c_r0c1_m1_n18)
      // core r0c1 m1 n19
      %a_r0c1_m1_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n19)
      %b_r0c1_m1_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n19)
      %c_r0c1_m1_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115904 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n19)
      aiex.dma_await_task(%c_r0c1_m1_n19)
      aiex.dma_free_task(%a_r0c1_m1_n19)
      aiex.dma_free_task(%b_r0c1_m1_n19)
      aiex.dma_free_task(%c_r0c1_m1_n19)
      // core r0c1 m1 n20
      %a_r0c1_m1_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n20)
      %b_r0c1_m1_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n20)
      %c_r0c1_m1_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 115968 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n20)
      aiex.dma_await_task(%c_r0c1_m1_n20)
      aiex.dma_free_task(%a_r0c1_m1_n20)
      aiex.dma_free_task(%b_r0c1_m1_n20)
      aiex.dma_free_task(%c_r0c1_m1_n20)
      // core r0c1 m1 n21
      %a_r0c1_m1_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n21)
      %b_r0c1_m1_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n21)
      %c_r0c1_m1_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116032 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n21)
      aiex.dma_await_task(%c_r0c1_m1_n21)
      aiex.dma_free_task(%a_r0c1_m1_n21)
      aiex.dma_free_task(%b_r0c1_m1_n21)
      aiex.dma_free_task(%c_r0c1_m1_n21)
      // core r0c1 m1 n22
      %a_r0c1_m1_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n22)
      %b_r0c1_m1_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n22)
      %c_r0c1_m1_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116096 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n22)
      aiex.dma_await_task(%c_r0c1_m1_n22)
      aiex.dma_free_task(%a_r0c1_m1_n22)
      aiex.dma_free_task(%b_r0c1_m1_n22)
      aiex.dma_free_task(%c_r0c1_m1_n22)
      // core r0c1 m1 n23
      %a_r0c1_m1_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n23)
      %b_r0c1_m1_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n23)
      %c_r0c1_m1_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116160 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n23)
      aiex.dma_await_task(%c_r0c1_m1_n23)
      aiex.dma_free_task(%a_r0c1_m1_n23)
      aiex.dma_free_task(%b_r0c1_m1_n23)
      aiex.dma_free_task(%c_r0c1_m1_n23)
      // core r0c1 m1 n24
      %a_r0c1_m1_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n24)
      %b_r0c1_m1_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n24)
      %c_r0c1_m1_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116224 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n24)
      aiex.dma_await_task(%c_r0c1_m1_n24)
      aiex.dma_free_task(%a_r0c1_m1_n24)
      aiex.dma_free_task(%b_r0c1_m1_n24)
      aiex.dma_free_task(%c_r0c1_m1_n24)
      // core r0c1 m1 n25
      %a_r0c1_m1_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n25)
      %b_r0c1_m1_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n25)
      %c_r0c1_m1_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116288 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n25)
      aiex.dma_await_task(%c_r0c1_m1_n25)
      aiex.dma_free_task(%a_r0c1_m1_n25)
      aiex.dma_free_task(%b_r0c1_m1_n25)
      aiex.dma_free_task(%c_r0c1_m1_n25)
      // core r0c1 m1 n26
      %a_r0c1_m1_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n26)
      %b_r0c1_m1_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n26)
      %c_r0c1_m1_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116352 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n26)
      aiex.dma_await_task(%c_r0c1_m1_n26)
      aiex.dma_free_task(%a_r0c1_m1_n26)
      aiex.dma_free_task(%b_r0c1_m1_n26)
      aiex.dma_free_task(%c_r0c1_m1_n26)
      // core r0c1 m1 n27
      %a_r0c1_m1_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m1_n27)
      %b_r0c1_m1_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m1_n27)
      %c_r0c1_m1_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116416 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m1_n27)
      aiex.dma_await_task(%c_r0c1_m1_n27)
      aiex.dma_free_task(%a_r0c1_m1_n27)
      aiex.dma_free_task(%b_r0c1_m1_n27)
      aiex.dma_free_task(%c_r0c1_m1_n27)
      // core r0c1 m2 n14
      %a_r0c1_m2_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n14)
      %b_r0c1_m2_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n14)
      %c_r0c1_m2_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n14)
      aiex.dma_await_task(%c_r0c1_m2_n14)
      aiex.dma_free_task(%a_r0c1_m2_n14)
      aiex.dma_free_task(%b_r0c1_m2_n14)
      aiex.dma_free_task(%c_r0c1_m2_n14)
      // core r0c1 m2 n15
      %a_r0c1_m2_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n15)
      %b_r0c1_m2_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n15)
      %c_r0c1_m2_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n15)
      aiex.dma_await_task(%c_r0c1_m2_n15)
      aiex.dma_free_task(%a_r0c1_m2_n15)
      aiex.dma_free_task(%b_r0c1_m2_n15)
      aiex.dma_free_task(%c_r0c1_m2_n15)
      // core r0c1 m2 n16
      %a_r0c1_m2_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n16)
      %b_r0c1_m2_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n16)
      %c_r0c1_m2_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n16)
      aiex.dma_await_task(%c_r0c1_m2_n16)
      aiex.dma_free_task(%a_r0c1_m2_n16)
      aiex.dma_free_task(%b_r0c1_m2_n16)
      aiex.dma_free_task(%c_r0c1_m2_n16)
      // core r0c1 m2 n17
      %a_r0c1_m2_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n17)
      %b_r0c1_m2_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n17)
      %c_r0c1_m2_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230464 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n17)
      aiex.dma_await_task(%c_r0c1_m2_n17)
      aiex.dma_free_task(%a_r0c1_m2_n17)
      aiex.dma_free_task(%b_r0c1_m2_n17)
      aiex.dma_free_task(%c_r0c1_m2_n17)
      // core r0c1 m2 n18
      %a_r0c1_m2_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n18)
      %b_r0c1_m2_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n18)
      %c_r0c1_m2_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230528 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n18)
      aiex.dma_await_task(%c_r0c1_m2_n18)
      aiex.dma_free_task(%a_r0c1_m2_n18)
      aiex.dma_free_task(%b_r0c1_m2_n18)
      aiex.dma_free_task(%c_r0c1_m2_n18)
      // core r0c1 m2 n19
      %a_r0c1_m2_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n19)
      %b_r0c1_m2_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n19)
      %c_r0c1_m2_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230592 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n19)
      aiex.dma_await_task(%c_r0c1_m2_n19)
      aiex.dma_free_task(%a_r0c1_m2_n19)
      aiex.dma_free_task(%b_r0c1_m2_n19)
      aiex.dma_free_task(%c_r0c1_m2_n19)
      // core r0c1 m2 n20
      %a_r0c1_m2_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n20)
      %b_r0c1_m2_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n20)
      %c_r0c1_m2_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230656 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n20)
      aiex.dma_await_task(%c_r0c1_m2_n20)
      aiex.dma_free_task(%a_r0c1_m2_n20)
      aiex.dma_free_task(%b_r0c1_m2_n20)
      aiex.dma_free_task(%c_r0c1_m2_n20)
      // core r0c1 m2 n21
      %a_r0c1_m2_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n21)
      %b_r0c1_m2_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n21)
      %c_r0c1_m2_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230720 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n21)
      aiex.dma_await_task(%c_r0c1_m2_n21)
      aiex.dma_free_task(%a_r0c1_m2_n21)
      aiex.dma_free_task(%b_r0c1_m2_n21)
      aiex.dma_free_task(%c_r0c1_m2_n21)
      // core r0c1 m2 n22
      %a_r0c1_m2_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n22)
      %b_r0c1_m2_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n22)
      %c_r0c1_m2_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230784 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n22)
      aiex.dma_await_task(%c_r0c1_m2_n22)
      aiex.dma_free_task(%a_r0c1_m2_n22)
      aiex.dma_free_task(%b_r0c1_m2_n22)
      aiex.dma_free_task(%c_r0c1_m2_n22)
      // core r0c1 m2 n23
      %a_r0c1_m2_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n23)
      %b_r0c1_m2_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n23)
      %c_r0c1_m2_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230848 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n23)
      aiex.dma_await_task(%c_r0c1_m2_n23)
      aiex.dma_free_task(%a_r0c1_m2_n23)
      aiex.dma_free_task(%b_r0c1_m2_n23)
      aiex.dma_free_task(%c_r0c1_m2_n23)
      // core r0c1 m2 n24
      %a_r0c1_m2_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n24)
      %b_r0c1_m2_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n24)
      %c_r0c1_m2_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230912 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n24)
      aiex.dma_await_task(%c_r0c1_m2_n24)
      aiex.dma_free_task(%a_r0c1_m2_n24)
      aiex.dma_free_task(%b_r0c1_m2_n24)
      aiex.dma_free_task(%c_r0c1_m2_n24)
      // core r0c1 m2 n25
      %a_r0c1_m2_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n25)
      %b_r0c1_m2_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n25)
      %c_r0c1_m2_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 230976 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n25)
      aiex.dma_await_task(%c_r0c1_m2_n25)
      aiex.dma_free_task(%a_r0c1_m2_n25)
      aiex.dma_free_task(%b_r0c1_m2_n25)
      aiex.dma_free_task(%c_r0c1_m2_n25)
      // core r0c1 m2 n26
      %a_r0c1_m2_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n26)
      %b_r0c1_m2_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n26)
      %c_r0c1_m2_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231040 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n26)
      aiex.dma_await_task(%c_r0c1_m2_n26)
      aiex.dma_free_task(%a_r0c1_m2_n26)
      aiex.dma_free_task(%b_r0c1_m2_n26)
      aiex.dma_free_task(%c_r0c1_m2_n26)
      // core r0c1 m2 n27
      %a_r0c1_m2_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m2_n27)
      %b_r0c1_m2_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m2_n27)
      %c_r0c1_m2_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231104 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m2_n27)
      aiex.dma_await_task(%c_r0c1_m2_n27)
      aiex.dma_free_task(%a_r0c1_m2_n27)
      aiex.dma_free_task(%b_r0c1_m2_n27)
      aiex.dma_free_task(%c_r0c1_m2_n27)
      // core r0c1 m3 n14
      %a_r0c1_m3_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n14)
      %b_r0c1_m3_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n14)
      %c_r0c1_m3_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 344960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n14)
      aiex.dma_await_task(%c_r0c1_m3_n14)
      aiex.dma_free_task(%a_r0c1_m3_n14)
      aiex.dma_free_task(%b_r0c1_m3_n14)
      aiex.dma_free_task(%c_r0c1_m3_n14)
      // core r0c1 m3 n15
      %a_r0c1_m3_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n15)
      %b_r0c1_m3_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n15)
      %c_r0c1_m3_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n15)
      aiex.dma_await_task(%c_r0c1_m3_n15)
      aiex.dma_free_task(%a_r0c1_m3_n15)
      aiex.dma_free_task(%b_r0c1_m3_n15)
      aiex.dma_free_task(%c_r0c1_m3_n15)
      // core r0c1 m3 n16
      %a_r0c1_m3_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n16)
      %b_r0c1_m3_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n16)
      %c_r0c1_m3_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n16)
      aiex.dma_await_task(%c_r0c1_m3_n16)
      aiex.dma_free_task(%a_r0c1_m3_n16)
      aiex.dma_free_task(%b_r0c1_m3_n16)
      aiex.dma_free_task(%c_r0c1_m3_n16)
      // core r0c1 m3 n17
      %a_r0c1_m3_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n17)
      %b_r0c1_m3_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n17)
      %c_r0c1_m3_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n17)
      aiex.dma_await_task(%c_r0c1_m3_n17)
      aiex.dma_free_task(%a_r0c1_m3_n17)
      aiex.dma_free_task(%b_r0c1_m3_n17)
      aiex.dma_free_task(%c_r0c1_m3_n17)
      // core r0c1 m3 n18
      %a_r0c1_m3_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n18)
      %b_r0c1_m3_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n18)
      %c_r0c1_m3_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345216 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n18)
      aiex.dma_await_task(%c_r0c1_m3_n18)
      aiex.dma_free_task(%a_r0c1_m3_n18)
      aiex.dma_free_task(%b_r0c1_m3_n18)
      aiex.dma_free_task(%c_r0c1_m3_n18)
      // core r0c1 m3 n19
      %a_r0c1_m3_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n19)
      %b_r0c1_m3_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n19)
      %c_r0c1_m3_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345280 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n19)
      aiex.dma_await_task(%c_r0c1_m3_n19)
      aiex.dma_free_task(%a_r0c1_m3_n19)
      aiex.dma_free_task(%b_r0c1_m3_n19)
      aiex.dma_free_task(%c_r0c1_m3_n19)
      // core r0c1 m3 n20
      %a_r0c1_m3_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n20)
      %b_r0c1_m3_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n20)
      %c_r0c1_m3_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345344 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n20)
      aiex.dma_await_task(%c_r0c1_m3_n20)
      aiex.dma_free_task(%a_r0c1_m3_n20)
      aiex.dma_free_task(%b_r0c1_m3_n20)
      aiex.dma_free_task(%c_r0c1_m3_n20)
      // core r0c1 m3 n21
      %a_r0c1_m3_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n21)
      %b_r0c1_m3_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n21)
      %c_r0c1_m3_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345408 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n21)
      aiex.dma_await_task(%c_r0c1_m3_n21)
      aiex.dma_free_task(%a_r0c1_m3_n21)
      aiex.dma_free_task(%b_r0c1_m3_n21)
      aiex.dma_free_task(%c_r0c1_m3_n21)
      // core r0c1 m3 n22
      %a_r0c1_m3_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n22)
      %b_r0c1_m3_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n22)
      %c_r0c1_m3_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345472 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n22)
      aiex.dma_await_task(%c_r0c1_m3_n22)
      aiex.dma_free_task(%a_r0c1_m3_n22)
      aiex.dma_free_task(%b_r0c1_m3_n22)
      aiex.dma_free_task(%c_r0c1_m3_n22)
      // core r0c1 m3 n23
      %a_r0c1_m3_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n23)
      %b_r0c1_m3_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n23)
      %c_r0c1_m3_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345536 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n23)
      aiex.dma_await_task(%c_r0c1_m3_n23)
      aiex.dma_free_task(%a_r0c1_m3_n23)
      aiex.dma_free_task(%b_r0c1_m3_n23)
      aiex.dma_free_task(%c_r0c1_m3_n23)
      // core r0c1 m3 n24
      %a_r0c1_m3_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n24)
      %b_r0c1_m3_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n24)
      %c_r0c1_m3_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345600 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n24)
      aiex.dma_await_task(%c_r0c1_m3_n24)
      aiex.dma_free_task(%a_r0c1_m3_n24)
      aiex.dma_free_task(%b_r0c1_m3_n24)
      aiex.dma_free_task(%c_r0c1_m3_n24)
      // core r0c1 m3 n25
      %a_r0c1_m3_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n25)
      %b_r0c1_m3_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n25)
      %c_r0c1_m3_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345664 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n25)
      aiex.dma_await_task(%c_r0c1_m3_n25)
      aiex.dma_free_task(%a_r0c1_m3_n25)
      aiex.dma_free_task(%b_r0c1_m3_n25)
      aiex.dma_free_task(%c_r0c1_m3_n25)
      // core r0c1 m3 n26
      %a_r0c1_m3_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n26)
      %b_r0c1_m3_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n26)
      %c_r0c1_m3_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345728 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n26)
      aiex.dma_await_task(%c_r0c1_m3_n26)
      aiex.dma_free_task(%a_r0c1_m3_n26)
      aiex.dma_free_task(%b_r0c1_m3_n26)
      aiex.dma_free_task(%c_r0c1_m3_n26)
      // core r0c1 m3 n27
      %a_r0c1_m3_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m3_n27)
      %b_r0c1_m3_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m3_n27)
      %c_r0c1_m3_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345792 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m3_n27)
      aiex.dma_await_task(%c_r0c1_m3_n27)
      aiex.dma_free_task(%a_r0c1_m3_n27)
      aiex.dma_free_task(%b_r0c1_m3_n27)
      aiex.dma_free_task(%c_r0c1_m3_n27)
      // core r0c1 m4 n14
      %a_r0c1_m4_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n14)
      %b_r0c1_m4_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n14)
      %c_r0c1_m4_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n14)
      aiex.dma_await_task(%c_r0c1_m4_n14)
      aiex.dma_free_task(%a_r0c1_m4_n14)
      aiex.dma_free_task(%b_r0c1_m4_n14)
      aiex.dma_free_task(%c_r0c1_m4_n14)
      // core r0c1 m4 n15
      %a_r0c1_m4_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n15)
      %b_r0c1_m4_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n15)
      %c_r0c1_m4_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n15)
      aiex.dma_await_task(%c_r0c1_m4_n15)
      aiex.dma_free_task(%a_r0c1_m4_n15)
      aiex.dma_free_task(%b_r0c1_m4_n15)
      aiex.dma_free_task(%c_r0c1_m4_n15)
      // core r0c1 m4 n16
      %a_r0c1_m4_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n16)
      %b_r0c1_m4_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n16)
      %c_r0c1_m4_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n16)
      aiex.dma_await_task(%c_r0c1_m4_n16)
      aiex.dma_free_task(%a_r0c1_m4_n16)
      aiex.dma_free_task(%b_r0c1_m4_n16)
      aiex.dma_free_task(%c_r0c1_m4_n16)
      // core r0c1 m4 n17
      %a_r0c1_m4_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n17)
      %b_r0c1_m4_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n17)
      %c_r0c1_m4_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459840 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n17)
      aiex.dma_await_task(%c_r0c1_m4_n17)
      aiex.dma_free_task(%a_r0c1_m4_n17)
      aiex.dma_free_task(%b_r0c1_m4_n17)
      aiex.dma_free_task(%c_r0c1_m4_n17)
      // core r0c1 m4 n18
      %a_r0c1_m4_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n18)
      %b_r0c1_m4_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n18)
      %c_r0c1_m4_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459904 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n18)
      aiex.dma_await_task(%c_r0c1_m4_n18)
      aiex.dma_free_task(%a_r0c1_m4_n18)
      aiex.dma_free_task(%b_r0c1_m4_n18)
      aiex.dma_free_task(%c_r0c1_m4_n18)
      // core r0c1 m4 n19
      %a_r0c1_m4_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n19)
      %b_r0c1_m4_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n19)
      %c_r0c1_m4_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 459968 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n19)
      aiex.dma_await_task(%c_r0c1_m4_n19)
      aiex.dma_free_task(%a_r0c1_m4_n19)
      aiex.dma_free_task(%b_r0c1_m4_n19)
      aiex.dma_free_task(%c_r0c1_m4_n19)
      // core r0c1 m4 n20
      %a_r0c1_m4_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n20)
      %b_r0c1_m4_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n20)
      %c_r0c1_m4_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460032 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n20)
      aiex.dma_await_task(%c_r0c1_m4_n20)
      aiex.dma_free_task(%a_r0c1_m4_n20)
      aiex.dma_free_task(%b_r0c1_m4_n20)
      aiex.dma_free_task(%c_r0c1_m4_n20)
      // core r0c1 m4 n21
      %a_r0c1_m4_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n21)
      %b_r0c1_m4_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n21)
      %c_r0c1_m4_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460096 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n21)
      aiex.dma_await_task(%c_r0c1_m4_n21)
      aiex.dma_free_task(%a_r0c1_m4_n21)
      aiex.dma_free_task(%b_r0c1_m4_n21)
      aiex.dma_free_task(%c_r0c1_m4_n21)
      // core r0c1 m4 n22
      %a_r0c1_m4_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n22)
      %b_r0c1_m4_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n22)
      %c_r0c1_m4_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460160 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n22)
      aiex.dma_await_task(%c_r0c1_m4_n22)
      aiex.dma_free_task(%a_r0c1_m4_n22)
      aiex.dma_free_task(%b_r0c1_m4_n22)
      aiex.dma_free_task(%c_r0c1_m4_n22)
      // core r0c1 m4 n23
      %a_r0c1_m4_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n23)
      %b_r0c1_m4_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n23)
      %c_r0c1_m4_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460224 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n23)
      aiex.dma_await_task(%c_r0c1_m4_n23)
      aiex.dma_free_task(%a_r0c1_m4_n23)
      aiex.dma_free_task(%b_r0c1_m4_n23)
      aiex.dma_free_task(%c_r0c1_m4_n23)
      // core r0c1 m4 n24
      %a_r0c1_m4_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n24)
      %b_r0c1_m4_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n24)
      %c_r0c1_m4_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460288 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n24)
      aiex.dma_await_task(%c_r0c1_m4_n24)
      aiex.dma_free_task(%a_r0c1_m4_n24)
      aiex.dma_free_task(%b_r0c1_m4_n24)
      aiex.dma_free_task(%c_r0c1_m4_n24)
      // core r0c1 m4 n25
      %a_r0c1_m4_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n25)
      %b_r0c1_m4_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n25)
      %c_r0c1_m4_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460352 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n25)
      aiex.dma_await_task(%c_r0c1_m4_n25)
      aiex.dma_free_task(%a_r0c1_m4_n25)
      aiex.dma_free_task(%b_r0c1_m4_n25)
      aiex.dma_free_task(%c_r0c1_m4_n25)
      // core r0c1 m4 n26
      %a_r0c1_m4_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n26)
      %b_r0c1_m4_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n26)
      %c_r0c1_m4_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460416 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n26)
      aiex.dma_await_task(%c_r0c1_m4_n26)
      aiex.dma_free_task(%a_r0c1_m4_n26)
      aiex.dma_free_task(%b_r0c1_m4_n26)
      aiex.dma_free_task(%c_r0c1_m4_n26)
      // core r0c1 m4 n27
      %a_r0c1_m4_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m4_n27)
      %b_r0c1_m4_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m4_n27)
      %c_r0c1_m4_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460480 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m4_n27)
      aiex.dma_await_task(%c_r0c1_m4_n27)
      aiex.dma_free_task(%a_r0c1_m4_n27)
      aiex.dma_free_task(%b_r0c1_m4_n27)
      aiex.dma_free_task(%c_r0c1_m4_n27)
      // core r0c1 m5 n14
      %a_r0c1_m5_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n14)
      %b_r0c1_m5_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n14)
      %c_r0c1_m5_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n14)
      aiex.dma_await_task(%c_r0c1_m5_n14)
      aiex.dma_free_task(%a_r0c1_m5_n14)
      aiex.dma_free_task(%b_r0c1_m5_n14)
      aiex.dma_free_task(%c_r0c1_m5_n14)
      // core r0c1 m5 n15
      %a_r0c1_m5_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n15)
      %b_r0c1_m5_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n15)
      %c_r0c1_m5_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n15)
      aiex.dma_await_task(%c_r0c1_m5_n15)
      aiex.dma_free_task(%a_r0c1_m5_n15)
      aiex.dma_free_task(%b_r0c1_m5_n15)
      aiex.dma_free_task(%c_r0c1_m5_n15)
      // core r0c1 m5 n16
      %a_r0c1_m5_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n16)
      %b_r0c1_m5_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n16)
      %c_r0c1_m5_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574464 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n16)
      aiex.dma_await_task(%c_r0c1_m5_n16)
      aiex.dma_free_task(%a_r0c1_m5_n16)
      aiex.dma_free_task(%b_r0c1_m5_n16)
      aiex.dma_free_task(%c_r0c1_m5_n16)
      // core r0c1 m5 n17
      %a_r0c1_m5_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n17)
      %b_r0c1_m5_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n17)
      %c_r0c1_m5_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574528 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n17)
      aiex.dma_await_task(%c_r0c1_m5_n17)
      aiex.dma_free_task(%a_r0c1_m5_n17)
      aiex.dma_free_task(%b_r0c1_m5_n17)
      aiex.dma_free_task(%c_r0c1_m5_n17)
      // core r0c1 m5 n18
      %a_r0c1_m5_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n18)
      %b_r0c1_m5_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n18)
      %c_r0c1_m5_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574592 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n18)
      aiex.dma_await_task(%c_r0c1_m5_n18)
      aiex.dma_free_task(%a_r0c1_m5_n18)
      aiex.dma_free_task(%b_r0c1_m5_n18)
      aiex.dma_free_task(%c_r0c1_m5_n18)
      // core r0c1 m5 n19
      %a_r0c1_m5_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n19)
      %b_r0c1_m5_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n19)
      %c_r0c1_m5_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574656 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n19)
      aiex.dma_await_task(%c_r0c1_m5_n19)
      aiex.dma_free_task(%a_r0c1_m5_n19)
      aiex.dma_free_task(%b_r0c1_m5_n19)
      aiex.dma_free_task(%c_r0c1_m5_n19)
      // core r0c1 m5 n20
      %a_r0c1_m5_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n20)
      %b_r0c1_m5_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n20)
      %c_r0c1_m5_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574720 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n20)
      aiex.dma_await_task(%c_r0c1_m5_n20)
      aiex.dma_free_task(%a_r0c1_m5_n20)
      aiex.dma_free_task(%b_r0c1_m5_n20)
      aiex.dma_free_task(%c_r0c1_m5_n20)
      // core r0c1 m5 n21
      %a_r0c1_m5_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n21)
      %b_r0c1_m5_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n21)
      %c_r0c1_m5_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574784 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n21)
      aiex.dma_await_task(%c_r0c1_m5_n21)
      aiex.dma_free_task(%a_r0c1_m5_n21)
      aiex.dma_free_task(%b_r0c1_m5_n21)
      aiex.dma_free_task(%c_r0c1_m5_n21)
      // core r0c1 m5 n22
      %a_r0c1_m5_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n22)
      %b_r0c1_m5_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n22)
      %c_r0c1_m5_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574848 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n22)
      aiex.dma_await_task(%c_r0c1_m5_n22)
      aiex.dma_free_task(%a_r0c1_m5_n22)
      aiex.dma_free_task(%b_r0c1_m5_n22)
      aiex.dma_free_task(%c_r0c1_m5_n22)
      // core r0c1 m5 n23
      %a_r0c1_m5_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n23)
      %b_r0c1_m5_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n23)
      %c_r0c1_m5_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574912 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n23)
      aiex.dma_await_task(%c_r0c1_m5_n23)
      aiex.dma_free_task(%a_r0c1_m5_n23)
      aiex.dma_free_task(%b_r0c1_m5_n23)
      aiex.dma_free_task(%c_r0c1_m5_n23)
      // core r0c1 m5 n24
      %a_r0c1_m5_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n24)
      %b_r0c1_m5_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n24)
      %c_r0c1_m5_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 574976 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n24)
      aiex.dma_await_task(%c_r0c1_m5_n24)
      aiex.dma_free_task(%a_r0c1_m5_n24)
      aiex.dma_free_task(%b_r0c1_m5_n24)
      aiex.dma_free_task(%c_r0c1_m5_n24)
      // core r0c1 m5 n25
      %a_r0c1_m5_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n25)
      %b_r0c1_m5_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n25)
      %c_r0c1_m5_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575040 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n25)
      aiex.dma_await_task(%c_r0c1_m5_n25)
      aiex.dma_free_task(%a_r0c1_m5_n25)
      aiex.dma_free_task(%b_r0c1_m5_n25)
      aiex.dma_free_task(%c_r0c1_m5_n25)
      // core r0c1 m5 n26
      %a_r0c1_m5_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n26)
      %b_r0c1_m5_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n26)
      %c_r0c1_m5_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575104 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n26)
      aiex.dma_await_task(%c_r0c1_m5_n26)
      aiex.dma_free_task(%a_r0c1_m5_n26)
      aiex.dma_free_task(%b_r0c1_m5_n26)
      aiex.dma_free_task(%c_r0c1_m5_n26)
      // core r0c1 m5 n27
      %a_r0c1_m5_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m5_n27)
      %b_r0c1_m5_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m5_n27)
      %c_r0c1_m5_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575168 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m5_n27)
      aiex.dma_await_task(%c_r0c1_m5_n27)
      aiex.dma_free_task(%a_r0c1_m5_n27)
      aiex.dma_free_task(%b_r0c1_m5_n27)
      aiex.dma_free_task(%c_r0c1_m5_n27)
      // core r0c1 m6 n14
      %a_r0c1_m6_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n14)
      %b_r0c1_m6_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n14)
      %c_r0c1_m6_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n14)
      aiex.dma_await_task(%c_r0c1_m6_n14)
      aiex.dma_free_task(%a_r0c1_m6_n14)
      aiex.dma_free_task(%b_r0c1_m6_n14)
      aiex.dma_free_task(%c_r0c1_m6_n14)
      // core r0c1 m6 n15
      %a_r0c1_m6_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n15)
      %b_r0c1_m6_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n15)
      %c_r0c1_m6_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n15)
      aiex.dma_await_task(%c_r0c1_m6_n15)
      aiex.dma_free_task(%a_r0c1_m6_n15)
      aiex.dma_free_task(%b_r0c1_m6_n15)
      aiex.dma_free_task(%c_r0c1_m6_n15)
      // core r0c1 m6 n16
      %a_r0c1_m6_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n16)
      %b_r0c1_m6_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n16)
      %c_r0c1_m6_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n16)
      aiex.dma_await_task(%c_r0c1_m6_n16)
      aiex.dma_free_task(%a_r0c1_m6_n16)
      aiex.dma_free_task(%b_r0c1_m6_n16)
      aiex.dma_free_task(%c_r0c1_m6_n16)
      // core r0c1 m6 n17
      %a_r0c1_m6_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n17)
      %b_r0c1_m6_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n17)
      %c_r0c1_m6_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689216 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n17)
      aiex.dma_await_task(%c_r0c1_m6_n17)
      aiex.dma_free_task(%a_r0c1_m6_n17)
      aiex.dma_free_task(%b_r0c1_m6_n17)
      aiex.dma_free_task(%c_r0c1_m6_n17)
      // core r0c1 m6 n18
      %a_r0c1_m6_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n18)
      %b_r0c1_m6_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n18)
      %c_r0c1_m6_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689280 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n18)
      aiex.dma_await_task(%c_r0c1_m6_n18)
      aiex.dma_free_task(%a_r0c1_m6_n18)
      aiex.dma_free_task(%b_r0c1_m6_n18)
      aiex.dma_free_task(%c_r0c1_m6_n18)
      // core r0c1 m6 n19
      %a_r0c1_m6_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n19)
      %b_r0c1_m6_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n19)
      %c_r0c1_m6_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689344 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n19)
      aiex.dma_await_task(%c_r0c1_m6_n19)
      aiex.dma_free_task(%a_r0c1_m6_n19)
      aiex.dma_free_task(%b_r0c1_m6_n19)
      aiex.dma_free_task(%c_r0c1_m6_n19)
      // core r0c1 m6 n20
      %a_r0c1_m6_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n20)
      %b_r0c1_m6_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n20)
      %c_r0c1_m6_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689408 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n20)
      aiex.dma_await_task(%c_r0c1_m6_n20)
      aiex.dma_free_task(%a_r0c1_m6_n20)
      aiex.dma_free_task(%b_r0c1_m6_n20)
      aiex.dma_free_task(%c_r0c1_m6_n20)
      // core r0c1 m6 n21
      %a_r0c1_m6_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n21)
      %b_r0c1_m6_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n21)
      %c_r0c1_m6_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689472 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n21)
      aiex.dma_await_task(%c_r0c1_m6_n21)
      aiex.dma_free_task(%a_r0c1_m6_n21)
      aiex.dma_free_task(%b_r0c1_m6_n21)
      aiex.dma_free_task(%c_r0c1_m6_n21)
      // core r0c1 m6 n22
      %a_r0c1_m6_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n22)
      %b_r0c1_m6_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n22)
      %c_r0c1_m6_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689536 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n22)
      aiex.dma_await_task(%c_r0c1_m6_n22)
      aiex.dma_free_task(%a_r0c1_m6_n22)
      aiex.dma_free_task(%b_r0c1_m6_n22)
      aiex.dma_free_task(%c_r0c1_m6_n22)
      // core r0c1 m6 n23
      %a_r0c1_m6_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n23)
      %b_r0c1_m6_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n23)
      %c_r0c1_m6_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689600 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n23)
      aiex.dma_await_task(%c_r0c1_m6_n23)
      aiex.dma_free_task(%a_r0c1_m6_n23)
      aiex.dma_free_task(%b_r0c1_m6_n23)
      aiex.dma_free_task(%c_r0c1_m6_n23)
      // core r0c1 m6 n24
      %a_r0c1_m6_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n24)
      %b_r0c1_m6_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n24)
      %c_r0c1_m6_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689664 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n24)
      aiex.dma_await_task(%c_r0c1_m6_n24)
      aiex.dma_free_task(%a_r0c1_m6_n24)
      aiex.dma_free_task(%b_r0c1_m6_n24)
      aiex.dma_free_task(%c_r0c1_m6_n24)
      // core r0c1 m6 n25
      %a_r0c1_m6_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n25)
      %b_r0c1_m6_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n25)
      %c_r0c1_m6_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689728 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n25)
      aiex.dma_await_task(%c_r0c1_m6_n25)
      aiex.dma_free_task(%a_r0c1_m6_n25)
      aiex.dma_free_task(%b_r0c1_m6_n25)
      aiex.dma_free_task(%c_r0c1_m6_n25)
      // core r0c1 m6 n26
      %a_r0c1_m6_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n26)
      %b_r0c1_m6_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n26)
      %c_r0c1_m6_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689792 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n26)
      aiex.dma_await_task(%c_r0c1_m6_n26)
      aiex.dma_free_task(%a_r0c1_m6_n26)
      aiex.dma_free_task(%b_r0c1_m6_n26)
      aiex.dma_free_task(%c_r0c1_m6_n26)
      // core r0c1 m6 n27
      %a_r0c1_m6_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m6_n27)
      %b_r0c1_m6_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m6_n27)
      %c_r0c1_m6_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689856 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m6_n27)
      aiex.dma_await_task(%c_r0c1_m6_n27)
      aiex.dma_free_task(%a_r0c1_m6_n27)
      aiex.dma_free_task(%b_r0c1_m6_n27)
      aiex.dma_free_task(%c_r0c1_m6_n27)
      // core r0c1 m7 n14
      %a_r0c1_m7_n14 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n14)
      %b_r0c1_m7_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n14)
      %c_r0c1_m7_n14 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n14)
      aiex.dma_await_task(%c_r0c1_m7_n14)
      aiex.dma_free_task(%a_r0c1_m7_n14)
      aiex.dma_free_task(%b_r0c1_m7_n14)
      aiex.dma_free_task(%c_r0c1_m7_n14)
      // core r0c1 m7 n15
      %a_r0c1_m7_n15 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n15)
      %b_r0c1_m7_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n15)
      %c_r0c1_m7_n15 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n15)
      aiex.dma_await_task(%c_r0c1_m7_n15)
      aiex.dma_free_task(%a_r0c1_m7_n15)
      aiex.dma_free_task(%b_r0c1_m7_n15)
      aiex.dma_free_task(%c_r0c1_m7_n15)
      // core r0c1 m7 n16
      %a_r0c1_m7_n16 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n16)
      %b_r0c1_m7_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n16)
      %c_r0c1_m7_n16 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803840 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n16)
      aiex.dma_await_task(%c_r0c1_m7_n16)
      aiex.dma_free_task(%a_r0c1_m7_n16)
      aiex.dma_free_task(%b_r0c1_m7_n16)
      aiex.dma_free_task(%c_r0c1_m7_n16)
      // core r0c1 m7 n17
      %a_r0c1_m7_n17 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n17)
      %b_r0c1_m7_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n17)
      %c_r0c1_m7_n17 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803904 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n17)
      aiex.dma_await_task(%c_r0c1_m7_n17)
      aiex.dma_free_task(%a_r0c1_m7_n17)
      aiex.dma_free_task(%b_r0c1_m7_n17)
      aiex.dma_free_task(%c_r0c1_m7_n17)
      // core r0c1 m7 n18
      %a_r0c1_m7_n18 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n18)
      %b_r0c1_m7_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n18)
      %c_r0c1_m7_n18 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 803968 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n18)
      aiex.dma_await_task(%c_r0c1_m7_n18)
      aiex.dma_free_task(%a_r0c1_m7_n18)
      aiex.dma_free_task(%b_r0c1_m7_n18)
      aiex.dma_free_task(%c_r0c1_m7_n18)
      // core r0c1 m7 n19
      %a_r0c1_m7_n19 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n19)
      %b_r0c1_m7_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n19)
      %c_r0c1_m7_n19 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804032 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n19)
      aiex.dma_await_task(%c_r0c1_m7_n19)
      aiex.dma_free_task(%a_r0c1_m7_n19)
      aiex.dma_free_task(%b_r0c1_m7_n19)
      aiex.dma_free_task(%c_r0c1_m7_n19)
      // core r0c1 m7 n20
      %a_r0c1_m7_n20 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n20)
      %b_r0c1_m7_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n20)
      %c_r0c1_m7_n20 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804096 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n20)
      aiex.dma_await_task(%c_r0c1_m7_n20)
      aiex.dma_free_task(%a_r0c1_m7_n20)
      aiex.dma_free_task(%b_r0c1_m7_n20)
      aiex.dma_free_task(%c_r0c1_m7_n20)
      // core r0c1 m7 n21
      %a_r0c1_m7_n21 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n21)
      %b_r0c1_m7_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n21)
      %c_r0c1_m7_n21 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804160 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n21)
      aiex.dma_await_task(%c_r0c1_m7_n21)
      aiex.dma_free_task(%a_r0c1_m7_n21)
      aiex.dma_free_task(%b_r0c1_m7_n21)
      aiex.dma_free_task(%c_r0c1_m7_n21)
      // core r0c1 m7 n22
      %a_r0c1_m7_n22 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n22)
      %b_r0c1_m7_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n22)
      %c_r0c1_m7_n22 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804224 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n22)
      aiex.dma_await_task(%c_r0c1_m7_n22)
      aiex.dma_free_task(%a_r0c1_m7_n22)
      aiex.dma_free_task(%b_r0c1_m7_n22)
      aiex.dma_free_task(%c_r0c1_m7_n22)
      // core r0c1 m7 n23
      %a_r0c1_m7_n23 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n23)
      %b_r0c1_m7_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n23)
      %c_r0c1_m7_n23 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804288 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n23)
      aiex.dma_await_task(%c_r0c1_m7_n23)
      aiex.dma_free_task(%a_r0c1_m7_n23)
      aiex.dma_free_task(%b_r0c1_m7_n23)
      aiex.dma_free_task(%c_r0c1_m7_n23)
      // core r0c1 m7 n24
      %a_r0c1_m7_n24 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n24)
      %b_r0c1_m7_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n24)
      %c_r0c1_m7_n24 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804352 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n24)
      aiex.dma_await_task(%c_r0c1_m7_n24)
      aiex.dma_free_task(%a_r0c1_m7_n24)
      aiex.dma_free_task(%b_r0c1_m7_n24)
      aiex.dma_free_task(%c_r0c1_m7_n24)
      // core r0c1 m7 n25
      %a_r0c1_m7_n25 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n25)
      %b_r0c1_m7_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n25)
      %c_r0c1_m7_n25 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804416 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n25)
      aiex.dma_await_task(%c_r0c1_m7_n25)
      aiex.dma_free_task(%a_r0c1_m7_n25)
      aiex.dma_free_task(%b_r0c1_m7_n25)
      aiex.dma_free_task(%c_r0c1_m7_n25)
      // core r0c1 m7 n26
      %a_r0c1_m7_n26 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n26)
      %b_r0c1_m7_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n26)
      %c_r0c1_m7_n26 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804480 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n26)
      aiex.dma_await_task(%c_r0c1_m7_n26)
      aiex.dma_free_task(%a_r0c1_m7_n26)
      aiex.dma_free_task(%b_r0c1_m7_n26)
      aiex.dma_free_task(%c_r0c1_m7_n26)
      // core r0c1 m7 n27
      %a_r0c1_m7_n27 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c1_m7_n27)
      %b_r0c1_m7_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c1_m7_n27)
      %c_r0c1_m7_n27 = aiex.dma_configure_task_for @C2_r0_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804544 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c1_m7_n27)
      aiex.dma_await_task(%c_r0c1_m7_n27)
      aiex.dma_free_task(%a_r0c1_m7_n27)
      aiex.dma_free_task(%b_r0c1_m7_n27)
      aiex.dma_free_task(%c_r0c1_m7_n27)
      // core r0c2 m0 n28
      %a_r0c2_m0_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n28)
      %b_r0c2_m0_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n28)
      %c_r0c2_m0_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1792 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n28)
      aiex.dma_await_task(%c_r0c2_m0_n28)
      aiex.dma_free_task(%a_r0c2_m0_n28)
      aiex.dma_free_task(%b_r0c2_m0_n28)
      aiex.dma_free_task(%c_r0c2_m0_n28)
      // core r0c2 m0 n29
      %a_r0c2_m0_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n29)
      %b_r0c2_m0_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n29)
      %c_r0c2_m0_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1856 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n29)
      aiex.dma_await_task(%c_r0c2_m0_n29)
      aiex.dma_free_task(%a_r0c2_m0_n29)
      aiex.dma_free_task(%b_r0c2_m0_n29)
      aiex.dma_free_task(%c_r0c2_m0_n29)
      // core r0c2 m0 n30
      %a_r0c2_m0_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n30)
      %b_r0c2_m0_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n30)
      %c_r0c2_m0_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1920 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n30)
      aiex.dma_await_task(%c_r0c2_m0_n30)
      aiex.dma_free_task(%a_r0c2_m0_n30)
      aiex.dma_free_task(%b_r0c2_m0_n30)
      aiex.dma_free_task(%c_r0c2_m0_n30)
      // core r0c2 m0 n31
      %a_r0c2_m0_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n31)
      %b_r0c2_m0_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n31)
      %c_r0c2_m0_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1984 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n31)
      aiex.dma_await_task(%c_r0c2_m0_n31)
      aiex.dma_free_task(%a_r0c2_m0_n31)
      aiex.dma_free_task(%b_r0c2_m0_n31)
      aiex.dma_free_task(%c_r0c2_m0_n31)
      // core r0c2 m0 n32
      %a_r0c2_m0_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n32)
      %b_r0c2_m0_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n32)
      %c_r0c2_m0_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2048 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n32)
      aiex.dma_await_task(%c_r0c2_m0_n32)
      aiex.dma_free_task(%a_r0c2_m0_n32)
      aiex.dma_free_task(%b_r0c2_m0_n32)
      aiex.dma_free_task(%c_r0c2_m0_n32)
      // core r0c2 m0 n33
      %a_r0c2_m0_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n33)
      %b_r0c2_m0_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n33)
      %c_r0c2_m0_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2112 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n33)
      aiex.dma_await_task(%c_r0c2_m0_n33)
      aiex.dma_free_task(%a_r0c2_m0_n33)
      aiex.dma_free_task(%b_r0c2_m0_n33)
      aiex.dma_free_task(%c_r0c2_m0_n33)
      // core r0c2 m0 n34
      %a_r0c2_m0_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n34)
      %b_r0c2_m0_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n34)
      %c_r0c2_m0_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2176 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n34)
      aiex.dma_await_task(%c_r0c2_m0_n34)
      aiex.dma_free_task(%a_r0c2_m0_n34)
      aiex.dma_free_task(%b_r0c2_m0_n34)
      aiex.dma_free_task(%c_r0c2_m0_n34)
      // core r0c2 m0 n35
      %a_r0c2_m0_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n35)
      %b_r0c2_m0_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n35)
      %c_r0c2_m0_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2240 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n35)
      aiex.dma_await_task(%c_r0c2_m0_n35)
      aiex.dma_free_task(%a_r0c2_m0_n35)
      aiex.dma_free_task(%b_r0c2_m0_n35)
      aiex.dma_free_task(%c_r0c2_m0_n35)
      // core r0c2 m0 n36
      %a_r0c2_m0_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n36)
      %b_r0c2_m0_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n36)
      %c_r0c2_m0_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2304 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n36)
      aiex.dma_await_task(%c_r0c2_m0_n36)
      aiex.dma_free_task(%a_r0c2_m0_n36)
      aiex.dma_free_task(%b_r0c2_m0_n36)
      aiex.dma_free_task(%c_r0c2_m0_n36)
      // core r0c2 m0 n37
      %a_r0c2_m0_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n37)
      %b_r0c2_m0_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n37)
      %c_r0c2_m0_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2368 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n37)
      aiex.dma_await_task(%c_r0c2_m0_n37)
      aiex.dma_free_task(%a_r0c2_m0_n37)
      aiex.dma_free_task(%b_r0c2_m0_n37)
      aiex.dma_free_task(%c_r0c2_m0_n37)
      // core r0c2 m0 n38
      %a_r0c2_m0_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n38)
      %b_r0c2_m0_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n38)
      %c_r0c2_m0_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2432 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n38)
      aiex.dma_await_task(%c_r0c2_m0_n38)
      aiex.dma_free_task(%a_r0c2_m0_n38)
      aiex.dma_free_task(%b_r0c2_m0_n38)
      aiex.dma_free_task(%c_r0c2_m0_n38)
      // core r0c2 m0 n39
      %a_r0c2_m0_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n39)
      %b_r0c2_m0_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n39)
      %c_r0c2_m0_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2496 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n39)
      aiex.dma_await_task(%c_r0c2_m0_n39)
      aiex.dma_free_task(%a_r0c2_m0_n39)
      aiex.dma_free_task(%b_r0c2_m0_n39)
      aiex.dma_free_task(%c_r0c2_m0_n39)
      // core r0c2 m0 n40
      %a_r0c2_m0_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n40)
      %b_r0c2_m0_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n40)
      %c_r0c2_m0_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2560 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n40)
      aiex.dma_await_task(%c_r0c2_m0_n40)
      aiex.dma_free_task(%a_r0c2_m0_n40)
      aiex.dma_free_task(%b_r0c2_m0_n40)
      aiex.dma_free_task(%c_r0c2_m0_n40)
      // core r0c2 m0 n41
      %a_r0c2_m0_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m0_n41)
      %b_r0c2_m0_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m0_n41)
      %c_r0c2_m0_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2624 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m0_n41)
      aiex.dma_await_task(%c_r0c2_m0_n41)
      aiex.dma_free_task(%a_r0c2_m0_n41)
      aiex.dma_free_task(%b_r0c2_m0_n41)
      aiex.dma_free_task(%c_r0c2_m0_n41)
      // core r0c2 m1 n28
      %a_r0c2_m1_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n28)
      %b_r0c2_m1_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n28)
      %c_r0c2_m1_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116480 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n28)
      aiex.dma_await_task(%c_r0c2_m1_n28)
      aiex.dma_free_task(%a_r0c2_m1_n28)
      aiex.dma_free_task(%b_r0c2_m1_n28)
      aiex.dma_free_task(%c_r0c2_m1_n28)
      // core r0c2 m1 n29
      %a_r0c2_m1_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n29)
      %b_r0c2_m1_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n29)
      %c_r0c2_m1_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116544 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n29)
      aiex.dma_await_task(%c_r0c2_m1_n29)
      aiex.dma_free_task(%a_r0c2_m1_n29)
      aiex.dma_free_task(%b_r0c2_m1_n29)
      aiex.dma_free_task(%c_r0c2_m1_n29)
      // core r0c2 m1 n30
      %a_r0c2_m1_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n30)
      %b_r0c2_m1_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n30)
      %c_r0c2_m1_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116608 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n30)
      aiex.dma_await_task(%c_r0c2_m1_n30)
      aiex.dma_free_task(%a_r0c2_m1_n30)
      aiex.dma_free_task(%b_r0c2_m1_n30)
      aiex.dma_free_task(%c_r0c2_m1_n30)
      // core r0c2 m1 n31
      %a_r0c2_m1_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n31)
      %b_r0c2_m1_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n31)
      %c_r0c2_m1_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116672 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n31)
      aiex.dma_await_task(%c_r0c2_m1_n31)
      aiex.dma_free_task(%a_r0c2_m1_n31)
      aiex.dma_free_task(%b_r0c2_m1_n31)
      aiex.dma_free_task(%c_r0c2_m1_n31)
      // core r0c2 m1 n32
      %a_r0c2_m1_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n32)
      %b_r0c2_m1_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n32)
      %c_r0c2_m1_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116736 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n32)
      aiex.dma_await_task(%c_r0c2_m1_n32)
      aiex.dma_free_task(%a_r0c2_m1_n32)
      aiex.dma_free_task(%b_r0c2_m1_n32)
      aiex.dma_free_task(%c_r0c2_m1_n32)
      // core r0c2 m1 n33
      %a_r0c2_m1_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n33)
      %b_r0c2_m1_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n33)
      %c_r0c2_m1_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116800 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n33)
      aiex.dma_await_task(%c_r0c2_m1_n33)
      aiex.dma_free_task(%a_r0c2_m1_n33)
      aiex.dma_free_task(%b_r0c2_m1_n33)
      aiex.dma_free_task(%c_r0c2_m1_n33)
      // core r0c2 m1 n34
      %a_r0c2_m1_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n34)
      %b_r0c2_m1_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n34)
      %c_r0c2_m1_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116864 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n34)
      aiex.dma_await_task(%c_r0c2_m1_n34)
      aiex.dma_free_task(%a_r0c2_m1_n34)
      aiex.dma_free_task(%b_r0c2_m1_n34)
      aiex.dma_free_task(%c_r0c2_m1_n34)
      // core r0c2 m1 n35
      %a_r0c2_m1_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n35)
      %b_r0c2_m1_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n35)
      %c_r0c2_m1_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116928 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n35)
      aiex.dma_await_task(%c_r0c2_m1_n35)
      aiex.dma_free_task(%a_r0c2_m1_n35)
      aiex.dma_free_task(%b_r0c2_m1_n35)
      aiex.dma_free_task(%c_r0c2_m1_n35)
      // core r0c2 m1 n36
      %a_r0c2_m1_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n36)
      %b_r0c2_m1_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n36)
      %c_r0c2_m1_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 116992 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n36)
      aiex.dma_await_task(%c_r0c2_m1_n36)
      aiex.dma_free_task(%a_r0c2_m1_n36)
      aiex.dma_free_task(%b_r0c2_m1_n36)
      aiex.dma_free_task(%c_r0c2_m1_n36)
      // core r0c2 m1 n37
      %a_r0c2_m1_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n37)
      %b_r0c2_m1_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n37)
      %c_r0c2_m1_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117056 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n37)
      aiex.dma_await_task(%c_r0c2_m1_n37)
      aiex.dma_free_task(%a_r0c2_m1_n37)
      aiex.dma_free_task(%b_r0c2_m1_n37)
      aiex.dma_free_task(%c_r0c2_m1_n37)
      // core r0c2 m1 n38
      %a_r0c2_m1_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n38)
      %b_r0c2_m1_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n38)
      %c_r0c2_m1_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117120 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n38)
      aiex.dma_await_task(%c_r0c2_m1_n38)
      aiex.dma_free_task(%a_r0c2_m1_n38)
      aiex.dma_free_task(%b_r0c2_m1_n38)
      aiex.dma_free_task(%c_r0c2_m1_n38)
      // core r0c2 m1 n39
      %a_r0c2_m1_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n39)
      %b_r0c2_m1_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n39)
      %c_r0c2_m1_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117184 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n39)
      aiex.dma_await_task(%c_r0c2_m1_n39)
      aiex.dma_free_task(%a_r0c2_m1_n39)
      aiex.dma_free_task(%b_r0c2_m1_n39)
      aiex.dma_free_task(%c_r0c2_m1_n39)
      // core r0c2 m1 n40
      %a_r0c2_m1_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n40)
      %b_r0c2_m1_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n40)
      %c_r0c2_m1_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117248 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n40)
      aiex.dma_await_task(%c_r0c2_m1_n40)
      aiex.dma_free_task(%a_r0c2_m1_n40)
      aiex.dma_free_task(%b_r0c2_m1_n40)
      aiex.dma_free_task(%c_r0c2_m1_n40)
      // core r0c2 m1 n41
      %a_r0c2_m1_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m1_n41)
      %b_r0c2_m1_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m1_n41)
      %c_r0c2_m1_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117312 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m1_n41)
      aiex.dma_await_task(%c_r0c2_m1_n41)
      aiex.dma_free_task(%a_r0c2_m1_n41)
      aiex.dma_free_task(%b_r0c2_m1_n41)
      aiex.dma_free_task(%c_r0c2_m1_n41)
      // core r0c2 m2 n28
      %a_r0c2_m2_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n28)
      %b_r0c2_m2_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n28)
      %c_r0c2_m2_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231168 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n28)
      aiex.dma_await_task(%c_r0c2_m2_n28)
      aiex.dma_free_task(%a_r0c2_m2_n28)
      aiex.dma_free_task(%b_r0c2_m2_n28)
      aiex.dma_free_task(%c_r0c2_m2_n28)
      // core r0c2 m2 n29
      %a_r0c2_m2_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n29)
      %b_r0c2_m2_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n29)
      %c_r0c2_m2_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231232 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n29)
      aiex.dma_await_task(%c_r0c2_m2_n29)
      aiex.dma_free_task(%a_r0c2_m2_n29)
      aiex.dma_free_task(%b_r0c2_m2_n29)
      aiex.dma_free_task(%c_r0c2_m2_n29)
      // core r0c2 m2 n30
      %a_r0c2_m2_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n30)
      %b_r0c2_m2_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n30)
      %c_r0c2_m2_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231296 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n30)
      aiex.dma_await_task(%c_r0c2_m2_n30)
      aiex.dma_free_task(%a_r0c2_m2_n30)
      aiex.dma_free_task(%b_r0c2_m2_n30)
      aiex.dma_free_task(%c_r0c2_m2_n30)
      // core r0c2 m2 n31
      %a_r0c2_m2_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n31)
      %b_r0c2_m2_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n31)
      %c_r0c2_m2_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231360 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n31)
      aiex.dma_await_task(%c_r0c2_m2_n31)
      aiex.dma_free_task(%a_r0c2_m2_n31)
      aiex.dma_free_task(%b_r0c2_m2_n31)
      aiex.dma_free_task(%c_r0c2_m2_n31)
      // core r0c2 m2 n32
      %a_r0c2_m2_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n32)
      %b_r0c2_m2_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n32)
      %c_r0c2_m2_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231424 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n32)
      aiex.dma_await_task(%c_r0c2_m2_n32)
      aiex.dma_free_task(%a_r0c2_m2_n32)
      aiex.dma_free_task(%b_r0c2_m2_n32)
      aiex.dma_free_task(%c_r0c2_m2_n32)
      // core r0c2 m2 n33
      %a_r0c2_m2_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n33)
      %b_r0c2_m2_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n33)
      %c_r0c2_m2_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231488 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n33)
      aiex.dma_await_task(%c_r0c2_m2_n33)
      aiex.dma_free_task(%a_r0c2_m2_n33)
      aiex.dma_free_task(%b_r0c2_m2_n33)
      aiex.dma_free_task(%c_r0c2_m2_n33)
      // core r0c2 m2 n34
      %a_r0c2_m2_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n34)
      %b_r0c2_m2_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n34)
      %c_r0c2_m2_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231552 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n34)
      aiex.dma_await_task(%c_r0c2_m2_n34)
      aiex.dma_free_task(%a_r0c2_m2_n34)
      aiex.dma_free_task(%b_r0c2_m2_n34)
      aiex.dma_free_task(%c_r0c2_m2_n34)
      // core r0c2 m2 n35
      %a_r0c2_m2_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n35)
      %b_r0c2_m2_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n35)
      %c_r0c2_m2_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231616 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n35)
      aiex.dma_await_task(%c_r0c2_m2_n35)
      aiex.dma_free_task(%a_r0c2_m2_n35)
      aiex.dma_free_task(%b_r0c2_m2_n35)
      aiex.dma_free_task(%c_r0c2_m2_n35)
      // core r0c2 m2 n36
      %a_r0c2_m2_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n36)
      %b_r0c2_m2_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n36)
      %c_r0c2_m2_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231680 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n36)
      aiex.dma_await_task(%c_r0c2_m2_n36)
      aiex.dma_free_task(%a_r0c2_m2_n36)
      aiex.dma_free_task(%b_r0c2_m2_n36)
      aiex.dma_free_task(%c_r0c2_m2_n36)
      // core r0c2 m2 n37
      %a_r0c2_m2_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n37)
      %b_r0c2_m2_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n37)
      %c_r0c2_m2_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231744 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n37)
      aiex.dma_await_task(%c_r0c2_m2_n37)
      aiex.dma_free_task(%a_r0c2_m2_n37)
      aiex.dma_free_task(%b_r0c2_m2_n37)
      aiex.dma_free_task(%c_r0c2_m2_n37)
      // core r0c2 m2 n38
      %a_r0c2_m2_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n38)
      %b_r0c2_m2_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n38)
      %c_r0c2_m2_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231808 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n38)
      aiex.dma_await_task(%c_r0c2_m2_n38)
      aiex.dma_free_task(%a_r0c2_m2_n38)
      aiex.dma_free_task(%b_r0c2_m2_n38)
      aiex.dma_free_task(%c_r0c2_m2_n38)
      // core r0c2 m2 n39
      %a_r0c2_m2_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n39)
      %b_r0c2_m2_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n39)
      %c_r0c2_m2_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231872 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n39)
      aiex.dma_await_task(%c_r0c2_m2_n39)
      aiex.dma_free_task(%a_r0c2_m2_n39)
      aiex.dma_free_task(%b_r0c2_m2_n39)
      aiex.dma_free_task(%c_r0c2_m2_n39)
      // core r0c2 m2 n40
      %a_r0c2_m2_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n40)
      %b_r0c2_m2_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n40)
      %c_r0c2_m2_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 231936 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n40)
      aiex.dma_await_task(%c_r0c2_m2_n40)
      aiex.dma_free_task(%a_r0c2_m2_n40)
      aiex.dma_free_task(%b_r0c2_m2_n40)
      aiex.dma_free_task(%c_r0c2_m2_n40)
      // core r0c2 m2 n41
      %a_r0c2_m2_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m2_n41)
      %b_r0c2_m2_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m2_n41)
      %c_r0c2_m2_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232000 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m2_n41)
      aiex.dma_await_task(%c_r0c2_m2_n41)
      aiex.dma_free_task(%a_r0c2_m2_n41)
      aiex.dma_free_task(%b_r0c2_m2_n41)
      aiex.dma_free_task(%c_r0c2_m2_n41)
      // core r0c2 m3 n28
      %a_r0c2_m3_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n28)
      %b_r0c2_m3_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n28)
      %c_r0c2_m3_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345856 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n28)
      aiex.dma_await_task(%c_r0c2_m3_n28)
      aiex.dma_free_task(%a_r0c2_m3_n28)
      aiex.dma_free_task(%b_r0c2_m3_n28)
      aiex.dma_free_task(%c_r0c2_m3_n28)
      // core r0c2 m3 n29
      %a_r0c2_m3_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n29)
      %b_r0c2_m3_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n29)
      %c_r0c2_m3_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345920 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n29)
      aiex.dma_await_task(%c_r0c2_m3_n29)
      aiex.dma_free_task(%a_r0c2_m3_n29)
      aiex.dma_free_task(%b_r0c2_m3_n29)
      aiex.dma_free_task(%c_r0c2_m3_n29)
      // core r0c2 m3 n30
      %a_r0c2_m3_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n30)
      %b_r0c2_m3_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n30)
      %c_r0c2_m3_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 345984 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n30)
      aiex.dma_await_task(%c_r0c2_m3_n30)
      aiex.dma_free_task(%a_r0c2_m3_n30)
      aiex.dma_free_task(%b_r0c2_m3_n30)
      aiex.dma_free_task(%c_r0c2_m3_n30)
      // core r0c2 m3 n31
      %a_r0c2_m3_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n31)
      %b_r0c2_m3_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n31)
      %c_r0c2_m3_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346048 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n31)
      aiex.dma_await_task(%c_r0c2_m3_n31)
      aiex.dma_free_task(%a_r0c2_m3_n31)
      aiex.dma_free_task(%b_r0c2_m3_n31)
      aiex.dma_free_task(%c_r0c2_m3_n31)
      // core r0c2 m3 n32
      %a_r0c2_m3_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n32)
      %b_r0c2_m3_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n32)
      %c_r0c2_m3_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346112 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n32)
      aiex.dma_await_task(%c_r0c2_m3_n32)
      aiex.dma_free_task(%a_r0c2_m3_n32)
      aiex.dma_free_task(%b_r0c2_m3_n32)
      aiex.dma_free_task(%c_r0c2_m3_n32)
      // core r0c2 m3 n33
      %a_r0c2_m3_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n33)
      %b_r0c2_m3_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n33)
      %c_r0c2_m3_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346176 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n33)
      aiex.dma_await_task(%c_r0c2_m3_n33)
      aiex.dma_free_task(%a_r0c2_m3_n33)
      aiex.dma_free_task(%b_r0c2_m3_n33)
      aiex.dma_free_task(%c_r0c2_m3_n33)
      // core r0c2 m3 n34
      %a_r0c2_m3_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n34)
      %b_r0c2_m3_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n34)
      %c_r0c2_m3_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346240 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n34)
      aiex.dma_await_task(%c_r0c2_m3_n34)
      aiex.dma_free_task(%a_r0c2_m3_n34)
      aiex.dma_free_task(%b_r0c2_m3_n34)
      aiex.dma_free_task(%c_r0c2_m3_n34)
      // core r0c2 m3 n35
      %a_r0c2_m3_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n35)
      %b_r0c2_m3_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n35)
      %c_r0c2_m3_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346304 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n35)
      aiex.dma_await_task(%c_r0c2_m3_n35)
      aiex.dma_free_task(%a_r0c2_m3_n35)
      aiex.dma_free_task(%b_r0c2_m3_n35)
      aiex.dma_free_task(%c_r0c2_m3_n35)
      // core r0c2 m3 n36
      %a_r0c2_m3_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n36)
      %b_r0c2_m3_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n36)
      %c_r0c2_m3_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346368 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n36)
      aiex.dma_await_task(%c_r0c2_m3_n36)
      aiex.dma_free_task(%a_r0c2_m3_n36)
      aiex.dma_free_task(%b_r0c2_m3_n36)
      aiex.dma_free_task(%c_r0c2_m3_n36)
      // core r0c2 m3 n37
      %a_r0c2_m3_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n37)
      %b_r0c2_m3_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n37)
      %c_r0c2_m3_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346432 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n37)
      aiex.dma_await_task(%c_r0c2_m3_n37)
      aiex.dma_free_task(%a_r0c2_m3_n37)
      aiex.dma_free_task(%b_r0c2_m3_n37)
      aiex.dma_free_task(%c_r0c2_m3_n37)
      // core r0c2 m3 n38
      %a_r0c2_m3_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n38)
      %b_r0c2_m3_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n38)
      %c_r0c2_m3_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346496 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n38)
      aiex.dma_await_task(%c_r0c2_m3_n38)
      aiex.dma_free_task(%a_r0c2_m3_n38)
      aiex.dma_free_task(%b_r0c2_m3_n38)
      aiex.dma_free_task(%c_r0c2_m3_n38)
      // core r0c2 m3 n39
      %a_r0c2_m3_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n39)
      %b_r0c2_m3_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n39)
      %c_r0c2_m3_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346560 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n39)
      aiex.dma_await_task(%c_r0c2_m3_n39)
      aiex.dma_free_task(%a_r0c2_m3_n39)
      aiex.dma_free_task(%b_r0c2_m3_n39)
      aiex.dma_free_task(%c_r0c2_m3_n39)
      // core r0c2 m3 n40
      %a_r0c2_m3_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n40)
      %b_r0c2_m3_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n40)
      %c_r0c2_m3_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346624 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n40)
      aiex.dma_await_task(%c_r0c2_m3_n40)
      aiex.dma_free_task(%a_r0c2_m3_n40)
      aiex.dma_free_task(%b_r0c2_m3_n40)
      aiex.dma_free_task(%c_r0c2_m3_n40)
      // core r0c2 m3 n41
      %a_r0c2_m3_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m3_n41)
      %b_r0c2_m3_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m3_n41)
      %c_r0c2_m3_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m3_n41)
      aiex.dma_await_task(%c_r0c2_m3_n41)
      aiex.dma_free_task(%a_r0c2_m3_n41)
      aiex.dma_free_task(%b_r0c2_m3_n41)
      aiex.dma_free_task(%c_r0c2_m3_n41)
      // core r0c2 m4 n28
      %a_r0c2_m4_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n28)
      %b_r0c2_m4_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n28)
      %c_r0c2_m4_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460544 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n28)
      aiex.dma_await_task(%c_r0c2_m4_n28)
      aiex.dma_free_task(%a_r0c2_m4_n28)
      aiex.dma_free_task(%b_r0c2_m4_n28)
      aiex.dma_free_task(%c_r0c2_m4_n28)
      // core r0c2 m4 n29
      %a_r0c2_m4_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n29)
      %b_r0c2_m4_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n29)
      %c_r0c2_m4_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460608 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n29)
      aiex.dma_await_task(%c_r0c2_m4_n29)
      aiex.dma_free_task(%a_r0c2_m4_n29)
      aiex.dma_free_task(%b_r0c2_m4_n29)
      aiex.dma_free_task(%c_r0c2_m4_n29)
      // core r0c2 m4 n30
      %a_r0c2_m4_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n30)
      %b_r0c2_m4_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n30)
      %c_r0c2_m4_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460672 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n30)
      aiex.dma_await_task(%c_r0c2_m4_n30)
      aiex.dma_free_task(%a_r0c2_m4_n30)
      aiex.dma_free_task(%b_r0c2_m4_n30)
      aiex.dma_free_task(%c_r0c2_m4_n30)
      // core r0c2 m4 n31
      %a_r0c2_m4_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n31)
      %b_r0c2_m4_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n31)
      %c_r0c2_m4_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460736 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n31)
      aiex.dma_await_task(%c_r0c2_m4_n31)
      aiex.dma_free_task(%a_r0c2_m4_n31)
      aiex.dma_free_task(%b_r0c2_m4_n31)
      aiex.dma_free_task(%c_r0c2_m4_n31)
      // core r0c2 m4 n32
      %a_r0c2_m4_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n32)
      %b_r0c2_m4_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n32)
      %c_r0c2_m4_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460800 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n32)
      aiex.dma_await_task(%c_r0c2_m4_n32)
      aiex.dma_free_task(%a_r0c2_m4_n32)
      aiex.dma_free_task(%b_r0c2_m4_n32)
      aiex.dma_free_task(%c_r0c2_m4_n32)
      // core r0c2 m4 n33
      %a_r0c2_m4_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n33)
      %b_r0c2_m4_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n33)
      %c_r0c2_m4_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460864 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n33)
      aiex.dma_await_task(%c_r0c2_m4_n33)
      aiex.dma_free_task(%a_r0c2_m4_n33)
      aiex.dma_free_task(%b_r0c2_m4_n33)
      aiex.dma_free_task(%c_r0c2_m4_n33)
      // core r0c2 m4 n34
      %a_r0c2_m4_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n34)
      %b_r0c2_m4_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n34)
      %c_r0c2_m4_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460928 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n34)
      aiex.dma_await_task(%c_r0c2_m4_n34)
      aiex.dma_free_task(%a_r0c2_m4_n34)
      aiex.dma_free_task(%b_r0c2_m4_n34)
      aiex.dma_free_task(%c_r0c2_m4_n34)
      // core r0c2 m4 n35
      %a_r0c2_m4_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n35)
      %b_r0c2_m4_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n35)
      %c_r0c2_m4_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 460992 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n35)
      aiex.dma_await_task(%c_r0c2_m4_n35)
      aiex.dma_free_task(%a_r0c2_m4_n35)
      aiex.dma_free_task(%b_r0c2_m4_n35)
      aiex.dma_free_task(%c_r0c2_m4_n35)
      // core r0c2 m4 n36
      %a_r0c2_m4_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n36)
      %b_r0c2_m4_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n36)
      %c_r0c2_m4_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461056 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n36)
      aiex.dma_await_task(%c_r0c2_m4_n36)
      aiex.dma_free_task(%a_r0c2_m4_n36)
      aiex.dma_free_task(%b_r0c2_m4_n36)
      aiex.dma_free_task(%c_r0c2_m4_n36)
      // core r0c2 m4 n37
      %a_r0c2_m4_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n37)
      %b_r0c2_m4_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n37)
      %c_r0c2_m4_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461120 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n37)
      aiex.dma_await_task(%c_r0c2_m4_n37)
      aiex.dma_free_task(%a_r0c2_m4_n37)
      aiex.dma_free_task(%b_r0c2_m4_n37)
      aiex.dma_free_task(%c_r0c2_m4_n37)
      // core r0c2 m4 n38
      %a_r0c2_m4_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n38)
      %b_r0c2_m4_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n38)
      %c_r0c2_m4_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461184 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n38)
      aiex.dma_await_task(%c_r0c2_m4_n38)
      aiex.dma_free_task(%a_r0c2_m4_n38)
      aiex.dma_free_task(%b_r0c2_m4_n38)
      aiex.dma_free_task(%c_r0c2_m4_n38)
      // core r0c2 m4 n39
      %a_r0c2_m4_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n39)
      %b_r0c2_m4_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n39)
      %c_r0c2_m4_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461248 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n39)
      aiex.dma_await_task(%c_r0c2_m4_n39)
      aiex.dma_free_task(%a_r0c2_m4_n39)
      aiex.dma_free_task(%b_r0c2_m4_n39)
      aiex.dma_free_task(%c_r0c2_m4_n39)
      // core r0c2 m4 n40
      %a_r0c2_m4_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n40)
      %b_r0c2_m4_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n40)
      %c_r0c2_m4_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461312 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n40)
      aiex.dma_await_task(%c_r0c2_m4_n40)
      aiex.dma_free_task(%a_r0c2_m4_n40)
      aiex.dma_free_task(%b_r0c2_m4_n40)
      aiex.dma_free_task(%c_r0c2_m4_n40)
      // core r0c2 m4 n41
      %a_r0c2_m4_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m4_n41)
      %b_r0c2_m4_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m4_n41)
      %c_r0c2_m4_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461376 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m4_n41)
      aiex.dma_await_task(%c_r0c2_m4_n41)
      aiex.dma_free_task(%a_r0c2_m4_n41)
      aiex.dma_free_task(%b_r0c2_m4_n41)
      aiex.dma_free_task(%c_r0c2_m4_n41)
      // core r0c2 m5 n28
      %a_r0c2_m5_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n28)
      %b_r0c2_m5_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n28)
      %c_r0c2_m5_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575232 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n28)
      aiex.dma_await_task(%c_r0c2_m5_n28)
      aiex.dma_free_task(%a_r0c2_m5_n28)
      aiex.dma_free_task(%b_r0c2_m5_n28)
      aiex.dma_free_task(%c_r0c2_m5_n28)
      // core r0c2 m5 n29
      %a_r0c2_m5_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n29)
      %b_r0c2_m5_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n29)
      %c_r0c2_m5_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575296 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n29)
      aiex.dma_await_task(%c_r0c2_m5_n29)
      aiex.dma_free_task(%a_r0c2_m5_n29)
      aiex.dma_free_task(%b_r0c2_m5_n29)
      aiex.dma_free_task(%c_r0c2_m5_n29)
      // core r0c2 m5 n30
      %a_r0c2_m5_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n30)
      %b_r0c2_m5_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n30)
      %c_r0c2_m5_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575360 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n30)
      aiex.dma_await_task(%c_r0c2_m5_n30)
      aiex.dma_free_task(%a_r0c2_m5_n30)
      aiex.dma_free_task(%b_r0c2_m5_n30)
      aiex.dma_free_task(%c_r0c2_m5_n30)
      // core r0c2 m5 n31
      %a_r0c2_m5_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n31)
      %b_r0c2_m5_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n31)
      %c_r0c2_m5_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575424 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n31)
      aiex.dma_await_task(%c_r0c2_m5_n31)
      aiex.dma_free_task(%a_r0c2_m5_n31)
      aiex.dma_free_task(%b_r0c2_m5_n31)
      aiex.dma_free_task(%c_r0c2_m5_n31)
      // core r0c2 m5 n32
      %a_r0c2_m5_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n32)
      %b_r0c2_m5_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n32)
      %c_r0c2_m5_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575488 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n32)
      aiex.dma_await_task(%c_r0c2_m5_n32)
      aiex.dma_free_task(%a_r0c2_m5_n32)
      aiex.dma_free_task(%b_r0c2_m5_n32)
      aiex.dma_free_task(%c_r0c2_m5_n32)
      // core r0c2 m5 n33
      %a_r0c2_m5_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n33)
      %b_r0c2_m5_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n33)
      %c_r0c2_m5_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575552 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n33)
      aiex.dma_await_task(%c_r0c2_m5_n33)
      aiex.dma_free_task(%a_r0c2_m5_n33)
      aiex.dma_free_task(%b_r0c2_m5_n33)
      aiex.dma_free_task(%c_r0c2_m5_n33)
      // core r0c2 m5 n34
      %a_r0c2_m5_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n34)
      %b_r0c2_m5_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n34)
      %c_r0c2_m5_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575616 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n34)
      aiex.dma_await_task(%c_r0c2_m5_n34)
      aiex.dma_free_task(%a_r0c2_m5_n34)
      aiex.dma_free_task(%b_r0c2_m5_n34)
      aiex.dma_free_task(%c_r0c2_m5_n34)
      // core r0c2 m5 n35
      %a_r0c2_m5_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n35)
      %b_r0c2_m5_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n35)
      %c_r0c2_m5_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575680 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n35)
      aiex.dma_await_task(%c_r0c2_m5_n35)
      aiex.dma_free_task(%a_r0c2_m5_n35)
      aiex.dma_free_task(%b_r0c2_m5_n35)
      aiex.dma_free_task(%c_r0c2_m5_n35)
      // core r0c2 m5 n36
      %a_r0c2_m5_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n36)
      %b_r0c2_m5_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n36)
      %c_r0c2_m5_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575744 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n36)
      aiex.dma_await_task(%c_r0c2_m5_n36)
      aiex.dma_free_task(%a_r0c2_m5_n36)
      aiex.dma_free_task(%b_r0c2_m5_n36)
      aiex.dma_free_task(%c_r0c2_m5_n36)
      // core r0c2 m5 n37
      %a_r0c2_m5_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n37)
      %b_r0c2_m5_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n37)
      %c_r0c2_m5_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575808 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n37)
      aiex.dma_await_task(%c_r0c2_m5_n37)
      aiex.dma_free_task(%a_r0c2_m5_n37)
      aiex.dma_free_task(%b_r0c2_m5_n37)
      aiex.dma_free_task(%c_r0c2_m5_n37)
      // core r0c2 m5 n38
      %a_r0c2_m5_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n38)
      %b_r0c2_m5_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n38)
      %c_r0c2_m5_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575872 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n38)
      aiex.dma_await_task(%c_r0c2_m5_n38)
      aiex.dma_free_task(%a_r0c2_m5_n38)
      aiex.dma_free_task(%b_r0c2_m5_n38)
      aiex.dma_free_task(%c_r0c2_m5_n38)
      // core r0c2 m5 n39
      %a_r0c2_m5_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n39)
      %b_r0c2_m5_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n39)
      %c_r0c2_m5_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 575936 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n39)
      aiex.dma_await_task(%c_r0c2_m5_n39)
      aiex.dma_free_task(%a_r0c2_m5_n39)
      aiex.dma_free_task(%b_r0c2_m5_n39)
      aiex.dma_free_task(%c_r0c2_m5_n39)
      // core r0c2 m5 n40
      %a_r0c2_m5_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n40)
      %b_r0c2_m5_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n40)
      %c_r0c2_m5_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576000 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n40)
      aiex.dma_await_task(%c_r0c2_m5_n40)
      aiex.dma_free_task(%a_r0c2_m5_n40)
      aiex.dma_free_task(%b_r0c2_m5_n40)
      aiex.dma_free_task(%c_r0c2_m5_n40)
      // core r0c2 m5 n41
      %a_r0c2_m5_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m5_n41)
      %b_r0c2_m5_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m5_n41)
      %c_r0c2_m5_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576064 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m5_n41)
      aiex.dma_await_task(%c_r0c2_m5_n41)
      aiex.dma_free_task(%a_r0c2_m5_n41)
      aiex.dma_free_task(%b_r0c2_m5_n41)
      aiex.dma_free_task(%c_r0c2_m5_n41)
      // core r0c2 m6 n28
      %a_r0c2_m6_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n28)
      %b_r0c2_m6_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n28)
      %c_r0c2_m6_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689920 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n28)
      aiex.dma_await_task(%c_r0c2_m6_n28)
      aiex.dma_free_task(%a_r0c2_m6_n28)
      aiex.dma_free_task(%b_r0c2_m6_n28)
      aiex.dma_free_task(%c_r0c2_m6_n28)
      // core r0c2 m6 n29
      %a_r0c2_m6_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n29)
      %b_r0c2_m6_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n29)
      %c_r0c2_m6_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 689984 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n29)
      aiex.dma_await_task(%c_r0c2_m6_n29)
      aiex.dma_free_task(%a_r0c2_m6_n29)
      aiex.dma_free_task(%b_r0c2_m6_n29)
      aiex.dma_free_task(%c_r0c2_m6_n29)
      // core r0c2 m6 n30
      %a_r0c2_m6_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n30)
      %b_r0c2_m6_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n30)
      %c_r0c2_m6_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690048 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n30)
      aiex.dma_await_task(%c_r0c2_m6_n30)
      aiex.dma_free_task(%a_r0c2_m6_n30)
      aiex.dma_free_task(%b_r0c2_m6_n30)
      aiex.dma_free_task(%c_r0c2_m6_n30)
      // core r0c2 m6 n31
      %a_r0c2_m6_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n31)
      %b_r0c2_m6_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n31)
      %c_r0c2_m6_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690112 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n31)
      aiex.dma_await_task(%c_r0c2_m6_n31)
      aiex.dma_free_task(%a_r0c2_m6_n31)
      aiex.dma_free_task(%b_r0c2_m6_n31)
      aiex.dma_free_task(%c_r0c2_m6_n31)
      // core r0c2 m6 n32
      %a_r0c2_m6_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n32)
      %b_r0c2_m6_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n32)
      %c_r0c2_m6_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690176 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n32)
      aiex.dma_await_task(%c_r0c2_m6_n32)
      aiex.dma_free_task(%a_r0c2_m6_n32)
      aiex.dma_free_task(%b_r0c2_m6_n32)
      aiex.dma_free_task(%c_r0c2_m6_n32)
      // core r0c2 m6 n33
      %a_r0c2_m6_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n33)
      %b_r0c2_m6_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n33)
      %c_r0c2_m6_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690240 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n33)
      aiex.dma_await_task(%c_r0c2_m6_n33)
      aiex.dma_free_task(%a_r0c2_m6_n33)
      aiex.dma_free_task(%b_r0c2_m6_n33)
      aiex.dma_free_task(%c_r0c2_m6_n33)
      // core r0c2 m6 n34
      %a_r0c2_m6_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n34)
      %b_r0c2_m6_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n34)
      %c_r0c2_m6_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690304 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n34)
      aiex.dma_await_task(%c_r0c2_m6_n34)
      aiex.dma_free_task(%a_r0c2_m6_n34)
      aiex.dma_free_task(%b_r0c2_m6_n34)
      aiex.dma_free_task(%c_r0c2_m6_n34)
      // core r0c2 m6 n35
      %a_r0c2_m6_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n35)
      %b_r0c2_m6_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n35)
      %c_r0c2_m6_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690368 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n35)
      aiex.dma_await_task(%c_r0c2_m6_n35)
      aiex.dma_free_task(%a_r0c2_m6_n35)
      aiex.dma_free_task(%b_r0c2_m6_n35)
      aiex.dma_free_task(%c_r0c2_m6_n35)
      // core r0c2 m6 n36
      %a_r0c2_m6_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n36)
      %b_r0c2_m6_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n36)
      %c_r0c2_m6_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690432 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n36)
      aiex.dma_await_task(%c_r0c2_m6_n36)
      aiex.dma_free_task(%a_r0c2_m6_n36)
      aiex.dma_free_task(%b_r0c2_m6_n36)
      aiex.dma_free_task(%c_r0c2_m6_n36)
      // core r0c2 m6 n37
      %a_r0c2_m6_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n37)
      %b_r0c2_m6_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n37)
      %c_r0c2_m6_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690496 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n37)
      aiex.dma_await_task(%c_r0c2_m6_n37)
      aiex.dma_free_task(%a_r0c2_m6_n37)
      aiex.dma_free_task(%b_r0c2_m6_n37)
      aiex.dma_free_task(%c_r0c2_m6_n37)
      // core r0c2 m6 n38
      %a_r0c2_m6_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n38)
      %b_r0c2_m6_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n38)
      %c_r0c2_m6_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690560 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n38)
      aiex.dma_await_task(%c_r0c2_m6_n38)
      aiex.dma_free_task(%a_r0c2_m6_n38)
      aiex.dma_free_task(%b_r0c2_m6_n38)
      aiex.dma_free_task(%c_r0c2_m6_n38)
      // core r0c2 m6 n39
      %a_r0c2_m6_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n39)
      %b_r0c2_m6_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n39)
      %c_r0c2_m6_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690624 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n39)
      aiex.dma_await_task(%c_r0c2_m6_n39)
      aiex.dma_free_task(%a_r0c2_m6_n39)
      aiex.dma_free_task(%b_r0c2_m6_n39)
      aiex.dma_free_task(%c_r0c2_m6_n39)
      // core r0c2 m6 n40
      %a_r0c2_m6_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n40)
      %b_r0c2_m6_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n40)
      %c_r0c2_m6_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n40)
      aiex.dma_await_task(%c_r0c2_m6_n40)
      aiex.dma_free_task(%a_r0c2_m6_n40)
      aiex.dma_free_task(%b_r0c2_m6_n40)
      aiex.dma_free_task(%c_r0c2_m6_n40)
      // core r0c2 m6 n41
      %a_r0c2_m6_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m6_n41)
      %b_r0c2_m6_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m6_n41)
      %c_r0c2_m6_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m6_n41)
      aiex.dma_await_task(%c_r0c2_m6_n41)
      aiex.dma_free_task(%a_r0c2_m6_n41)
      aiex.dma_free_task(%b_r0c2_m6_n41)
      aiex.dma_free_task(%c_r0c2_m6_n41)
      // core r0c2 m7 n28
      %a_r0c2_m7_n28 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n28)
      %b_r0c2_m7_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n28)
      %c_r0c2_m7_n28 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804608 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n28)
      aiex.dma_await_task(%c_r0c2_m7_n28)
      aiex.dma_free_task(%a_r0c2_m7_n28)
      aiex.dma_free_task(%b_r0c2_m7_n28)
      aiex.dma_free_task(%c_r0c2_m7_n28)
      // core r0c2 m7 n29
      %a_r0c2_m7_n29 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n29)
      %b_r0c2_m7_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n29)
      %c_r0c2_m7_n29 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804672 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n29)
      aiex.dma_await_task(%c_r0c2_m7_n29)
      aiex.dma_free_task(%a_r0c2_m7_n29)
      aiex.dma_free_task(%b_r0c2_m7_n29)
      aiex.dma_free_task(%c_r0c2_m7_n29)
      // core r0c2 m7 n30
      %a_r0c2_m7_n30 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n30)
      %b_r0c2_m7_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n30)
      %c_r0c2_m7_n30 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804736 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n30)
      aiex.dma_await_task(%c_r0c2_m7_n30)
      aiex.dma_free_task(%a_r0c2_m7_n30)
      aiex.dma_free_task(%b_r0c2_m7_n30)
      aiex.dma_free_task(%c_r0c2_m7_n30)
      // core r0c2 m7 n31
      %a_r0c2_m7_n31 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n31)
      %b_r0c2_m7_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n31)
      %c_r0c2_m7_n31 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804800 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n31)
      aiex.dma_await_task(%c_r0c2_m7_n31)
      aiex.dma_free_task(%a_r0c2_m7_n31)
      aiex.dma_free_task(%b_r0c2_m7_n31)
      aiex.dma_free_task(%c_r0c2_m7_n31)
      // core r0c2 m7 n32
      %a_r0c2_m7_n32 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n32)
      %b_r0c2_m7_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n32)
      %c_r0c2_m7_n32 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804864 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n32)
      aiex.dma_await_task(%c_r0c2_m7_n32)
      aiex.dma_free_task(%a_r0c2_m7_n32)
      aiex.dma_free_task(%b_r0c2_m7_n32)
      aiex.dma_free_task(%c_r0c2_m7_n32)
      // core r0c2 m7 n33
      %a_r0c2_m7_n33 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n33)
      %b_r0c2_m7_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n33)
      %c_r0c2_m7_n33 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804928 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n33)
      aiex.dma_await_task(%c_r0c2_m7_n33)
      aiex.dma_free_task(%a_r0c2_m7_n33)
      aiex.dma_free_task(%b_r0c2_m7_n33)
      aiex.dma_free_task(%c_r0c2_m7_n33)
      // core r0c2 m7 n34
      %a_r0c2_m7_n34 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n34)
      %b_r0c2_m7_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n34)
      %c_r0c2_m7_n34 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 804992 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n34)
      aiex.dma_await_task(%c_r0c2_m7_n34)
      aiex.dma_free_task(%a_r0c2_m7_n34)
      aiex.dma_free_task(%b_r0c2_m7_n34)
      aiex.dma_free_task(%c_r0c2_m7_n34)
      // core r0c2 m7 n35
      %a_r0c2_m7_n35 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n35)
      %b_r0c2_m7_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n35)
      %c_r0c2_m7_n35 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805056 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n35)
      aiex.dma_await_task(%c_r0c2_m7_n35)
      aiex.dma_free_task(%a_r0c2_m7_n35)
      aiex.dma_free_task(%b_r0c2_m7_n35)
      aiex.dma_free_task(%c_r0c2_m7_n35)
      // core r0c2 m7 n36
      %a_r0c2_m7_n36 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n36)
      %b_r0c2_m7_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n36)
      %c_r0c2_m7_n36 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805120 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n36)
      aiex.dma_await_task(%c_r0c2_m7_n36)
      aiex.dma_free_task(%a_r0c2_m7_n36)
      aiex.dma_free_task(%b_r0c2_m7_n36)
      aiex.dma_free_task(%c_r0c2_m7_n36)
      // core r0c2 m7 n37
      %a_r0c2_m7_n37 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n37)
      %b_r0c2_m7_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n37)
      %c_r0c2_m7_n37 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805184 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n37)
      aiex.dma_await_task(%c_r0c2_m7_n37)
      aiex.dma_free_task(%a_r0c2_m7_n37)
      aiex.dma_free_task(%b_r0c2_m7_n37)
      aiex.dma_free_task(%c_r0c2_m7_n37)
      // core r0c2 m7 n38
      %a_r0c2_m7_n38 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n38)
      %b_r0c2_m7_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n38)
      %c_r0c2_m7_n38 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805248 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n38)
      aiex.dma_await_task(%c_r0c2_m7_n38)
      aiex.dma_free_task(%a_r0c2_m7_n38)
      aiex.dma_free_task(%b_r0c2_m7_n38)
      aiex.dma_free_task(%c_r0c2_m7_n38)
      // core r0c2 m7 n39
      %a_r0c2_m7_n39 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n39)
      %b_r0c2_m7_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n39)
      %c_r0c2_m7_n39 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805312 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n39)
      aiex.dma_await_task(%c_r0c2_m7_n39)
      aiex.dma_free_task(%a_r0c2_m7_n39)
      aiex.dma_free_task(%b_r0c2_m7_n39)
      aiex.dma_free_task(%c_r0c2_m7_n39)
      // core r0c2 m7 n40
      %a_r0c2_m7_n40 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n40)
      %b_r0c2_m7_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n40)
      %c_r0c2_m7_n40 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805376 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n40)
      aiex.dma_await_task(%c_r0c2_m7_n40)
      aiex.dma_free_task(%a_r0c2_m7_n40)
      aiex.dma_free_task(%b_r0c2_m7_n40)
      aiex.dma_free_task(%c_r0c2_m7_n40)
      // core r0c2 m7 n41
      %a_r0c2_m7_n41 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c2_m7_n41)
      %b_r0c2_m7_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c2_m7_n41)
      %c_r0c2_m7_n41 = aiex.dma_configure_task_for @C2_r0_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c2_m7_n41)
      aiex.dma_await_task(%c_r0c2_m7_n41)
      aiex.dma_free_task(%a_r0c2_m7_n41)
      aiex.dma_free_task(%b_r0c2_m7_n41)
      aiex.dma_free_task(%c_r0c2_m7_n41)
      // core r0c3 m0 n42
      %a_r0c3_m0_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n42)
      %b_r0c3_m0_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n42)
      %c_r0c3_m0_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n42)
      aiex.dma_await_task(%c_r0c3_m0_n42)
      aiex.dma_free_task(%a_r0c3_m0_n42)
      aiex.dma_free_task(%b_r0c3_m0_n42)
      aiex.dma_free_task(%c_r0c3_m0_n42)
      // core r0c3 m0 n43
      %a_r0c3_m0_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n43)
      %b_r0c3_m0_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n43)
      %c_r0c3_m0_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n43)
      aiex.dma_await_task(%c_r0c3_m0_n43)
      aiex.dma_free_task(%a_r0c3_m0_n43)
      aiex.dma_free_task(%b_r0c3_m0_n43)
      aiex.dma_free_task(%c_r0c3_m0_n43)
      // core r0c3 m0 n44
      %a_r0c3_m0_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n44)
      %b_r0c3_m0_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n44)
      %c_r0c3_m0_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n44)
      aiex.dma_await_task(%c_r0c3_m0_n44)
      aiex.dma_free_task(%a_r0c3_m0_n44)
      aiex.dma_free_task(%b_r0c3_m0_n44)
      aiex.dma_free_task(%c_r0c3_m0_n44)
      // core r0c3 m0 n45
      %a_r0c3_m0_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n45)
      %b_r0c3_m0_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n45)
      %c_r0c3_m0_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n45)
      aiex.dma_await_task(%c_r0c3_m0_n45)
      aiex.dma_free_task(%a_r0c3_m0_n45)
      aiex.dma_free_task(%b_r0c3_m0_n45)
      aiex.dma_free_task(%c_r0c3_m0_n45)
      // core r0c3 m0 n46
      %a_r0c3_m0_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n46)
      %b_r0c3_m0_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n46)
      %c_r0c3_m0_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 2944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n46)
      aiex.dma_await_task(%c_r0c3_m0_n46)
      aiex.dma_free_task(%a_r0c3_m0_n46)
      aiex.dma_free_task(%b_r0c3_m0_n46)
      aiex.dma_free_task(%c_r0c3_m0_n46)
      // core r0c3 m0 n47
      %a_r0c3_m0_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n47)
      %b_r0c3_m0_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n47)
      %c_r0c3_m0_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n47)
      aiex.dma_await_task(%c_r0c3_m0_n47)
      aiex.dma_free_task(%a_r0c3_m0_n47)
      aiex.dma_free_task(%b_r0c3_m0_n47)
      aiex.dma_free_task(%c_r0c3_m0_n47)
      // core r0c3 m0 n48
      %a_r0c3_m0_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n48)
      %b_r0c3_m0_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n48)
      %c_r0c3_m0_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n48)
      aiex.dma_await_task(%c_r0c3_m0_n48)
      aiex.dma_free_task(%a_r0c3_m0_n48)
      aiex.dma_free_task(%b_r0c3_m0_n48)
      aiex.dma_free_task(%c_r0c3_m0_n48)
      // core r0c3 m0 n49
      %a_r0c3_m0_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n49)
      %b_r0c3_m0_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n49)
      %c_r0c3_m0_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n49)
      aiex.dma_await_task(%c_r0c3_m0_n49)
      aiex.dma_free_task(%a_r0c3_m0_n49)
      aiex.dma_free_task(%b_r0c3_m0_n49)
      aiex.dma_free_task(%c_r0c3_m0_n49)
      // core r0c3 m0 n50
      %a_r0c3_m0_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n50)
      %b_r0c3_m0_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n50)
      %c_r0c3_m0_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n50)
      aiex.dma_await_task(%c_r0c3_m0_n50)
      aiex.dma_free_task(%a_r0c3_m0_n50)
      aiex.dma_free_task(%b_r0c3_m0_n50)
      aiex.dma_free_task(%c_r0c3_m0_n50)
      // core r0c3 m0 n51
      %a_r0c3_m0_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n51)
      %b_r0c3_m0_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n51)
      %c_r0c3_m0_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n51)
      aiex.dma_await_task(%c_r0c3_m0_n51)
      aiex.dma_free_task(%a_r0c3_m0_n51)
      aiex.dma_free_task(%b_r0c3_m0_n51)
      aiex.dma_free_task(%c_r0c3_m0_n51)
      // core r0c3 m0 n52
      %a_r0c3_m0_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n52)
      %b_r0c3_m0_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n52)
      %c_r0c3_m0_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n52)
      aiex.dma_await_task(%c_r0c3_m0_n52)
      aiex.dma_free_task(%a_r0c3_m0_n52)
      aiex.dma_free_task(%b_r0c3_m0_n52)
      aiex.dma_free_task(%c_r0c3_m0_n52)
      // core r0c3 m0 n53
      %a_r0c3_m0_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n53)
      %b_r0c3_m0_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n53)
      %c_r0c3_m0_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n53)
      aiex.dma_await_task(%c_r0c3_m0_n53)
      aiex.dma_free_task(%a_r0c3_m0_n53)
      aiex.dma_free_task(%b_r0c3_m0_n53)
      aiex.dma_free_task(%c_r0c3_m0_n53)
      // core r0c3 m0 n54
      %a_r0c3_m0_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n54)
      %b_r0c3_m0_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n54)
      %c_r0c3_m0_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n54)
      aiex.dma_await_task(%c_r0c3_m0_n54)
      aiex.dma_free_task(%a_r0c3_m0_n54)
      aiex.dma_free_task(%b_r0c3_m0_n54)
      aiex.dma_free_task(%c_r0c3_m0_n54)
      // core r0c3 m0 n55
      %a_r0c3_m0_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 0 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m0_n55)
      %b_r0c3_m0_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m0_n55)
      %c_r0c3_m0_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 3520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m0_n55)
      aiex.dma_await_task(%c_r0c3_m0_n55)
      aiex.dma_free_task(%a_r0c3_m0_n55)
      aiex.dma_free_task(%b_r0c3_m0_n55)
      aiex.dma_free_task(%c_r0c3_m0_n55)
      // core r0c3 m1 n42
      %a_r0c3_m1_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n42)
      %b_r0c3_m1_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n42)
      %c_r0c3_m1_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117376 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n42)
      aiex.dma_await_task(%c_r0c3_m1_n42)
      aiex.dma_free_task(%a_r0c3_m1_n42)
      aiex.dma_free_task(%b_r0c3_m1_n42)
      aiex.dma_free_task(%c_r0c3_m1_n42)
      // core r0c3 m1 n43
      %a_r0c3_m1_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n43)
      %b_r0c3_m1_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n43)
      %c_r0c3_m1_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n43)
      aiex.dma_await_task(%c_r0c3_m1_n43)
      aiex.dma_free_task(%a_r0c3_m1_n43)
      aiex.dma_free_task(%b_r0c3_m1_n43)
      aiex.dma_free_task(%c_r0c3_m1_n43)
      // core r0c3 m1 n44
      %a_r0c3_m1_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n44)
      %b_r0c3_m1_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n44)
      %c_r0c3_m1_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n44)
      aiex.dma_await_task(%c_r0c3_m1_n44)
      aiex.dma_free_task(%a_r0c3_m1_n44)
      aiex.dma_free_task(%b_r0c3_m1_n44)
      aiex.dma_free_task(%c_r0c3_m1_n44)
      // core r0c3 m1 n45
      %a_r0c3_m1_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n45)
      %b_r0c3_m1_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n45)
      %c_r0c3_m1_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n45)
      aiex.dma_await_task(%c_r0c3_m1_n45)
      aiex.dma_free_task(%a_r0c3_m1_n45)
      aiex.dma_free_task(%b_r0c3_m1_n45)
      aiex.dma_free_task(%c_r0c3_m1_n45)
      // core r0c3 m1 n46
      %a_r0c3_m1_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n46)
      %b_r0c3_m1_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n46)
      %c_r0c3_m1_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n46)
      aiex.dma_await_task(%c_r0c3_m1_n46)
      aiex.dma_free_task(%a_r0c3_m1_n46)
      aiex.dma_free_task(%b_r0c3_m1_n46)
      aiex.dma_free_task(%c_r0c3_m1_n46)
      // core r0c3 m1 n47
      %a_r0c3_m1_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n47)
      %b_r0c3_m1_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n47)
      %c_r0c3_m1_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n47)
      aiex.dma_await_task(%c_r0c3_m1_n47)
      aiex.dma_free_task(%a_r0c3_m1_n47)
      aiex.dma_free_task(%b_r0c3_m1_n47)
      aiex.dma_free_task(%c_r0c3_m1_n47)
      // core r0c3 m1 n48
      %a_r0c3_m1_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n48)
      %b_r0c3_m1_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n48)
      %c_r0c3_m1_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n48)
      aiex.dma_await_task(%c_r0c3_m1_n48)
      aiex.dma_free_task(%a_r0c3_m1_n48)
      aiex.dma_free_task(%b_r0c3_m1_n48)
      aiex.dma_free_task(%c_r0c3_m1_n48)
      // core r0c3 m1 n49
      %a_r0c3_m1_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n49)
      %b_r0c3_m1_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n49)
      %c_r0c3_m1_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n49)
      aiex.dma_await_task(%c_r0c3_m1_n49)
      aiex.dma_free_task(%a_r0c3_m1_n49)
      aiex.dma_free_task(%b_r0c3_m1_n49)
      aiex.dma_free_task(%c_r0c3_m1_n49)
      // core r0c3 m1 n50
      %a_r0c3_m1_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n50)
      %b_r0c3_m1_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n50)
      %c_r0c3_m1_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n50)
      aiex.dma_await_task(%c_r0c3_m1_n50)
      aiex.dma_free_task(%a_r0c3_m1_n50)
      aiex.dma_free_task(%b_r0c3_m1_n50)
      aiex.dma_free_task(%c_r0c3_m1_n50)
      // core r0c3 m1 n51
      %a_r0c3_m1_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n51)
      %b_r0c3_m1_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n51)
      %c_r0c3_m1_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 117952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n51)
      aiex.dma_await_task(%c_r0c3_m1_n51)
      aiex.dma_free_task(%a_r0c3_m1_n51)
      aiex.dma_free_task(%b_r0c3_m1_n51)
      aiex.dma_free_task(%c_r0c3_m1_n51)
      // core r0c3 m1 n52
      %a_r0c3_m1_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n52)
      %b_r0c3_m1_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n52)
      %c_r0c3_m1_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 118016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n52)
      aiex.dma_await_task(%c_r0c3_m1_n52)
      aiex.dma_free_task(%a_r0c3_m1_n52)
      aiex.dma_free_task(%b_r0c3_m1_n52)
      aiex.dma_free_task(%c_r0c3_m1_n52)
      // core r0c3 m1 n53
      %a_r0c3_m1_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n53)
      %b_r0c3_m1_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n53)
      %c_r0c3_m1_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 118080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n53)
      aiex.dma_await_task(%c_r0c3_m1_n53)
      aiex.dma_free_task(%a_r0c3_m1_n53)
      aiex.dma_free_task(%b_r0c3_m1_n53)
      aiex.dma_free_task(%c_r0c3_m1_n53)
      // core r0c3 m1 n54
      %a_r0c3_m1_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n54)
      %b_r0c3_m1_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n54)
      %c_r0c3_m1_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 118144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n54)
      aiex.dma_await_task(%c_r0c3_m1_n54)
      aiex.dma_free_task(%a_r0c3_m1_n54)
      aiex.dma_free_task(%b_r0c3_m1_n54)
      aiex.dma_free_task(%c_r0c3_m1_n54)
      // core r0c3 m1 n55
      %a_r0c3_m1_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 32768 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m1_n55)
      %b_r0c3_m1_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m1_n55)
      %c_r0c3_m1_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 118208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m1_n55)
      aiex.dma_await_task(%c_r0c3_m1_n55)
      aiex.dma_free_task(%a_r0c3_m1_n55)
      aiex.dma_free_task(%b_r0c3_m1_n55)
      aiex.dma_free_task(%c_r0c3_m1_n55)
      // core r0c3 m2 n42
      %a_r0c3_m2_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n42)
      %b_r0c3_m2_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n42)
      %c_r0c3_m2_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232064 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n42)
      aiex.dma_await_task(%c_r0c3_m2_n42)
      aiex.dma_free_task(%a_r0c3_m2_n42)
      aiex.dma_free_task(%b_r0c3_m2_n42)
      aiex.dma_free_task(%c_r0c3_m2_n42)
      // core r0c3 m2 n43
      %a_r0c3_m2_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n43)
      %b_r0c3_m2_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n43)
      %c_r0c3_m2_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n43)
      aiex.dma_await_task(%c_r0c3_m2_n43)
      aiex.dma_free_task(%a_r0c3_m2_n43)
      aiex.dma_free_task(%b_r0c3_m2_n43)
      aiex.dma_free_task(%c_r0c3_m2_n43)
      // core r0c3 m2 n44
      %a_r0c3_m2_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n44)
      %b_r0c3_m2_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n44)
      %c_r0c3_m2_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n44)
      aiex.dma_await_task(%c_r0c3_m2_n44)
      aiex.dma_free_task(%a_r0c3_m2_n44)
      aiex.dma_free_task(%b_r0c3_m2_n44)
      aiex.dma_free_task(%c_r0c3_m2_n44)
      // core r0c3 m2 n45
      %a_r0c3_m2_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n45)
      %b_r0c3_m2_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n45)
      %c_r0c3_m2_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n45)
      aiex.dma_await_task(%c_r0c3_m2_n45)
      aiex.dma_free_task(%a_r0c3_m2_n45)
      aiex.dma_free_task(%b_r0c3_m2_n45)
      aiex.dma_free_task(%c_r0c3_m2_n45)
      // core r0c3 m2 n46
      %a_r0c3_m2_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n46)
      %b_r0c3_m2_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n46)
      %c_r0c3_m2_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n46)
      aiex.dma_await_task(%c_r0c3_m2_n46)
      aiex.dma_free_task(%a_r0c3_m2_n46)
      aiex.dma_free_task(%b_r0c3_m2_n46)
      aiex.dma_free_task(%c_r0c3_m2_n46)
      // core r0c3 m2 n47
      %a_r0c3_m2_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n47)
      %b_r0c3_m2_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n47)
      %c_r0c3_m2_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n47)
      aiex.dma_await_task(%c_r0c3_m2_n47)
      aiex.dma_free_task(%a_r0c3_m2_n47)
      aiex.dma_free_task(%b_r0c3_m2_n47)
      aiex.dma_free_task(%c_r0c3_m2_n47)
      // core r0c3 m2 n48
      %a_r0c3_m2_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n48)
      %b_r0c3_m2_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n48)
      %c_r0c3_m2_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n48)
      aiex.dma_await_task(%c_r0c3_m2_n48)
      aiex.dma_free_task(%a_r0c3_m2_n48)
      aiex.dma_free_task(%b_r0c3_m2_n48)
      aiex.dma_free_task(%c_r0c3_m2_n48)
      // core r0c3 m2 n49
      %a_r0c3_m2_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n49)
      %b_r0c3_m2_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n49)
      %c_r0c3_m2_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n49)
      aiex.dma_await_task(%c_r0c3_m2_n49)
      aiex.dma_free_task(%a_r0c3_m2_n49)
      aiex.dma_free_task(%b_r0c3_m2_n49)
      aiex.dma_free_task(%c_r0c3_m2_n49)
      // core r0c3 m2 n50
      %a_r0c3_m2_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n50)
      %b_r0c3_m2_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n50)
      %c_r0c3_m2_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n50)
      aiex.dma_await_task(%c_r0c3_m2_n50)
      aiex.dma_free_task(%a_r0c3_m2_n50)
      aiex.dma_free_task(%b_r0c3_m2_n50)
      aiex.dma_free_task(%c_r0c3_m2_n50)
      // core r0c3 m2 n51
      %a_r0c3_m2_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n51)
      %b_r0c3_m2_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n51)
      %c_r0c3_m2_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n51)
      aiex.dma_await_task(%c_r0c3_m2_n51)
      aiex.dma_free_task(%a_r0c3_m2_n51)
      aiex.dma_free_task(%b_r0c3_m2_n51)
      aiex.dma_free_task(%c_r0c3_m2_n51)
      // core r0c3 m2 n52
      %a_r0c3_m2_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n52)
      %b_r0c3_m2_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n52)
      %c_r0c3_m2_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n52)
      aiex.dma_await_task(%c_r0c3_m2_n52)
      aiex.dma_free_task(%a_r0c3_m2_n52)
      aiex.dma_free_task(%b_r0c3_m2_n52)
      aiex.dma_free_task(%c_r0c3_m2_n52)
      // core r0c3 m2 n53
      %a_r0c3_m2_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n53)
      %b_r0c3_m2_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n53)
      %c_r0c3_m2_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n53)
      aiex.dma_await_task(%c_r0c3_m2_n53)
      aiex.dma_free_task(%a_r0c3_m2_n53)
      aiex.dma_free_task(%b_r0c3_m2_n53)
      aiex.dma_free_task(%c_r0c3_m2_n53)
      // core r0c3 m2 n54
      %a_r0c3_m2_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n54)
      %b_r0c3_m2_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n54)
      %c_r0c3_m2_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n54)
      aiex.dma_await_task(%c_r0c3_m2_n54)
      aiex.dma_free_task(%a_r0c3_m2_n54)
      aiex.dma_free_task(%b_r0c3_m2_n54)
      aiex.dma_free_task(%c_r0c3_m2_n54)
      // core r0c3 m2 n55
      %a_r0c3_m2_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 65536 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m2_n55)
      %b_r0c3_m2_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m2_n55)
      %c_r0c3_m2_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 232896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m2_n55)
      aiex.dma_await_task(%c_r0c3_m2_n55)
      aiex.dma_free_task(%a_r0c3_m2_n55)
      aiex.dma_free_task(%b_r0c3_m2_n55)
      aiex.dma_free_task(%c_r0c3_m2_n55)
      // core r0c3 m3 n42
      %a_r0c3_m3_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n42)
      %b_r0c3_m3_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n42)
      %c_r0c3_m3_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n42)
      aiex.dma_await_task(%c_r0c3_m3_n42)
      aiex.dma_free_task(%a_r0c3_m3_n42)
      aiex.dma_free_task(%b_r0c3_m3_n42)
      aiex.dma_free_task(%c_r0c3_m3_n42)
      // core r0c3 m3 n43
      %a_r0c3_m3_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n43)
      %b_r0c3_m3_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n43)
      %c_r0c3_m3_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n43)
      aiex.dma_await_task(%c_r0c3_m3_n43)
      aiex.dma_free_task(%a_r0c3_m3_n43)
      aiex.dma_free_task(%b_r0c3_m3_n43)
      aiex.dma_free_task(%c_r0c3_m3_n43)
      // core r0c3 m3 n44
      %a_r0c3_m3_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n44)
      %b_r0c3_m3_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n44)
      %c_r0c3_m3_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n44)
      aiex.dma_await_task(%c_r0c3_m3_n44)
      aiex.dma_free_task(%a_r0c3_m3_n44)
      aiex.dma_free_task(%b_r0c3_m3_n44)
      aiex.dma_free_task(%c_r0c3_m3_n44)
      // core r0c3 m3 n45
      %a_r0c3_m3_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n45)
      %b_r0c3_m3_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n45)
      %c_r0c3_m3_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 346944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n45)
      aiex.dma_await_task(%c_r0c3_m3_n45)
      aiex.dma_free_task(%a_r0c3_m3_n45)
      aiex.dma_free_task(%b_r0c3_m3_n45)
      aiex.dma_free_task(%c_r0c3_m3_n45)
      // core r0c3 m3 n46
      %a_r0c3_m3_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n46)
      %b_r0c3_m3_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n46)
      %c_r0c3_m3_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n46)
      aiex.dma_await_task(%c_r0c3_m3_n46)
      aiex.dma_free_task(%a_r0c3_m3_n46)
      aiex.dma_free_task(%b_r0c3_m3_n46)
      aiex.dma_free_task(%c_r0c3_m3_n46)
      // core r0c3 m3 n47
      %a_r0c3_m3_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n47)
      %b_r0c3_m3_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n47)
      %c_r0c3_m3_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n47)
      aiex.dma_await_task(%c_r0c3_m3_n47)
      aiex.dma_free_task(%a_r0c3_m3_n47)
      aiex.dma_free_task(%b_r0c3_m3_n47)
      aiex.dma_free_task(%c_r0c3_m3_n47)
      // core r0c3 m3 n48
      %a_r0c3_m3_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n48)
      %b_r0c3_m3_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n48)
      %c_r0c3_m3_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n48)
      aiex.dma_await_task(%c_r0c3_m3_n48)
      aiex.dma_free_task(%a_r0c3_m3_n48)
      aiex.dma_free_task(%b_r0c3_m3_n48)
      aiex.dma_free_task(%c_r0c3_m3_n48)
      // core r0c3 m3 n49
      %a_r0c3_m3_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n49)
      %b_r0c3_m3_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n49)
      %c_r0c3_m3_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n49)
      aiex.dma_await_task(%c_r0c3_m3_n49)
      aiex.dma_free_task(%a_r0c3_m3_n49)
      aiex.dma_free_task(%b_r0c3_m3_n49)
      aiex.dma_free_task(%c_r0c3_m3_n49)
      // core r0c3 m3 n50
      %a_r0c3_m3_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n50)
      %b_r0c3_m3_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n50)
      %c_r0c3_m3_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n50)
      aiex.dma_await_task(%c_r0c3_m3_n50)
      aiex.dma_free_task(%a_r0c3_m3_n50)
      aiex.dma_free_task(%b_r0c3_m3_n50)
      aiex.dma_free_task(%c_r0c3_m3_n50)
      // core r0c3 m3 n51
      %a_r0c3_m3_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n51)
      %b_r0c3_m3_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n51)
      %c_r0c3_m3_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n51)
      aiex.dma_await_task(%c_r0c3_m3_n51)
      aiex.dma_free_task(%a_r0c3_m3_n51)
      aiex.dma_free_task(%b_r0c3_m3_n51)
      aiex.dma_free_task(%c_r0c3_m3_n51)
      // core r0c3 m3 n52
      %a_r0c3_m3_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n52)
      %b_r0c3_m3_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n52)
      %c_r0c3_m3_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n52)
      aiex.dma_await_task(%c_r0c3_m3_n52)
      aiex.dma_free_task(%a_r0c3_m3_n52)
      aiex.dma_free_task(%b_r0c3_m3_n52)
      aiex.dma_free_task(%c_r0c3_m3_n52)
      // core r0c3 m3 n53
      %a_r0c3_m3_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n53)
      %b_r0c3_m3_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n53)
      %c_r0c3_m3_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n53)
      aiex.dma_await_task(%c_r0c3_m3_n53)
      aiex.dma_free_task(%a_r0c3_m3_n53)
      aiex.dma_free_task(%b_r0c3_m3_n53)
      aiex.dma_free_task(%c_r0c3_m3_n53)
      // core r0c3 m3 n54
      %a_r0c3_m3_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n54)
      %b_r0c3_m3_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n54)
      %c_r0c3_m3_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n54)
      aiex.dma_await_task(%c_r0c3_m3_n54)
      aiex.dma_free_task(%a_r0c3_m3_n54)
      aiex.dma_free_task(%b_r0c3_m3_n54)
      aiex.dma_free_task(%c_r0c3_m3_n54)
      // core r0c3 m3 n55
      %a_r0c3_m3_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 98304 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m3_n55)
      %b_r0c3_m3_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m3_n55)
      %c_r0c3_m3_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 347584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m3_n55)
      aiex.dma_await_task(%c_r0c3_m3_n55)
      aiex.dma_free_task(%a_r0c3_m3_n55)
      aiex.dma_free_task(%b_r0c3_m3_n55)
      aiex.dma_free_task(%c_r0c3_m3_n55)
      // core r0c3 m4 n42
      %a_r0c3_m4_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n42)
      %b_r0c3_m4_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n42)
      %c_r0c3_m4_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n42)
      aiex.dma_await_task(%c_r0c3_m4_n42)
      aiex.dma_free_task(%a_r0c3_m4_n42)
      aiex.dma_free_task(%b_r0c3_m4_n42)
      aiex.dma_free_task(%c_r0c3_m4_n42)
      // core r0c3 m4 n43
      %a_r0c3_m4_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n43)
      %b_r0c3_m4_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n43)
      %c_r0c3_m4_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n43)
      aiex.dma_await_task(%c_r0c3_m4_n43)
      aiex.dma_free_task(%a_r0c3_m4_n43)
      aiex.dma_free_task(%b_r0c3_m4_n43)
      aiex.dma_free_task(%c_r0c3_m4_n43)
      // core r0c3 m4 n44
      %a_r0c3_m4_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n44)
      %b_r0c3_m4_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n44)
      %c_r0c3_m4_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n44)
      aiex.dma_await_task(%c_r0c3_m4_n44)
      aiex.dma_free_task(%a_r0c3_m4_n44)
      aiex.dma_free_task(%b_r0c3_m4_n44)
      aiex.dma_free_task(%c_r0c3_m4_n44)
      // core r0c3 m4 n45
      %a_r0c3_m4_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n45)
      %b_r0c3_m4_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n45)
      %c_r0c3_m4_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n45)
      aiex.dma_await_task(%c_r0c3_m4_n45)
      aiex.dma_free_task(%a_r0c3_m4_n45)
      aiex.dma_free_task(%b_r0c3_m4_n45)
      aiex.dma_free_task(%c_r0c3_m4_n45)
      // core r0c3 m4 n46
      %a_r0c3_m4_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n46)
      %b_r0c3_m4_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n46)
      %c_r0c3_m4_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n46)
      aiex.dma_await_task(%c_r0c3_m4_n46)
      aiex.dma_free_task(%a_r0c3_m4_n46)
      aiex.dma_free_task(%b_r0c3_m4_n46)
      aiex.dma_free_task(%c_r0c3_m4_n46)
      // core r0c3 m4 n47
      %a_r0c3_m4_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n47)
      %b_r0c3_m4_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n47)
      %c_r0c3_m4_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n47)
      aiex.dma_await_task(%c_r0c3_m4_n47)
      aiex.dma_free_task(%a_r0c3_m4_n47)
      aiex.dma_free_task(%b_r0c3_m4_n47)
      aiex.dma_free_task(%c_r0c3_m4_n47)
      // core r0c3 m4 n48
      %a_r0c3_m4_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n48)
      %b_r0c3_m4_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n48)
      %c_r0c3_m4_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n48)
      aiex.dma_await_task(%c_r0c3_m4_n48)
      aiex.dma_free_task(%a_r0c3_m4_n48)
      aiex.dma_free_task(%b_r0c3_m4_n48)
      aiex.dma_free_task(%c_r0c3_m4_n48)
      // core r0c3 m4 n49
      %a_r0c3_m4_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n49)
      %b_r0c3_m4_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n49)
      %c_r0c3_m4_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n49)
      aiex.dma_await_task(%c_r0c3_m4_n49)
      aiex.dma_free_task(%a_r0c3_m4_n49)
      aiex.dma_free_task(%b_r0c3_m4_n49)
      aiex.dma_free_task(%c_r0c3_m4_n49)
      // core r0c3 m4 n50
      %a_r0c3_m4_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n50)
      %b_r0c3_m4_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n50)
      %c_r0c3_m4_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 461952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n50)
      aiex.dma_await_task(%c_r0c3_m4_n50)
      aiex.dma_free_task(%a_r0c3_m4_n50)
      aiex.dma_free_task(%b_r0c3_m4_n50)
      aiex.dma_free_task(%c_r0c3_m4_n50)
      // core r0c3 m4 n51
      %a_r0c3_m4_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n51)
      %b_r0c3_m4_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n51)
      %c_r0c3_m4_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 462016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n51)
      aiex.dma_await_task(%c_r0c3_m4_n51)
      aiex.dma_free_task(%a_r0c3_m4_n51)
      aiex.dma_free_task(%b_r0c3_m4_n51)
      aiex.dma_free_task(%c_r0c3_m4_n51)
      // core r0c3 m4 n52
      %a_r0c3_m4_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n52)
      %b_r0c3_m4_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n52)
      %c_r0c3_m4_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 462080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n52)
      aiex.dma_await_task(%c_r0c3_m4_n52)
      aiex.dma_free_task(%a_r0c3_m4_n52)
      aiex.dma_free_task(%b_r0c3_m4_n52)
      aiex.dma_free_task(%c_r0c3_m4_n52)
      // core r0c3 m4 n53
      %a_r0c3_m4_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n53)
      %b_r0c3_m4_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n53)
      %c_r0c3_m4_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 462144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n53)
      aiex.dma_await_task(%c_r0c3_m4_n53)
      aiex.dma_free_task(%a_r0c3_m4_n53)
      aiex.dma_free_task(%b_r0c3_m4_n53)
      aiex.dma_free_task(%c_r0c3_m4_n53)
      // core r0c3 m4 n54
      %a_r0c3_m4_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n54)
      %b_r0c3_m4_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n54)
      %c_r0c3_m4_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 462208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n54)
      aiex.dma_await_task(%c_r0c3_m4_n54)
      aiex.dma_free_task(%a_r0c3_m4_n54)
      aiex.dma_free_task(%b_r0c3_m4_n54)
      aiex.dma_free_task(%c_r0c3_m4_n54)
      // core r0c3 m4 n55
      %a_r0c3_m4_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 131072 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m4_n55)
      %b_r0c3_m4_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m4_n55)
      %c_r0c3_m4_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 462272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m4_n55)
      aiex.dma_await_task(%c_r0c3_m4_n55)
      aiex.dma_free_task(%a_r0c3_m4_n55)
      aiex.dma_free_task(%b_r0c3_m4_n55)
      aiex.dma_free_task(%c_r0c3_m4_n55)
      // core r0c3 m5 n42
      %a_r0c3_m5_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n42)
      %b_r0c3_m5_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n42)
      %c_r0c3_m5_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n42)
      aiex.dma_await_task(%c_r0c3_m5_n42)
      aiex.dma_free_task(%a_r0c3_m5_n42)
      aiex.dma_free_task(%b_r0c3_m5_n42)
      aiex.dma_free_task(%c_r0c3_m5_n42)
      // core r0c3 m5 n43
      %a_r0c3_m5_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n43)
      %b_r0c3_m5_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n43)
      %c_r0c3_m5_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n43)
      aiex.dma_await_task(%c_r0c3_m5_n43)
      aiex.dma_free_task(%a_r0c3_m5_n43)
      aiex.dma_free_task(%b_r0c3_m5_n43)
      aiex.dma_free_task(%c_r0c3_m5_n43)
      // core r0c3 m5 n44
      %a_r0c3_m5_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n44)
      %b_r0c3_m5_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n44)
      %c_r0c3_m5_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n44)
      aiex.dma_await_task(%c_r0c3_m5_n44)
      aiex.dma_free_task(%a_r0c3_m5_n44)
      aiex.dma_free_task(%b_r0c3_m5_n44)
      aiex.dma_free_task(%c_r0c3_m5_n44)
      // core r0c3 m5 n45
      %a_r0c3_m5_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n45)
      %b_r0c3_m5_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n45)
      %c_r0c3_m5_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n45)
      aiex.dma_await_task(%c_r0c3_m5_n45)
      aiex.dma_free_task(%a_r0c3_m5_n45)
      aiex.dma_free_task(%b_r0c3_m5_n45)
      aiex.dma_free_task(%c_r0c3_m5_n45)
      // core r0c3 m5 n46
      %a_r0c3_m5_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n46)
      %b_r0c3_m5_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n46)
      %c_r0c3_m5_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n46)
      aiex.dma_await_task(%c_r0c3_m5_n46)
      aiex.dma_free_task(%a_r0c3_m5_n46)
      aiex.dma_free_task(%b_r0c3_m5_n46)
      aiex.dma_free_task(%c_r0c3_m5_n46)
      // core r0c3 m5 n47
      %a_r0c3_m5_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n47)
      %b_r0c3_m5_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n47)
      %c_r0c3_m5_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n47)
      aiex.dma_await_task(%c_r0c3_m5_n47)
      aiex.dma_free_task(%a_r0c3_m5_n47)
      aiex.dma_free_task(%b_r0c3_m5_n47)
      aiex.dma_free_task(%c_r0c3_m5_n47)
      // core r0c3 m5 n48
      %a_r0c3_m5_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n48)
      %b_r0c3_m5_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n48)
      %c_r0c3_m5_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n48)
      aiex.dma_await_task(%c_r0c3_m5_n48)
      aiex.dma_free_task(%a_r0c3_m5_n48)
      aiex.dma_free_task(%b_r0c3_m5_n48)
      aiex.dma_free_task(%c_r0c3_m5_n48)
      // core r0c3 m5 n49
      %a_r0c3_m5_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n49)
      %b_r0c3_m5_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n49)
      %c_r0c3_m5_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n49)
      aiex.dma_await_task(%c_r0c3_m5_n49)
      aiex.dma_free_task(%a_r0c3_m5_n49)
      aiex.dma_free_task(%b_r0c3_m5_n49)
      aiex.dma_free_task(%c_r0c3_m5_n49)
      // core r0c3 m5 n50
      %a_r0c3_m5_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n50)
      %b_r0c3_m5_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n50)
      %c_r0c3_m5_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n50)
      aiex.dma_await_task(%c_r0c3_m5_n50)
      aiex.dma_free_task(%a_r0c3_m5_n50)
      aiex.dma_free_task(%b_r0c3_m5_n50)
      aiex.dma_free_task(%c_r0c3_m5_n50)
      // core r0c3 m5 n51
      %a_r0c3_m5_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n51)
      %b_r0c3_m5_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n51)
      %c_r0c3_m5_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n51)
      aiex.dma_await_task(%c_r0c3_m5_n51)
      aiex.dma_free_task(%a_r0c3_m5_n51)
      aiex.dma_free_task(%b_r0c3_m5_n51)
      aiex.dma_free_task(%c_r0c3_m5_n51)
      // core r0c3 m5 n52
      %a_r0c3_m5_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n52)
      %b_r0c3_m5_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n52)
      %c_r0c3_m5_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n52)
      aiex.dma_await_task(%c_r0c3_m5_n52)
      aiex.dma_free_task(%a_r0c3_m5_n52)
      aiex.dma_free_task(%b_r0c3_m5_n52)
      aiex.dma_free_task(%c_r0c3_m5_n52)
      // core r0c3 m5 n53
      %a_r0c3_m5_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n53)
      %b_r0c3_m5_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n53)
      %c_r0c3_m5_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n53)
      aiex.dma_await_task(%c_r0c3_m5_n53)
      aiex.dma_free_task(%a_r0c3_m5_n53)
      aiex.dma_free_task(%b_r0c3_m5_n53)
      aiex.dma_free_task(%c_r0c3_m5_n53)
      // core r0c3 m5 n54
      %a_r0c3_m5_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n54)
      %b_r0c3_m5_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n54)
      %c_r0c3_m5_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n54)
      aiex.dma_await_task(%c_r0c3_m5_n54)
      aiex.dma_free_task(%a_r0c3_m5_n54)
      aiex.dma_free_task(%b_r0c3_m5_n54)
      aiex.dma_free_task(%c_r0c3_m5_n54)
      // core r0c3 m5 n55
      %a_r0c3_m5_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 163840 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m5_n55)
      %b_r0c3_m5_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m5_n55)
      %c_r0c3_m5_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 576960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m5_n55)
      aiex.dma_await_task(%c_r0c3_m5_n55)
      aiex.dma_free_task(%a_r0c3_m5_n55)
      aiex.dma_free_task(%b_r0c3_m5_n55)
      aiex.dma_free_task(%c_r0c3_m5_n55)
      // core r0c3 m6 n42
      %a_r0c3_m6_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n42)
      %b_r0c3_m6_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n42)
      %c_r0c3_m6_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n42)
      aiex.dma_await_task(%c_r0c3_m6_n42)
      aiex.dma_free_task(%a_r0c3_m6_n42)
      aiex.dma_free_task(%b_r0c3_m6_n42)
      aiex.dma_free_task(%c_r0c3_m6_n42)
      // core r0c3 m6 n43
      %a_r0c3_m6_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n43)
      %b_r0c3_m6_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n43)
      %c_r0c3_m6_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n43)
      aiex.dma_await_task(%c_r0c3_m6_n43)
      aiex.dma_free_task(%a_r0c3_m6_n43)
      aiex.dma_free_task(%b_r0c3_m6_n43)
      aiex.dma_free_task(%c_r0c3_m6_n43)
      // core r0c3 m6 n44
      %a_r0c3_m6_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n44)
      %b_r0c3_m6_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n44)
      %c_r0c3_m6_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 690944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n44)
      aiex.dma_await_task(%c_r0c3_m6_n44)
      aiex.dma_free_task(%a_r0c3_m6_n44)
      aiex.dma_free_task(%b_r0c3_m6_n44)
      aiex.dma_free_task(%c_r0c3_m6_n44)
      // core r0c3 m6 n45
      %a_r0c3_m6_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n45)
      %b_r0c3_m6_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n45)
      %c_r0c3_m6_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n45)
      aiex.dma_await_task(%c_r0c3_m6_n45)
      aiex.dma_free_task(%a_r0c3_m6_n45)
      aiex.dma_free_task(%b_r0c3_m6_n45)
      aiex.dma_free_task(%c_r0c3_m6_n45)
      // core r0c3 m6 n46
      %a_r0c3_m6_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n46)
      %b_r0c3_m6_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n46)
      %c_r0c3_m6_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n46)
      aiex.dma_await_task(%c_r0c3_m6_n46)
      aiex.dma_free_task(%a_r0c3_m6_n46)
      aiex.dma_free_task(%b_r0c3_m6_n46)
      aiex.dma_free_task(%c_r0c3_m6_n46)
      // core r0c3 m6 n47
      %a_r0c3_m6_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n47)
      %b_r0c3_m6_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n47)
      %c_r0c3_m6_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n47)
      aiex.dma_await_task(%c_r0c3_m6_n47)
      aiex.dma_free_task(%a_r0c3_m6_n47)
      aiex.dma_free_task(%b_r0c3_m6_n47)
      aiex.dma_free_task(%c_r0c3_m6_n47)
      // core r0c3 m6 n48
      %a_r0c3_m6_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n48)
      %b_r0c3_m6_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n48)
      %c_r0c3_m6_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n48)
      aiex.dma_await_task(%c_r0c3_m6_n48)
      aiex.dma_free_task(%a_r0c3_m6_n48)
      aiex.dma_free_task(%b_r0c3_m6_n48)
      aiex.dma_free_task(%c_r0c3_m6_n48)
      // core r0c3 m6 n49
      %a_r0c3_m6_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n49)
      %b_r0c3_m6_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n49)
      %c_r0c3_m6_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n49)
      aiex.dma_await_task(%c_r0c3_m6_n49)
      aiex.dma_free_task(%a_r0c3_m6_n49)
      aiex.dma_free_task(%b_r0c3_m6_n49)
      aiex.dma_free_task(%c_r0c3_m6_n49)
      // core r0c3 m6 n50
      %a_r0c3_m6_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n50)
      %b_r0c3_m6_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n50)
      %c_r0c3_m6_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n50)
      aiex.dma_await_task(%c_r0c3_m6_n50)
      aiex.dma_free_task(%a_r0c3_m6_n50)
      aiex.dma_free_task(%b_r0c3_m6_n50)
      aiex.dma_free_task(%c_r0c3_m6_n50)
      // core r0c3 m6 n51
      %a_r0c3_m6_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n51)
      %b_r0c3_m6_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n51)
      %c_r0c3_m6_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n51)
      aiex.dma_await_task(%c_r0c3_m6_n51)
      aiex.dma_free_task(%a_r0c3_m6_n51)
      aiex.dma_free_task(%b_r0c3_m6_n51)
      aiex.dma_free_task(%c_r0c3_m6_n51)
      // core r0c3 m6 n52
      %a_r0c3_m6_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n52)
      %b_r0c3_m6_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n52)
      %c_r0c3_m6_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n52)
      aiex.dma_await_task(%c_r0c3_m6_n52)
      aiex.dma_free_task(%a_r0c3_m6_n52)
      aiex.dma_free_task(%b_r0c3_m6_n52)
      aiex.dma_free_task(%c_r0c3_m6_n52)
      // core r0c3 m6 n53
      %a_r0c3_m6_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n53)
      %b_r0c3_m6_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n53)
      %c_r0c3_m6_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n53)
      aiex.dma_await_task(%c_r0c3_m6_n53)
      aiex.dma_free_task(%a_r0c3_m6_n53)
      aiex.dma_free_task(%b_r0c3_m6_n53)
      aiex.dma_free_task(%c_r0c3_m6_n53)
      // core r0c3 m6 n54
      %a_r0c3_m6_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n54)
      %b_r0c3_m6_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n54)
      %c_r0c3_m6_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n54)
      aiex.dma_await_task(%c_r0c3_m6_n54)
      aiex.dma_free_task(%a_r0c3_m6_n54)
      aiex.dma_free_task(%b_r0c3_m6_n54)
      aiex.dma_free_task(%c_r0c3_m6_n54)
      // core r0c3 m6 n55
      %a_r0c3_m6_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 196608 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m6_n55)
      %b_r0c3_m6_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m6_n55)
      %c_r0c3_m6_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 691648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m6_n55)
      aiex.dma_await_task(%c_r0c3_m6_n55)
      aiex.dma_free_task(%a_r0c3_m6_n55)
      aiex.dma_free_task(%b_r0c3_m6_n55)
      aiex.dma_free_task(%c_r0c3_m6_n55)
      // core r0c3 m7 n42
      %a_r0c3_m7_n42 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n42)
      %b_r0c3_m7_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n42)
      %c_r0c3_m7_n42 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n42)
      aiex.dma_await_task(%c_r0c3_m7_n42)
      aiex.dma_free_task(%a_r0c3_m7_n42)
      aiex.dma_free_task(%b_r0c3_m7_n42)
      aiex.dma_free_task(%c_r0c3_m7_n42)
      // core r0c3 m7 n43
      %a_r0c3_m7_n43 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n43)
      %b_r0c3_m7_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n43)
      %c_r0c3_m7_n43 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n43)
      aiex.dma_await_task(%c_r0c3_m7_n43)
      aiex.dma_free_task(%a_r0c3_m7_n43)
      aiex.dma_free_task(%b_r0c3_m7_n43)
      aiex.dma_free_task(%c_r0c3_m7_n43)
      // core r0c3 m7 n44
      %a_r0c3_m7_n44 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n44)
      %b_r0c3_m7_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n44)
      %c_r0c3_m7_n44 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n44)
      aiex.dma_await_task(%c_r0c3_m7_n44)
      aiex.dma_free_task(%a_r0c3_m7_n44)
      aiex.dma_free_task(%b_r0c3_m7_n44)
      aiex.dma_free_task(%c_r0c3_m7_n44)
      // core r0c3 m7 n45
      %a_r0c3_m7_n45 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n45)
      %b_r0c3_m7_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n45)
      %c_r0c3_m7_n45 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n45)
      aiex.dma_await_task(%c_r0c3_m7_n45)
      aiex.dma_free_task(%a_r0c3_m7_n45)
      aiex.dma_free_task(%b_r0c3_m7_n45)
      aiex.dma_free_task(%c_r0c3_m7_n45)
      // core r0c3 m7 n46
      %a_r0c3_m7_n46 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n46)
      %b_r0c3_m7_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n46)
      %c_r0c3_m7_n46 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n46)
      aiex.dma_await_task(%c_r0c3_m7_n46)
      aiex.dma_free_task(%a_r0c3_m7_n46)
      aiex.dma_free_task(%b_r0c3_m7_n46)
      aiex.dma_free_task(%c_r0c3_m7_n46)
      // core r0c3 m7 n47
      %a_r0c3_m7_n47 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n47)
      %b_r0c3_m7_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n47)
      %c_r0c3_m7_n47 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n47)
      aiex.dma_await_task(%c_r0c3_m7_n47)
      aiex.dma_free_task(%a_r0c3_m7_n47)
      aiex.dma_free_task(%b_r0c3_m7_n47)
      aiex.dma_free_task(%c_r0c3_m7_n47)
      // core r0c3 m7 n48
      %a_r0c3_m7_n48 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n48)
      %b_r0c3_m7_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n48)
      %c_r0c3_m7_n48 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n48)
      aiex.dma_await_task(%c_r0c3_m7_n48)
      aiex.dma_free_task(%a_r0c3_m7_n48)
      aiex.dma_free_task(%b_r0c3_m7_n48)
      aiex.dma_free_task(%c_r0c3_m7_n48)
      // core r0c3 m7 n49
      %a_r0c3_m7_n49 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n49)
      %b_r0c3_m7_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n49)
      %c_r0c3_m7_n49 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 805952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n49)
      aiex.dma_await_task(%c_r0c3_m7_n49)
      aiex.dma_free_task(%a_r0c3_m7_n49)
      aiex.dma_free_task(%b_r0c3_m7_n49)
      aiex.dma_free_task(%c_r0c3_m7_n49)
      // core r0c3 m7 n50
      %a_r0c3_m7_n50 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n50)
      %b_r0c3_m7_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n50)
      %c_r0c3_m7_n50 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 806016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n50)
      aiex.dma_await_task(%c_r0c3_m7_n50)
      aiex.dma_free_task(%a_r0c3_m7_n50)
      aiex.dma_free_task(%b_r0c3_m7_n50)
      aiex.dma_free_task(%c_r0c3_m7_n50)
      // core r0c3 m7 n51
      %a_r0c3_m7_n51 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n51)
      %b_r0c3_m7_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n51)
      %c_r0c3_m7_n51 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 806080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n51)
      aiex.dma_await_task(%c_r0c3_m7_n51)
      aiex.dma_free_task(%a_r0c3_m7_n51)
      aiex.dma_free_task(%b_r0c3_m7_n51)
      aiex.dma_free_task(%c_r0c3_m7_n51)
      // core r0c3 m7 n52
      %a_r0c3_m7_n52 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n52)
      %b_r0c3_m7_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n52)
      %c_r0c3_m7_n52 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 806144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n52)
      aiex.dma_await_task(%c_r0c3_m7_n52)
      aiex.dma_free_task(%a_r0c3_m7_n52)
      aiex.dma_free_task(%b_r0c3_m7_n52)
      aiex.dma_free_task(%c_r0c3_m7_n52)
      // core r0c3 m7 n53
      %a_r0c3_m7_n53 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n53)
      %b_r0c3_m7_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n53)
      %c_r0c3_m7_n53 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 806208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n53)
      aiex.dma_await_task(%c_r0c3_m7_n53)
      aiex.dma_free_task(%a_r0c3_m7_n53)
      aiex.dma_free_task(%b_r0c3_m7_n53)
      aiex.dma_free_task(%c_r0c3_m7_n53)
      // core r0c3 m7 n54
      %a_r0c3_m7_n54 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n54)
      %b_r0c3_m7_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n54)
      %c_r0c3_m7_n54 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 806272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n54)
      aiex.dma_await_task(%c_r0c3_m7_n54)
      aiex.dma_free_task(%a_r0c3_m7_n54)
      aiex.dma_free_task(%b_r0c3_m7_n54)
      aiex.dma_free_task(%c_r0c3_m7_n54)
      // core r0c3 m7 n55
      %a_r0c3_m7_n55 = aiex.dma_configure_task_for @A_r0 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 229376 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r0c3_m7_n55)
      %b_r0c3_m7_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r0c3_m7_n55)
      %c_r0c3_m7_n55 = aiex.dma_configure_task_for @C2_r0_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 806336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r0c3_m7_n55)
      aiex.dma_await_task(%c_r0c3_m7_n55)
      aiex.dma_free_task(%a_r0c3_m7_n55)
      aiex.dma_free_task(%b_r0c3_m7_n55)
      aiex.dma_free_task(%c_r0c3_m7_n55)
      // core r1c0 m8 n0
      %a_r1c0_m8_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n0)
      %b_r1c0_m8_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n0)
      %c_r1c0_m8_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n0)
      aiex.dma_await_task(%c_r1c0_m8_n0)
      aiex.dma_free_task(%a_r1c0_m8_n0)
      aiex.dma_free_task(%b_r1c0_m8_n0)
      aiex.dma_free_task(%c_r1c0_m8_n0)
      // core r1c0 m8 n1
      %a_r1c0_m8_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n1)
      %b_r1c0_m8_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n1)
      %c_r1c0_m8_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n1)
      aiex.dma_await_task(%c_r1c0_m8_n1)
      aiex.dma_free_task(%a_r1c0_m8_n1)
      aiex.dma_free_task(%b_r1c0_m8_n1)
      aiex.dma_free_task(%c_r1c0_m8_n1)
      // core r1c0 m8 n2
      %a_r1c0_m8_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n2)
      %b_r1c0_m8_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n2)
      %c_r1c0_m8_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n2)
      aiex.dma_await_task(%c_r1c0_m8_n2)
      aiex.dma_free_task(%a_r1c0_m8_n2)
      aiex.dma_free_task(%b_r1c0_m8_n2)
      aiex.dma_free_task(%c_r1c0_m8_n2)
      // core r1c0 m8 n3
      %a_r1c0_m8_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n3)
      %b_r1c0_m8_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n3)
      %c_r1c0_m8_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n3)
      aiex.dma_await_task(%c_r1c0_m8_n3)
      aiex.dma_free_task(%a_r1c0_m8_n3)
      aiex.dma_free_task(%b_r1c0_m8_n3)
      aiex.dma_free_task(%c_r1c0_m8_n3)
      // core r1c0 m8 n4
      %a_r1c0_m8_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n4)
      %b_r1c0_m8_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n4)
      %c_r1c0_m8_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n4)
      aiex.dma_await_task(%c_r1c0_m8_n4)
      aiex.dma_free_task(%a_r1c0_m8_n4)
      aiex.dma_free_task(%b_r1c0_m8_n4)
      aiex.dma_free_task(%c_r1c0_m8_n4)
      // core r1c0 m8 n5
      %a_r1c0_m8_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n5)
      %b_r1c0_m8_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n5)
      %c_r1c0_m8_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n5)
      aiex.dma_await_task(%c_r1c0_m8_n5)
      aiex.dma_free_task(%a_r1c0_m8_n5)
      aiex.dma_free_task(%b_r1c0_m8_n5)
      aiex.dma_free_task(%c_r1c0_m8_n5)
      // core r1c0 m8 n6
      %a_r1c0_m8_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n6)
      %b_r1c0_m8_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n6)
      %c_r1c0_m8_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n6)
      aiex.dma_await_task(%c_r1c0_m8_n6)
      aiex.dma_free_task(%a_r1c0_m8_n6)
      aiex.dma_free_task(%b_r1c0_m8_n6)
      aiex.dma_free_task(%c_r1c0_m8_n6)
      // core r1c0 m8 n7
      %a_r1c0_m8_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n7)
      %b_r1c0_m8_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n7)
      %c_r1c0_m8_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 917952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n7)
      aiex.dma_await_task(%c_r1c0_m8_n7)
      aiex.dma_free_task(%a_r1c0_m8_n7)
      aiex.dma_free_task(%b_r1c0_m8_n7)
      aiex.dma_free_task(%c_r1c0_m8_n7)
      // core r1c0 m8 n8
      %a_r1c0_m8_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n8)
      %b_r1c0_m8_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n8)
      %c_r1c0_m8_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n8)
      aiex.dma_await_task(%c_r1c0_m8_n8)
      aiex.dma_free_task(%a_r1c0_m8_n8)
      aiex.dma_free_task(%b_r1c0_m8_n8)
      aiex.dma_free_task(%c_r1c0_m8_n8)
      // core r1c0 m8 n9
      %a_r1c0_m8_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n9)
      %b_r1c0_m8_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n9)
      %c_r1c0_m8_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n9)
      aiex.dma_await_task(%c_r1c0_m8_n9)
      aiex.dma_free_task(%a_r1c0_m8_n9)
      aiex.dma_free_task(%b_r1c0_m8_n9)
      aiex.dma_free_task(%c_r1c0_m8_n9)
      // core r1c0 m8 n10
      %a_r1c0_m8_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n10)
      %b_r1c0_m8_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n10)
      %c_r1c0_m8_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n10)
      aiex.dma_await_task(%c_r1c0_m8_n10)
      aiex.dma_free_task(%a_r1c0_m8_n10)
      aiex.dma_free_task(%b_r1c0_m8_n10)
      aiex.dma_free_task(%c_r1c0_m8_n10)
      // core r1c0 m8 n11
      %a_r1c0_m8_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n11)
      %b_r1c0_m8_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n11)
      %c_r1c0_m8_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n11)
      aiex.dma_await_task(%c_r1c0_m8_n11)
      aiex.dma_free_task(%a_r1c0_m8_n11)
      aiex.dma_free_task(%b_r1c0_m8_n11)
      aiex.dma_free_task(%c_r1c0_m8_n11)
      // core r1c0 m8 n12
      %a_r1c0_m8_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n12)
      %b_r1c0_m8_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n12)
      %c_r1c0_m8_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n12)
      aiex.dma_await_task(%c_r1c0_m8_n12)
      aiex.dma_free_task(%a_r1c0_m8_n12)
      aiex.dma_free_task(%b_r1c0_m8_n12)
      aiex.dma_free_task(%c_r1c0_m8_n12)
      // core r1c0 m8 n13
      %a_r1c0_m8_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m8_n13)
      %b_r1c0_m8_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m8_n13)
      %c_r1c0_m8_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m8_n13)
      aiex.dma_await_task(%c_r1c0_m8_n13)
      aiex.dma_free_task(%a_r1c0_m8_n13)
      aiex.dma_free_task(%b_r1c0_m8_n13)
      aiex.dma_free_task(%c_r1c0_m8_n13)
      // core r1c0 m9 n0
      %a_r1c0_m9_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n0)
      %b_r1c0_m9_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n0)
      %c_r1c0_m9_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n0)
      aiex.dma_await_task(%c_r1c0_m9_n0)
      aiex.dma_free_task(%a_r1c0_m9_n0)
      aiex.dma_free_task(%b_r1c0_m9_n0)
      aiex.dma_free_task(%c_r1c0_m9_n0)
      // core r1c0 m9 n1
      %a_r1c0_m9_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n1)
      %b_r1c0_m9_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n1)
      %c_r1c0_m9_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n1)
      aiex.dma_await_task(%c_r1c0_m9_n1)
      aiex.dma_free_task(%a_r1c0_m9_n1)
      aiex.dma_free_task(%b_r1c0_m9_n1)
      aiex.dma_free_task(%c_r1c0_m9_n1)
      // core r1c0 m9 n2
      %a_r1c0_m9_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n2)
      %b_r1c0_m9_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n2)
      %c_r1c0_m9_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n2)
      aiex.dma_await_task(%c_r1c0_m9_n2)
      aiex.dma_free_task(%a_r1c0_m9_n2)
      aiex.dma_free_task(%b_r1c0_m9_n2)
      aiex.dma_free_task(%c_r1c0_m9_n2)
      // core r1c0 m9 n3
      %a_r1c0_m9_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n3)
      %b_r1c0_m9_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n3)
      %c_r1c0_m9_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n3)
      aiex.dma_await_task(%c_r1c0_m9_n3)
      aiex.dma_free_task(%a_r1c0_m9_n3)
      aiex.dma_free_task(%b_r1c0_m9_n3)
      aiex.dma_free_task(%c_r1c0_m9_n3)
      // core r1c0 m9 n4
      %a_r1c0_m9_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n4)
      %b_r1c0_m9_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n4)
      %c_r1c0_m9_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n4)
      aiex.dma_await_task(%c_r1c0_m9_n4)
      aiex.dma_free_task(%a_r1c0_m9_n4)
      aiex.dma_free_task(%b_r1c0_m9_n4)
      aiex.dma_free_task(%c_r1c0_m9_n4)
      // core r1c0 m9 n5
      %a_r1c0_m9_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n5)
      %b_r1c0_m9_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n5)
      %c_r1c0_m9_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n5)
      aiex.dma_await_task(%c_r1c0_m9_n5)
      aiex.dma_free_task(%a_r1c0_m9_n5)
      aiex.dma_free_task(%b_r1c0_m9_n5)
      aiex.dma_free_task(%c_r1c0_m9_n5)
      // core r1c0 m9 n6
      %a_r1c0_m9_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n6)
      %b_r1c0_m9_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n6)
      %c_r1c0_m9_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n6)
      aiex.dma_await_task(%c_r1c0_m9_n6)
      aiex.dma_free_task(%a_r1c0_m9_n6)
      aiex.dma_free_task(%b_r1c0_m9_n6)
      aiex.dma_free_task(%c_r1c0_m9_n6)
      // core r1c0 m9 n7
      %a_r1c0_m9_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n7)
      %b_r1c0_m9_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n7)
      %c_r1c0_m9_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n7)
      aiex.dma_await_task(%c_r1c0_m9_n7)
      aiex.dma_free_task(%a_r1c0_m9_n7)
      aiex.dma_free_task(%b_r1c0_m9_n7)
      aiex.dma_free_task(%c_r1c0_m9_n7)
      // core r1c0 m9 n8
      %a_r1c0_m9_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n8)
      %b_r1c0_m9_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n8)
      %c_r1c0_m9_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n8)
      aiex.dma_await_task(%c_r1c0_m9_n8)
      aiex.dma_free_task(%a_r1c0_m9_n8)
      aiex.dma_free_task(%b_r1c0_m9_n8)
      aiex.dma_free_task(%c_r1c0_m9_n8)
      // core r1c0 m9 n9
      %a_r1c0_m9_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n9)
      %b_r1c0_m9_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n9)
      %c_r1c0_m9_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n9)
      aiex.dma_await_task(%c_r1c0_m9_n9)
      aiex.dma_free_task(%a_r1c0_m9_n9)
      aiex.dma_free_task(%b_r1c0_m9_n9)
      aiex.dma_free_task(%c_r1c0_m9_n9)
      // core r1c0 m9 n10
      %a_r1c0_m9_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n10)
      %b_r1c0_m9_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n10)
      %c_r1c0_m9_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n10)
      aiex.dma_await_task(%c_r1c0_m9_n10)
      aiex.dma_free_task(%a_r1c0_m9_n10)
      aiex.dma_free_task(%b_r1c0_m9_n10)
      aiex.dma_free_task(%c_r1c0_m9_n10)
      // core r1c0 m9 n11
      %a_r1c0_m9_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n11)
      %b_r1c0_m9_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n11)
      %c_r1c0_m9_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n11)
      aiex.dma_await_task(%c_r1c0_m9_n11)
      aiex.dma_free_task(%a_r1c0_m9_n11)
      aiex.dma_free_task(%b_r1c0_m9_n11)
      aiex.dma_free_task(%c_r1c0_m9_n11)
      // core r1c0 m9 n12
      %a_r1c0_m9_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n12)
      %b_r1c0_m9_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n12)
      %c_r1c0_m9_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1032960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n12)
      aiex.dma_await_task(%c_r1c0_m9_n12)
      aiex.dma_free_task(%a_r1c0_m9_n12)
      aiex.dma_free_task(%b_r1c0_m9_n12)
      aiex.dma_free_task(%c_r1c0_m9_n12)
      // core r1c0 m9 n13
      %a_r1c0_m9_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m9_n13)
      %b_r1c0_m9_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m9_n13)
      %c_r1c0_m9_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m9_n13)
      aiex.dma_await_task(%c_r1c0_m9_n13)
      aiex.dma_free_task(%a_r1c0_m9_n13)
      aiex.dma_free_task(%b_r1c0_m9_n13)
      aiex.dma_free_task(%c_r1c0_m9_n13)
      // core r1c0 m10 n0
      %a_r1c0_m10_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n0)
      %b_r1c0_m10_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n0)
      %c_r1c0_m10_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1146880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n0)
      aiex.dma_await_task(%c_r1c0_m10_n0)
      aiex.dma_free_task(%a_r1c0_m10_n0)
      aiex.dma_free_task(%b_r1c0_m10_n0)
      aiex.dma_free_task(%c_r1c0_m10_n0)
      // core r1c0 m10 n1
      %a_r1c0_m10_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n1)
      %b_r1c0_m10_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n1)
      %c_r1c0_m10_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1146944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n1)
      aiex.dma_await_task(%c_r1c0_m10_n1)
      aiex.dma_free_task(%a_r1c0_m10_n1)
      aiex.dma_free_task(%b_r1c0_m10_n1)
      aiex.dma_free_task(%c_r1c0_m10_n1)
      // core r1c0 m10 n2
      %a_r1c0_m10_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n2)
      %b_r1c0_m10_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n2)
      %c_r1c0_m10_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n2)
      aiex.dma_await_task(%c_r1c0_m10_n2)
      aiex.dma_free_task(%a_r1c0_m10_n2)
      aiex.dma_free_task(%b_r1c0_m10_n2)
      aiex.dma_free_task(%c_r1c0_m10_n2)
      // core r1c0 m10 n3
      %a_r1c0_m10_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n3)
      %b_r1c0_m10_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n3)
      %c_r1c0_m10_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n3)
      aiex.dma_await_task(%c_r1c0_m10_n3)
      aiex.dma_free_task(%a_r1c0_m10_n3)
      aiex.dma_free_task(%b_r1c0_m10_n3)
      aiex.dma_free_task(%c_r1c0_m10_n3)
      // core r1c0 m10 n4
      %a_r1c0_m10_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n4)
      %b_r1c0_m10_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n4)
      %c_r1c0_m10_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n4)
      aiex.dma_await_task(%c_r1c0_m10_n4)
      aiex.dma_free_task(%a_r1c0_m10_n4)
      aiex.dma_free_task(%b_r1c0_m10_n4)
      aiex.dma_free_task(%c_r1c0_m10_n4)
      // core r1c0 m10 n5
      %a_r1c0_m10_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n5)
      %b_r1c0_m10_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n5)
      %c_r1c0_m10_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n5)
      aiex.dma_await_task(%c_r1c0_m10_n5)
      aiex.dma_free_task(%a_r1c0_m10_n5)
      aiex.dma_free_task(%b_r1c0_m10_n5)
      aiex.dma_free_task(%c_r1c0_m10_n5)
      // core r1c0 m10 n6
      %a_r1c0_m10_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n6)
      %b_r1c0_m10_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n6)
      %c_r1c0_m10_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n6)
      aiex.dma_await_task(%c_r1c0_m10_n6)
      aiex.dma_free_task(%a_r1c0_m10_n6)
      aiex.dma_free_task(%b_r1c0_m10_n6)
      aiex.dma_free_task(%c_r1c0_m10_n6)
      // core r1c0 m10 n7
      %a_r1c0_m10_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n7)
      %b_r1c0_m10_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n7)
      %c_r1c0_m10_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n7)
      aiex.dma_await_task(%c_r1c0_m10_n7)
      aiex.dma_free_task(%a_r1c0_m10_n7)
      aiex.dma_free_task(%b_r1c0_m10_n7)
      aiex.dma_free_task(%c_r1c0_m10_n7)
      // core r1c0 m10 n8
      %a_r1c0_m10_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n8)
      %b_r1c0_m10_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n8)
      %c_r1c0_m10_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n8)
      aiex.dma_await_task(%c_r1c0_m10_n8)
      aiex.dma_free_task(%a_r1c0_m10_n8)
      aiex.dma_free_task(%b_r1c0_m10_n8)
      aiex.dma_free_task(%c_r1c0_m10_n8)
      // core r1c0 m10 n9
      %a_r1c0_m10_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n9)
      %b_r1c0_m10_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n9)
      %c_r1c0_m10_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n9)
      aiex.dma_await_task(%c_r1c0_m10_n9)
      aiex.dma_free_task(%a_r1c0_m10_n9)
      aiex.dma_free_task(%b_r1c0_m10_n9)
      aiex.dma_free_task(%c_r1c0_m10_n9)
      // core r1c0 m10 n10
      %a_r1c0_m10_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n10)
      %b_r1c0_m10_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n10)
      %c_r1c0_m10_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n10)
      aiex.dma_await_task(%c_r1c0_m10_n10)
      aiex.dma_free_task(%a_r1c0_m10_n10)
      aiex.dma_free_task(%b_r1c0_m10_n10)
      aiex.dma_free_task(%c_r1c0_m10_n10)
      // core r1c0 m10 n11
      %a_r1c0_m10_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n11)
      %b_r1c0_m10_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n11)
      %c_r1c0_m10_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n11)
      aiex.dma_await_task(%c_r1c0_m10_n11)
      aiex.dma_free_task(%a_r1c0_m10_n11)
      aiex.dma_free_task(%b_r1c0_m10_n11)
      aiex.dma_free_task(%c_r1c0_m10_n11)
      // core r1c0 m10 n12
      %a_r1c0_m10_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n12)
      %b_r1c0_m10_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n12)
      %c_r1c0_m10_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n12)
      aiex.dma_await_task(%c_r1c0_m10_n12)
      aiex.dma_free_task(%a_r1c0_m10_n12)
      aiex.dma_free_task(%b_r1c0_m10_n12)
      aiex.dma_free_task(%c_r1c0_m10_n12)
      // core r1c0 m10 n13
      %a_r1c0_m10_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m10_n13)
      %b_r1c0_m10_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m10_n13)
      %c_r1c0_m10_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m10_n13)
      aiex.dma_await_task(%c_r1c0_m10_n13)
      aiex.dma_free_task(%a_r1c0_m10_n13)
      aiex.dma_free_task(%b_r1c0_m10_n13)
      aiex.dma_free_task(%c_r1c0_m10_n13)
      // core r1c0 m11 n0
      %a_r1c0_m11_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n0)
      %b_r1c0_m11_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n0)
      %c_r1c0_m11_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n0)
      aiex.dma_await_task(%c_r1c0_m11_n0)
      aiex.dma_free_task(%a_r1c0_m11_n0)
      aiex.dma_free_task(%b_r1c0_m11_n0)
      aiex.dma_free_task(%c_r1c0_m11_n0)
      // core r1c0 m11 n1
      %a_r1c0_m11_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n1)
      %b_r1c0_m11_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n1)
      %c_r1c0_m11_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n1)
      aiex.dma_await_task(%c_r1c0_m11_n1)
      aiex.dma_free_task(%a_r1c0_m11_n1)
      aiex.dma_free_task(%b_r1c0_m11_n1)
      aiex.dma_free_task(%c_r1c0_m11_n1)
      // core r1c0 m11 n2
      %a_r1c0_m11_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n2)
      %b_r1c0_m11_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n2)
      %c_r1c0_m11_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n2)
      aiex.dma_await_task(%c_r1c0_m11_n2)
      aiex.dma_free_task(%a_r1c0_m11_n2)
      aiex.dma_free_task(%b_r1c0_m11_n2)
      aiex.dma_free_task(%c_r1c0_m11_n2)
      // core r1c0 m11 n3
      %a_r1c0_m11_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n3)
      %b_r1c0_m11_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n3)
      %c_r1c0_m11_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n3)
      aiex.dma_await_task(%c_r1c0_m11_n3)
      aiex.dma_free_task(%a_r1c0_m11_n3)
      aiex.dma_free_task(%b_r1c0_m11_n3)
      aiex.dma_free_task(%c_r1c0_m11_n3)
      // core r1c0 m11 n4
      %a_r1c0_m11_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n4)
      %b_r1c0_m11_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n4)
      %c_r1c0_m11_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n4)
      aiex.dma_await_task(%c_r1c0_m11_n4)
      aiex.dma_free_task(%a_r1c0_m11_n4)
      aiex.dma_free_task(%b_r1c0_m11_n4)
      aiex.dma_free_task(%c_r1c0_m11_n4)
      // core r1c0 m11 n5
      %a_r1c0_m11_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n5)
      %b_r1c0_m11_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n5)
      %c_r1c0_m11_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n5)
      aiex.dma_await_task(%c_r1c0_m11_n5)
      aiex.dma_free_task(%a_r1c0_m11_n5)
      aiex.dma_free_task(%b_r1c0_m11_n5)
      aiex.dma_free_task(%c_r1c0_m11_n5)
      // core r1c0 m11 n6
      %a_r1c0_m11_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n6)
      %b_r1c0_m11_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n6)
      %c_r1c0_m11_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1261952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n6)
      aiex.dma_await_task(%c_r1c0_m11_n6)
      aiex.dma_free_task(%a_r1c0_m11_n6)
      aiex.dma_free_task(%b_r1c0_m11_n6)
      aiex.dma_free_task(%c_r1c0_m11_n6)
      // core r1c0 m11 n7
      %a_r1c0_m11_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n7)
      %b_r1c0_m11_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n7)
      %c_r1c0_m11_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n7)
      aiex.dma_await_task(%c_r1c0_m11_n7)
      aiex.dma_free_task(%a_r1c0_m11_n7)
      aiex.dma_free_task(%b_r1c0_m11_n7)
      aiex.dma_free_task(%c_r1c0_m11_n7)
      // core r1c0 m11 n8
      %a_r1c0_m11_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n8)
      %b_r1c0_m11_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n8)
      %c_r1c0_m11_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n8)
      aiex.dma_await_task(%c_r1c0_m11_n8)
      aiex.dma_free_task(%a_r1c0_m11_n8)
      aiex.dma_free_task(%b_r1c0_m11_n8)
      aiex.dma_free_task(%c_r1c0_m11_n8)
      // core r1c0 m11 n9
      %a_r1c0_m11_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n9)
      %b_r1c0_m11_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n9)
      %c_r1c0_m11_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n9)
      aiex.dma_await_task(%c_r1c0_m11_n9)
      aiex.dma_free_task(%a_r1c0_m11_n9)
      aiex.dma_free_task(%b_r1c0_m11_n9)
      aiex.dma_free_task(%c_r1c0_m11_n9)
      // core r1c0 m11 n10
      %a_r1c0_m11_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n10)
      %b_r1c0_m11_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n10)
      %c_r1c0_m11_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n10)
      aiex.dma_await_task(%c_r1c0_m11_n10)
      aiex.dma_free_task(%a_r1c0_m11_n10)
      aiex.dma_free_task(%b_r1c0_m11_n10)
      aiex.dma_free_task(%c_r1c0_m11_n10)
      // core r1c0 m11 n11
      %a_r1c0_m11_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n11)
      %b_r1c0_m11_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n11)
      %c_r1c0_m11_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n11)
      aiex.dma_await_task(%c_r1c0_m11_n11)
      aiex.dma_free_task(%a_r1c0_m11_n11)
      aiex.dma_free_task(%b_r1c0_m11_n11)
      aiex.dma_free_task(%c_r1c0_m11_n11)
      // core r1c0 m11 n12
      %a_r1c0_m11_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n12)
      %b_r1c0_m11_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n12)
      %c_r1c0_m11_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n12)
      aiex.dma_await_task(%c_r1c0_m11_n12)
      aiex.dma_free_task(%a_r1c0_m11_n12)
      aiex.dma_free_task(%b_r1c0_m11_n12)
      aiex.dma_free_task(%c_r1c0_m11_n12)
      // core r1c0 m11 n13
      %a_r1c0_m11_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m11_n13)
      %b_r1c0_m11_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m11_n13)
      %c_r1c0_m11_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m11_n13)
      aiex.dma_await_task(%c_r1c0_m11_n13)
      aiex.dma_free_task(%a_r1c0_m11_n13)
      aiex.dma_free_task(%b_r1c0_m11_n13)
      aiex.dma_free_task(%c_r1c0_m11_n13)
      // core r1c0 m12 n0
      %a_r1c0_m12_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n0)
      %b_r1c0_m12_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n0)
      %c_r1c0_m12_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n0)
      aiex.dma_await_task(%c_r1c0_m12_n0)
      aiex.dma_free_task(%a_r1c0_m12_n0)
      aiex.dma_free_task(%b_r1c0_m12_n0)
      aiex.dma_free_task(%c_r1c0_m12_n0)
      // core r1c0 m12 n1
      %a_r1c0_m12_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n1)
      %b_r1c0_m12_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n1)
      %c_r1c0_m12_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n1)
      aiex.dma_await_task(%c_r1c0_m12_n1)
      aiex.dma_free_task(%a_r1c0_m12_n1)
      aiex.dma_free_task(%b_r1c0_m12_n1)
      aiex.dma_free_task(%c_r1c0_m12_n1)
      // core r1c0 m12 n2
      %a_r1c0_m12_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n2)
      %b_r1c0_m12_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n2)
      %c_r1c0_m12_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n2)
      aiex.dma_await_task(%c_r1c0_m12_n2)
      aiex.dma_free_task(%a_r1c0_m12_n2)
      aiex.dma_free_task(%b_r1c0_m12_n2)
      aiex.dma_free_task(%c_r1c0_m12_n2)
      // core r1c0 m12 n3
      %a_r1c0_m12_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n3)
      %b_r1c0_m12_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n3)
      %c_r1c0_m12_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n3)
      aiex.dma_await_task(%c_r1c0_m12_n3)
      aiex.dma_free_task(%a_r1c0_m12_n3)
      aiex.dma_free_task(%b_r1c0_m12_n3)
      aiex.dma_free_task(%c_r1c0_m12_n3)
      // core r1c0 m12 n4
      %a_r1c0_m12_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n4)
      %b_r1c0_m12_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n4)
      %c_r1c0_m12_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n4)
      aiex.dma_await_task(%c_r1c0_m12_n4)
      aiex.dma_free_task(%a_r1c0_m12_n4)
      aiex.dma_free_task(%b_r1c0_m12_n4)
      aiex.dma_free_task(%c_r1c0_m12_n4)
      // core r1c0 m12 n5
      %a_r1c0_m12_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n5)
      %b_r1c0_m12_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n5)
      %c_r1c0_m12_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n5)
      aiex.dma_await_task(%c_r1c0_m12_n5)
      aiex.dma_free_task(%a_r1c0_m12_n5)
      aiex.dma_free_task(%b_r1c0_m12_n5)
      aiex.dma_free_task(%c_r1c0_m12_n5)
      // core r1c0 m12 n6
      %a_r1c0_m12_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n6)
      %b_r1c0_m12_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n6)
      %c_r1c0_m12_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n6)
      aiex.dma_await_task(%c_r1c0_m12_n6)
      aiex.dma_free_task(%a_r1c0_m12_n6)
      aiex.dma_free_task(%b_r1c0_m12_n6)
      aiex.dma_free_task(%c_r1c0_m12_n6)
      // core r1c0 m12 n7
      %a_r1c0_m12_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n7)
      %b_r1c0_m12_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n7)
      %c_r1c0_m12_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n7)
      aiex.dma_await_task(%c_r1c0_m12_n7)
      aiex.dma_free_task(%a_r1c0_m12_n7)
      aiex.dma_free_task(%b_r1c0_m12_n7)
      aiex.dma_free_task(%c_r1c0_m12_n7)
      // core r1c0 m12 n8
      %a_r1c0_m12_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n8)
      %b_r1c0_m12_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n8)
      %c_r1c0_m12_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n8)
      aiex.dma_await_task(%c_r1c0_m12_n8)
      aiex.dma_free_task(%a_r1c0_m12_n8)
      aiex.dma_free_task(%b_r1c0_m12_n8)
      aiex.dma_free_task(%c_r1c0_m12_n8)
      // core r1c0 m12 n9
      %a_r1c0_m12_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n9)
      %b_r1c0_m12_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n9)
      %c_r1c0_m12_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n9)
      aiex.dma_await_task(%c_r1c0_m12_n9)
      aiex.dma_free_task(%a_r1c0_m12_n9)
      aiex.dma_free_task(%b_r1c0_m12_n9)
      aiex.dma_free_task(%c_r1c0_m12_n9)
      // core r1c0 m12 n10
      %a_r1c0_m12_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n10)
      %b_r1c0_m12_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n10)
      %c_r1c0_m12_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n10)
      aiex.dma_await_task(%c_r1c0_m12_n10)
      aiex.dma_free_task(%a_r1c0_m12_n10)
      aiex.dma_free_task(%b_r1c0_m12_n10)
      aiex.dma_free_task(%c_r1c0_m12_n10)
      // core r1c0 m12 n11
      %a_r1c0_m12_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n11)
      %b_r1c0_m12_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n11)
      %c_r1c0_m12_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1376960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n11)
      aiex.dma_await_task(%c_r1c0_m12_n11)
      aiex.dma_free_task(%a_r1c0_m12_n11)
      aiex.dma_free_task(%b_r1c0_m12_n11)
      aiex.dma_free_task(%c_r1c0_m12_n11)
      // core r1c0 m12 n12
      %a_r1c0_m12_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n12)
      %b_r1c0_m12_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n12)
      %c_r1c0_m12_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n12)
      aiex.dma_await_task(%c_r1c0_m12_n12)
      aiex.dma_free_task(%a_r1c0_m12_n12)
      aiex.dma_free_task(%b_r1c0_m12_n12)
      aiex.dma_free_task(%c_r1c0_m12_n12)
      // core r1c0 m12 n13
      %a_r1c0_m12_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m12_n13)
      %b_r1c0_m12_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m12_n13)
      %c_r1c0_m12_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m12_n13)
      aiex.dma_await_task(%c_r1c0_m12_n13)
      aiex.dma_free_task(%a_r1c0_m12_n13)
      aiex.dma_free_task(%b_r1c0_m12_n13)
      aiex.dma_free_task(%c_r1c0_m12_n13)
      // core r1c0 m13 n0
      %a_r1c0_m13_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n0)
      %b_r1c0_m13_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n0)
      %c_r1c0_m13_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1490944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n0)
      aiex.dma_await_task(%c_r1c0_m13_n0)
      aiex.dma_free_task(%a_r1c0_m13_n0)
      aiex.dma_free_task(%b_r1c0_m13_n0)
      aiex.dma_free_task(%c_r1c0_m13_n0)
      // core r1c0 m13 n1
      %a_r1c0_m13_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n1)
      %b_r1c0_m13_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n1)
      %c_r1c0_m13_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n1)
      aiex.dma_await_task(%c_r1c0_m13_n1)
      aiex.dma_free_task(%a_r1c0_m13_n1)
      aiex.dma_free_task(%b_r1c0_m13_n1)
      aiex.dma_free_task(%c_r1c0_m13_n1)
      // core r1c0 m13 n2
      %a_r1c0_m13_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n2)
      %b_r1c0_m13_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n2)
      %c_r1c0_m13_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n2)
      aiex.dma_await_task(%c_r1c0_m13_n2)
      aiex.dma_free_task(%a_r1c0_m13_n2)
      aiex.dma_free_task(%b_r1c0_m13_n2)
      aiex.dma_free_task(%c_r1c0_m13_n2)
      // core r1c0 m13 n3
      %a_r1c0_m13_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n3)
      %b_r1c0_m13_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n3)
      %c_r1c0_m13_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n3)
      aiex.dma_await_task(%c_r1c0_m13_n3)
      aiex.dma_free_task(%a_r1c0_m13_n3)
      aiex.dma_free_task(%b_r1c0_m13_n3)
      aiex.dma_free_task(%c_r1c0_m13_n3)
      // core r1c0 m13 n4
      %a_r1c0_m13_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n4)
      %b_r1c0_m13_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n4)
      %c_r1c0_m13_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n4)
      aiex.dma_await_task(%c_r1c0_m13_n4)
      aiex.dma_free_task(%a_r1c0_m13_n4)
      aiex.dma_free_task(%b_r1c0_m13_n4)
      aiex.dma_free_task(%c_r1c0_m13_n4)
      // core r1c0 m13 n5
      %a_r1c0_m13_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n5)
      %b_r1c0_m13_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n5)
      %c_r1c0_m13_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n5)
      aiex.dma_await_task(%c_r1c0_m13_n5)
      aiex.dma_free_task(%a_r1c0_m13_n5)
      aiex.dma_free_task(%b_r1c0_m13_n5)
      aiex.dma_free_task(%c_r1c0_m13_n5)
      // core r1c0 m13 n6
      %a_r1c0_m13_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n6)
      %b_r1c0_m13_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n6)
      %c_r1c0_m13_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n6)
      aiex.dma_await_task(%c_r1c0_m13_n6)
      aiex.dma_free_task(%a_r1c0_m13_n6)
      aiex.dma_free_task(%b_r1c0_m13_n6)
      aiex.dma_free_task(%c_r1c0_m13_n6)
      // core r1c0 m13 n7
      %a_r1c0_m13_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n7)
      %b_r1c0_m13_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n7)
      %c_r1c0_m13_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n7)
      aiex.dma_await_task(%c_r1c0_m13_n7)
      aiex.dma_free_task(%a_r1c0_m13_n7)
      aiex.dma_free_task(%b_r1c0_m13_n7)
      aiex.dma_free_task(%c_r1c0_m13_n7)
      // core r1c0 m13 n8
      %a_r1c0_m13_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n8)
      %b_r1c0_m13_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n8)
      %c_r1c0_m13_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n8)
      aiex.dma_await_task(%c_r1c0_m13_n8)
      aiex.dma_free_task(%a_r1c0_m13_n8)
      aiex.dma_free_task(%b_r1c0_m13_n8)
      aiex.dma_free_task(%c_r1c0_m13_n8)
      // core r1c0 m13 n9
      %a_r1c0_m13_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n9)
      %b_r1c0_m13_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n9)
      %c_r1c0_m13_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n9)
      aiex.dma_await_task(%c_r1c0_m13_n9)
      aiex.dma_free_task(%a_r1c0_m13_n9)
      aiex.dma_free_task(%b_r1c0_m13_n9)
      aiex.dma_free_task(%c_r1c0_m13_n9)
      // core r1c0 m13 n10
      %a_r1c0_m13_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n10)
      %b_r1c0_m13_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n10)
      %c_r1c0_m13_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n10)
      aiex.dma_await_task(%c_r1c0_m13_n10)
      aiex.dma_free_task(%a_r1c0_m13_n10)
      aiex.dma_free_task(%b_r1c0_m13_n10)
      aiex.dma_free_task(%c_r1c0_m13_n10)
      // core r1c0 m13 n11
      %a_r1c0_m13_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n11)
      %b_r1c0_m13_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n11)
      %c_r1c0_m13_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n11)
      aiex.dma_await_task(%c_r1c0_m13_n11)
      aiex.dma_free_task(%a_r1c0_m13_n11)
      aiex.dma_free_task(%b_r1c0_m13_n11)
      aiex.dma_free_task(%c_r1c0_m13_n11)
      // core r1c0 m13 n12
      %a_r1c0_m13_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n12)
      %b_r1c0_m13_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n12)
      %c_r1c0_m13_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n12)
      aiex.dma_await_task(%c_r1c0_m13_n12)
      aiex.dma_free_task(%a_r1c0_m13_n12)
      aiex.dma_free_task(%b_r1c0_m13_n12)
      aiex.dma_free_task(%c_r1c0_m13_n12)
      // core r1c0 m13 n13
      %a_r1c0_m13_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m13_n13)
      %b_r1c0_m13_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m13_n13)
      %c_r1c0_m13_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m13_n13)
      aiex.dma_await_task(%c_r1c0_m13_n13)
      aiex.dma_free_task(%a_r1c0_m13_n13)
      aiex.dma_free_task(%b_r1c0_m13_n13)
      aiex.dma_free_task(%c_r1c0_m13_n13)
      // core r1c0 m14 n0
      %a_r1c0_m14_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n0)
      %b_r1c0_m14_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n0)
      %c_r1c0_m14_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1605632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n0)
      aiex.dma_await_task(%c_r1c0_m14_n0)
      aiex.dma_free_task(%a_r1c0_m14_n0)
      aiex.dma_free_task(%b_r1c0_m14_n0)
      aiex.dma_free_task(%c_r1c0_m14_n0)
      // core r1c0 m14 n1
      %a_r1c0_m14_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n1)
      %b_r1c0_m14_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n1)
      %c_r1c0_m14_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1605696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n1)
      aiex.dma_await_task(%c_r1c0_m14_n1)
      aiex.dma_free_task(%a_r1c0_m14_n1)
      aiex.dma_free_task(%b_r1c0_m14_n1)
      aiex.dma_free_task(%c_r1c0_m14_n1)
      // core r1c0 m14 n2
      %a_r1c0_m14_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n2)
      %b_r1c0_m14_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n2)
      %c_r1c0_m14_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1605760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n2)
      aiex.dma_await_task(%c_r1c0_m14_n2)
      aiex.dma_free_task(%a_r1c0_m14_n2)
      aiex.dma_free_task(%b_r1c0_m14_n2)
      aiex.dma_free_task(%c_r1c0_m14_n2)
      // core r1c0 m14 n3
      %a_r1c0_m14_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n3)
      %b_r1c0_m14_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n3)
      %c_r1c0_m14_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1605824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n3)
      aiex.dma_await_task(%c_r1c0_m14_n3)
      aiex.dma_free_task(%a_r1c0_m14_n3)
      aiex.dma_free_task(%b_r1c0_m14_n3)
      aiex.dma_free_task(%c_r1c0_m14_n3)
      // core r1c0 m14 n4
      %a_r1c0_m14_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n4)
      %b_r1c0_m14_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n4)
      %c_r1c0_m14_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1605888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n4)
      aiex.dma_await_task(%c_r1c0_m14_n4)
      aiex.dma_free_task(%a_r1c0_m14_n4)
      aiex.dma_free_task(%b_r1c0_m14_n4)
      aiex.dma_free_task(%c_r1c0_m14_n4)
      // core r1c0 m14 n5
      %a_r1c0_m14_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n5)
      %b_r1c0_m14_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n5)
      %c_r1c0_m14_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1605952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n5)
      aiex.dma_await_task(%c_r1c0_m14_n5)
      aiex.dma_free_task(%a_r1c0_m14_n5)
      aiex.dma_free_task(%b_r1c0_m14_n5)
      aiex.dma_free_task(%c_r1c0_m14_n5)
      // core r1c0 m14 n6
      %a_r1c0_m14_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n6)
      %b_r1c0_m14_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n6)
      %c_r1c0_m14_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n6)
      aiex.dma_await_task(%c_r1c0_m14_n6)
      aiex.dma_free_task(%a_r1c0_m14_n6)
      aiex.dma_free_task(%b_r1c0_m14_n6)
      aiex.dma_free_task(%c_r1c0_m14_n6)
      // core r1c0 m14 n7
      %a_r1c0_m14_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n7)
      %b_r1c0_m14_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n7)
      %c_r1c0_m14_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n7)
      aiex.dma_await_task(%c_r1c0_m14_n7)
      aiex.dma_free_task(%a_r1c0_m14_n7)
      aiex.dma_free_task(%b_r1c0_m14_n7)
      aiex.dma_free_task(%c_r1c0_m14_n7)
      // core r1c0 m14 n8
      %a_r1c0_m14_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n8)
      %b_r1c0_m14_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n8)
      %c_r1c0_m14_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n8)
      aiex.dma_await_task(%c_r1c0_m14_n8)
      aiex.dma_free_task(%a_r1c0_m14_n8)
      aiex.dma_free_task(%b_r1c0_m14_n8)
      aiex.dma_free_task(%c_r1c0_m14_n8)
      // core r1c0 m14 n9
      %a_r1c0_m14_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n9)
      %b_r1c0_m14_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n9)
      %c_r1c0_m14_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n9)
      aiex.dma_await_task(%c_r1c0_m14_n9)
      aiex.dma_free_task(%a_r1c0_m14_n9)
      aiex.dma_free_task(%b_r1c0_m14_n9)
      aiex.dma_free_task(%c_r1c0_m14_n9)
      // core r1c0 m14 n10
      %a_r1c0_m14_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n10)
      %b_r1c0_m14_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n10)
      %c_r1c0_m14_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n10)
      aiex.dma_await_task(%c_r1c0_m14_n10)
      aiex.dma_free_task(%a_r1c0_m14_n10)
      aiex.dma_free_task(%b_r1c0_m14_n10)
      aiex.dma_free_task(%c_r1c0_m14_n10)
      // core r1c0 m14 n11
      %a_r1c0_m14_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n11)
      %b_r1c0_m14_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n11)
      %c_r1c0_m14_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n11)
      aiex.dma_await_task(%c_r1c0_m14_n11)
      aiex.dma_free_task(%a_r1c0_m14_n11)
      aiex.dma_free_task(%b_r1c0_m14_n11)
      aiex.dma_free_task(%c_r1c0_m14_n11)
      // core r1c0 m14 n12
      %a_r1c0_m14_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n12)
      %b_r1c0_m14_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n12)
      %c_r1c0_m14_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n12)
      aiex.dma_await_task(%c_r1c0_m14_n12)
      aiex.dma_free_task(%a_r1c0_m14_n12)
      aiex.dma_free_task(%b_r1c0_m14_n12)
      aiex.dma_free_task(%c_r1c0_m14_n12)
      // core r1c0 m14 n13
      %a_r1c0_m14_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m14_n13)
      %b_r1c0_m14_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m14_n13)
      %c_r1c0_m14_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606464 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m14_n13)
      aiex.dma_await_task(%c_r1c0_m14_n13)
      aiex.dma_free_task(%a_r1c0_m14_n13)
      aiex.dma_free_task(%b_r1c0_m14_n13)
      aiex.dma_free_task(%c_r1c0_m14_n13)
      // core r1c0 m15 n0
      %a_r1c0_m15_n0 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n0)
      %b_r1c0_m15_n0 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 0 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n0)
      %c_r1c0_m15_n0 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n0)
      aiex.dma_await_task(%c_r1c0_m15_n0)
      aiex.dma_free_task(%a_r1c0_m15_n0)
      aiex.dma_free_task(%b_r1c0_m15_n0)
      aiex.dma_free_task(%c_r1c0_m15_n0)
      // core r1c0 m15 n1
      %a_r1c0_m15_n1 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n1)
      %b_r1c0_m15_n1 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 64 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n1)
      %c_r1c0_m15_n1 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n1)
      aiex.dma_await_task(%c_r1c0_m15_n1)
      aiex.dma_free_task(%a_r1c0_m15_n1)
      aiex.dma_free_task(%b_r1c0_m15_n1)
      aiex.dma_free_task(%c_r1c0_m15_n1)
      // core r1c0 m15 n2
      %a_r1c0_m15_n2 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n2)
      %b_r1c0_m15_n2 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 128 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n2)
      %c_r1c0_m15_n2 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n2)
      aiex.dma_await_task(%c_r1c0_m15_n2)
      aiex.dma_free_task(%a_r1c0_m15_n2)
      aiex.dma_free_task(%b_r1c0_m15_n2)
      aiex.dma_free_task(%c_r1c0_m15_n2)
      // core r1c0 m15 n3
      %a_r1c0_m15_n3 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n3)
      %b_r1c0_m15_n3 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 192 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n3)
      %c_r1c0_m15_n3 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n3)
      aiex.dma_await_task(%c_r1c0_m15_n3)
      aiex.dma_free_task(%a_r1c0_m15_n3)
      aiex.dma_free_task(%b_r1c0_m15_n3)
      aiex.dma_free_task(%c_r1c0_m15_n3)
      // core r1c0 m15 n4
      %a_r1c0_m15_n4 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n4)
      %b_r1c0_m15_n4 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 256 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n4)
      %c_r1c0_m15_n4 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n4)
      aiex.dma_await_task(%c_r1c0_m15_n4)
      aiex.dma_free_task(%a_r1c0_m15_n4)
      aiex.dma_free_task(%b_r1c0_m15_n4)
      aiex.dma_free_task(%c_r1c0_m15_n4)
      // core r1c0 m15 n5
      %a_r1c0_m15_n5 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n5)
      %b_r1c0_m15_n5 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 320 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n5)
      %c_r1c0_m15_n5 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n5)
      aiex.dma_await_task(%c_r1c0_m15_n5)
      aiex.dma_free_task(%a_r1c0_m15_n5)
      aiex.dma_free_task(%b_r1c0_m15_n5)
      aiex.dma_free_task(%c_r1c0_m15_n5)
      // core r1c0 m15 n6
      %a_r1c0_m15_n6 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n6)
      %b_r1c0_m15_n6 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 384 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n6)
      %c_r1c0_m15_n6 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n6)
      aiex.dma_await_task(%c_r1c0_m15_n6)
      aiex.dma_free_task(%a_r1c0_m15_n6)
      aiex.dma_free_task(%b_r1c0_m15_n6)
      aiex.dma_free_task(%c_r1c0_m15_n6)
      // core r1c0 m15 n7
      %a_r1c0_m15_n7 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n7)
      %b_r1c0_m15_n7 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 448 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n7)
      %c_r1c0_m15_n7 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n7)
      aiex.dma_await_task(%c_r1c0_m15_n7)
      aiex.dma_free_task(%a_r1c0_m15_n7)
      aiex.dma_free_task(%b_r1c0_m15_n7)
      aiex.dma_free_task(%c_r1c0_m15_n7)
      // core r1c0 m15 n8
      %a_r1c0_m15_n8 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n8)
      %b_r1c0_m15_n8 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 512 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n8)
      %c_r1c0_m15_n8 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n8)
      aiex.dma_await_task(%c_r1c0_m15_n8)
      aiex.dma_free_task(%a_r1c0_m15_n8)
      aiex.dma_free_task(%b_r1c0_m15_n8)
      aiex.dma_free_task(%c_r1c0_m15_n8)
      // core r1c0 m15 n9
      %a_r1c0_m15_n9 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n9)
      %b_r1c0_m15_n9 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 576 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n9)
      %c_r1c0_m15_n9 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n9)
      aiex.dma_await_task(%c_r1c0_m15_n9)
      aiex.dma_free_task(%a_r1c0_m15_n9)
      aiex.dma_free_task(%b_r1c0_m15_n9)
      aiex.dma_free_task(%c_r1c0_m15_n9)
      // core r1c0 m15 n10
      %a_r1c0_m15_n10 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n10)
      %b_r1c0_m15_n10 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 640 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n10)
      %c_r1c0_m15_n10 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1720960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n10)
      aiex.dma_await_task(%c_r1c0_m15_n10)
      aiex.dma_free_task(%a_r1c0_m15_n10)
      aiex.dma_free_task(%b_r1c0_m15_n10)
      aiex.dma_free_task(%c_r1c0_m15_n10)
      // core r1c0 m15 n11
      %a_r1c0_m15_n11 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n11)
      %b_r1c0_m15_n11 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 704 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n11)
      %c_r1c0_m15_n11 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n11)
      aiex.dma_await_task(%c_r1c0_m15_n11)
      aiex.dma_free_task(%a_r1c0_m15_n11)
      aiex.dma_free_task(%b_r1c0_m15_n11)
      aiex.dma_free_task(%c_r1c0_m15_n11)
      // core r1c0 m15 n12
      %a_r1c0_m15_n12 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n12)
      %b_r1c0_m15_n12 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 768 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n12)
      %c_r1c0_m15_n12 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n12)
      aiex.dma_await_task(%c_r1c0_m15_n12)
      aiex.dma_free_task(%a_r1c0_m15_n12)
      aiex.dma_free_task(%b_r1c0_m15_n12)
      aiex.dma_free_task(%c_r1c0_m15_n12)
      // core r1c0 m15 n13
      %a_r1c0_m15_n13 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c0_m15_n13)
      %b_r1c0_m15_n13 = aiex.dma_configure_task_for @B_c0 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 832 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c0_m15_n13)
      %c_r1c0_m15_n13 = aiex.dma_configure_task_for @C2_r1_c0 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c0_m15_n13)
      aiex.dma_await_task(%c_r1c0_m15_n13)
      aiex.dma_free_task(%a_r1c0_m15_n13)
      aiex.dma_free_task(%b_r1c0_m15_n13)
      aiex.dma_free_task(%c_r1c0_m15_n13)
      // core r1c1 m8 n14
      %a_r1c1_m8_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n14)
      %b_r1c1_m8_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n14)
      %c_r1c1_m8_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n14)
      aiex.dma_await_task(%c_r1c1_m8_n14)
      aiex.dma_free_task(%a_r1c1_m8_n14)
      aiex.dma_free_task(%b_r1c1_m8_n14)
      aiex.dma_free_task(%c_r1c1_m8_n14)
      // core r1c1 m8 n15
      %a_r1c1_m8_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n15)
      %b_r1c1_m8_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n15)
      %c_r1c1_m8_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918464 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n15)
      aiex.dma_await_task(%c_r1c1_m8_n15)
      aiex.dma_free_task(%a_r1c1_m8_n15)
      aiex.dma_free_task(%b_r1c1_m8_n15)
      aiex.dma_free_task(%c_r1c1_m8_n15)
      // core r1c1 m8 n16
      %a_r1c1_m8_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n16)
      %b_r1c1_m8_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n16)
      %c_r1c1_m8_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918528 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n16)
      aiex.dma_await_task(%c_r1c1_m8_n16)
      aiex.dma_free_task(%a_r1c1_m8_n16)
      aiex.dma_free_task(%b_r1c1_m8_n16)
      aiex.dma_free_task(%c_r1c1_m8_n16)
      // core r1c1 m8 n17
      %a_r1c1_m8_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n17)
      %b_r1c1_m8_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n17)
      %c_r1c1_m8_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918592 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n17)
      aiex.dma_await_task(%c_r1c1_m8_n17)
      aiex.dma_free_task(%a_r1c1_m8_n17)
      aiex.dma_free_task(%b_r1c1_m8_n17)
      aiex.dma_free_task(%c_r1c1_m8_n17)
      // core r1c1 m8 n18
      %a_r1c1_m8_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n18)
      %b_r1c1_m8_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n18)
      %c_r1c1_m8_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918656 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n18)
      aiex.dma_await_task(%c_r1c1_m8_n18)
      aiex.dma_free_task(%a_r1c1_m8_n18)
      aiex.dma_free_task(%b_r1c1_m8_n18)
      aiex.dma_free_task(%c_r1c1_m8_n18)
      // core r1c1 m8 n19
      %a_r1c1_m8_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n19)
      %b_r1c1_m8_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n19)
      %c_r1c1_m8_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918720 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n19)
      aiex.dma_await_task(%c_r1c1_m8_n19)
      aiex.dma_free_task(%a_r1c1_m8_n19)
      aiex.dma_free_task(%b_r1c1_m8_n19)
      aiex.dma_free_task(%c_r1c1_m8_n19)
      // core r1c1 m8 n20
      %a_r1c1_m8_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n20)
      %b_r1c1_m8_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n20)
      %c_r1c1_m8_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918784 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n20)
      aiex.dma_await_task(%c_r1c1_m8_n20)
      aiex.dma_free_task(%a_r1c1_m8_n20)
      aiex.dma_free_task(%b_r1c1_m8_n20)
      aiex.dma_free_task(%c_r1c1_m8_n20)
      // core r1c1 m8 n21
      %a_r1c1_m8_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n21)
      %b_r1c1_m8_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n21)
      %c_r1c1_m8_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918848 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n21)
      aiex.dma_await_task(%c_r1c1_m8_n21)
      aiex.dma_free_task(%a_r1c1_m8_n21)
      aiex.dma_free_task(%b_r1c1_m8_n21)
      aiex.dma_free_task(%c_r1c1_m8_n21)
      // core r1c1 m8 n22
      %a_r1c1_m8_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n22)
      %b_r1c1_m8_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n22)
      %c_r1c1_m8_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918912 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n22)
      aiex.dma_await_task(%c_r1c1_m8_n22)
      aiex.dma_free_task(%a_r1c1_m8_n22)
      aiex.dma_free_task(%b_r1c1_m8_n22)
      aiex.dma_free_task(%c_r1c1_m8_n22)
      // core r1c1 m8 n23
      %a_r1c1_m8_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n23)
      %b_r1c1_m8_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n23)
      %c_r1c1_m8_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 918976 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n23)
      aiex.dma_await_task(%c_r1c1_m8_n23)
      aiex.dma_free_task(%a_r1c1_m8_n23)
      aiex.dma_free_task(%b_r1c1_m8_n23)
      aiex.dma_free_task(%c_r1c1_m8_n23)
      // core r1c1 m8 n24
      %a_r1c1_m8_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n24)
      %b_r1c1_m8_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n24)
      %c_r1c1_m8_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919040 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n24)
      aiex.dma_await_task(%c_r1c1_m8_n24)
      aiex.dma_free_task(%a_r1c1_m8_n24)
      aiex.dma_free_task(%b_r1c1_m8_n24)
      aiex.dma_free_task(%c_r1c1_m8_n24)
      // core r1c1 m8 n25
      %a_r1c1_m8_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n25)
      %b_r1c1_m8_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n25)
      %c_r1c1_m8_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919104 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n25)
      aiex.dma_await_task(%c_r1c1_m8_n25)
      aiex.dma_free_task(%a_r1c1_m8_n25)
      aiex.dma_free_task(%b_r1c1_m8_n25)
      aiex.dma_free_task(%c_r1c1_m8_n25)
      // core r1c1 m8 n26
      %a_r1c1_m8_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n26)
      %b_r1c1_m8_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n26)
      %c_r1c1_m8_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919168 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n26)
      aiex.dma_await_task(%c_r1c1_m8_n26)
      aiex.dma_free_task(%a_r1c1_m8_n26)
      aiex.dma_free_task(%b_r1c1_m8_n26)
      aiex.dma_free_task(%c_r1c1_m8_n26)
      // core r1c1 m8 n27
      %a_r1c1_m8_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m8_n27)
      %b_r1c1_m8_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m8_n27)
      %c_r1c1_m8_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919232 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m8_n27)
      aiex.dma_await_task(%c_r1c1_m8_n27)
      aiex.dma_free_task(%a_r1c1_m8_n27)
      aiex.dma_free_task(%b_r1c1_m8_n27)
      aiex.dma_free_task(%c_r1c1_m8_n27)
      // core r1c1 m9 n14
      %a_r1c1_m9_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n14)
      %b_r1c1_m9_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n14)
      %c_r1c1_m9_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n14)
      aiex.dma_await_task(%c_r1c1_m9_n14)
      aiex.dma_free_task(%a_r1c1_m9_n14)
      aiex.dma_free_task(%b_r1c1_m9_n14)
      aiex.dma_free_task(%c_r1c1_m9_n14)
      // core r1c1 m9 n15
      %a_r1c1_m9_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n15)
      %b_r1c1_m9_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n15)
      %c_r1c1_m9_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n15)
      aiex.dma_await_task(%c_r1c1_m9_n15)
      aiex.dma_free_task(%a_r1c1_m9_n15)
      aiex.dma_free_task(%b_r1c1_m9_n15)
      aiex.dma_free_task(%c_r1c1_m9_n15)
      // core r1c1 m9 n16
      %a_r1c1_m9_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n16)
      %b_r1c1_m9_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n16)
      %c_r1c1_m9_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033216 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n16)
      aiex.dma_await_task(%c_r1c1_m9_n16)
      aiex.dma_free_task(%a_r1c1_m9_n16)
      aiex.dma_free_task(%b_r1c1_m9_n16)
      aiex.dma_free_task(%c_r1c1_m9_n16)
      // core r1c1 m9 n17
      %a_r1c1_m9_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n17)
      %b_r1c1_m9_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n17)
      %c_r1c1_m9_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033280 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n17)
      aiex.dma_await_task(%c_r1c1_m9_n17)
      aiex.dma_free_task(%a_r1c1_m9_n17)
      aiex.dma_free_task(%b_r1c1_m9_n17)
      aiex.dma_free_task(%c_r1c1_m9_n17)
      // core r1c1 m9 n18
      %a_r1c1_m9_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n18)
      %b_r1c1_m9_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n18)
      %c_r1c1_m9_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033344 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n18)
      aiex.dma_await_task(%c_r1c1_m9_n18)
      aiex.dma_free_task(%a_r1c1_m9_n18)
      aiex.dma_free_task(%b_r1c1_m9_n18)
      aiex.dma_free_task(%c_r1c1_m9_n18)
      // core r1c1 m9 n19
      %a_r1c1_m9_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n19)
      %b_r1c1_m9_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n19)
      %c_r1c1_m9_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033408 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n19)
      aiex.dma_await_task(%c_r1c1_m9_n19)
      aiex.dma_free_task(%a_r1c1_m9_n19)
      aiex.dma_free_task(%b_r1c1_m9_n19)
      aiex.dma_free_task(%c_r1c1_m9_n19)
      // core r1c1 m9 n20
      %a_r1c1_m9_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n20)
      %b_r1c1_m9_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n20)
      %c_r1c1_m9_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033472 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n20)
      aiex.dma_await_task(%c_r1c1_m9_n20)
      aiex.dma_free_task(%a_r1c1_m9_n20)
      aiex.dma_free_task(%b_r1c1_m9_n20)
      aiex.dma_free_task(%c_r1c1_m9_n20)
      // core r1c1 m9 n21
      %a_r1c1_m9_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n21)
      %b_r1c1_m9_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n21)
      %c_r1c1_m9_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033536 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n21)
      aiex.dma_await_task(%c_r1c1_m9_n21)
      aiex.dma_free_task(%a_r1c1_m9_n21)
      aiex.dma_free_task(%b_r1c1_m9_n21)
      aiex.dma_free_task(%c_r1c1_m9_n21)
      // core r1c1 m9 n22
      %a_r1c1_m9_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n22)
      %b_r1c1_m9_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n22)
      %c_r1c1_m9_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033600 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n22)
      aiex.dma_await_task(%c_r1c1_m9_n22)
      aiex.dma_free_task(%a_r1c1_m9_n22)
      aiex.dma_free_task(%b_r1c1_m9_n22)
      aiex.dma_free_task(%c_r1c1_m9_n22)
      // core r1c1 m9 n23
      %a_r1c1_m9_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n23)
      %b_r1c1_m9_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n23)
      %c_r1c1_m9_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033664 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n23)
      aiex.dma_await_task(%c_r1c1_m9_n23)
      aiex.dma_free_task(%a_r1c1_m9_n23)
      aiex.dma_free_task(%b_r1c1_m9_n23)
      aiex.dma_free_task(%c_r1c1_m9_n23)
      // core r1c1 m9 n24
      %a_r1c1_m9_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n24)
      %b_r1c1_m9_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n24)
      %c_r1c1_m9_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033728 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n24)
      aiex.dma_await_task(%c_r1c1_m9_n24)
      aiex.dma_free_task(%a_r1c1_m9_n24)
      aiex.dma_free_task(%b_r1c1_m9_n24)
      aiex.dma_free_task(%c_r1c1_m9_n24)
      // core r1c1 m9 n25
      %a_r1c1_m9_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n25)
      %b_r1c1_m9_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n25)
      %c_r1c1_m9_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033792 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n25)
      aiex.dma_await_task(%c_r1c1_m9_n25)
      aiex.dma_free_task(%a_r1c1_m9_n25)
      aiex.dma_free_task(%b_r1c1_m9_n25)
      aiex.dma_free_task(%c_r1c1_m9_n25)
      // core r1c1 m9 n26
      %a_r1c1_m9_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n26)
      %b_r1c1_m9_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n26)
      %c_r1c1_m9_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033856 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n26)
      aiex.dma_await_task(%c_r1c1_m9_n26)
      aiex.dma_free_task(%a_r1c1_m9_n26)
      aiex.dma_free_task(%b_r1c1_m9_n26)
      aiex.dma_free_task(%c_r1c1_m9_n26)
      // core r1c1 m9 n27
      %a_r1c1_m9_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m9_n27)
      %b_r1c1_m9_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m9_n27)
      %c_r1c1_m9_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033920 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m9_n27)
      aiex.dma_await_task(%c_r1c1_m9_n27)
      aiex.dma_free_task(%a_r1c1_m9_n27)
      aiex.dma_free_task(%b_r1c1_m9_n27)
      aiex.dma_free_task(%c_r1c1_m9_n27)
      // core r1c1 m10 n14
      %a_r1c1_m10_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n14)
      %b_r1c1_m10_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n14)
      %c_r1c1_m10_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n14)
      aiex.dma_await_task(%c_r1c1_m10_n14)
      aiex.dma_free_task(%a_r1c1_m10_n14)
      aiex.dma_free_task(%b_r1c1_m10_n14)
      aiex.dma_free_task(%c_r1c1_m10_n14)
      // core r1c1 m10 n15
      %a_r1c1_m10_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n15)
      %b_r1c1_m10_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n15)
      %c_r1c1_m10_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147840 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n15)
      aiex.dma_await_task(%c_r1c1_m10_n15)
      aiex.dma_free_task(%a_r1c1_m10_n15)
      aiex.dma_free_task(%b_r1c1_m10_n15)
      aiex.dma_free_task(%c_r1c1_m10_n15)
      // core r1c1 m10 n16
      %a_r1c1_m10_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n16)
      %b_r1c1_m10_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n16)
      %c_r1c1_m10_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147904 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n16)
      aiex.dma_await_task(%c_r1c1_m10_n16)
      aiex.dma_free_task(%a_r1c1_m10_n16)
      aiex.dma_free_task(%b_r1c1_m10_n16)
      aiex.dma_free_task(%c_r1c1_m10_n16)
      // core r1c1 m10 n17
      %a_r1c1_m10_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n17)
      %b_r1c1_m10_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n17)
      %c_r1c1_m10_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1147968 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n17)
      aiex.dma_await_task(%c_r1c1_m10_n17)
      aiex.dma_free_task(%a_r1c1_m10_n17)
      aiex.dma_free_task(%b_r1c1_m10_n17)
      aiex.dma_free_task(%c_r1c1_m10_n17)
      // core r1c1 m10 n18
      %a_r1c1_m10_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n18)
      %b_r1c1_m10_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n18)
      %c_r1c1_m10_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148032 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n18)
      aiex.dma_await_task(%c_r1c1_m10_n18)
      aiex.dma_free_task(%a_r1c1_m10_n18)
      aiex.dma_free_task(%b_r1c1_m10_n18)
      aiex.dma_free_task(%c_r1c1_m10_n18)
      // core r1c1 m10 n19
      %a_r1c1_m10_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n19)
      %b_r1c1_m10_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n19)
      %c_r1c1_m10_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148096 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n19)
      aiex.dma_await_task(%c_r1c1_m10_n19)
      aiex.dma_free_task(%a_r1c1_m10_n19)
      aiex.dma_free_task(%b_r1c1_m10_n19)
      aiex.dma_free_task(%c_r1c1_m10_n19)
      // core r1c1 m10 n20
      %a_r1c1_m10_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n20)
      %b_r1c1_m10_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n20)
      %c_r1c1_m10_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148160 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n20)
      aiex.dma_await_task(%c_r1c1_m10_n20)
      aiex.dma_free_task(%a_r1c1_m10_n20)
      aiex.dma_free_task(%b_r1c1_m10_n20)
      aiex.dma_free_task(%c_r1c1_m10_n20)
      // core r1c1 m10 n21
      %a_r1c1_m10_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n21)
      %b_r1c1_m10_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n21)
      %c_r1c1_m10_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148224 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n21)
      aiex.dma_await_task(%c_r1c1_m10_n21)
      aiex.dma_free_task(%a_r1c1_m10_n21)
      aiex.dma_free_task(%b_r1c1_m10_n21)
      aiex.dma_free_task(%c_r1c1_m10_n21)
      // core r1c1 m10 n22
      %a_r1c1_m10_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n22)
      %b_r1c1_m10_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n22)
      %c_r1c1_m10_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148288 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n22)
      aiex.dma_await_task(%c_r1c1_m10_n22)
      aiex.dma_free_task(%a_r1c1_m10_n22)
      aiex.dma_free_task(%b_r1c1_m10_n22)
      aiex.dma_free_task(%c_r1c1_m10_n22)
      // core r1c1 m10 n23
      %a_r1c1_m10_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n23)
      %b_r1c1_m10_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n23)
      %c_r1c1_m10_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148352 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n23)
      aiex.dma_await_task(%c_r1c1_m10_n23)
      aiex.dma_free_task(%a_r1c1_m10_n23)
      aiex.dma_free_task(%b_r1c1_m10_n23)
      aiex.dma_free_task(%c_r1c1_m10_n23)
      // core r1c1 m10 n24
      %a_r1c1_m10_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n24)
      %b_r1c1_m10_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n24)
      %c_r1c1_m10_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148416 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n24)
      aiex.dma_await_task(%c_r1c1_m10_n24)
      aiex.dma_free_task(%a_r1c1_m10_n24)
      aiex.dma_free_task(%b_r1c1_m10_n24)
      aiex.dma_free_task(%c_r1c1_m10_n24)
      // core r1c1 m10 n25
      %a_r1c1_m10_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n25)
      %b_r1c1_m10_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n25)
      %c_r1c1_m10_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148480 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n25)
      aiex.dma_await_task(%c_r1c1_m10_n25)
      aiex.dma_free_task(%a_r1c1_m10_n25)
      aiex.dma_free_task(%b_r1c1_m10_n25)
      aiex.dma_free_task(%c_r1c1_m10_n25)
      // core r1c1 m10 n26
      %a_r1c1_m10_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n26)
      %b_r1c1_m10_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n26)
      %c_r1c1_m10_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148544 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n26)
      aiex.dma_await_task(%c_r1c1_m10_n26)
      aiex.dma_free_task(%a_r1c1_m10_n26)
      aiex.dma_free_task(%b_r1c1_m10_n26)
      aiex.dma_free_task(%c_r1c1_m10_n26)
      // core r1c1 m10 n27
      %a_r1c1_m10_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m10_n27)
      %b_r1c1_m10_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m10_n27)
      %c_r1c1_m10_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148608 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m10_n27)
      aiex.dma_await_task(%c_r1c1_m10_n27)
      aiex.dma_free_task(%a_r1c1_m10_n27)
      aiex.dma_free_task(%b_r1c1_m10_n27)
      aiex.dma_free_task(%c_r1c1_m10_n27)
      // core r1c1 m11 n14
      %a_r1c1_m11_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n14)
      %b_r1c1_m11_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n14)
      %c_r1c1_m11_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262464 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n14)
      aiex.dma_await_task(%c_r1c1_m11_n14)
      aiex.dma_free_task(%a_r1c1_m11_n14)
      aiex.dma_free_task(%b_r1c1_m11_n14)
      aiex.dma_free_task(%c_r1c1_m11_n14)
      // core r1c1 m11 n15
      %a_r1c1_m11_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n15)
      %b_r1c1_m11_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n15)
      %c_r1c1_m11_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262528 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n15)
      aiex.dma_await_task(%c_r1c1_m11_n15)
      aiex.dma_free_task(%a_r1c1_m11_n15)
      aiex.dma_free_task(%b_r1c1_m11_n15)
      aiex.dma_free_task(%c_r1c1_m11_n15)
      // core r1c1 m11 n16
      %a_r1c1_m11_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n16)
      %b_r1c1_m11_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n16)
      %c_r1c1_m11_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262592 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n16)
      aiex.dma_await_task(%c_r1c1_m11_n16)
      aiex.dma_free_task(%a_r1c1_m11_n16)
      aiex.dma_free_task(%b_r1c1_m11_n16)
      aiex.dma_free_task(%c_r1c1_m11_n16)
      // core r1c1 m11 n17
      %a_r1c1_m11_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n17)
      %b_r1c1_m11_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n17)
      %c_r1c1_m11_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262656 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n17)
      aiex.dma_await_task(%c_r1c1_m11_n17)
      aiex.dma_free_task(%a_r1c1_m11_n17)
      aiex.dma_free_task(%b_r1c1_m11_n17)
      aiex.dma_free_task(%c_r1c1_m11_n17)
      // core r1c1 m11 n18
      %a_r1c1_m11_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n18)
      %b_r1c1_m11_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n18)
      %c_r1c1_m11_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262720 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n18)
      aiex.dma_await_task(%c_r1c1_m11_n18)
      aiex.dma_free_task(%a_r1c1_m11_n18)
      aiex.dma_free_task(%b_r1c1_m11_n18)
      aiex.dma_free_task(%c_r1c1_m11_n18)
      // core r1c1 m11 n19
      %a_r1c1_m11_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n19)
      %b_r1c1_m11_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n19)
      %c_r1c1_m11_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262784 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n19)
      aiex.dma_await_task(%c_r1c1_m11_n19)
      aiex.dma_free_task(%a_r1c1_m11_n19)
      aiex.dma_free_task(%b_r1c1_m11_n19)
      aiex.dma_free_task(%c_r1c1_m11_n19)
      // core r1c1 m11 n20
      %a_r1c1_m11_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n20)
      %b_r1c1_m11_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n20)
      %c_r1c1_m11_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262848 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n20)
      aiex.dma_await_task(%c_r1c1_m11_n20)
      aiex.dma_free_task(%a_r1c1_m11_n20)
      aiex.dma_free_task(%b_r1c1_m11_n20)
      aiex.dma_free_task(%c_r1c1_m11_n20)
      // core r1c1 m11 n21
      %a_r1c1_m11_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n21)
      %b_r1c1_m11_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n21)
      %c_r1c1_m11_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262912 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n21)
      aiex.dma_await_task(%c_r1c1_m11_n21)
      aiex.dma_free_task(%a_r1c1_m11_n21)
      aiex.dma_free_task(%b_r1c1_m11_n21)
      aiex.dma_free_task(%c_r1c1_m11_n21)
      // core r1c1 m11 n22
      %a_r1c1_m11_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n22)
      %b_r1c1_m11_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n22)
      %c_r1c1_m11_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1262976 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n22)
      aiex.dma_await_task(%c_r1c1_m11_n22)
      aiex.dma_free_task(%a_r1c1_m11_n22)
      aiex.dma_free_task(%b_r1c1_m11_n22)
      aiex.dma_free_task(%c_r1c1_m11_n22)
      // core r1c1 m11 n23
      %a_r1c1_m11_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n23)
      %b_r1c1_m11_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n23)
      %c_r1c1_m11_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263040 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n23)
      aiex.dma_await_task(%c_r1c1_m11_n23)
      aiex.dma_free_task(%a_r1c1_m11_n23)
      aiex.dma_free_task(%b_r1c1_m11_n23)
      aiex.dma_free_task(%c_r1c1_m11_n23)
      // core r1c1 m11 n24
      %a_r1c1_m11_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n24)
      %b_r1c1_m11_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n24)
      %c_r1c1_m11_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263104 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n24)
      aiex.dma_await_task(%c_r1c1_m11_n24)
      aiex.dma_free_task(%a_r1c1_m11_n24)
      aiex.dma_free_task(%b_r1c1_m11_n24)
      aiex.dma_free_task(%c_r1c1_m11_n24)
      // core r1c1 m11 n25
      %a_r1c1_m11_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n25)
      %b_r1c1_m11_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n25)
      %c_r1c1_m11_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263168 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n25)
      aiex.dma_await_task(%c_r1c1_m11_n25)
      aiex.dma_free_task(%a_r1c1_m11_n25)
      aiex.dma_free_task(%b_r1c1_m11_n25)
      aiex.dma_free_task(%c_r1c1_m11_n25)
      // core r1c1 m11 n26
      %a_r1c1_m11_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n26)
      %b_r1c1_m11_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n26)
      %c_r1c1_m11_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263232 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n26)
      aiex.dma_await_task(%c_r1c1_m11_n26)
      aiex.dma_free_task(%a_r1c1_m11_n26)
      aiex.dma_free_task(%b_r1c1_m11_n26)
      aiex.dma_free_task(%c_r1c1_m11_n26)
      // core r1c1 m11 n27
      %a_r1c1_m11_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m11_n27)
      %b_r1c1_m11_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m11_n27)
      %c_r1c1_m11_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263296 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m11_n27)
      aiex.dma_await_task(%c_r1c1_m11_n27)
      aiex.dma_free_task(%a_r1c1_m11_n27)
      aiex.dma_free_task(%b_r1c1_m11_n27)
      aiex.dma_free_task(%c_r1c1_m11_n27)
      // core r1c1 m12 n14
      %a_r1c1_m12_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n14)
      %b_r1c1_m12_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n14)
      %c_r1c1_m12_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n14)
      aiex.dma_await_task(%c_r1c1_m12_n14)
      aiex.dma_free_task(%a_r1c1_m12_n14)
      aiex.dma_free_task(%b_r1c1_m12_n14)
      aiex.dma_free_task(%c_r1c1_m12_n14)
      // core r1c1 m12 n15
      %a_r1c1_m12_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n15)
      %b_r1c1_m12_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n15)
      %c_r1c1_m12_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377216 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n15)
      aiex.dma_await_task(%c_r1c1_m12_n15)
      aiex.dma_free_task(%a_r1c1_m12_n15)
      aiex.dma_free_task(%b_r1c1_m12_n15)
      aiex.dma_free_task(%c_r1c1_m12_n15)
      // core r1c1 m12 n16
      %a_r1c1_m12_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n16)
      %b_r1c1_m12_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n16)
      %c_r1c1_m12_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377280 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n16)
      aiex.dma_await_task(%c_r1c1_m12_n16)
      aiex.dma_free_task(%a_r1c1_m12_n16)
      aiex.dma_free_task(%b_r1c1_m12_n16)
      aiex.dma_free_task(%c_r1c1_m12_n16)
      // core r1c1 m12 n17
      %a_r1c1_m12_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n17)
      %b_r1c1_m12_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n17)
      %c_r1c1_m12_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377344 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n17)
      aiex.dma_await_task(%c_r1c1_m12_n17)
      aiex.dma_free_task(%a_r1c1_m12_n17)
      aiex.dma_free_task(%b_r1c1_m12_n17)
      aiex.dma_free_task(%c_r1c1_m12_n17)
      // core r1c1 m12 n18
      %a_r1c1_m12_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n18)
      %b_r1c1_m12_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n18)
      %c_r1c1_m12_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377408 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n18)
      aiex.dma_await_task(%c_r1c1_m12_n18)
      aiex.dma_free_task(%a_r1c1_m12_n18)
      aiex.dma_free_task(%b_r1c1_m12_n18)
      aiex.dma_free_task(%c_r1c1_m12_n18)
      // core r1c1 m12 n19
      %a_r1c1_m12_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n19)
      %b_r1c1_m12_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n19)
      %c_r1c1_m12_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377472 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n19)
      aiex.dma_await_task(%c_r1c1_m12_n19)
      aiex.dma_free_task(%a_r1c1_m12_n19)
      aiex.dma_free_task(%b_r1c1_m12_n19)
      aiex.dma_free_task(%c_r1c1_m12_n19)
      // core r1c1 m12 n20
      %a_r1c1_m12_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n20)
      %b_r1c1_m12_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n20)
      %c_r1c1_m12_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377536 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n20)
      aiex.dma_await_task(%c_r1c1_m12_n20)
      aiex.dma_free_task(%a_r1c1_m12_n20)
      aiex.dma_free_task(%b_r1c1_m12_n20)
      aiex.dma_free_task(%c_r1c1_m12_n20)
      // core r1c1 m12 n21
      %a_r1c1_m12_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n21)
      %b_r1c1_m12_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n21)
      %c_r1c1_m12_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377600 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n21)
      aiex.dma_await_task(%c_r1c1_m12_n21)
      aiex.dma_free_task(%a_r1c1_m12_n21)
      aiex.dma_free_task(%b_r1c1_m12_n21)
      aiex.dma_free_task(%c_r1c1_m12_n21)
      // core r1c1 m12 n22
      %a_r1c1_m12_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n22)
      %b_r1c1_m12_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n22)
      %c_r1c1_m12_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377664 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n22)
      aiex.dma_await_task(%c_r1c1_m12_n22)
      aiex.dma_free_task(%a_r1c1_m12_n22)
      aiex.dma_free_task(%b_r1c1_m12_n22)
      aiex.dma_free_task(%c_r1c1_m12_n22)
      // core r1c1 m12 n23
      %a_r1c1_m12_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n23)
      %b_r1c1_m12_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n23)
      %c_r1c1_m12_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377728 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n23)
      aiex.dma_await_task(%c_r1c1_m12_n23)
      aiex.dma_free_task(%a_r1c1_m12_n23)
      aiex.dma_free_task(%b_r1c1_m12_n23)
      aiex.dma_free_task(%c_r1c1_m12_n23)
      // core r1c1 m12 n24
      %a_r1c1_m12_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n24)
      %b_r1c1_m12_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n24)
      %c_r1c1_m12_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377792 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n24)
      aiex.dma_await_task(%c_r1c1_m12_n24)
      aiex.dma_free_task(%a_r1c1_m12_n24)
      aiex.dma_free_task(%b_r1c1_m12_n24)
      aiex.dma_free_task(%c_r1c1_m12_n24)
      // core r1c1 m12 n25
      %a_r1c1_m12_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n25)
      %b_r1c1_m12_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n25)
      %c_r1c1_m12_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377856 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n25)
      aiex.dma_await_task(%c_r1c1_m12_n25)
      aiex.dma_free_task(%a_r1c1_m12_n25)
      aiex.dma_free_task(%b_r1c1_m12_n25)
      aiex.dma_free_task(%c_r1c1_m12_n25)
      // core r1c1 m12 n26
      %a_r1c1_m12_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n26)
      %b_r1c1_m12_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n26)
      %c_r1c1_m12_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377920 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n26)
      aiex.dma_await_task(%c_r1c1_m12_n26)
      aiex.dma_free_task(%a_r1c1_m12_n26)
      aiex.dma_free_task(%b_r1c1_m12_n26)
      aiex.dma_free_task(%c_r1c1_m12_n26)
      // core r1c1 m12 n27
      %a_r1c1_m12_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m12_n27)
      %b_r1c1_m12_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m12_n27)
      %c_r1c1_m12_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1377984 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m12_n27)
      aiex.dma_await_task(%c_r1c1_m12_n27)
      aiex.dma_free_task(%a_r1c1_m12_n27)
      aiex.dma_free_task(%b_r1c1_m12_n27)
      aiex.dma_free_task(%c_r1c1_m12_n27)
      // core r1c1 m13 n14
      %a_r1c1_m13_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n14)
      %b_r1c1_m13_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n14)
      %c_r1c1_m13_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491840 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n14)
      aiex.dma_await_task(%c_r1c1_m13_n14)
      aiex.dma_free_task(%a_r1c1_m13_n14)
      aiex.dma_free_task(%b_r1c1_m13_n14)
      aiex.dma_free_task(%c_r1c1_m13_n14)
      // core r1c1 m13 n15
      %a_r1c1_m13_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n15)
      %b_r1c1_m13_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n15)
      %c_r1c1_m13_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491904 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n15)
      aiex.dma_await_task(%c_r1c1_m13_n15)
      aiex.dma_free_task(%a_r1c1_m13_n15)
      aiex.dma_free_task(%b_r1c1_m13_n15)
      aiex.dma_free_task(%c_r1c1_m13_n15)
      // core r1c1 m13 n16
      %a_r1c1_m13_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n16)
      %b_r1c1_m13_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n16)
      %c_r1c1_m13_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1491968 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n16)
      aiex.dma_await_task(%c_r1c1_m13_n16)
      aiex.dma_free_task(%a_r1c1_m13_n16)
      aiex.dma_free_task(%b_r1c1_m13_n16)
      aiex.dma_free_task(%c_r1c1_m13_n16)
      // core r1c1 m13 n17
      %a_r1c1_m13_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n17)
      %b_r1c1_m13_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n17)
      %c_r1c1_m13_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492032 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n17)
      aiex.dma_await_task(%c_r1c1_m13_n17)
      aiex.dma_free_task(%a_r1c1_m13_n17)
      aiex.dma_free_task(%b_r1c1_m13_n17)
      aiex.dma_free_task(%c_r1c1_m13_n17)
      // core r1c1 m13 n18
      %a_r1c1_m13_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n18)
      %b_r1c1_m13_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n18)
      %c_r1c1_m13_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492096 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n18)
      aiex.dma_await_task(%c_r1c1_m13_n18)
      aiex.dma_free_task(%a_r1c1_m13_n18)
      aiex.dma_free_task(%b_r1c1_m13_n18)
      aiex.dma_free_task(%c_r1c1_m13_n18)
      // core r1c1 m13 n19
      %a_r1c1_m13_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n19)
      %b_r1c1_m13_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n19)
      %c_r1c1_m13_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492160 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n19)
      aiex.dma_await_task(%c_r1c1_m13_n19)
      aiex.dma_free_task(%a_r1c1_m13_n19)
      aiex.dma_free_task(%b_r1c1_m13_n19)
      aiex.dma_free_task(%c_r1c1_m13_n19)
      // core r1c1 m13 n20
      %a_r1c1_m13_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n20)
      %b_r1c1_m13_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n20)
      %c_r1c1_m13_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492224 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n20)
      aiex.dma_await_task(%c_r1c1_m13_n20)
      aiex.dma_free_task(%a_r1c1_m13_n20)
      aiex.dma_free_task(%b_r1c1_m13_n20)
      aiex.dma_free_task(%c_r1c1_m13_n20)
      // core r1c1 m13 n21
      %a_r1c1_m13_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n21)
      %b_r1c1_m13_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n21)
      %c_r1c1_m13_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492288 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n21)
      aiex.dma_await_task(%c_r1c1_m13_n21)
      aiex.dma_free_task(%a_r1c1_m13_n21)
      aiex.dma_free_task(%b_r1c1_m13_n21)
      aiex.dma_free_task(%c_r1c1_m13_n21)
      // core r1c1 m13 n22
      %a_r1c1_m13_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n22)
      %b_r1c1_m13_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n22)
      %c_r1c1_m13_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492352 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n22)
      aiex.dma_await_task(%c_r1c1_m13_n22)
      aiex.dma_free_task(%a_r1c1_m13_n22)
      aiex.dma_free_task(%b_r1c1_m13_n22)
      aiex.dma_free_task(%c_r1c1_m13_n22)
      // core r1c1 m13 n23
      %a_r1c1_m13_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n23)
      %b_r1c1_m13_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n23)
      %c_r1c1_m13_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492416 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n23)
      aiex.dma_await_task(%c_r1c1_m13_n23)
      aiex.dma_free_task(%a_r1c1_m13_n23)
      aiex.dma_free_task(%b_r1c1_m13_n23)
      aiex.dma_free_task(%c_r1c1_m13_n23)
      // core r1c1 m13 n24
      %a_r1c1_m13_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n24)
      %b_r1c1_m13_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n24)
      %c_r1c1_m13_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492480 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n24)
      aiex.dma_await_task(%c_r1c1_m13_n24)
      aiex.dma_free_task(%a_r1c1_m13_n24)
      aiex.dma_free_task(%b_r1c1_m13_n24)
      aiex.dma_free_task(%c_r1c1_m13_n24)
      // core r1c1 m13 n25
      %a_r1c1_m13_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n25)
      %b_r1c1_m13_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n25)
      %c_r1c1_m13_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492544 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n25)
      aiex.dma_await_task(%c_r1c1_m13_n25)
      aiex.dma_free_task(%a_r1c1_m13_n25)
      aiex.dma_free_task(%b_r1c1_m13_n25)
      aiex.dma_free_task(%c_r1c1_m13_n25)
      // core r1c1 m13 n26
      %a_r1c1_m13_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n26)
      %b_r1c1_m13_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n26)
      %c_r1c1_m13_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492608 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n26)
      aiex.dma_await_task(%c_r1c1_m13_n26)
      aiex.dma_free_task(%a_r1c1_m13_n26)
      aiex.dma_free_task(%b_r1c1_m13_n26)
      aiex.dma_free_task(%c_r1c1_m13_n26)
      // core r1c1 m13 n27
      %a_r1c1_m13_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m13_n27)
      %b_r1c1_m13_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m13_n27)
      %c_r1c1_m13_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492672 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m13_n27)
      aiex.dma_await_task(%c_r1c1_m13_n27)
      aiex.dma_free_task(%a_r1c1_m13_n27)
      aiex.dma_free_task(%b_r1c1_m13_n27)
      aiex.dma_free_task(%c_r1c1_m13_n27)
      // core r1c1 m14 n14
      %a_r1c1_m14_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n14)
      %b_r1c1_m14_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n14)
      %c_r1c1_m14_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606528 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n14)
      aiex.dma_await_task(%c_r1c1_m14_n14)
      aiex.dma_free_task(%a_r1c1_m14_n14)
      aiex.dma_free_task(%b_r1c1_m14_n14)
      aiex.dma_free_task(%c_r1c1_m14_n14)
      // core r1c1 m14 n15
      %a_r1c1_m14_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n15)
      %b_r1c1_m14_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n15)
      %c_r1c1_m14_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606592 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n15)
      aiex.dma_await_task(%c_r1c1_m14_n15)
      aiex.dma_free_task(%a_r1c1_m14_n15)
      aiex.dma_free_task(%b_r1c1_m14_n15)
      aiex.dma_free_task(%c_r1c1_m14_n15)
      // core r1c1 m14 n16
      %a_r1c1_m14_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n16)
      %b_r1c1_m14_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n16)
      %c_r1c1_m14_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606656 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n16)
      aiex.dma_await_task(%c_r1c1_m14_n16)
      aiex.dma_free_task(%a_r1c1_m14_n16)
      aiex.dma_free_task(%b_r1c1_m14_n16)
      aiex.dma_free_task(%c_r1c1_m14_n16)
      // core r1c1 m14 n17
      %a_r1c1_m14_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n17)
      %b_r1c1_m14_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n17)
      %c_r1c1_m14_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606720 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n17)
      aiex.dma_await_task(%c_r1c1_m14_n17)
      aiex.dma_free_task(%a_r1c1_m14_n17)
      aiex.dma_free_task(%b_r1c1_m14_n17)
      aiex.dma_free_task(%c_r1c1_m14_n17)
      // core r1c1 m14 n18
      %a_r1c1_m14_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n18)
      %b_r1c1_m14_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n18)
      %c_r1c1_m14_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606784 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n18)
      aiex.dma_await_task(%c_r1c1_m14_n18)
      aiex.dma_free_task(%a_r1c1_m14_n18)
      aiex.dma_free_task(%b_r1c1_m14_n18)
      aiex.dma_free_task(%c_r1c1_m14_n18)
      // core r1c1 m14 n19
      %a_r1c1_m14_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n19)
      %b_r1c1_m14_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n19)
      %c_r1c1_m14_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606848 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n19)
      aiex.dma_await_task(%c_r1c1_m14_n19)
      aiex.dma_free_task(%a_r1c1_m14_n19)
      aiex.dma_free_task(%b_r1c1_m14_n19)
      aiex.dma_free_task(%c_r1c1_m14_n19)
      // core r1c1 m14 n20
      %a_r1c1_m14_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n20)
      %b_r1c1_m14_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n20)
      %c_r1c1_m14_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606912 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n20)
      aiex.dma_await_task(%c_r1c1_m14_n20)
      aiex.dma_free_task(%a_r1c1_m14_n20)
      aiex.dma_free_task(%b_r1c1_m14_n20)
      aiex.dma_free_task(%c_r1c1_m14_n20)
      // core r1c1 m14 n21
      %a_r1c1_m14_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n21)
      %b_r1c1_m14_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n21)
      %c_r1c1_m14_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1606976 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n21)
      aiex.dma_await_task(%c_r1c1_m14_n21)
      aiex.dma_free_task(%a_r1c1_m14_n21)
      aiex.dma_free_task(%b_r1c1_m14_n21)
      aiex.dma_free_task(%c_r1c1_m14_n21)
      // core r1c1 m14 n22
      %a_r1c1_m14_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n22)
      %b_r1c1_m14_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n22)
      %c_r1c1_m14_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607040 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n22)
      aiex.dma_await_task(%c_r1c1_m14_n22)
      aiex.dma_free_task(%a_r1c1_m14_n22)
      aiex.dma_free_task(%b_r1c1_m14_n22)
      aiex.dma_free_task(%c_r1c1_m14_n22)
      // core r1c1 m14 n23
      %a_r1c1_m14_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n23)
      %b_r1c1_m14_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n23)
      %c_r1c1_m14_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607104 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n23)
      aiex.dma_await_task(%c_r1c1_m14_n23)
      aiex.dma_free_task(%a_r1c1_m14_n23)
      aiex.dma_free_task(%b_r1c1_m14_n23)
      aiex.dma_free_task(%c_r1c1_m14_n23)
      // core r1c1 m14 n24
      %a_r1c1_m14_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n24)
      %b_r1c1_m14_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n24)
      %c_r1c1_m14_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607168 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n24)
      aiex.dma_await_task(%c_r1c1_m14_n24)
      aiex.dma_free_task(%a_r1c1_m14_n24)
      aiex.dma_free_task(%b_r1c1_m14_n24)
      aiex.dma_free_task(%c_r1c1_m14_n24)
      // core r1c1 m14 n25
      %a_r1c1_m14_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n25)
      %b_r1c1_m14_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n25)
      %c_r1c1_m14_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607232 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n25)
      aiex.dma_await_task(%c_r1c1_m14_n25)
      aiex.dma_free_task(%a_r1c1_m14_n25)
      aiex.dma_free_task(%b_r1c1_m14_n25)
      aiex.dma_free_task(%c_r1c1_m14_n25)
      // core r1c1 m14 n26
      %a_r1c1_m14_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n26)
      %b_r1c1_m14_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n26)
      %c_r1c1_m14_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607296 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n26)
      aiex.dma_await_task(%c_r1c1_m14_n26)
      aiex.dma_free_task(%a_r1c1_m14_n26)
      aiex.dma_free_task(%b_r1c1_m14_n26)
      aiex.dma_free_task(%c_r1c1_m14_n26)
      // core r1c1 m14 n27
      %a_r1c1_m14_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m14_n27)
      %b_r1c1_m14_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m14_n27)
      %c_r1c1_m14_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607360 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m14_n27)
      aiex.dma_await_task(%c_r1c1_m14_n27)
      aiex.dma_free_task(%a_r1c1_m14_n27)
      aiex.dma_free_task(%b_r1c1_m14_n27)
      aiex.dma_free_task(%c_r1c1_m14_n27)
      // core r1c1 m15 n14
      %a_r1c1_m15_n14 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n14)
      %b_r1c1_m15_n14 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 896 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n14)
      %c_r1c1_m15_n14 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721216 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n14)
      aiex.dma_await_task(%c_r1c1_m15_n14)
      aiex.dma_free_task(%a_r1c1_m15_n14)
      aiex.dma_free_task(%b_r1c1_m15_n14)
      aiex.dma_free_task(%c_r1c1_m15_n14)
      // core r1c1 m15 n15
      %a_r1c1_m15_n15 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n15)
      %b_r1c1_m15_n15 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 960 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n15)
      %c_r1c1_m15_n15 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721280 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n15)
      aiex.dma_await_task(%c_r1c1_m15_n15)
      aiex.dma_free_task(%a_r1c1_m15_n15)
      aiex.dma_free_task(%b_r1c1_m15_n15)
      aiex.dma_free_task(%c_r1c1_m15_n15)
      // core r1c1 m15 n16
      %a_r1c1_m15_n16 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n16)
      %b_r1c1_m15_n16 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1024 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n16)
      %c_r1c1_m15_n16 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721344 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n16)
      aiex.dma_await_task(%c_r1c1_m15_n16)
      aiex.dma_free_task(%a_r1c1_m15_n16)
      aiex.dma_free_task(%b_r1c1_m15_n16)
      aiex.dma_free_task(%c_r1c1_m15_n16)
      // core r1c1 m15 n17
      %a_r1c1_m15_n17 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n17)
      %b_r1c1_m15_n17 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1088 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n17)
      %c_r1c1_m15_n17 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721408 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n17)
      aiex.dma_await_task(%c_r1c1_m15_n17)
      aiex.dma_free_task(%a_r1c1_m15_n17)
      aiex.dma_free_task(%b_r1c1_m15_n17)
      aiex.dma_free_task(%c_r1c1_m15_n17)
      // core r1c1 m15 n18
      %a_r1c1_m15_n18 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n18)
      %b_r1c1_m15_n18 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1152 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n18)
      %c_r1c1_m15_n18 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721472 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n18)
      aiex.dma_await_task(%c_r1c1_m15_n18)
      aiex.dma_free_task(%a_r1c1_m15_n18)
      aiex.dma_free_task(%b_r1c1_m15_n18)
      aiex.dma_free_task(%c_r1c1_m15_n18)
      // core r1c1 m15 n19
      %a_r1c1_m15_n19 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n19)
      %b_r1c1_m15_n19 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1216 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n19)
      %c_r1c1_m15_n19 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721536 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n19)
      aiex.dma_await_task(%c_r1c1_m15_n19)
      aiex.dma_free_task(%a_r1c1_m15_n19)
      aiex.dma_free_task(%b_r1c1_m15_n19)
      aiex.dma_free_task(%c_r1c1_m15_n19)
      // core r1c1 m15 n20
      %a_r1c1_m15_n20 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n20)
      %b_r1c1_m15_n20 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1280 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n20)
      %c_r1c1_m15_n20 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721600 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n20)
      aiex.dma_await_task(%c_r1c1_m15_n20)
      aiex.dma_free_task(%a_r1c1_m15_n20)
      aiex.dma_free_task(%b_r1c1_m15_n20)
      aiex.dma_free_task(%c_r1c1_m15_n20)
      // core r1c1 m15 n21
      %a_r1c1_m15_n21 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n21)
      %b_r1c1_m15_n21 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1344 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n21)
      %c_r1c1_m15_n21 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721664 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n21)
      aiex.dma_await_task(%c_r1c1_m15_n21)
      aiex.dma_free_task(%a_r1c1_m15_n21)
      aiex.dma_free_task(%b_r1c1_m15_n21)
      aiex.dma_free_task(%c_r1c1_m15_n21)
      // core r1c1 m15 n22
      %a_r1c1_m15_n22 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n22)
      %b_r1c1_m15_n22 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1408 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n22)
      %c_r1c1_m15_n22 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721728 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n22)
      aiex.dma_await_task(%c_r1c1_m15_n22)
      aiex.dma_free_task(%a_r1c1_m15_n22)
      aiex.dma_free_task(%b_r1c1_m15_n22)
      aiex.dma_free_task(%c_r1c1_m15_n22)
      // core r1c1 m15 n23
      %a_r1c1_m15_n23 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n23)
      %b_r1c1_m15_n23 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1472 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n23)
      %c_r1c1_m15_n23 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721792 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n23)
      aiex.dma_await_task(%c_r1c1_m15_n23)
      aiex.dma_free_task(%a_r1c1_m15_n23)
      aiex.dma_free_task(%b_r1c1_m15_n23)
      aiex.dma_free_task(%c_r1c1_m15_n23)
      // core r1c1 m15 n24
      %a_r1c1_m15_n24 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n24)
      %b_r1c1_m15_n24 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1536 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n24)
      %c_r1c1_m15_n24 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721856 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n24)
      aiex.dma_await_task(%c_r1c1_m15_n24)
      aiex.dma_free_task(%a_r1c1_m15_n24)
      aiex.dma_free_task(%b_r1c1_m15_n24)
      aiex.dma_free_task(%c_r1c1_m15_n24)
      // core r1c1 m15 n25
      %a_r1c1_m15_n25 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n25)
      %b_r1c1_m15_n25 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1600 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n25)
      %c_r1c1_m15_n25 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721920 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n25)
      aiex.dma_await_task(%c_r1c1_m15_n25)
      aiex.dma_free_task(%a_r1c1_m15_n25)
      aiex.dma_free_task(%b_r1c1_m15_n25)
      aiex.dma_free_task(%c_r1c1_m15_n25)
      // core r1c1 m15 n26
      %a_r1c1_m15_n26 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n26)
      %b_r1c1_m15_n26 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1664 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n26)
      %c_r1c1_m15_n26 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1721984 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n26)
      aiex.dma_await_task(%c_r1c1_m15_n26)
      aiex.dma_free_task(%a_r1c1_m15_n26)
      aiex.dma_free_task(%b_r1c1_m15_n26)
      aiex.dma_free_task(%c_r1c1_m15_n26)
      // core r1c1 m15 n27
      %a_r1c1_m15_n27 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c1_m15_n27)
      %b_r1c1_m15_n27 = aiex.dma_configure_task_for @B_c1 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1728 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c1_m15_n27)
      %c_r1c1_m15_n27 = aiex.dma_configure_task_for @C2_r1_c1 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722048 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c1_m15_n27)
      aiex.dma_await_task(%c_r1c1_m15_n27)
      aiex.dma_free_task(%a_r1c1_m15_n27)
      aiex.dma_free_task(%b_r1c1_m15_n27)
      aiex.dma_free_task(%c_r1c1_m15_n27)
      // core r1c2 m8 n28
      %a_r1c2_m8_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n28)
      %b_r1c2_m8_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n28)
      %c_r1c2_m8_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919296 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n28)
      aiex.dma_await_task(%c_r1c2_m8_n28)
      aiex.dma_free_task(%a_r1c2_m8_n28)
      aiex.dma_free_task(%b_r1c2_m8_n28)
      aiex.dma_free_task(%c_r1c2_m8_n28)
      // core r1c2 m8 n29
      %a_r1c2_m8_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n29)
      %b_r1c2_m8_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n29)
      %c_r1c2_m8_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919360 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n29)
      aiex.dma_await_task(%c_r1c2_m8_n29)
      aiex.dma_free_task(%a_r1c2_m8_n29)
      aiex.dma_free_task(%b_r1c2_m8_n29)
      aiex.dma_free_task(%c_r1c2_m8_n29)
      // core r1c2 m8 n30
      %a_r1c2_m8_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n30)
      %b_r1c2_m8_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n30)
      %c_r1c2_m8_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919424 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n30)
      aiex.dma_await_task(%c_r1c2_m8_n30)
      aiex.dma_free_task(%a_r1c2_m8_n30)
      aiex.dma_free_task(%b_r1c2_m8_n30)
      aiex.dma_free_task(%c_r1c2_m8_n30)
      // core r1c2 m8 n31
      %a_r1c2_m8_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n31)
      %b_r1c2_m8_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n31)
      %c_r1c2_m8_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919488 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n31)
      aiex.dma_await_task(%c_r1c2_m8_n31)
      aiex.dma_free_task(%a_r1c2_m8_n31)
      aiex.dma_free_task(%b_r1c2_m8_n31)
      aiex.dma_free_task(%c_r1c2_m8_n31)
      // core r1c2 m8 n32
      %a_r1c2_m8_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n32)
      %b_r1c2_m8_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n32)
      %c_r1c2_m8_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919552 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n32)
      aiex.dma_await_task(%c_r1c2_m8_n32)
      aiex.dma_free_task(%a_r1c2_m8_n32)
      aiex.dma_free_task(%b_r1c2_m8_n32)
      aiex.dma_free_task(%c_r1c2_m8_n32)
      // core r1c2 m8 n33
      %a_r1c2_m8_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n33)
      %b_r1c2_m8_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n33)
      %c_r1c2_m8_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919616 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n33)
      aiex.dma_await_task(%c_r1c2_m8_n33)
      aiex.dma_free_task(%a_r1c2_m8_n33)
      aiex.dma_free_task(%b_r1c2_m8_n33)
      aiex.dma_free_task(%c_r1c2_m8_n33)
      // core r1c2 m8 n34
      %a_r1c2_m8_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n34)
      %b_r1c2_m8_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n34)
      %c_r1c2_m8_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919680 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n34)
      aiex.dma_await_task(%c_r1c2_m8_n34)
      aiex.dma_free_task(%a_r1c2_m8_n34)
      aiex.dma_free_task(%b_r1c2_m8_n34)
      aiex.dma_free_task(%c_r1c2_m8_n34)
      // core r1c2 m8 n35
      %a_r1c2_m8_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n35)
      %b_r1c2_m8_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n35)
      %c_r1c2_m8_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919744 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n35)
      aiex.dma_await_task(%c_r1c2_m8_n35)
      aiex.dma_free_task(%a_r1c2_m8_n35)
      aiex.dma_free_task(%b_r1c2_m8_n35)
      aiex.dma_free_task(%c_r1c2_m8_n35)
      // core r1c2 m8 n36
      %a_r1c2_m8_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n36)
      %b_r1c2_m8_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n36)
      %c_r1c2_m8_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919808 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n36)
      aiex.dma_await_task(%c_r1c2_m8_n36)
      aiex.dma_free_task(%a_r1c2_m8_n36)
      aiex.dma_free_task(%b_r1c2_m8_n36)
      aiex.dma_free_task(%c_r1c2_m8_n36)
      // core r1c2 m8 n37
      %a_r1c2_m8_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n37)
      %b_r1c2_m8_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n37)
      %c_r1c2_m8_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919872 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n37)
      aiex.dma_await_task(%c_r1c2_m8_n37)
      aiex.dma_free_task(%a_r1c2_m8_n37)
      aiex.dma_free_task(%b_r1c2_m8_n37)
      aiex.dma_free_task(%c_r1c2_m8_n37)
      // core r1c2 m8 n38
      %a_r1c2_m8_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n38)
      %b_r1c2_m8_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n38)
      %c_r1c2_m8_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 919936 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n38)
      aiex.dma_await_task(%c_r1c2_m8_n38)
      aiex.dma_free_task(%a_r1c2_m8_n38)
      aiex.dma_free_task(%b_r1c2_m8_n38)
      aiex.dma_free_task(%c_r1c2_m8_n38)
      // core r1c2 m8 n39
      %a_r1c2_m8_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n39)
      %b_r1c2_m8_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n39)
      %c_r1c2_m8_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920000 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n39)
      aiex.dma_await_task(%c_r1c2_m8_n39)
      aiex.dma_free_task(%a_r1c2_m8_n39)
      aiex.dma_free_task(%b_r1c2_m8_n39)
      aiex.dma_free_task(%c_r1c2_m8_n39)
      // core r1c2 m8 n40
      %a_r1c2_m8_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n40)
      %b_r1c2_m8_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n40)
      %c_r1c2_m8_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920064 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n40)
      aiex.dma_await_task(%c_r1c2_m8_n40)
      aiex.dma_free_task(%a_r1c2_m8_n40)
      aiex.dma_free_task(%b_r1c2_m8_n40)
      aiex.dma_free_task(%c_r1c2_m8_n40)
      // core r1c2 m8 n41
      %a_r1c2_m8_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m8_n41)
      %b_r1c2_m8_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m8_n41)
      %c_r1c2_m8_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m8_n41)
      aiex.dma_await_task(%c_r1c2_m8_n41)
      aiex.dma_free_task(%a_r1c2_m8_n41)
      aiex.dma_free_task(%b_r1c2_m8_n41)
      aiex.dma_free_task(%c_r1c2_m8_n41)
      // core r1c2 m9 n28
      %a_r1c2_m9_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n28)
      %b_r1c2_m9_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n28)
      %c_r1c2_m9_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1033984 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n28)
      aiex.dma_await_task(%c_r1c2_m9_n28)
      aiex.dma_free_task(%a_r1c2_m9_n28)
      aiex.dma_free_task(%b_r1c2_m9_n28)
      aiex.dma_free_task(%c_r1c2_m9_n28)
      // core r1c2 m9 n29
      %a_r1c2_m9_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n29)
      %b_r1c2_m9_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n29)
      %c_r1c2_m9_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034048 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n29)
      aiex.dma_await_task(%c_r1c2_m9_n29)
      aiex.dma_free_task(%a_r1c2_m9_n29)
      aiex.dma_free_task(%b_r1c2_m9_n29)
      aiex.dma_free_task(%c_r1c2_m9_n29)
      // core r1c2 m9 n30
      %a_r1c2_m9_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n30)
      %b_r1c2_m9_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n30)
      %c_r1c2_m9_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034112 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n30)
      aiex.dma_await_task(%c_r1c2_m9_n30)
      aiex.dma_free_task(%a_r1c2_m9_n30)
      aiex.dma_free_task(%b_r1c2_m9_n30)
      aiex.dma_free_task(%c_r1c2_m9_n30)
      // core r1c2 m9 n31
      %a_r1c2_m9_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n31)
      %b_r1c2_m9_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n31)
      %c_r1c2_m9_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034176 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n31)
      aiex.dma_await_task(%c_r1c2_m9_n31)
      aiex.dma_free_task(%a_r1c2_m9_n31)
      aiex.dma_free_task(%b_r1c2_m9_n31)
      aiex.dma_free_task(%c_r1c2_m9_n31)
      // core r1c2 m9 n32
      %a_r1c2_m9_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n32)
      %b_r1c2_m9_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n32)
      %c_r1c2_m9_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034240 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n32)
      aiex.dma_await_task(%c_r1c2_m9_n32)
      aiex.dma_free_task(%a_r1c2_m9_n32)
      aiex.dma_free_task(%b_r1c2_m9_n32)
      aiex.dma_free_task(%c_r1c2_m9_n32)
      // core r1c2 m9 n33
      %a_r1c2_m9_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n33)
      %b_r1c2_m9_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n33)
      %c_r1c2_m9_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034304 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n33)
      aiex.dma_await_task(%c_r1c2_m9_n33)
      aiex.dma_free_task(%a_r1c2_m9_n33)
      aiex.dma_free_task(%b_r1c2_m9_n33)
      aiex.dma_free_task(%c_r1c2_m9_n33)
      // core r1c2 m9 n34
      %a_r1c2_m9_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n34)
      %b_r1c2_m9_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n34)
      %c_r1c2_m9_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034368 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n34)
      aiex.dma_await_task(%c_r1c2_m9_n34)
      aiex.dma_free_task(%a_r1c2_m9_n34)
      aiex.dma_free_task(%b_r1c2_m9_n34)
      aiex.dma_free_task(%c_r1c2_m9_n34)
      // core r1c2 m9 n35
      %a_r1c2_m9_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n35)
      %b_r1c2_m9_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n35)
      %c_r1c2_m9_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034432 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n35)
      aiex.dma_await_task(%c_r1c2_m9_n35)
      aiex.dma_free_task(%a_r1c2_m9_n35)
      aiex.dma_free_task(%b_r1c2_m9_n35)
      aiex.dma_free_task(%c_r1c2_m9_n35)
      // core r1c2 m9 n36
      %a_r1c2_m9_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n36)
      %b_r1c2_m9_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n36)
      %c_r1c2_m9_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034496 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n36)
      aiex.dma_await_task(%c_r1c2_m9_n36)
      aiex.dma_free_task(%a_r1c2_m9_n36)
      aiex.dma_free_task(%b_r1c2_m9_n36)
      aiex.dma_free_task(%c_r1c2_m9_n36)
      // core r1c2 m9 n37
      %a_r1c2_m9_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n37)
      %b_r1c2_m9_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n37)
      %c_r1c2_m9_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034560 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n37)
      aiex.dma_await_task(%c_r1c2_m9_n37)
      aiex.dma_free_task(%a_r1c2_m9_n37)
      aiex.dma_free_task(%b_r1c2_m9_n37)
      aiex.dma_free_task(%c_r1c2_m9_n37)
      // core r1c2 m9 n38
      %a_r1c2_m9_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n38)
      %b_r1c2_m9_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n38)
      %c_r1c2_m9_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034624 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n38)
      aiex.dma_await_task(%c_r1c2_m9_n38)
      aiex.dma_free_task(%a_r1c2_m9_n38)
      aiex.dma_free_task(%b_r1c2_m9_n38)
      aiex.dma_free_task(%c_r1c2_m9_n38)
      // core r1c2 m9 n39
      %a_r1c2_m9_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n39)
      %b_r1c2_m9_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n39)
      %c_r1c2_m9_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n39)
      aiex.dma_await_task(%c_r1c2_m9_n39)
      aiex.dma_free_task(%a_r1c2_m9_n39)
      aiex.dma_free_task(%b_r1c2_m9_n39)
      aiex.dma_free_task(%c_r1c2_m9_n39)
      // core r1c2 m9 n40
      %a_r1c2_m9_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n40)
      %b_r1c2_m9_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n40)
      %c_r1c2_m9_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n40)
      aiex.dma_await_task(%c_r1c2_m9_n40)
      aiex.dma_free_task(%a_r1c2_m9_n40)
      aiex.dma_free_task(%b_r1c2_m9_n40)
      aiex.dma_free_task(%c_r1c2_m9_n40)
      // core r1c2 m9 n41
      %a_r1c2_m9_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m9_n41)
      %b_r1c2_m9_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m9_n41)
      %c_r1c2_m9_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m9_n41)
      aiex.dma_await_task(%c_r1c2_m9_n41)
      aiex.dma_free_task(%a_r1c2_m9_n41)
      aiex.dma_free_task(%b_r1c2_m9_n41)
      aiex.dma_free_task(%c_r1c2_m9_n41)
      // core r1c2 m10 n28
      %a_r1c2_m10_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n28)
      %b_r1c2_m10_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n28)
      %c_r1c2_m10_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148672 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n28)
      aiex.dma_await_task(%c_r1c2_m10_n28)
      aiex.dma_free_task(%a_r1c2_m10_n28)
      aiex.dma_free_task(%b_r1c2_m10_n28)
      aiex.dma_free_task(%c_r1c2_m10_n28)
      // core r1c2 m10 n29
      %a_r1c2_m10_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n29)
      %b_r1c2_m10_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n29)
      %c_r1c2_m10_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148736 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n29)
      aiex.dma_await_task(%c_r1c2_m10_n29)
      aiex.dma_free_task(%a_r1c2_m10_n29)
      aiex.dma_free_task(%b_r1c2_m10_n29)
      aiex.dma_free_task(%c_r1c2_m10_n29)
      // core r1c2 m10 n30
      %a_r1c2_m10_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n30)
      %b_r1c2_m10_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n30)
      %c_r1c2_m10_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148800 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n30)
      aiex.dma_await_task(%c_r1c2_m10_n30)
      aiex.dma_free_task(%a_r1c2_m10_n30)
      aiex.dma_free_task(%b_r1c2_m10_n30)
      aiex.dma_free_task(%c_r1c2_m10_n30)
      // core r1c2 m10 n31
      %a_r1c2_m10_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n31)
      %b_r1c2_m10_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n31)
      %c_r1c2_m10_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148864 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n31)
      aiex.dma_await_task(%c_r1c2_m10_n31)
      aiex.dma_free_task(%a_r1c2_m10_n31)
      aiex.dma_free_task(%b_r1c2_m10_n31)
      aiex.dma_free_task(%c_r1c2_m10_n31)
      // core r1c2 m10 n32
      %a_r1c2_m10_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n32)
      %b_r1c2_m10_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n32)
      %c_r1c2_m10_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148928 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n32)
      aiex.dma_await_task(%c_r1c2_m10_n32)
      aiex.dma_free_task(%a_r1c2_m10_n32)
      aiex.dma_free_task(%b_r1c2_m10_n32)
      aiex.dma_free_task(%c_r1c2_m10_n32)
      // core r1c2 m10 n33
      %a_r1c2_m10_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n33)
      %b_r1c2_m10_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n33)
      %c_r1c2_m10_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1148992 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n33)
      aiex.dma_await_task(%c_r1c2_m10_n33)
      aiex.dma_free_task(%a_r1c2_m10_n33)
      aiex.dma_free_task(%b_r1c2_m10_n33)
      aiex.dma_free_task(%c_r1c2_m10_n33)
      // core r1c2 m10 n34
      %a_r1c2_m10_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n34)
      %b_r1c2_m10_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n34)
      %c_r1c2_m10_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149056 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n34)
      aiex.dma_await_task(%c_r1c2_m10_n34)
      aiex.dma_free_task(%a_r1c2_m10_n34)
      aiex.dma_free_task(%b_r1c2_m10_n34)
      aiex.dma_free_task(%c_r1c2_m10_n34)
      // core r1c2 m10 n35
      %a_r1c2_m10_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n35)
      %b_r1c2_m10_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n35)
      %c_r1c2_m10_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149120 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n35)
      aiex.dma_await_task(%c_r1c2_m10_n35)
      aiex.dma_free_task(%a_r1c2_m10_n35)
      aiex.dma_free_task(%b_r1c2_m10_n35)
      aiex.dma_free_task(%c_r1c2_m10_n35)
      // core r1c2 m10 n36
      %a_r1c2_m10_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n36)
      %b_r1c2_m10_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n36)
      %c_r1c2_m10_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149184 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n36)
      aiex.dma_await_task(%c_r1c2_m10_n36)
      aiex.dma_free_task(%a_r1c2_m10_n36)
      aiex.dma_free_task(%b_r1c2_m10_n36)
      aiex.dma_free_task(%c_r1c2_m10_n36)
      // core r1c2 m10 n37
      %a_r1c2_m10_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n37)
      %b_r1c2_m10_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n37)
      %c_r1c2_m10_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149248 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n37)
      aiex.dma_await_task(%c_r1c2_m10_n37)
      aiex.dma_free_task(%a_r1c2_m10_n37)
      aiex.dma_free_task(%b_r1c2_m10_n37)
      aiex.dma_free_task(%c_r1c2_m10_n37)
      // core r1c2 m10 n38
      %a_r1c2_m10_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n38)
      %b_r1c2_m10_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n38)
      %c_r1c2_m10_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149312 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n38)
      aiex.dma_await_task(%c_r1c2_m10_n38)
      aiex.dma_free_task(%a_r1c2_m10_n38)
      aiex.dma_free_task(%b_r1c2_m10_n38)
      aiex.dma_free_task(%c_r1c2_m10_n38)
      // core r1c2 m10 n39
      %a_r1c2_m10_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n39)
      %b_r1c2_m10_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n39)
      %c_r1c2_m10_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149376 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n39)
      aiex.dma_await_task(%c_r1c2_m10_n39)
      aiex.dma_free_task(%a_r1c2_m10_n39)
      aiex.dma_free_task(%b_r1c2_m10_n39)
      aiex.dma_free_task(%c_r1c2_m10_n39)
      // core r1c2 m10 n40
      %a_r1c2_m10_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n40)
      %b_r1c2_m10_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n40)
      %c_r1c2_m10_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n40)
      aiex.dma_await_task(%c_r1c2_m10_n40)
      aiex.dma_free_task(%a_r1c2_m10_n40)
      aiex.dma_free_task(%b_r1c2_m10_n40)
      aiex.dma_free_task(%c_r1c2_m10_n40)
      // core r1c2 m10 n41
      %a_r1c2_m10_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m10_n41)
      %b_r1c2_m10_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m10_n41)
      %c_r1c2_m10_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m10_n41)
      aiex.dma_await_task(%c_r1c2_m10_n41)
      aiex.dma_free_task(%a_r1c2_m10_n41)
      aiex.dma_free_task(%b_r1c2_m10_n41)
      aiex.dma_free_task(%c_r1c2_m10_n41)
      // core r1c2 m11 n28
      %a_r1c2_m11_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n28)
      %b_r1c2_m11_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n28)
      %c_r1c2_m11_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263360 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n28)
      aiex.dma_await_task(%c_r1c2_m11_n28)
      aiex.dma_free_task(%a_r1c2_m11_n28)
      aiex.dma_free_task(%b_r1c2_m11_n28)
      aiex.dma_free_task(%c_r1c2_m11_n28)
      // core r1c2 m11 n29
      %a_r1c2_m11_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n29)
      %b_r1c2_m11_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n29)
      %c_r1c2_m11_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263424 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n29)
      aiex.dma_await_task(%c_r1c2_m11_n29)
      aiex.dma_free_task(%a_r1c2_m11_n29)
      aiex.dma_free_task(%b_r1c2_m11_n29)
      aiex.dma_free_task(%c_r1c2_m11_n29)
      // core r1c2 m11 n30
      %a_r1c2_m11_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n30)
      %b_r1c2_m11_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n30)
      %c_r1c2_m11_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263488 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n30)
      aiex.dma_await_task(%c_r1c2_m11_n30)
      aiex.dma_free_task(%a_r1c2_m11_n30)
      aiex.dma_free_task(%b_r1c2_m11_n30)
      aiex.dma_free_task(%c_r1c2_m11_n30)
      // core r1c2 m11 n31
      %a_r1c2_m11_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n31)
      %b_r1c2_m11_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n31)
      %c_r1c2_m11_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263552 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n31)
      aiex.dma_await_task(%c_r1c2_m11_n31)
      aiex.dma_free_task(%a_r1c2_m11_n31)
      aiex.dma_free_task(%b_r1c2_m11_n31)
      aiex.dma_free_task(%c_r1c2_m11_n31)
      // core r1c2 m11 n32
      %a_r1c2_m11_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n32)
      %b_r1c2_m11_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n32)
      %c_r1c2_m11_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263616 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n32)
      aiex.dma_await_task(%c_r1c2_m11_n32)
      aiex.dma_free_task(%a_r1c2_m11_n32)
      aiex.dma_free_task(%b_r1c2_m11_n32)
      aiex.dma_free_task(%c_r1c2_m11_n32)
      // core r1c2 m11 n33
      %a_r1c2_m11_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n33)
      %b_r1c2_m11_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n33)
      %c_r1c2_m11_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263680 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n33)
      aiex.dma_await_task(%c_r1c2_m11_n33)
      aiex.dma_free_task(%a_r1c2_m11_n33)
      aiex.dma_free_task(%b_r1c2_m11_n33)
      aiex.dma_free_task(%c_r1c2_m11_n33)
      // core r1c2 m11 n34
      %a_r1c2_m11_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n34)
      %b_r1c2_m11_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n34)
      %c_r1c2_m11_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263744 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n34)
      aiex.dma_await_task(%c_r1c2_m11_n34)
      aiex.dma_free_task(%a_r1c2_m11_n34)
      aiex.dma_free_task(%b_r1c2_m11_n34)
      aiex.dma_free_task(%c_r1c2_m11_n34)
      // core r1c2 m11 n35
      %a_r1c2_m11_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n35)
      %b_r1c2_m11_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n35)
      %c_r1c2_m11_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263808 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n35)
      aiex.dma_await_task(%c_r1c2_m11_n35)
      aiex.dma_free_task(%a_r1c2_m11_n35)
      aiex.dma_free_task(%b_r1c2_m11_n35)
      aiex.dma_free_task(%c_r1c2_m11_n35)
      // core r1c2 m11 n36
      %a_r1c2_m11_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n36)
      %b_r1c2_m11_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n36)
      %c_r1c2_m11_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263872 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n36)
      aiex.dma_await_task(%c_r1c2_m11_n36)
      aiex.dma_free_task(%a_r1c2_m11_n36)
      aiex.dma_free_task(%b_r1c2_m11_n36)
      aiex.dma_free_task(%c_r1c2_m11_n36)
      // core r1c2 m11 n37
      %a_r1c2_m11_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n37)
      %b_r1c2_m11_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n37)
      %c_r1c2_m11_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1263936 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n37)
      aiex.dma_await_task(%c_r1c2_m11_n37)
      aiex.dma_free_task(%a_r1c2_m11_n37)
      aiex.dma_free_task(%b_r1c2_m11_n37)
      aiex.dma_free_task(%c_r1c2_m11_n37)
      // core r1c2 m11 n38
      %a_r1c2_m11_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n38)
      %b_r1c2_m11_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n38)
      %c_r1c2_m11_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264000 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n38)
      aiex.dma_await_task(%c_r1c2_m11_n38)
      aiex.dma_free_task(%a_r1c2_m11_n38)
      aiex.dma_free_task(%b_r1c2_m11_n38)
      aiex.dma_free_task(%c_r1c2_m11_n38)
      // core r1c2 m11 n39
      %a_r1c2_m11_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n39)
      %b_r1c2_m11_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n39)
      %c_r1c2_m11_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264064 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n39)
      aiex.dma_await_task(%c_r1c2_m11_n39)
      aiex.dma_free_task(%a_r1c2_m11_n39)
      aiex.dma_free_task(%b_r1c2_m11_n39)
      aiex.dma_free_task(%c_r1c2_m11_n39)
      // core r1c2 m11 n40
      %a_r1c2_m11_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n40)
      %b_r1c2_m11_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n40)
      %c_r1c2_m11_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n40)
      aiex.dma_await_task(%c_r1c2_m11_n40)
      aiex.dma_free_task(%a_r1c2_m11_n40)
      aiex.dma_free_task(%b_r1c2_m11_n40)
      aiex.dma_free_task(%c_r1c2_m11_n40)
      // core r1c2 m11 n41
      %a_r1c2_m11_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m11_n41)
      %b_r1c2_m11_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m11_n41)
      %c_r1c2_m11_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m11_n41)
      aiex.dma_await_task(%c_r1c2_m11_n41)
      aiex.dma_free_task(%a_r1c2_m11_n41)
      aiex.dma_free_task(%b_r1c2_m11_n41)
      aiex.dma_free_task(%c_r1c2_m11_n41)
      // core r1c2 m12 n28
      %a_r1c2_m12_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n28)
      %b_r1c2_m12_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n28)
      %c_r1c2_m12_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378048 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n28)
      aiex.dma_await_task(%c_r1c2_m12_n28)
      aiex.dma_free_task(%a_r1c2_m12_n28)
      aiex.dma_free_task(%b_r1c2_m12_n28)
      aiex.dma_free_task(%c_r1c2_m12_n28)
      // core r1c2 m12 n29
      %a_r1c2_m12_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n29)
      %b_r1c2_m12_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n29)
      %c_r1c2_m12_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378112 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n29)
      aiex.dma_await_task(%c_r1c2_m12_n29)
      aiex.dma_free_task(%a_r1c2_m12_n29)
      aiex.dma_free_task(%b_r1c2_m12_n29)
      aiex.dma_free_task(%c_r1c2_m12_n29)
      // core r1c2 m12 n30
      %a_r1c2_m12_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n30)
      %b_r1c2_m12_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n30)
      %c_r1c2_m12_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378176 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n30)
      aiex.dma_await_task(%c_r1c2_m12_n30)
      aiex.dma_free_task(%a_r1c2_m12_n30)
      aiex.dma_free_task(%b_r1c2_m12_n30)
      aiex.dma_free_task(%c_r1c2_m12_n30)
      // core r1c2 m12 n31
      %a_r1c2_m12_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n31)
      %b_r1c2_m12_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n31)
      %c_r1c2_m12_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378240 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n31)
      aiex.dma_await_task(%c_r1c2_m12_n31)
      aiex.dma_free_task(%a_r1c2_m12_n31)
      aiex.dma_free_task(%b_r1c2_m12_n31)
      aiex.dma_free_task(%c_r1c2_m12_n31)
      // core r1c2 m12 n32
      %a_r1c2_m12_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n32)
      %b_r1c2_m12_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n32)
      %c_r1c2_m12_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378304 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n32)
      aiex.dma_await_task(%c_r1c2_m12_n32)
      aiex.dma_free_task(%a_r1c2_m12_n32)
      aiex.dma_free_task(%b_r1c2_m12_n32)
      aiex.dma_free_task(%c_r1c2_m12_n32)
      // core r1c2 m12 n33
      %a_r1c2_m12_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n33)
      %b_r1c2_m12_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n33)
      %c_r1c2_m12_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378368 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n33)
      aiex.dma_await_task(%c_r1c2_m12_n33)
      aiex.dma_free_task(%a_r1c2_m12_n33)
      aiex.dma_free_task(%b_r1c2_m12_n33)
      aiex.dma_free_task(%c_r1c2_m12_n33)
      // core r1c2 m12 n34
      %a_r1c2_m12_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n34)
      %b_r1c2_m12_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n34)
      %c_r1c2_m12_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378432 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n34)
      aiex.dma_await_task(%c_r1c2_m12_n34)
      aiex.dma_free_task(%a_r1c2_m12_n34)
      aiex.dma_free_task(%b_r1c2_m12_n34)
      aiex.dma_free_task(%c_r1c2_m12_n34)
      // core r1c2 m12 n35
      %a_r1c2_m12_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n35)
      %b_r1c2_m12_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n35)
      %c_r1c2_m12_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378496 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n35)
      aiex.dma_await_task(%c_r1c2_m12_n35)
      aiex.dma_free_task(%a_r1c2_m12_n35)
      aiex.dma_free_task(%b_r1c2_m12_n35)
      aiex.dma_free_task(%c_r1c2_m12_n35)
      // core r1c2 m12 n36
      %a_r1c2_m12_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n36)
      %b_r1c2_m12_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n36)
      %c_r1c2_m12_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378560 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n36)
      aiex.dma_await_task(%c_r1c2_m12_n36)
      aiex.dma_free_task(%a_r1c2_m12_n36)
      aiex.dma_free_task(%b_r1c2_m12_n36)
      aiex.dma_free_task(%c_r1c2_m12_n36)
      // core r1c2 m12 n37
      %a_r1c2_m12_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n37)
      %b_r1c2_m12_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n37)
      %c_r1c2_m12_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378624 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n37)
      aiex.dma_await_task(%c_r1c2_m12_n37)
      aiex.dma_free_task(%a_r1c2_m12_n37)
      aiex.dma_free_task(%b_r1c2_m12_n37)
      aiex.dma_free_task(%c_r1c2_m12_n37)
      // core r1c2 m12 n38
      %a_r1c2_m12_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n38)
      %b_r1c2_m12_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n38)
      %c_r1c2_m12_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n38)
      aiex.dma_await_task(%c_r1c2_m12_n38)
      aiex.dma_free_task(%a_r1c2_m12_n38)
      aiex.dma_free_task(%b_r1c2_m12_n38)
      aiex.dma_free_task(%c_r1c2_m12_n38)
      // core r1c2 m12 n39
      %a_r1c2_m12_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n39)
      %b_r1c2_m12_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n39)
      %c_r1c2_m12_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n39)
      aiex.dma_await_task(%c_r1c2_m12_n39)
      aiex.dma_free_task(%a_r1c2_m12_n39)
      aiex.dma_free_task(%b_r1c2_m12_n39)
      aiex.dma_free_task(%c_r1c2_m12_n39)
      // core r1c2 m12 n40
      %a_r1c2_m12_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n40)
      %b_r1c2_m12_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n40)
      %c_r1c2_m12_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n40)
      aiex.dma_await_task(%c_r1c2_m12_n40)
      aiex.dma_free_task(%a_r1c2_m12_n40)
      aiex.dma_free_task(%b_r1c2_m12_n40)
      aiex.dma_free_task(%c_r1c2_m12_n40)
      // core r1c2 m12 n41
      %a_r1c2_m12_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m12_n41)
      %b_r1c2_m12_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m12_n41)
      %c_r1c2_m12_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m12_n41)
      aiex.dma_await_task(%c_r1c2_m12_n41)
      aiex.dma_free_task(%a_r1c2_m12_n41)
      aiex.dma_free_task(%b_r1c2_m12_n41)
      aiex.dma_free_task(%c_r1c2_m12_n41)
      // core r1c2 m13 n28
      %a_r1c2_m13_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n28)
      %b_r1c2_m13_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n28)
      %c_r1c2_m13_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492736 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n28)
      aiex.dma_await_task(%c_r1c2_m13_n28)
      aiex.dma_free_task(%a_r1c2_m13_n28)
      aiex.dma_free_task(%b_r1c2_m13_n28)
      aiex.dma_free_task(%c_r1c2_m13_n28)
      // core r1c2 m13 n29
      %a_r1c2_m13_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n29)
      %b_r1c2_m13_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n29)
      %c_r1c2_m13_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492800 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n29)
      aiex.dma_await_task(%c_r1c2_m13_n29)
      aiex.dma_free_task(%a_r1c2_m13_n29)
      aiex.dma_free_task(%b_r1c2_m13_n29)
      aiex.dma_free_task(%c_r1c2_m13_n29)
      // core r1c2 m13 n30
      %a_r1c2_m13_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n30)
      %b_r1c2_m13_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n30)
      %c_r1c2_m13_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492864 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n30)
      aiex.dma_await_task(%c_r1c2_m13_n30)
      aiex.dma_free_task(%a_r1c2_m13_n30)
      aiex.dma_free_task(%b_r1c2_m13_n30)
      aiex.dma_free_task(%c_r1c2_m13_n30)
      // core r1c2 m13 n31
      %a_r1c2_m13_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n31)
      %b_r1c2_m13_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n31)
      %c_r1c2_m13_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492928 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n31)
      aiex.dma_await_task(%c_r1c2_m13_n31)
      aiex.dma_free_task(%a_r1c2_m13_n31)
      aiex.dma_free_task(%b_r1c2_m13_n31)
      aiex.dma_free_task(%c_r1c2_m13_n31)
      // core r1c2 m13 n32
      %a_r1c2_m13_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n32)
      %b_r1c2_m13_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n32)
      %c_r1c2_m13_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1492992 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n32)
      aiex.dma_await_task(%c_r1c2_m13_n32)
      aiex.dma_free_task(%a_r1c2_m13_n32)
      aiex.dma_free_task(%b_r1c2_m13_n32)
      aiex.dma_free_task(%c_r1c2_m13_n32)
      // core r1c2 m13 n33
      %a_r1c2_m13_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n33)
      %b_r1c2_m13_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n33)
      %c_r1c2_m13_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493056 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n33)
      aiex.dma_await_task(%c_r1c2_m13_n33)
      aiex.dma_free_task(%a_r1c2_m13_n33)
      aiex.dma_free_task(%b_r1c2_m13_n33)
      aiex.dma_free_task(%c_r1c2_m13_n33)
      // core r1c2 m13 n34
      %a_r1c2_m13_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n34)
      %b_r1c2_m13_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n34)
      %c_r1c2_m13_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493120 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n34)
      aiex.dma_await_task(%c_r1c2_m13_n34)
      aiex.dma_free_task(%a_r1c2_m13_n34)
      aiex.dma_free_task(%b_r1c2_m13_n34)
      aiex.dma_free_task(%c_r1c2_m13_n34)
      // core r1c2 m13 n35
      %a_r1c2_m13_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n35)
      %b_r1c2_m13_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n35)
      %c_r1c2_m13_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493184 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n35)
      aiex.dma_await_task(%c_r1c2_m13_n35)
      aiex.dma_free_task(%a_r1c2_m13_n35)
      aiex.dma_free_task(%b_r1c2_m13_n35)
      aiex.dma_free_task(%c_r1c2_m13_n35)
      // core r1c2 m13 n36
      %a_r1c2_m13_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n36)
      %b_r1c2_m13_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n36)
      %c_r1c2_m13_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493248 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n36)
      aiex.dma_await_task(%c_r1c2_m13_n36)
      aiex.dma_free_task(%a_r1c2_m13_n36)
      aiex.dma_free_task(%b_r1c2_m13_n36)
      aiex.dma_free_task(%c_r1c2_m13_n36)
      // core r1c2 m13 n37
      %a_r1c2_m13_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n37)
      %b_r1c2_m13_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n37)
      %c_r1c2_m13_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493312 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n37)
      aiex.dma_await_task(%c_r1c2_m13_n37)
      aiex.dma_free_task(%a_r1c2_m13_n37)
      aiex.dma_free_task(%b_r1c2_m13_n37)
      aiex.dma_free_task(%c_r1c2_m13_n37)
      // core r1c2 m13 n38
      %a_r1c2_m13_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n38)
      %b_r1c2_m13_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n38)
      %c_r1c2_m13_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493376 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n38)
      aiex.dma_await_task(%c_r1c2_m13_n38)
      aiex.dma_free_task(%a_r1c2_m13_n38)
      aiex.dma_free_task(%b_r1c2_m13_n38)
      aiex.dma_free_task(%c_r1c2_m13_n38)
      // core r1c2 m13 n39
      %a_r1c2_m13_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n39)
      %b_r1c2_m13_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n39)
      %c_r1c2_m13_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493440 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n39)
      aiex.dma_await_task(%c_r1c2_m13_n39)
      aiex.dma_free_task(%a_r1c2_m13_n39)
      aiex.dma_free_task(%b_r1c2_m13_n39)
      aiex.dma_free_task(%c_r1c2_m13_n39)
      // core r1c2 m13 n40
      %a_r1c2_m13_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n40)
      %b_r1c2_m13_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n40)
      %c_r1c2_m13_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493504 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n40)
      aiex.dma_await_task(%c_r1c2_m13_n40)
      aiex.dma_free_task(%a_r1c2_m13_n40)
      aiex.dma_free_task(%b_r1c2_m13_n40)
      aiex.dma_free_task(%c_r1c2_m13_n40)
      // core r1c2 m13 n41
      %a_r1c2_m13_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m13_n41)
      %b_r1c2_m13_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m13_n41)
      %c_r1c2_m13_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m13_n41)
      aiex.dma_await_task(%c_r1c2_m13_n41)
      aiex.dma_free_task(%a_r1c2_m13_n41)
      aiex.dma_free_task(%b_r1c2_m13_n41)
      aiex.dma_free_task(%c_r1c2_m13_n41)
      // core r1c2 m14 n28
      %a_r1c2_m14_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n28)
      %b_r1c2_m14_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n28)
      %c_r1c2_m14_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607424 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n28)
      aiex.dma_await_task(%c_r1c2_m14_n28)
      aiex.dma_free_task(%a_r1c2_m14_n28)
      aiex.dma_free_task(%b_r1c2_m14_n28)
      aiex.dma_free_task(%c_r1c2_m14_n28)
      // core r1c2 m14 n29
      %a_r1c2_m14_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n29)
      %b_r1c2_m14_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n29)
      %c_r1c2_m14_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607488 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n29)
      aiex.dma_await_task(%c_r1c2_m14_n29)
      aiex.dma_free_task(%a_r1c2_m14_n29)
      aiex.dma_free_task(%b_r1c2_m14_n29)
      aiex.dma_free_task(%c_r1c2_m14_n29)
      // core r1c2 m14 n30
      %a_r1c2_m14_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n30)
      %b_r1c2_m14_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n30)
      %c_r1c2_m14_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607552 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n30)
      aiex.dma_await_task(%c_r1c2_m14_n30)
      aiex.dma_free_task(%a_r1c2_m14_n30)
      aiex.dma_free_task(%b_r1c2_m14_n30)
      aiex.dma_free_task(%c_r1c2_m14_n30)
      // core r1c2 m14 n31
      %a_r1c2_m14_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n31)
      %b_r1c2_m14_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n31)
      %c_r1c2_m14_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607616 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n31)
      aiex.dma_await_task(%c_r1c2_m14_n31)
      aiex.dma_free_task(%a_r1c2_m14_n31)
      aiex.dma_free_task(%b_r1c2_m14_n31)
      aiex.dma_free_task(%c_r1c2_m14_n31)
      // core r1c2 m14 n32
      %a_r1c2_m14_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n32)
      %b_r1c2_m14_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n32)
      %c_r1c2_m14_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607680 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n32)
      aiex.dma_await_task(%c_r1c2_m14_n32)
      aiex.dma_free_task(%a_r1c2_m14_n32)
      aiex.dma_free_task(%b_r1c2_m14_n32)
      aiex.dma_free_task(%c_r1c2_m14_n32)
      // core r1c2 m14 n33
      %a_r1c2_m14_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n33)
      %b_r1c2_m14_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n33)
      %c_r1c2_m14_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607744 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n33)
      aiex.dma_await_task(%c_r1c2_m14_n33)
      aiex.dma_free_task(%a_r1c2_m14_n33)
      aiex.dma_free_task(%b_r1c2_m14_n33)
      aiex.dma_free_task(%c_r1c2_m14_n33)
      // core r1c2 m14 n34
      %a_r1c2_m14_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n34)
      %b_r1c2_m14_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n34)
      %c_r1c2_m14_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607808 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n34)
      aiex.dma_await_task(%c_r1c2_m14_n34)
      aiex.dma_free_task(%a_r1c2_m14_n34)
      aiex.dma_free_task(%b_r1c2_m14_n34)
      aiex.dma_free_task(%c_r1c2_m14_n34)
      // core r1c2 m14 n35
      %a_r1c2_m14_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n35)
      %b_r1c2_m14_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n35)
      %c_r1c2_m14_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607872 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n35)
      aiex.dma_await_task(%c_r1c2_m14_n35)
      aiex.dma_free_task(%a_r1c2_m14_n35)
      aiex.dma_free_task(%b_r1c2_m14_n35)
      aiex.dma_free_task(%c_r1c2_m14_n35)
      // core r1c2 m14 n36
      %a_r1c2_m14_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n36)
      %b_r1c2_m14_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n36)
      %c_r1c2_m14_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1607936 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n36)
      aiex.dma_await_task(%c_r1c2_m14_n36)
      aiex.dma_free_task(%a_r1c2_m14_n36)
      aiex.dma_free_task(%b_r1c2_m14_n36)
      aiex.dma_free_task(%c_r1c2_m14_n36)
      // core r1c2 m14 n37
      %a_r1c2_m14_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n37)
      %b_r1c2_m14_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n37)
      %c_r1c2_m14_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608000 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n37)
      aiex.dma_await_task(%c_r1c2_m14_n37)
      aiex.dma_free_task(%a_r1c2_m14_n37)
      aiex.dma_free_task(%b_r1c2_m14_n37)
      aiex.dma_free_task(%c_r1c2_m14_n37)
      // core r1c2 m14 n38
      %a_r1c2_m14_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n38)
      %b_r1c2_m14_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n38)
      %c_r1c2_m14_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608064 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n38)
      aiex.dma_await_task(%c_r1c2_m14_n38)
      aiex.dma_free_task(%a_r1c2_m14_n38)
      aiex.dma_free_task(%b_r1c2_m14_n38)
      aiex.dma_free_task(%c_r1c2_m14_n38)
      // core r1c2 m14 n39
      %a_r1c2_m14_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n39)
      %b_r1c2_m14_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n39)
      %c_r1c2_m14_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608128 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n39)
      aiex.dma_await_task(%c_r1c2_m14_n39)
      aiex.dma_free_task(%a_r1c2_m14_n39)
      aiex.dma_free_task(%b_r1c2_m14_n39)
      aiex.dma_free_task(%c_r1c2_m14_n39)
      // core r1c2 m14 n40
      %a_r1c2_m14_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n40)
      %b_r1c2_m14_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n40)
      %c_r1c2_m14_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n40)
      aiex.dma_await_task(%c_r1c2_m14_n40)
      aiex.dma_free_task(%a_r1c2_m14_n40)
      aiex.dma_free_task(%b_r1c2_m14_n40)
      aiex.dma_free_task(%c_r1c2_m14_n40)
      // core r1c2 m14 n41
      %a_r1c2_m14_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m14_n41)
      %b_r1c2_m14_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m14_n41)
      %c_r1c2_m14_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m14_n41)
      aiex.dma_await_task(%c_r1c2_m14_n41)
      aiex.dma_free_task(%a_r1c2_m14_n41)
      aiex.dma_free_task(%b_r1c2_m14_n41)
      aiex.dma_free_task(%c_r1c2_m14_n41)
      // core r1c2 m15 n28
      %a_r1c2_m15_n28 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n28)
      %b_r1c2_m15_n28 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1792 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n28)
      %c_r1c2_m15_n28 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722112 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n28)
      aiex.dma_await_task(%c_r1c2_m15_n28)
      aiex.dma_free_task(%a_r1c2_m15_n28)
      aiex.dma_free_task(%b_r1c2_m15_n28)
      aiex.dma_free_task(%c_r1c2_m15_n28)
      // core r1c2 m15 n29
      %a_r1c2_m15_n29 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n29)
      %b_r1c2_m15_n29 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1856 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n29)
      %c_r1c2_m15_n29 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722176 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n29)
      aiex.dma_await_task(%c_r1c2_m15_n29)
      aiex.dma_free_task(%a_r1c2_m15_n29)
      aiex.dma_free_task(%b_r1c2_m15_n29)
      aiex.dma_free_task(%c_r1c2_m15_n29)
      // core r1c2 m15 n30
      %a_r1c2_m15_n30 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n30)
      %b_r1c2_m15_n30 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1920 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n30)
      %c_r1c2_m15_n30 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722240 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n30)
      aiex.dma_await_task(%c_r1c2_m15_n30)
      aiex.dma_free_task(%a_r1c2_m15_n30)
      aiex.dma_free_task(%b_r1c2_m15_n30)
      aiex.dma_free_task(%c_r1c2_m15_n30)
      // core r1c2 m15 n31
      %a_r1c2_m15_n31 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n31)
      %b_r1c2_m15_n31 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 1984 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n31)
      %c_r1c2_m15_n31 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722304 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n31)
      aiex.dma_await_task(%c_r1c2_m15_n31)
      aiex.dma_free_task(%a_r1c2_m15_n31)
      aiex.dma_free_task(%b_r1c2_m15_n31)
      aiex.dma_free_task(%c_r1c2_m15_n31)
      // core r1c2 m15 n32
      %a_r1c2_m15_n32 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n32)
      %b_r1c2_m15_n32 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2048 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n32)
      %c_r1c2_m15_n32 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722368 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n32)
      aiex.dma_await_task(%c_r1c2_m15_n32)
      aiex.dma_free_task(%a_r1c2_m15_n32)
      aiex.dma_free_task(%b_r1c2_m15_n32)
      aiex.dma_free_task(%c_r1c2_m15_n32)
      // core r1c2 m15 n33
      %a_r1c2_m15_n33 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n33)
      %b_r1c2_m15_n33 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2112 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n33)
      %c_r1c2_m15_n33 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722432 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n33)
      aiex.dma_await_task(%c_r1c2_m15_n33)
      aiex.dma_free_task(%a_r1c2_m15_n33)
      aiex.dma_free_task(%b_r1c2_m15_n33)
      aiex.dma_free_task(%c_r1c2_m15_n33)
      // core r1c2 m15 n34
      %a_r1c2_m15_n34 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n34)
      %b_r1c2_m15_n34 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2176 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n34)
      %c_r1c2_m15_n34 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722496 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n34)
      aiex.dma_await_task(%c_r1c2_m15_n34)
      aiex.dma_free_task(%a_r1c2_m15_n34)
      aiex.dma_free_task(%b_r1c2_m15_n34)
      aiex.dma_free_task(%c_r1c2_m15_n34)
      // core r1c2 m15 n35
      %a_r1c2_m15_n35 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n35)
      %b_r1c2_m15_n35 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2240 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n35)
      %c_r1c2_m15_n35 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722560 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n35)
      aiex.dma_await_task(%c_r1c2_m15_n35)
      aiex.dma_free_task(%a_r1c2_m15_n35)
      aiex.dma_free_task(%b_r1c2_m15_n35)
      aiex.dma_free_task(%c_r1c2_m15_n35)
      // core r1c2 m15 n36
      %a_r1c2_m15_n36 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n36)
      %b_r1c2_m15_n36 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2304 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n36)
      %c_r1c2_m15_n36 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722624 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n36)
      aiex.dma_await_task(%c_r1c2_m15_n36)
      aiex.dma_free_task(%a_r1c2_m15_n36)
      aiex.dma_free_task(%b_r1c2_m15_n36)
      aiex.dma_free_task(%c_r1c2_m15_n36)
      // core r1c2 m15 n37
      %a_r1c2_m15_n37 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n37)
      %b_r1c2_m15_n37 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2368 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n37)
      %c_r1c2_m15_n37 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722688 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n37)
      aiex.dma_await_task(%c_r1c2_m15_n37)
      aiex.dma_free_task(%a_r1c2_m15_n37)
      aiex.dma_free_task(%b_r1c2_m15_n37)
      aiex.dma_free_task(%c_r1c2_m15_n37)
      // core r1c2 m15 n38
      %a_r1c2_m15_n38 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n38)
      %b_r1c2_m15_n38 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2432 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n38)
      %c_r1c2_m15_n38 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722752 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n38)
      aiex.dma_await_task(%c_r1c2_m15_n38)
      aiex.dma_free_task(%a_r1c2_m15_n38)
      aiex.dma_free_task(%b_r1c2_m15_n38)
      aiex.dma_free_task(%c_r1c2_m15_n38)
      // core r1c2 m15 n39
      %a_r1c2_m15_n39 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n39)
      %b_r1c2_m15_n39 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2496 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n39)
      %c_r1c2_m15_n39 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722816 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n39)
      aiex.dma_await_task(%c_r1c2_m15_n39)
      aiex.dma_free_task(%a_r1c2_m15_n39)
      aiex.dma_free_task(%b_r1c2_m15_n39)
      aiex.dma_free_task(%c_r1c2_m15_n39)
      // core r1c2 m15 n40
      %a_r1c2_m15_n40 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n40)
      %b_r1c2_m15_n40 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2560 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n40)
      %c_r1c2_m15_n40 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n40)
      aiex.dma_await_task(%c_r1c2_m15_n40)
      aiex.dma_free_task(%a_r1c2_m15_n40)
      aiex.dma_free_task(%b_r1c2_m15_n40)
      aiex.dma_free_task(%c_r1c2_m15_n40)
      // core r1c2 m15 n41
      %a_r1c2_m15_n41 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c2_m15_n41)
      %b_r1c2_m15_n41 = aiex.dma_configure_task_for @B_c2 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2624 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c2_m15_n41)
      %c_r1c2_m15_n41 = aiex.dma_configure_task_for @C2_r1_c2 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1722944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c2_m15_n41)
      aiex.dma_await_task(%c_r1c2_m15_n41)
      aiex.dma_free_task(%a_r1c2_m15_n41)
      aiex.dma_free_task(%b_r1c2_m15_n41)
      aiex.dma_free_task(%c_r1c2_m15_n41)
      // core r1c3 m8 n42
      %a_r1c3_m8_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n42)
      %b_r1c3_m8_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n42)
      %c_r1c3_m8_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920192 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n42)
      aiex.dma_await_task(%c_r1c3_m8_n42)
      aiex.dma_free_task(%a_r1c3_m8_n42)
      aiex.dma_free_task(%b_r1c3_m8_n42)
      aiex.dma_free_task(%c_r1c3_m8_n42)
      // core r1c3 m8 n43
      %a_r1c3_m8_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n43)
      %b_r1c3_m8_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n43)
      %c_r1c3_m8_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n43)
      aiex.dma_await_task(%c_r1c3_m8_n43)
      aiex.dma_free_task(%a_r1c3_m8_n43)
      aiex.dma_free_task(%b_r1c3_m8_n43)
      aiex.dma_free_task(%c_r1c3_m8_n43)
      // core r1c3 m8 n44
      %a_r1c3_m8_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n44)
      %b_r1c3_m8_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n44)
      %c_r1c3_m8_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n44)
      aiex.dma_await_task(%c_r1c3_m8_n44)
      aiex.dma_free_task(%a_r1c3_m8_n44)
      aiex.dma_free_task(%b_r1c3_m8_n44)
      aiex.dma_free_task(%c_r1c3_m8_n44)
      // core r1c3 m8 n45
      %a_r1c3_m8_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n45)
      %b_r1c3_m8_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n45)
      %c_r1c3_m8_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n45)
      aiex.dma_await_task(%c_r1c3_m8_n45)
      aiex.dma_free_task(%a_r1c3_m8_n45)
      aiex.dma_free_task(%b_r1c3_m8_n45)
      aiex.dma_free_task(%c_r1c3_m8_n45)
      // core r1c3 m8 n46
      %a_r1c3_m8_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n46)
      %b_r1c3_m8_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n46)
      %c_r1c3_m8_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n46)
      aiex.dma_await_task(%c_r1c3_m8_n46)
      aiex.dma_free_task(%a_r1c3_m8_n46)
      aiex.dma_free_task(%b_r1c3_m8_n46)
      aiex.dma_free_task(%c_r1c3_m8_n46)
      // core r1c3 m8 n47
      %a_r1c3_m8_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n47)
      %b_r1c3_m8_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n47)
      %c_r1c3_m8_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n47)
      aiex.dma_await_task(%c_r1c3_m8_n47)
      aiex.dma_free_task(%a_r1c3_m8_n47)
      aiex.dma_free_task(%b_r1c3_m8_n47)
      aiex.dma_free_task(%c_r1c3_m8_n47)
      // core r1c3 m8 n48
      %a_r1c3_m8_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n48)
      %b_r1c3_m8_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n48)
      %c_r1c3_m8_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n48)
      aiex.dma_await_task(%c_r1c3_m8_n48)
      aiex.dma_free_task(%a_r1c3_m8_n48)
      aiex.dma_free_task(%b_r1c3_m8_n48)
      aiex.dma_free_task(%c_r1c3_m8_n48)
      // core r1c3 m8 n49
      %a_r1c3_m8_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n49)
      %b_r1c3_m8_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n49)
      %c_r1c3_m8_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n49)
      aiex.dma_await_task(%c_r1c3_m8_n49)
      aiex.dma_free_task(%a_r1c3_m8_n49)
      aiex.dma_free_task(%b_r1c3_m8_n49)
      aiex.dma_free_task(%c_r1c3_m8_n49)
      // core r1c3 m8 n50
      %a_r1c3_m8_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n50)
      %b_r1c3_m8_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n50)
      %c_r1c3_m8_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n50)
      aiex.dma_await_task(%c_r1c3_m8_n50)
      aiex.dma_free_task(%a_r1c3_m8_n50)
      aiex.dma_free_task(%b_r1c3_m8_n50)
      aiex.dma_free_task(%c_r1c3_m8_n50)
      // core r1c3 m8 n51
      %a_r1c3_m8_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n51)
      %b_r1c3_m8_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n51)
      %c_r1c3_m8_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n51)
      aiex.dma_await_task(%c_r1c3_m8_n51)
      aiex.dma_free_task(%a_r1c3_m8_n51)
      aiex.dma_free_task(%b_r1c3_m8_n51)
      aiex.dma_free_task(%c_r1c3_m8_n51)
      // core r1c3 m8 n52
      %a_r1c3_m8_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n52)
      %b_r1c3_m8_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n52)
      %c_r1c3_m8_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n52)
      aiex.dma_await_task(%c_r1c3_m8_n52)
      aiex.dma_free_task(%a_r1c3_m8_n52)
      aiex.dma_free_task(%b_r1c3_m8_n52)
      aiex.dma_free_task(%c_r1c3_m8_n52)
      // core r1c3 m8 n53
      %a_r1c3_m8_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n53)
      %b_r1c3_m8_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n53)
      %c_r1c3_m8_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n53)
      aiex.dma_await_task(%c_r1c3_m8_n53)
      aiex.dma_free_task(%a_r1c3_m8_n53)
      aiex.dma_free_task(%b_r1c3_m8_n53)
      aiex.dma_free_task(%c_r1c3_m8_n53)
      // core r1c3 m8 n54
      %a_r1c3_m8_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n54)
      %b_r1c3_m8_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n54)
      %c_r1c3_m8_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 920960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n54)
      aiex.dma_await_task(%c_r1c3_m8_n54)
      aiex.dma_free_task(%a_r1c3_m8_n54)
      aiex.dma_free_task(%b_r1c3_m8_n54)
      aiex.dma_free_task(%c_r1c3_m8_n54)
      // core r1c3 m8 n55
      %a_r1c3_m8_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 262144 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m8_n55)
      %b_r1c3_m8_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m8_n55)
      %c_r1c3_m8_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 921024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m8_n55)
      aiex.dma_await_task(%c_r1c3_m8_n55)
      aiex.dma_free_task(%a_r1c3_m8_n55)
      aiex.dma_free_task(%b_r1c3_m8_n55)
      aiex.dma_free_task(%c_r1c3_m8_n55)
      // core r1c3 m9 n42
      %a_r1c3_m9_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n42)
      %b_r1c3_m9_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n42)
      %c_r1c3_m9_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034880 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n42)
      aiex.dma_await_task(%c_r1c3_m9_n42)
      aiex.dma_free_task(%a_r1c3_m9_n42)
      aiex.dma_free_task(%b_r1c3_m9_n42)
      aiex.dma_free_task(%c_r1c3_m9_n42)
      // core r1c3 m9 n43
      %a_r1c3_m9_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n43)
      %b_r1c3_m9_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n43)
      %c_r1c3_m9_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1034944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n43)
      aiex.dma_await_task(%c_r1c3_m9_n43)
      aiex.dma_free_task(%a_r1c3_m9_n43)
      aiex.dma_free_task(%b_r1c3_m9_n43)
      aiex.dma_free_task(%c_r1c3_m9_n43)
      // core r1c3 m9 n44
      %a_r1c3_m9_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n44)
      %b_r1c3_m9_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n44)
      %c_r1c3_m9_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n44)
      aiex.dma_await_task(%c_r1c3_m9_n44)
      aiex.dma_free_task(%a_r1c3_m9_n44)
      aiex.dma_free_task(%b_r1c3_m9_n44)
      aiex.dma_free_task(%c_r1c3_m9_n44)
      // core r1c3 m9 n45
      %a_r1c3_m9_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n45)
      %b_r1c3_m9_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n45)
      %c_r1c3_m9_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n45)
      aiex.dma_await_task(%c_r1c3_m9_n45)
      aiex.dma_free_task(%a_r1c3_m9_n45)
      aiex.dma_free_task(%b_r1c3_m9_n45)
      aiex.dma_free_task(%c_r1c3_m9_n45)
      // core r1c3 m9 n46
      %a_r1c3_m9_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n46)
      %b_r1c3_m9_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n46)
      %c_r1c3_m9_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n46)
      aiex.dma_await_task(%c_r1c3_m9_n46)
      aiex.dma_free_task(%a_r1c3_m9_n46)
      aiex.dma_free_task(%b_r1c3_m9_n46)
      aiex.dma_free_task(%c_r1c3_m9_n46)
      // core r1c3 m9 n47
      %a_r1c3_m9_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n47)
      %b_r1c3_m9_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n47)
      %c_r1c3_m9_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n47)
      aiex.dma_await_task(%c_r1c3_m9_n47)
      aiex.dma_free_task(%a_r1c3_m9_n47)
      aiex.dma_free_task(%b_r1c3_m9_n47)
      aiex.dma_free_task(%c_r1c3_m9_n47)
      // core r1c3 m9 n48
      %a_r1c3_m9_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n48)
      %b_r1c3_m9_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n48)
      %c_r1c3_m9_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n48)
      aiex.dma_await_task(%c_r1c3_m9_n48)
      aiex.dma_free_task(%a_r1c3_m9_n48)
      aiex.dma_free_task(%b_r1c3_m9_n48)
      aiex.dma_free_task(%c_r1c3_m9_n48)
      // core r1c3 m9 n49
      %a_r1c3_m9_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n49)
      %b_r1c3_m9_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n49)
      %c_r1c3_m9_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n49)
      aiex.dma_await_task(%c_r1c3_m9_n49)
      aiex.dma_free_task(%a_r1c3_m9_n49)
      aiex.dma_free_task(%b_r1c3_m9_n49)
      aiex.dma_free_task(%c_r1c3_m9_n49)
      // core r1c3 m9 n50
      %a_r1c3_m9_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n50)
      %b_r1c3_m9_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n50)
      %c_r1c3_m9_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n50)
      aiex.dma_await_task(%c_r1c3_m9_n50)
      aiex.dma_free_task(%a_r1c3_m9_n50)
      aiex.dma_free_task(%b_r1c3_m9_n50)
      aiex.dma_free_task(%c_r1c3_m9_n50)
      // core r1c3 m9 n51
      %a_r1c3_m9_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n51)
      %b_r1c3_m9_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n51)
      %c_r1c3_m9_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n51)
      aiex.dma_await_task(%c_r1c3_m9_n51)
      aiex.dma_free_task(%a_r1c3_m9_n51)
      aiex.dma_free_task(%b_r1c3_m9_n51)
      aiex.dma_free_task(%c_r1c3_m9_n51)
      // core r1c3 m9 n52
      %a_r1c3_m9_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n52)
      %b_r1c3_m9_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n52)
      %c_r1c3_m9_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n52)
      aiex.dma_await_task(%c_r1c3_m9_n52)
      aiex.dma_free_task(%a_r1c3_m9_n52)
      aiex.dma_free_task(%b_r1c3_m9_n52)
      aiex.dma_free_task(%c_r1c3_m9_n52)
      // core r1c3 m9 n53
      %a_r1c3_m9_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n53)
      %b_r1c3_m9_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n53)
      %c_r1c3_m9_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n53)
      aiex.dma_await_task(%c_r1c3_m9_n53)
      aiex.dma_free_task(%a_r1c3_m9_n53)
      aiex.dma_free_task(%b_r1c3_m9_n53)
      aiex.dma_free_task(%c_r1c3_m9_n53)
      // core r1c3 m9 n54
      %a_r1c3_m9_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n54)
      %b_r1c3_m9_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n54)
      %c_r1c3_m9_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n54)
      aiex.dma_await_task(%c_r1c3_m9_n54)
      aiex.dma_free_task(%a_r1c3_m9_n54)
      aiex.dma_free_task(%b_r1c3_m9_n54)
      aiex.dma_free_task(%c_r1c3_m9_n54)
      // core r1c3 m9 n55
      %a_r1c3_m9_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 294912 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m9_n55)
      %b_r1c3_m9_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m9_n55)
      %c_r1c3_m9_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1035712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m9_n55)
      aiex.dma_await_task(%c_r1c3_m9_n55)
      aiex.dma_free_task(%a_r1c3_m9_n55)
      aiex.dma_free_task(%b_r1c3_m9_n55)
      aiex.dma_free_task(%c_r1c3_m9_n55)
      // core r1c3 m10 n42
      %a_r1c3_m10_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n42)
      %b_r1c3_m10_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n42)
      %c_r1c3_m10_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149568 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n42)
      aiex.dma_await_task(%c_r1c3_m10_n42)
      aiex.dma_free_task(%a_r1c3_m10_n42)
      aiex.dma_free_task(%b_r1c3_m10_n42)
      aiex.dma_free_task(%c_r1c3_m10_n42)
      // core r1c3 m10 n43
      %a_r1c3_m10_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n43)
      %b_r1c3_m10_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n43)
      %c_r1c3_m10_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n43)
      aiex.dma_await_task(%c_r1c3_m10_n43)
      aiex.dma_free_task(%a_r1c3_m10_n43)
      aiex.dma_free_task(%b_r1c3_m10_n43)
      aiex.dma_free_task(%c_r1c3_m10_n43)
      // core r1c3 m10 n44
      %a_r1c3_m10_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n44)
      %b_r1c3_m10_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n44)
      %c_r1c3_m10_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n44)
      aiex.dma_await_task(%c_r1c3_m10_n44)
      aiex.dma_free_task(%a_r1c3_m10_n44)
      aiex.dma_free_task(%b_r1c3_m10_n44)
      aiex.dma_free_task(%c_r1c3_m10_n44)
      // core r1c3 m10 n45
      %a_r1c3_m10_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n45)
      %b_r1c3_m10_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n45)
      %c_r1c3_m10_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n45)
      aiex.dma_await_task(%c_r1c3_m10_n45)
      aiex.dma_free_task(%a_r1c3_m10_n45)
      aiex.dma_free_task(%b_r1c3_m10_n45)
      aiex.dma_free_task(%c_r1c3_m10_n45)
      // core r1c3 m10 n46
      %a_r1c3_m10_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n46)
      %b_r1c3_m10_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n46)
      %c_r1c3_m10_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n46)
      aiex.dma_await_task(%c_r1c3_m10_n46)
      aiex.dma_free_task(%a_r1c3_m10_n46)
      aiex.dma_free_task(%b_r1c3_m10_n46)
      aiex.dma_free_task(%c_r1c3_m10_n46)
      // core r1c3 m10 n47
      %a_r1c3_m10_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n47)
      %b_r1c3_m10_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n47)
      %c_r1c3_m10_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n47)
      aiex.dma_await_task(%c_r1c3_m10_n47)
      aiex.dma_free_task(%a_r1c3_m10_n47)
      aiex.dma_free_task(%b_r1c3_m10_n47)
      aiex.dma_free_task(%c_r1c3_m10_n47)
      // core r1c3 m10 n48
      %a_r1c3_m10_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n48)
      %b_r1c3_m10_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n48)
      %c_r1c3_m10_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1149952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n48)
      aiex.dma_await_task(%c_r1c3_m10_n48)
      aiex.dma_free_task(%a_r1c3_m10_n48)
      aiex.dma_free_task(%b_r1c3_m10_n48)
      aiex.dma_free_task(%c_r1c3_m10_n48)
      // core r1c3 m10 n49
      %a_r1c3_m10_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n49)
      %b_r1c3_m10_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n49)
      %c_r1c3_m10_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n49)
      aiex.dma_await_task(%c_r1c3_m10_n49)
      aiex.dma_free_task(%a_r1c3_m10_n49)
      aiex.dma_free_task(%b_r1c3_m10_n49)
      aiex.dma_free_task(%c_r1c3_m10_n49)
      // core r1c3 m10 n50
      %a_r1c3_m10_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n50)
      %b_r1c3_m10_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n50)
      %c_r1c3_m10_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n50)
      aiex.dma_await_task(%c_r1c3_m10_n50)
      aiex.dma_free_task(%a_r1c3_m10_n50)
      aiex.dma_free_task(%b_r1c3_m10_n50)
      aiex.dma_free_task(%c_r1c3_m10_n50)
      // core r1c3 m10 n51
      %a_r1c3_m10_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n51)
      %b_r1c3_m10_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n51)
      %c_r1c3_m10_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n51)
      aiex.dma_await_task(%c_r1c3_m10_n51)
      aiex.dma_free_task(%a_r1c3_m10_n51)
      aiex.dma_free_task(%b_r1c3_m10_n51)
      aiex.dma_free_task(%c_r1c3_m10_n51)
      // core r1c3 m10 n52
      %a_r1c3_m10_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n52)
      %b_r1c3_m10_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n52)
      %c_r1c3_m10_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n52)
      aiex.dma_await_task(%c_r1c3_m10_n52)
      aiex.dma_free_task(%a_r1c3_m10_n52)
      aiex.dma_free_task(%b_r1c3_m10_n52)
      aiex.dma_free_task(%c_r1c3_m10_n52)
      // core r1c3 m10 n53
      %a_r1c3_m10_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n53)
      %b_r1c3_m10_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n53)
      %c_r1c3_m10_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n53)
      aiex.dma_await_task(%c_r1c3_m10_n53)
      aiex.dma_free_task(%a_r1c3_m10_n53)
      aiex.dma_free_task(%b_r1c3_m10_n53)
      aiex.dma_free_task(%c_r1c3_m10_n53)
      // core r1c3 m10 n54
      %a_r1c3_m10_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n54)
      %b_r1c3_m10_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n54)
      %c_r1c3_m10_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n54)
      aiex.dma_await_task(%c_r1c3_m10_n54)
      aiex.dma_free_task(%a_r1c3_m10_n54)
      aiex.dma_free_task(%b_r1c3_m10_n54)
      aiex.dma_free_task(%c_r1c3_m10_n54)
      // core r1c3 m10 n55
      %a_r1c3_m10_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 327680 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m10_n55)
      %b_r1c3_m10_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m10_n55)
      %c_r1c3_m10_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1150400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m10_n55)
      aiex.dma_await_task(%c_r1c3_m10_n55)
      aiex.dma_free_task(%a_r1c3_m10_n55)
      aiex.dma_free_task(%b_r1c3_m10_n55)
      aiex.dma_free_task(%c_r1c3_m10_n55)
      // core r1c3 m11 n42
      %a_r1c3_m11_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n42)
      %b_r1c3_m11_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n42)
      %c_r1c3_m11_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264256 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n42)
      aiex.dma_await_task(%c_r1c3_m11_n42)
      aiex.dma_free_task(%a_r1c3_m11_n42)
      aiex.dma_free_task(%b_r1c3_m11_n42)
      aiex.dma_free_task(%c_r1c3_m11_n42)
      // core r1c3 m11 n43
      %a_r1c3_m11_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n43)
      %b_r1c3_m11_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n43)
      %c_r1c3_m11_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n43)
      aiex.dma_await_task(%c_r1c3_m11_n43)
      aiex.dma_free_task(%a_r1c3_m11_n43)
      aiex.dma_free_task(%b_r1c3_m11_n43)
      aiex.dma_free_task(%c_r1c3_m11_n43)
      // core r1c3 m11 n44
      %a_r1c3_m11_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n44)
      %b_r1c3_m11_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n44)
      %c_r1c3_m11_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n44)
      aiex.dma_await_task(%c_r1c3_m11_n44)
      aiex.dma_free_task(%a_r1c3_m11_n44)
      aiex.dma_free_task(%b_r1c3_m11_n44)
      aiex.dma_free_task(%c_r1c3_m11_n44)
      // core r1c3 m11 n45
      %a_r1c3_m11_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n45)
      %b_r1c3_m11_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n45)
      %c_r1c3_m11_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n45)
      aiex.dma_await_task(%c_r1c3_m11_n45)
      aiex.dma_free_task(%a_r1c3_m11_n45)
      aiex.dma_free_task(%b_r1c3_m11_n45)
      aiex.dma_free_task(%c_r1c3_m11_n45)
      // core r1c3 m11 n46
      %a_r1c3_m11_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n46)
      %b_r1c3_m11_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n46)
      %c_r1c3_m11_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n46)
      aiex.dma_await_task(%c_r1c3_m11_n46)
      aiex.dma_free_task(%a_r1c3_m11_n46)
      aiex.dma_free_task(%b_r1c3_m11_n46)
      aiex.dma_free_task(%c_r1c3_m11_n46)
      // core r1c3 m11 n47
      %a_r1c3_m11_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n47)
      %b_r1c3_m11_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n47)
      %c_r1c3_m11_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n47)
      aiex.dma_await_task(%c_r1c3_m11_n47)
      aiex.dma_free_task(%a_r1c3_m11_n47)
      aiex.dma_free_task(%b_r1c3_m11_n47)
      aiex.dma_free_task(%c_r1c3_m11_n47)
      // core r1c3 m11 n48
      %a_r1c3_m11_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n48)
      %b_r1c3_m11_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n48)
      %c_r1c3_m11_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n48)
      aiex.dma_await_task(%c_r1c3_m11_n48)
      aiex.dma_free_task(%a_r1c3_m11_n48)
      aiex.dma_free_task(%b_r1c3_m11_n48)
      aiex.dma_free_task(%c_r1c3_m11_n48)
      // core r1c3 m11 n49
      %a_r1c3_m11_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n49)
      %b_r1c3_m11_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n49)
      %c_r1c3_m11_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n49)
      aiex.dma_await_task(%c_r1c3_m11_n49)
      aiex.dma_free_task(%a_r1c3_m11_n49)
      aiex.dma_free_task(%b_r1c3_m11_n49)
      aiex.dma_free_task(%c_r1c3_m11_n49)
      // core r1c3 m11 n50
      %a_r1c3_m11_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n50)
      %b_r1c3_m11_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n50)
      %c_r1c3_m11_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n50)
      aiex.dma_await_task(%c_r1c3_m11_n50)
      aiex.dma_free_task(%a_r1c3_m11_n50)
      aiex.dma_free_task(%b_r1c3_m11_n50)
      aiex.dma_free_task(%c_r1c3_m11_n50)
      // core r1c3 m11 n51
      %a_r1c3_m11_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n51)
      %b_r1c3_m11_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n51)
      %c_r1c3_m11_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n51)
      aiex.dma_await_task(%c_r1c3_m11_n51)
      aiex.dma_free_task(%a_r1c3_m11_n51)
      aiex.dma_free_task(%b_r1c3_m11_n51)
      aiex.dma_free_task(%c_r1c3_m11_n51)
      // core r1c3 m11 n52
      %a_r1c3_m11_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n52)
      %b_r1c3_m11_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n52)
      %c_r1c3_m11_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n52)
      aiex.dma_await_task(%c_r1c3_m11_n52)
      aiex.dma_free_task(%a_r1c3_m11_n52)
      aiex.dma_free_task(%b_r1c3_m11_n52)
      aiex.dma_free_task(%c_r1c3_m11_n52)
      // core r1c3 m11 n53
      %a_r1c3_m11_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n53)
      %b_r1c3_m11_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n53)
      %c_r1c3_m11_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1264960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n53)
      aiex.dma_await_task(%c_r1c3_m11_n53)
      aiex.dma_free_task(%a_r1c3_m11_n53)
      aiex.dma_free_task(%b_r1c3_m11_n53)
      aiex.dma_free_task(%c_r1c3_m11_n53)
      // core r1c3 m11 n54
      %a_r1c3_m11_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n54)
      %b_r1c3_m11_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n54)
      %c_r1c3_m11_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1265024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n54)
      aiex.dma_await_task(%c_r1c3_m11_n54)
      aiex.dma_free_task(%a_r1c3_m11_n54)
      aiex.dma_free_task(%b_r1c3_m11_n54)
      aiex.dma_free_task(%c_r1c3_m11_n54)
      // core r1c3 m11 n55
      %a_r1c3_m11_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 360448 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m11_n55)
      %b_r1c3_m11_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m11_n55)
      %c_r1c3_m11_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1265088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m11_n55)
      aiex.dma_await_task(%c_r1c3_m11_n55)
      aiex.dma_free_task(%a_r1c3_m11_n55)
      aiex.dma_free_task(%b_r1c3_m11_n55)
      aiex.dma_free_task(%c_r1c3_m11_n55)
      // core r1c3 m12 n42
      %a_r1c3_m12_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n42)
      %b_r1c3_m12_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n42)
      %c_r1c3_m12_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1378944 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n42)
      aiex.dma_await_task(%c_r1c3_m12_n42)
      aiex.dma_free_task(%a_r1c3_m12_n42)
      aiex.dma_free_task(%b_r1c3_m12_n42)
      aiex.dma_free_task(%c_r1c3_m12_n42)
      // core r1c3 m12 n43
      %a_r1c3_m12_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n43)
      %b_r1c3_m12_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n43)
      %c_r1c3_m12_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n43)
      aiex.dma_await_task(%c_r1c3_m12_n43)
      aiex.dma_free_task(%a_r1c3_m12_n43)
      aiex.dma_free_task(%b_r1c3_m12_n43)
      aiex.dma_free_task(%c_r1c3_m12_n43)
      // core r1c3 m12 n44
      %a_r1c3_m12_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n44)
      %b_r1c3_m12_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n44)
      %c_r1c3_m12_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n44)
      aiex.dma_await_task(%c_r1c3_m12_n44)
      aiex.dma_free_task(%a_r1c3_m12_n44)
      aiex.dma_free_task(%b_r1c3_m12_n44)
      aiex.dma_free_task(%c_r1c3_m12_n44)
      // core r1c3 m12 n45
      %a_r1c3_m12_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n45)
      %b_r1c3_m12_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n45)
      %c_r1c3_m12_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n45)
      aiex.dma_await_task(%c_r1c3_m12_n45)
      aiex.dma_free_task(%a_r1c3_m12_n45)
      aiex.dma_free_task(%b_r1c3_m12_n45)
      aiex.dma_free_task(%c_r1c3_m12_n45)
      // core r1c3 m12 n46
      %a_r1c3_m12_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n46)
      %b_r1c3_m12_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n46)
      %c_r1c3_m12_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n46)
      aiex.dma_await_task(%c_r1c3_m12_n46)
      aiex.dma_free_task(%a_r1c3_m12_n46)
      aiex.dma_free_task(%b_r1c3_m12_n46)
      aiex.dma_free_task(%c_r1c3_m12_n46)
      // core r1c3 m12 n47
      %a_r1c3_m12_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n47)
      %b_r1c3_m12_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n47)
      %c_r1c3_m12_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n47)
      aiex.dma_await_task(%c_r1c3_m12_n47)
      aiex.dma_free_task(%a_r1c3_m12_n47)
      aiex.dma_free_task(%b_r1c3_m12_n47)
      aiex.dma_free_task(%c_r1c3_m12_n47)
      // core r1c3 m12 n48
      %a_r1c3_m12_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n48)
      %b_r1c3_m12_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n48)
      %c_r1c3_m12_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n48)
      aiex.dma_await_task(%c_r1c3_m12_n48)
      aiex.dma_free_task(%a_r1c3_m12_n48)
      aiex.dma_free_task(%b_r1c3_m12_n48)
      aiex.dma_free_task(%c_r1c3_m12_n48)
      // core r1c3 m12 n49
      %a_r1c3_m12_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n49)
      %b_r1c3_m12_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n49)
      %c_r1c3_m12_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n49)
      aiex.dma_await_task(%c_r1c3_m12_n49)
      aiex.dma_free_task(%a_r1c3_m12_n49)
      aiex.dma_free_task(%b_r1c3_m12_n49)
      aiex.dma_free_task(%c_r1c3_m12_n49)
      // core r1c3 m12 n50
      %a_r1c3_m12_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n50)
      %b_r1c3_m12_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n50)
      %c_r1c3_m12_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n50)
      aiex.dma_await_task(%c_r1c3_m12_n50)
      aiex.dma_free_task(%a_r1c3_m12_n50)
      aiex.dma_free_task(%b_r1c3_m12_n50)
      aiex.dma_free_task(%c_r1c3_m12_n50)
      // core r1c3 m12 n51
      %a_r1c3_m12_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n51)
      %b_r1c3_m12_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n51)
      %c_r1c3_m12_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n51)
      aiex.dma_await_task(%c_r1c3_m12_n51)
      aiex.dma_free_task(%a_r1c3_m12_n51)
      aiex.dma_free_task(%b_r1c3_m12_n51)
      aiex.dma_free_task(%c_r1c3_m12_n51)
      // core r1c3 m12 n52
      %a_r1c3_m12_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n52)
      %b_r1c3_m12_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n52)
      %c_r1c3_m12_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n52)
      aiex.dma_await_task(%c_r1c3_m12_n52)
      aiex.dma_free_task(%a_r1c3_m12_n52)
      aiex.dma_free_task(%b_r1c3_m12_n52)
      aiex.dma_free_task(%c_r1c3_m12_n52)
      // core r1c3 m12 n53
      %a_r1c3_m12_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n53)
      %b_r1c3_m12_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n53)
      %c_r1c3_m12_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n53)
      aiex.dma_await_task(%c_r1c3_m12_n53)
      aiex.dma_free_task(%a_r1c3_m12_n53)
      aiex.dma_free_task(%b_r1c3_m12_n53)
      aiex.dma_free_task(%c_r1c3_m12_n53)
      // core r1c3 m12 n54
      %a_r1c3_m12_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n54)
      %b_r1c3_m12_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n54)
      %c_r1c3_m12_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n54)
      aiex.dma_await_task(%c_r1c3_m12_n54)
      aiex.dma_free_task(%a_r1c3_m12_n54)
      aiex.dma_free_task(%b_r1c3_m12_n54)
      aiex.dma_free_task(%c_r1c3_m12_n54)
      // core r1c3 m12 n55
      %a_r1c3_m12_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 393216 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m12_n55)
      %b_r1c3_m12_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m12_n55)
      %c_r1c3_m12_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1379776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m12_n55)
      aiex.dma_await_task(%c_r1c3_m12_n55)
      aiex.dma_free_task(%a_r1c3_m12_n55)
      aiex.dma_free_task(%b_r1c3_m12_n55)
      aiex.dma_free_task(%c_r1c3_m12_n55)
      // core r1c3 m13 n42
      %a_r1c3_m13_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n42)
      %b_r1c3_m13_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n42)
      %c_r1c3_m13_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493632 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n42)
      aiex.dma_await_task(%c_r1c3_m13_n42)
      aiex.dma_free_task(%a_r1c3_m13_n42)
      aiex.dma_free_task(%b_r1c3_m13_n42)
      aiex.dma_free_task(%c_r1c3_m13_n42)
      // core r1c3 m13 n43
      %a_r1c3_m13_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n43)
      %b_r1c3_m13_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n43)
      %c_r1c3_m13_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493696 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n43)
      aiex.dma_await_task(%c_r1c3_m13_n43)
      aiex.dma_free_task(%a_r1c3_m13_n43)
      aiex.dma_free_task(%b_r1c3_m13_n43)
      aiex.dma_free_task(%c_r1c3_m13_n43)
      // core r1c3 m13 n44
      %a_r1c3_m13_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n44)
      %b_r1c3_m13_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n44)
      %c_r1c3_m13_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493760 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n44)
      aiex.dma_await_task(%c_r1c3_m13_n44)
      aiex.dma_free_task(%a_r1c3_m13_n44)
      aiex.dma_free_task(%b_r1c3_m13_n44)
      aiex.dma_free_task(%c_r1c3_m13_n44)
      // core r1c3 m13 n45
      %a_r1c3_m13_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n45)
      %b_r1c3_m13_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n45)
      %c_r1c3_m13_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493824 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n45)
      aiex.dma_await_task(%c_r1c3_m13_n45)
      aiex.dma_free_task(%a_r1c3_m13_n45)
      aiex.dma_free_task(%b_r1c3_m13_n45)
      aiex.dma_free_task(%c_r1c3_m13_n45)
      // core r1c3 m13 n46
      %a_r1c3_m13_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n46)
      %b_r1c3_m13_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n46)
      %c_r1c3_m13_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493888 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n46)
      aiex.dma_await_task(%c_r1c3_m13_n46)
      aiex.dma_free_task(%a_r1c3_m13_n46)
      aiex.dma_free_task(%b_r1c3_m13_n46)
      aiex.dma_free_task(%c_r1c3_m13_n46)
      // core r1c3 m13 n47
      %a_r1c3_m13_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n47)
      %b_r1c3_m13_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n47)
      %c_r1c3_m13_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1493952 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n47)
      aiex.dma_await_task(%c_r1c3_m13_n47)
      aiex.dma_free_task(%a_r1c3_m13_n47)
      aiex.dma_free_task(%b_r1c3_m13_n47)
      aiex.dma_free_task(%c_r1c3_m13_n47)
      // core r1c3 m13 n48
      %a_r1c3_m13_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n48)
      %b_r1c3_m13_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n48)
      %c_r1c3_m13_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494016 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n48)
      aiex.dma_await_task(%c_r1c3_m13_n48)
      aiex.dma_free_task(%a_r1c3_m13_n48)
      aiex.dma_free_task(%b_r1c3_m13_n48)
      aiex.dma_free_task(%c_r1c3_m13_n48)
      // core r1c3 m13 n49
      %a_r1c3_m13_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n49)
      %b_r1c3_m13_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n49)
      %c_r1c3_m13_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494080 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n49)
      aiex.dma_await_task(%c_r1c3_m13_n49)
      aiex.dma_free_task(%a_r1c3_m13_n49)
      aiex.dma_free_task(%b_r1c3_m13_n49)
      aiex.dma_free_task(%c_r1c3_m13_n49)
      // core r1c3 m13 n50
      %a_r1c3_m13_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n50)
      %b_r1c3_m13_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n50)
      %c_r1c3_m13_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494144 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n50)
      aiex.dma_await_task(%c_r1c3_m13_n50)
      aiex.dma_free_task(%a_r1c3_m13_n50)
      aiex.dma_free_task(%b_r1c3_m13_n50)
      aiex.dma_free_task(%c_r1c3_m13_n50)
      // core r1c3 m13 n51
      %a_r1c3_m13_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n51)
      %b_r1c3_m13_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n51)
      %c_r1c3_m13_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494208 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n51)
      aiex.dma_await_task(%c_r1c3_m13_n51)
      aiex.dma_free_task(%a_r1c3_m13_n51)
      aiex.dma_free_task(%b_r1c3_m13_n51)
      aiex.dma_free_task(%c_r1c3_m13_n51)
      // core r1c3 m13 n52
      %a_r1c3_m13_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n52)
      %b_r1c3_m13_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n52)
      %c_r1c3_m13_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494272 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n52)
      aiex.dma_await_task(%c_r1c3_m13_n52)
      aiex.dma_free_task(%a_r1c3_m13_n52)
      aiex.dma_free_task(%b_r1c3_m13_n52)
      aiex.dma_free_task(%c_r1c3_m13_n52)
      // core r1c3 m13 n53
      %a_r1c3_m13_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n53)
      %b_r1c3_m13_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n53)
      %c_r1c3_m13_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494336 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n53)
      aiex.dma_await_task(%c_r1c3_m13_n53)
      aiex.dma_free_task(%a_r1c3_m13_n53)
      aiex.dma_free_task(%b_r1c3_m13_n53)
      aiex.dma_free_task(%c_r1c3_m13_n53)
      // core r1c3 m13 n54
      %a_r1c3_m13_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n54)
      %b_r1c3_m13_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n54)
      %c_r1c3_m13_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494400 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n54)
      aiex.dma_await_task(%c_r1c3_m13_n54)
      aiex.dma_free_task(%a_r1c3_m13_n54)
      aiex.dma_free_task(%b_r1c3_m13_n54)
      aiex.dma_free_task(%c_r1c3_m13_n54)
      // core r1c3 m13 n55
      %a_r1c3_m13_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 425984 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m13_n55)
      %b_r1c3_m13_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m13_n55)
      %c_r1c3_m13_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1494464 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m13_n55)
      aiex.dma_await_task(%c_r1c3_m13_n55)
      aiex.dma_free_task(%a_r1c3_m13_n55)
      aiex.dma_free_task(%b_r1c3_m13_n55)
      aiex.dma_free_task(%c_r1c3_m13_n55)
      // core r1c3 m14 n42
      %a_r1c3_m14_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n42)
      %b_r1c3_m14_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n42)
      %c_r1c3_m14_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608320 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n42)
      aiex.dma_await_task(%c_r1c3_m14_n42)
      aiex.dma_free_task(%a_r1c3_m14_n42)
      aiex.dma_free_task(%b_r1c3_m14_n42)
      aiex.dma_free_task(%c_r1c3_m14_n42)
      // core r1c3 m14 n43
      %a_r1c3_m14_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n43)
      %b_r1c3_m14_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n43)
      %c_r1c3_m14_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608384 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n43)
      aiex.dma_await_task(%c_r1c3_m14_n43)
      aiex.dma_free_task(%a_r1c3_m14_n43)
      aiex.dma_free_task(%b_r1c3_m14_n43)
      aiex.dma_free_task(%c_r1c3_m14_n43)
      // core r1c3 m14 n44
      %a_r1c3_m14_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n44)
      %b_r1c3_m14_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n44)
      %c_r1c3_m14_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608448 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n44)
      aiex.dma_await_task(%c_r1c3_m14_n44)
      aiex.dma_free_task(%a_r1c3_m14_n44)
      aiex.dma_free_task(%b_r1c3_m14_n44)
      aiex.dma_free_task(%c_r1c3_m14_n44)
      // core r1c3 m14 n45
      %a_r1c3_m14_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n45)
      %b_r1c3_m14_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n45)
      %c_r1c3_m14_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608512 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n45)
      aiex.dma_await_task(%c_r1c3_m14_n45)
      aiex.dma_free_task(%a_r1c3_m14_n45)
      aiex.dma_free_task(%b_r1c3_m14_n45)
      aiex.dma_free_task(%c_r1c3_m14_n45)
      // core r1c3 m14 n46
      %a_r1c3_m14_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n46)
      %b_r1c3_m14_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n46)
      %c_r1c3_m14_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608576 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n46)
      aiex.dma_await_task(%c_r1c3_m14_n46)
      aiex.dma_free_task(%a_r1c3_m14_n46)
      aiex.dma_free_task(%b_r1c3_m14_n46)
      aiex.dma_free_task(%c_r1c3_m14_n46)
      // core r1c3 m14 n47
      %a_r1c3_m14_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n47)
      %b_r1c3_m14_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n47)
      %c_r1c3_m14_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608640 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n47)
      aiex.dma_await_task(%c_r1c3_m14_n47)
      aiex.dma_free_task(%a_r1c3_m14_n47)
      aiex.dma_free_task(%b_r1c3_m14_n47)
      aiex.dma_free_task(%c_r1c3_m14_n47)
      // core r1c3 m14 n48
      %a_r1c3_m14_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n48)
      %b_r1c3_m14_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n48)
      %c_r1c3_m14_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608704 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n48)
      aiex.dma_await_task(%c_r1c3_m14_n48)
      aiex.dma_free_task(%a_r1c3_m14_n48)
      aiex.dma_free_task(%b_r1c3_m14_n48)
      aiex.dma_free_task(%c_r1c3_m14_n48)
      // core r1c3 m14 n49
      %a_r1c3_m14_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n49)
      %b_r1c3_m14_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n49)
      %c_r1c3_m14_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608768 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n49)
      aiex.dma_await_task(%c_r1c3_m14_n49)
      aiex.dma_free_task(%a_r1c3_m14_n49)
      aiex.dma_free_task(%b_r1c3_m14_n49)
      aiex.dma_free_task(%c_r1c3_m14_n49)
      // core r1c3 m14 n50
      %a_r1c3_m14_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n50)
      %b_r1c3_m14_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n50)
      %c_r1c3_m14_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608832 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n50)
      aiex.dma_await_task(%c_r1c3_m14_n50)
      aiex.dma_free_task(%a_r1c3_m14_n50)
      aiex.dma_free_task(%b_r1c3_m14_n50)
      aiex.dma_free_task(%c_r1c3_m14_n50)
      // core r1c3 m14 n51
      %a_r1c3_m14_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n51)
      %b_r1c3_m14_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n51)
      %c_r1c3_m14_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608896 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n51)
      aiex.dma_await_task(%c_r1c3_m14_n51)
      aiex.dma_free_task(%a_r1c3_m14_n51)
      aiex.dma_free_task(%b_r1c3_m14_n51)
      aiex.dma_free_task(%c_r1c3_m14_n51)
      // core r1c3 m14 n52
      %a_r1c3_m14_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n52)
      %b_r1c3_m14_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n52)
      %c_r1c3_m14_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1608960 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n52)
      aiex.dma_await_task(%c_r1c3_m14_n52)
      aiex.dma_free_task(%a_r1c3_m14_n52)
      aiex.dma_free_task(%b_r1c3_m14_n52)
      aiex.dma_free_task(%c_r1c3_m14_n52)
      // core r1c3 m14 n53
      %a_r1c3_m14_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n53)
      %b_r1c3_m14_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n53)
      %c_r1c3_m14_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1609024 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n53)
      aiex.dma_await_task(%c_r1c3_m14_n53)
      aiex.dma_free_task(%a_r1c3_m14_n53)
      aiex.dma_free_task(%b_r1c3_m14_n53)
      aiex.dma_free_task(%c_r1c3_m14_n53)
      // core r1c3 m14 n54
      %a_r1c3_m14_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n54)
      %b_r1c3_m14_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n54)
      %c_r1c3_m14_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1609088 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n54)
      aiex.dma_await_task(%c_r1c3_m14_n54)
      aiex.dma_free_task(%a_r1c3_m14_n54)
      aiex.dma_free_task(%b_r1c3_m14_n54)
      aiex.dma_free_task(%c_r1c3_m14_n54)
      // core r1c3 m14 n55
      %a_r1c3_m14_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 458752 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m14_n55)
      %b_r1c3_m14_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m14_n55)
      %c_r1c3_m14_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1609152 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m14_n55)
      aiex.dma_await_task(%c_r1c3_m14_n55)
      aiex.dma_free_task(%a_r1c3_m14_n55)
      aiex.dma_free_task(%b_r1c3_m14_n55)
      aiex.dma_free_task(%c_r1c3_m14_n55)
      // core r1c3 m15 n42
      %a_r1c3_m15_n42 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n42)
      %b_r1c3_m15_n42 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2688 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n42)
      %c_r1c3_m15_n42 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723008 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n42)
      aiex.dma_await_task(%c_r1c3_m15_n42)
      aiex.dma_free_task(%a_r1c3_m15_n42)
      aiex.dma_free_task(%b_r1c3_m15_n42)
      aiex.dma_free_task(%c_r1c3_m15_n42)
      // core r1c3 m15 n43
      %a_r1c3_m15_n43 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n43)
      %b_r1c3_m15_n43 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2752 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n43)
      %c_r1c3_m15_n43 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723072 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n43)
      aiex.dma_await_task(%c_r1c3_m15_n43)
      aiex.dma_free_task(%a_r1c3_m15_n43)
      aiex.dma_free_task(%b_r1c3_m15_n43)
      aiex.dma_free_task(%c_r1c3_m15_n43)
      // core r1c3 m15 n44
      %a_r1c3_m15_n44 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n44)
      %b_r1c3_m15_n44 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2816 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n44)
      %c_r1c3_m15_n44 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723136 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n44)
      aiex.dma_await_task(%c_r1c3_m15_n44)
      aiex.dma_free_task(%a_r1c3_m15_n44)
      aiex.dma_free_task(%b_r1c3_m15_n44)
      aiex.dma_free_task(%c_r1c3_m15_n44)
      // core r1c3 m15 n45
      %a_r1c3_m15_n45 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n45)
      %b_r1c3_m15_n45 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2880 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n45)
      %c_r1c3_m15_n45 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723200 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n45)
      aiex.dma_await_task(%c_r1c3_m15_n45)
      aiex.dma_free_task(%a_r1c3_m15_n45)
      aiex.dma_free_task(%b_r1c3_m15_n45)
      aiex.dma_free_task(%c_r1c3_m15_n45)
      // core r1c3 m15 n46
      %a_r1c3_m15_n46 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n46)
      %b_r1c3_m15_n46 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 2944 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n46)
      %c_r1c3_m15_n46 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723264 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n46)
      aiex.dma_await_task(%c_r1c3_m15_n46)
      aiex.dma_free_task(%a_r1c3_m15_n46)
      aiex.dma_free_task(%b_r1c3_m15_n46)
      aiex.dma_free_task(%c_r1c3_m15_n46)
      // core r1c3 m15 n47
      %a_r1c3_m15_n47 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n47)
      %b_r1c3_m15_n47 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3008 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n47)
      %c_r1c3_m15_n47 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723328 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n47)
      aiex.dma_await_task(%c_r1c3_m15_n47)
      aiex.dma_free_task(%a_r1c3_m15_n47)
      aiex.dma_free_task(%b_r1c3_m15_n47)
      aiex.dma_free_task(%c_r1c3_m15_n47)
      // core r1c3 m15 n48
      %a_r1c3_m15_n48 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n48)
      %b_r1c3_m15_n48 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3072 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n48)
      %c_r1c3_m15_n48 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723392 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n48)
      aiex.dma_await_task(%c_r1c3_m15_n48)
      aiex.dma_free_task(%a_r1c3_m15_n48)
      aiex.dma_free_task(%b_r1c3_m15_n48)
      aiex.dma_free_task(%c_r1c3_m15_n48)
      // core r1c3 m15 n49
      %a_r1c3_m15_n49 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n49)
      %b_r1c3_m15_n49 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3136 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n49)
      %c_r1c3_m15_n49 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723456 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n49)
      aiex.dma_await_task(%c_r1c3_m15_n49)
      aiex.dma_free_task(%a_r1c3_m15_n49)
      aiex.dma_free_task(%b_r1c3_m15_n49)
      aiex.dma_free_task(%c_r1c3_m15_n49)
      // core r1c3 m15 n50
      %a_r1c3_m15_n50 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n50)
      %b_r1c3_m15_n50 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3200 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n50)
      %c_r1c3_m15_n50 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723520 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n50)
      aiex.dma_await_task(%c_r1c3_m15_n50)
      aiex.dma_free_task(%a_r1c3_m15_n50)
      aiex.dma_free_task(%b_r1c3_m15_n50)
      aiex.dma_free_task(%c_r1c3_m15_n50)
      // core r1c3 m15 n51
      %a_r1c3_m15_n51 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n51)
      %b_r1c3_m15_n51 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3264 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n51)
      %c_r1c3_m15_n51 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723584 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n51)
      aiex.dma_await_task(%c_r1c3_m15_n51)
      aiex.dma_free_task(%a_r1c3_m15_n51)
      aiex.dma_free_task(%b_r1c3_m15_n51)
      aiex.dma_free_task(%c_r1c3_m15_n51)
      // core r1c3 m15 n52
      %a_r1c3_m15_n52 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n52)
      %b_r1c3_m15_n52 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3328 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n52)
      %c_r1c3_m15_n52 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723648 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n52)
      aiex.dma_await_task(%c_r1c3_m15_n52)
      aiex.dma_free_task(%a_r1c3_m15_n52)
      aiex.dma_free_task(%b_r1c3_m15_n52)
      aiex.dma_free_task(%c_r1c3_m15_n52)
      // core r1c3 m15 n53
      %a_r1c3_m15_n53 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n53)
      %b_r1c3_m15_n53 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3392 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n53)
      %c_r1c3_m15_n53 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723712 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n53)
      aiex.dma_await_task(%c_r1c3_m15_n53)
      aiex.dma_free_task(%a_r1c3_m15_n53)
      aiex.dma_free_task(%b_r1c3_m15_n53)
      aiex.dma_free_task(%c_r1c3_m15_n53)
      // core r1c3 m15 n54
      %a_r1c3_m15_n54 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n54)
      %b_r1c3_m15_n54 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3456 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n54)
      %c_r1c3_m15_n54 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723776 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n54)
      aiex.dma_await_task(%c_r1c3_m15_n54)
      aiex.dma_free_task(%a_r1c3_m15_n54)
      aiex.dma_free_task(%b_r1c3_m15_n54)
      aiex.dma_free_task(%c_r1c3_m15_n54)
      // core r1c3 m15 n55
      %a_r1c3_m15_n55 = aiex.dma_configure_task_for @A_r1 {
        aie.dma_bd(%A : memref<512x1024xi8> offset = 491520 len = 8192 sizes = [1, 32, 256] strides = [0, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%a_r1c3_m15_n55)
      %b_r1c3_m15_n55 = aiex.dma_configure_task_for @B_c3 {
        aie.dma_bd(%B : memref<1024x3584xi8> offset = 3520 len = 16384 sizes = [1, 256, 64] strides = [0, 3584, 1])
        aie.end
      }
      aiex.dma_start_task(%b_r1c3_m15_n55)
      %c_r1c3_m15_n55 = aiex.dma_configure_task_for @C2_r1_c3 {
        aie.dma_bd(%Cout : memref<512x3584xi32> offset = 1723840 len = 2048 sizes = [1, 32, 64] strides = [0, 3584, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%c_r1c3_m15_n55)
      aiex.dma_await_task(%c_r1c3_m15_n55)
      aiex.dma_free_task(%a_r1c3_m15_n55)
      aiex.dma_free_task(%b_r1c3_m15_n55)
      aiex.dma_free_task(%c_r1c3_m15_n55)
    }
  }
}