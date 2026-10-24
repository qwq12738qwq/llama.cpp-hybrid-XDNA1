with open("/workspace/llama.cpp/experiments/phase39/bench_q8_c4.py", "r") as f:
    content = f.read()

old_pack_a_vec = '''def pack_A_vec(A, As, M, K, m_tile, k_chunk):
    """Pack A for vector mac kernels (q8_chunk_vec format).
    Layout: all chunks for each row tile.
    Tile (mt, kc, blk) = [m*KT int8][m f32] where KT=k_chunk"""
    m, KT = m_tile, k_chunk
    nb_per_chunk = KT // 32
    n_chunks = K // KT
    a_tile = m * KT + m * 4  # per chunk
    n_mt = M // m
    total_tiles = n_mt * n_chunks
    Ap = np.empty((total_tiles * a_tile,), np.int8)
    
    for mt in range(n_mt):
        for ch in range(n_chunks):
            blk_offset = ch * nb_per_chunk
            t = Ap[(mt * n_chunks + ch) * a_tile:(mt * n_chunks + ch + 1) * a_tile]
            # int8 data for this chunk
            for i in range(m):
                for l in range(KT):
                    t[i * KT + l] = A[mt * m + i, ch * KT + l]
            # float32 scales for this chunk
            ts = t[m * KT:].view(np.float32)
            for i in range(m):
                ts[i] = As[mt * m + i, blk_offset:(blk_offset + nb_per_chunk)].mean()  # WRONG - need per-block scales
    return Ap'''

new_pack_a_vec = '''def pack_A_vec(A, As, M, K, m_tile, k_chunk):
    """Pack A for vector mac kernels (q8_chunk_vec format).
    Layout: all chunks for each row tile.
    Tile (mt, kc, blk) = [m*KT int8][m*nblk f32] where KT=k_chunk, nblk=KT/32"""
    m, KT = m_tile, k_chunk
    nb_per_chunk = KT // 32
    n_chunks = K // KT
    a_tile = m * KT + m * nb_per_chunk * 4  # per chunk
    n_mt = M // m
    total_tiles = n_mt * n_chunks
    Ap = np.empty((total_tiles * a_tile,), np.int8)
    
    for mt in range(n_mt):
        for ch in range(n_chunks):
            blk_offset = ch * nb_per_chunk
            t = Ap[(mt * n_chunks + ch) * a_tile:(mt * n_chunks + ch + 1) * a_tile]
            # int8 data for this chunk
            for i in range(m):
                for l in range(KT):
                    t[i * KT + l] = A[mt * m + i, ch * KT + l]
            # float32 scales for this chunk
            ts = t[m * KT:].view(np.float32)
            for i in range(m):
                ts[i * nb_per_chunk:(i + 1) * nb_per_chunk] = As[mt * m + i, blk_offset:(blk_offset + nb_per_chunk)]
    return Ap'''

if old_pack_a_vec in content:
    content = content.replace(old_pack_a_vec, new_pack_a_vec)
    with open("/workspace/llama.cpp/experiments/phase39/bench_q8_c4.py", "w") as f:
        f.write(content)
    print("fixed pack_A_vec")
else:
    print("ERROR: old pack_A_vec not found")
