# Podman Storage Audit - Phase 6A

**Date:** 2026-09-08
**Root FS Usage:** 96% (12G/13G used, 580M free) - CRITICAL

## Images
| REPOSITORY | TAG | IMAGE ID | CREATED | SIZE |
|------------|-----|----------|---------|------|
| localhost/npu-dev | mlir | 6d3d2be79faa | 46 hours ago | 651 MB |
| localhost/npu-dev | base | 410671d9e585 | 46 hours ago | 645 MB |
| docker.io/library/ubuntu | 26.04 | af52039db3f8 | 3 weeks ago | 112 MB |

**Total Images:** 3 (2 active, 651.2 MB total, 644.8 MB reclaimable)

## Containers
| CONTAINER ID | IMAGE | COMMAND | CREATED | STATUS | NAME |
|--------------|-------|---------|---------|--------|------|
| 722ed8f0c282 | localhost/npu-dev:mlir | sleep infinity | 43 hours ago | Exited (0) 32 hours ago | zen_cori |
| 77c49275c1c5 | localhost/npu-dev:mlir | sleep infinity | 30 hours ago | Exited (0) 292 years ago | zen_combined |
| a3334d84bd4d | docker.io/library/ubuntu:26.04 | sleep infinity | 30 hours ago | Exited (0) 292 years ago | zen_gpu |
| b82008ec3f4a | localhost/npu-dev:mlir | sleep infinity | 27 hours ago | Up 19 hours | zen_overlap |
| c74fff92bee8 | localhost/npu-dev:mlir | sleep infinity | 19 hours ago | Up 19 hours | quirky_lederberg |
| 1411bf20dc50 | localhost/npu-dev:mlir | bash -c MLIR_AIE_... | 8 minutes ago | Up 8 minutes | sweet_germain |

**Total Containers:** 6 (3 active, 1.399 GB total, 389.3 MB reclaimable)

## Volumes
| DRIVER | VOLUME NAME |
|--------|-------------|
| local | npu-mlir-data |

**Total Volumes:** 1 (2.488 GB, 0B reclaimable)

## Largest Consumers
1. **npu-mlir-data volume**: 2.488 GB (MLIR/AIE build artifacts)
2. **npu-dev:mlir image**: 651 MB (MLIR development image)
3. **Container writable layers**: ~389 MB reclaimable

## Recommendations
- **DO NOT install additional large packages** - root FS at 96%
- **DO NOT rebuild Podman images** unless absolutely necessary
- Consider removing stopped containers (zen_cori, zen_combined, zen_gpu) to reclaim ~389 MB
- Model must remain on external storage (/home/ubuntu12738/Storage/Models/)
- No model downloads into container or root filesystem

## Phase 6A Status: COMPLETE
Storage understood. No unnecessary installations will be performed.