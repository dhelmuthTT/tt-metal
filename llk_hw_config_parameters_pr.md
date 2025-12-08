# LLK Hardware Configuration Parameters - Unified API

This PR consolidates multiple specialized LLK HW configuration functions into three main unified functions across Wormhole B0 and Blackhole architectures.

## `_llk_math_hw_configure_`

| Parameter | Type | Description | Arch |
|-----------|------|-------------|------|
| `srca_data_format` | `std::uint32_t` | Source A data format | WH/BH |
| `srcb_data_format` | `std::uint32_t` | Source B data format | WH/BH |
| `int8_math_enabled` | `uint` (computed) | Enable INT8 math mode | WH/BH |

### Architecture-Specific Differences:
- **WH**: Programs ALU format registers for both sources
- **BH**: Only programs INT8 math enable flag (ALU format is inferred in BH)
- **BH**: Additionally sets DEST_ACCESS_CFG_zeroacc_absolute_tile_mode to 1 (legacy mode for ZEROACC)

## `_llk_pack_hw_configure_`

| Parameter | Type | Description | Arch |
|-----------|------|-------------|------|
| `is_fp32_dest_acc_en` | `bool` (template) | FP32 destination accumulation enable | WH/BH |
| `untilize` | `bool` (template) | Untilize mode enable | WH/BH |
| `tilize` | `bool` (template) | Tilize mode enable | BH |
| `pack_src_format` | `std::uint32_t` | Source format for packer | WH/BH |
| `pack_dst_format` | `std::uint32_t` | Destination format for packer | WH/BH |
| `tile_size` | `std::uint32_t` | Size of tile in bytes | WH/BH |
| `face_r_dim` | `std::uint32_t` | Face row dimension (default: FACE_R_DIM) | WH/BH |
| `tile_c_dim` | `std::uint32_t` | Tile column dimension (default: TILE_C_DIM) | BH |
| `num_faces` | `std::uint32_t` | Number of faces (must be 1, 2, or 4) | WH/BH |
| `partial_face` | `bool` | Partial face processing | WH/BH |
| `narrow_tile` | `bool` | Narrow tile processing | WH/BH |
| `relu_config` | `std::uint32_t` | ReLU configuration (default: 0) | WH/BH |

### Architecture-Specific Differences:
- **BH**: Supports additional `tilize` template parameter and `tile_c_dim` parameter
- **WH**: Uses `face_r_dim` for face dimensions
- **BH**: Uses both `face_r_dim` and `tile_c_dim` for more flexible tile configurations

## `_llk_unpack_hw_configure_`

| Parameter | Type | Description | Arch |
|-----------|------|-------------|------|
| `is_fp32_dest_acc_en` | `bool` (template) | FP32 destination accumulation enable | WH/BH |
| `unpA_src_format` | `std::uint32_t` | Unpacker A source format | WH/BH |
| `unpB_src_format` | `std::uint32_t` | Unpacker B source format | WH/BH |
| `unpA_dst_format` | `std::uint32_t` | Unpacker A destination format | WH/BH |
| `unpB_dst_format` | `std::uint32_t` | Unpacker B destination format | WH/BH |
| `unpA_face_r_dim` | `std::uint32_t` | Unpacker A face row dimension (default: FACE_R_DIM) | WH/BH |
| `unpB_face_r_dim` | `std::uint32_t` | Unpacker B face row dimension (default: FACE_R_DIM) | WH/BH |
| `unpA_num_faces` | `std::uint32_t` | Unpacker A number of faces (must be 1, 2, or 4) | WH/BH |
| `unpB_num_faces` | `std::uint32_t` | Unpacker B number of faces (must be 1, 2, or 4) | WH/BH |

### Architecture-Specific Differences:
- **Both architectures**: Identical function signature and behavior
- Calls `configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>()` with the same parameters

## Specialized Functions

### `_llk_unpack_configure_stoch_rnd_`

| Parameter | Type | Description | Arch |
|-----------|------|-------------|------|
| `stoch_rnd_mode` | `StochRndType` (template) | Stochastic rounding mode | WH/BH |

**StochRndType Values:**
- `All`: Enables FPU and Packer stochastic rounding
- `Fpu`: Enables only FPU stochastic rounding
- `Pack`: Enables only Packer stochastic rounding

### `_llk_math_reconfig_remap_` (BH Only)

| Parameter | Type | Description | Arch |
|-----------|------|-------------|------|
| `remap_enable` | `bool` | Enable address remapping for untilize operations | BH |

## Summary

This unification effort consolidates multiple specialized HW configuration functions into three main functions:

1. **`_llk_math_hw_configure_`**: Configures math unit data formats and INT8 math mode
2. **`_llk_pack_hw_configure_`**: Configures packer with comprehensive tile and format settings  
3. **`_llk_unpack_hw_configure_`**: Configures both unpackers with format and dimension settings

**Key Benefits:**
- **Reduced API complexity**: Fewer HW configuration calls required
- **Cleaner interface**: Consolidated parameters with sensible defaults
- **Full backward compatibility**: All existing functionality preserved
- **Cross-architecture support**: Unified API works across both Wormhole B0 and Blackhole

**Specialized functions** handle specific features:
- **`_llk_unpack_configure_stoch_rnd_`**: Stochastic rounding configuration (both architectures)
- **`_llk_math_reconfig_remap_`**: Address remapping for untilize operations (BH only)

This provides a cleaner, more maintainable LLK API while preserving full functionality across both architectures.
