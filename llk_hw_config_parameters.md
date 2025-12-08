# LLK Hardware Configuration Parameters

This document lists the parameters that each HW configuration function sets across different architectures (WH = Wormhole B0, BH = Blackhole).

## `_llk_math_hw_configure_`

| Parameter | Type | Description | Arch | Register/Configuration | Notes |
|-----------|------|-------------|------|----------------------|-------|
| `srca_data_format` | `std::uint32_t` | Source A data format | WH/BH | ALU_FORMAT_SPEC_REG0_SrcA | Runtime parameter |
| `srcb_data_format` | `std::uint32_t` | Source B data format | WH/BH | ALU_FORMAT_SPEC_REG1_SrcB | Runtime parameter |
| `int8_math_enabled` | `uint` (computed) | Enable INT8 math mode | WH/BH | ALU_ACC_CTRL_INT8_math_enabled | Computed from format parameters |

### Architecture-Specific Differences:
- **WH**: Programs ALU format registers for both sources
- **BH**: Only programs INT8 math enable flag (ALU format is inferred in BH)
- **BH**: Additionally sets DEST_ACCESS_CFG_zeroacc_absolute_tile_mode to 1 (legacy mode for ZEROACC)

## `_llk_pack_hw_configure_`

| Parameter | Type | Description | Arch | Register/Configuration | Notes |
|-----------|------|-------------|------|----------------------|-------|
| `is_fp32_dest_acc_en` | `bool` (template) | FP32 destination accumulation enable | WH/BH | PCK_DEST_RD_CTRL_Read_32b_data | Template parameter |
| `untilize` | `bool` (template) | Untilize mode enable | WH/BH | Address mode configuration | Template parameter |
| `tilize` | `bool` (template) | Tilize mode enable | BH | Address mode configuration | Template parameter (BH only) |
| `pack_src_format` | `std::uint32_t` | Source format for packer | WH/BH | THCON_SECx_REG1_In_data_format, ALU_FORMAT_SPEC_REG2_Dstacc | Runtime parameter |
| `pack_dst_format` | `std::uint32_t` | Destination format for packer | WH/BH | THCON_SECx_REG1_Out_data_format, THCON_SEC0_REG1_Pac_LF8_4b_exp (BH) | Runtime parameter |
| `tile_size` | `std::uint32_t` | Size of tile in bytes | WH/BH | p_gpr_pack::TILE_HEADER | Runtime parameter |
| `face_r_dim` | `std::uint32_t` | Face row dimension | WH/BH | PACK_COUNTERS_SEC0_pack_reads_per_xy_plane | Default: FACE_R_DIM |
| `tile_c_dim` | `std::uint32_t` | Tile column dimension | BH | Packer stride configuration | Default: TILE_C_DIM (BH only) |
| `num_faces` | `std::uint32_t` | Number of faces | WH/BH | THCON_SECx_REG1_Exp_section_size | Must be 1, 2, or 4 |
| `partial_face` | `bool` | Partial face processing | WH/BH | THCON_SECx_REG1_Exp_section_size | Runtime parameter |
| `narrow_tile` | `bool` | Narrow tile processing | WH/BH | p_setadc::PAC x-dimension | Runtime parameter |
| `relu_config` | `std::uint32_t` | ReLU configuration | WH/BH | STACC_RELU_ApplyRelu, STACC_RELU_ReluThreshold | Default: 0 |

### Architecture-Specific Differences:
- **BH**: Supports additional `tilize` template parameter and `tile_c_dim` parameter
- **WH**: Uses `face_r_dim` for face dimensions
- **BH**: Uses both `face_r_dim` and `tile_c_dim` for more flexible tile configurations

## `_llk_unpack_hw_configure_`

| Parameter | Type | Description | Arch | Register/Configuration | Notes |
|-----------|------|-------------|------|----------------------|-------|
| `is_fp32_dest_acc_en` | `bool` (template) | FP32 destination accumulation enable | WH/BH | ALU_ACC_CTRL_Fp32_enabled, ALU_ACC_CTRL_SFPU_Fp32_enabled | Template parameter |
| `unpA_src_format` | `std::uint32_t` | Unpacker A source format | WH/BH | THCON_SEC0_REG0_TileDescriptor, THCON_SEC0_REG1_Unp_LF8_4b_exp (BH), ALU_FORMAT_SPEC_REG0_SrcAUnsigned | Runtime parameter |
| `unpB_src_format` | `std::uint32_t` | Unpacker B source format | WH/BH | THCON_SEC1_REG0_TileDescriptor, THCON_SEC1_REG1_Unp_LF8_4b_exp (BH), ALU_FORMAT_SPEC_REG0_SrcBUnsigned | Runtime parameter |
| `unpA_dst_format` | `std::uint32_t` | Unpacker A destination format | WH/BH | THCON_SEC0_REG2_Out_data_format, UNP0_ADDR_CTRL_ZW_REG_1_Zstride | Runtime parameter |
| `unpB_dst_format` | `std::uint32_t` | Unpacker B destination format | WH/BH | THCON_SEC1_REG2_Out_data_format, UNP1_ADDR_CTRL_ZW_REG_1_Zstride | Runtime parameter |
| `unpA_face_r_dim` | `std::uint32_t` | Unpacker A face row dimension | WH/BH | THCON_SEC0_REG5_Tile_x_dim_cntx0, p_setadc::UNP_A x-dimension | Default: FACE_R_DIM |
| `unpB_face_r_dim` | `std::uint32_t` | Unpacker B face row dimension | WH/BH | THCON_SEC1_REG0_TileDescriptor x_dim, p_setadc::UNP_B x-dimension | Default: FACE_R_DIM |
| `unpA_num_faces` | `std::uint32_t` | Unpacker A number of faces | WH/BH | THCON_SEC0_REG0_TileDescriptor z_dim | Must be 1, 2, or 4 |
| `unpB_num_faces` | `std::uint32_t` | Unpacker B number of faces | WH/BH | THCON_SEC1_REG0_TileDescriptor z_dim | Must be 1, 2, or 4 |

### Architecture-Specific Differences:
- **Both architectures**: Identical function signature and behavior
- Calls `configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>()` with the same parameters

## Specialized Functions

### `_llk_unpack_configure_stoch_rnd_`

| Parameter | Type | Description | Arch | Register/Configuration | Notes |
|-----------|------|-------------|------|----------------------|-------|
| `stoch_rnd_mode` | `StochRndType` (template) | Stochastic rounding mode | WH/BH | ALU_ROUNDING_MODE_Fpu_srnd_en, ALU_ROUNDING_MODE_Gasket_srnd_en, ALU_ROUNDING_MODE_Packer_srnd_en | Template parameter |

**Configured Register Flags:**
- `ALU_ROUNDING_MODE_Fpu_srnd_en`: FPU stochastic rounding enable
- `ALU_ROUNDING_MODE_Gasket_srnd_en`: Gasket stochastic rounding enable  
- `ALU_ROUNDING_MODE_Packer_srnd_en`: Packer stochastic rounding enable

**StochRndType Values:**
- `All`: Enables FPU and Packer stochastic rounding
- `Fpu`: Enables only FPU stochastic rounding
- `Pack`: Enables only Packer stochastic rounding

### `_llk_math_reconfig_remap_` (BH Only)

| Parameter | Type | Description | Arch | Register/Configuration | Notes |
|-----------|------|-------------|------|----------------------|-------|
| `remap_enable` | `bool` | Enable address remapping | BH | DEST_ACCESS_CFG_remap_addrs, DEST_ACCESS_CFG_swizzle_32b | Runtime parameter |

**Configured Register Settings:**
- `DEST_ACCESS_CFG_remap_addrs`: Address remapping enable/disable
- `DEST_ACCESS_CFG_swizzle_32b`: 32-bit swizzle enable/disable (same value as remap_enable)
- Includes synchronization: Waits for all DEST accesses and packs to finish before changing configuration
- **Use Case**: Required for untilize mode which needs dest read access with stride of 16

## Summary

The unified HW configuration approach consolidates multiple specialized configuration functions into three main functions:

1. **`_llk_math_hw_configure_`**: Configures math unit data formats and INT8 math mode
2. **`_llk_pack_hw_configure_`**: Configures packer with comprehensive tile and format settings
3. **`_llk_unpack_hw_configure_`**: Configures both unpackers with format and dimension settings

The specialized functions handle specific features:
- **`_llk_unpack_configure_stoch_rnd_`**: Stochastic rounding configuration (both architectures)
- **`_llk_math_reconfig_remap_`**: Address remapping for untilize operations (BH only)

This unification reduces the number of HW configuration calls and provides a cleaner API while maintaining full functionality across both Wormhole B0 and Blackhole architectures.
