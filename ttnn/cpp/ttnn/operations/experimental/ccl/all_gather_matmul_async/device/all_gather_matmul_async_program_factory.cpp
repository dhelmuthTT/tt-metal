// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"  //TODO: migrate this code to use new matmul API. This code relies on the old matmul struct
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/core_coord.hpp>
#include <unordered_map>
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn {
struct AllGatherAsync {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;
    bool use_all_gather_async_llama_sharded;
    bool use_optimal_ccl_for_llama;
    const std::optional<GlobalSemaphore>& barrier_semaphore;
    bool using_persistent_buffers;
    std::optional<uint32_t> chunks_per_sync;
    std::optional<uint32_t> num_workers_per_link;
    std::optional<uint32_t> num_buffers_per_channel;
    bool reverse_order;
    const std::optional<CoreRangeSet> sub_core_grid;

    AllGatherAsync(
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis,
        bool use_all_gather_async_llama_sharded,
        bool use_optimal_ccl_for_llama,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        bool using_persistent_buffers,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel,
        bool reverse_order = false,
        const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt) :
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis),
        use_all_gather_async_llama_sharded(use_all_gather_async_llama_sharded),
        use_optimal_ccl_for_llama(use_optimal_ccl_for_llama),
        barrier_semaphore(barrier_semaphore),
        using_persistent_buffers(using_persistent_buffers),
        chunks_per_sync(chunks_per_sync),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel),
        reverse_order(reverse_order),
        sub_core_grid(sub_core_grid) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("using_persistent_buffers", using_persistent_buffers);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("use_all_gather_async_llama_sharded", use_all_gather_async_llama_sharded);
        attrs.emplace_back("use_optimal_ccl_for_llama", use_optimal_ccl_for_llama);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("chunks_per_sync", chunks_per_sync);
        attrs.emplace_back("num_worker_per_link", num_workers_per_link);
        attrs.emplace_back("num_buffers_per_channel", num_buffers_per_channel);
        attrs.emplace_back("reverse_order", reverse_order);
        attrs.emplace_back("sub_core_grid", sub_core_grid);
        return attrs;
    }
};

using namespace tt::constants;
namespace detail {

uint32_t all_gather_async_core_count_per_link_2(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    return (num_workers_per_direction + num_mux_cores_per_direction_per_link) * num_directions_per_link;
}

uint32_t default_workers_2(
    const MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    ttnn::ccl::Topology topology,
    uint32_t output_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    auto sd_id = sub_device_id.value_or(mesh_device.get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    if (sub_core_grid.has_value()) {
        subdevice_core_range_set = subdevice_core_range_set.intersection(sub_core_grid.value());
    }
    uint32_t num_cores = subdevice_core_range_set.num_cores();
    // Above 4 workers we start getting performance drops, so we limit to 4 workers or less, depending on the number of
    // available cores This was determined by the sweep
    // tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_T3K.py
    ttnn::SmallVector<uint32_t> candidate_worker_counts;
    // if per link data moved is greater than 0.25 MB, we search greedily for 4 workers, otherwise we search greedily
    // for 2 workers. for ring, half the data is moved per link, so we divide by 2
    double data_moved_per_link_bytes = double(output_data_size_bytes) * (ring_size - 1) / ring_size / num_links /
                                       (topology == ttnn::ccl::Topology::Ring ? 2 : 1);
    // At a single packet size (4KB) we should just have one worker with the optimal packet size and save on mux
    // overheads At 256KB we observe that the perf improves if we have four workers per link
    constexpr double DATA_THRESHOLD = 256.0 * 1024;
    constexpr double SINGLE_PACKET_THRESHOLD = 4.0 * 1024;
    if (data_moved_per_link_bytes > DATA_THRESHOLD) {
        candidate_worker_counts = {4, 2, 1};
    } else if (data_moved_per_link_bytes <= SINGLE_PACKET_THRESHOLD) {
        candidate_worker_counts = {1};
    } else {
        candidate_worker_counts = {2, 1};
    }
    for (auto worker_count : candidate_worker_counts) {
        uint32_t core_count =
            num_links * all_gather_async_core_count_per_link_2(
                            worker_count, num_directions_per_link, num_mux_cores_per_direction_per_link);
        if (num_cores >= core_count) {
            log_trace(
                tt::LogOp,
                "data_moved_per_link_bytes: {} and worker_count: {}",
                data_moved_per_link_bytes,
                worker_count);
            return worker_count;
        }
    }
    TT_THROW(
        "Not enough cores available on the subdevice or device for the requested match the number of links {}",
        num_links);
}
}  // namespace detail

void fabric_mux_connection_ct_args_2(
    const uint32_t num_workers_per_direction,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));  // fabric_mux_num_buffers_per_channel
    worker_ct_args.push_back(
        mux_kernel_config.get_buffer_size_bytes(channel_type));        // fabric_mux_channel_buffer_size_bytes
    worker_ct_args.push_back(mux_kernel_config.get_status_address());  // fabric_mux_status_address
    worker_ct_args.push_back(
        mux_kernel_config.get_termination_signal_address());  // fabric_mux_termination_signal_address
    worker_ct_args.push_back(num_workers_per_direction);      // num_mux_clients
}

void fabric_mux_connection_rt_args_2(
    const bool mux_connection_valid,
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(mux_connection_valid);   // mux_connection_valid
    worker_rt_args.push_back(is_termination_master);  // is_termination_master
    worker_rt_args.push_back(mux_virtual_core.x);     // fabric_mux_x
    worker_rt_args.push_back(mux_virtual_core.y);     // fabric_mux_y
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_base_address(channel_type, worker_id));  // fabric_mux_channel_base_address
    worker_rt_args.push_back(
        mux_kernel_config.get_connection_info_address(channel_type, worker_id));  // fabric_mux_connection_info_address
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(
        channel_type, worker_id));  // fabric_mux_connection_handshake_address
    worker_rt_args.push_back(
        mux_kernel_config.get_flow_control_address(channel_type, worker_id));  // fabric_mux_flow_control_address
    worker_rt_args.push_back(
        mux_kernel_config.get_buffer_index_address(channel_type, worker_id));  // fabric_mux_buffer_index_address
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));  // fabric_mux_channel_id
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // termination_sync_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);                    // termination_master_noc_x
    worker_rt_args.push_back(termination_master_virtual_core.y);                    // termination_master_noc_y
}

AllGatherProgramArtifacts build_all_gather_async_minimal_default_program_artifacts_2(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device not found");

    // When reverse_order is enabled, tensor width must be divisible by 32*num_devices for proper sharding
    if (reverse_order) {
        uint32_t tensor_width = output_tensor_shape[3];
        uint32_t required_divisor = 32 * ring_size;
        TT_FATAL(
            tensor_width % required_divisor == 0,
            "When reverse_order=true, tensor width ({}) must be divisible by 32*num_devices (32*{} = {})",
            tensor_width,
            ring_size,
            required_divisor);
    }

    // op hyperparams
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    // Get worker cores
    // 2 senders (reader + writer) per direction (forward, reverse_order) per link
    uint32_t output_data_size_bytes = output_tensor.buffer()->size();
    uint32_t num_workers_per_direction = num_workers_per_direction_opt.value_or(detail::default_workers_2(
        *mesh_device,
        sub_device_id,
        topology,
        output_data_size_bytes,
        num_links,
        ring_size,
        num_directions_per_link,
        num_mux_cores_per_direction_per_link,
        sub_core_grid));
    uint32_t num_cores_per_link = detail::all_gather_async_core_count_per_link_2(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    bool fuse_op = fused_op_signaler.has_value();

    // Need a separate signaler for the sender workers, to handle the first tensor slice that is locally available
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;
    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, false);
    auto [unicast_forward_args, unicast_backward_args] = ttnn::ccl::get_forward_backward_line_unicast_configuration(
        topology, sender_device_coord, forward_coord, backward_coord, mesh_device);
    auto [barrier_mcast_forward_args, barrier_mcast_backward_args] =
        ttnn::ccl::get_forward_backward_line_mcast_configuration(
            topology,
            sender_device_coord,
            forward_coord,
            backward_coord,
            topology == ttnn::ccl::Topology::Linear ? num_targets_forward : ring_size - 1,
            topology == ttnn::ccl::Topology::Linear ? num_targets_backward : ring_size - 1,
            mesh_device);

    TT_FATAL(
        !((topology == ttnn::ccl::Topology::Linear) && fuse_op),
        "linear is not support when using fused for all-gather");
    const auto [all_core_range, all_cores] = ttnn::ccl::choose_worker_cores(
        num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset, sub_core_grid);

    std::vector<CoreRange> sender_worker_core_ranges;
    std::vector<CoreRange> mux_core_ranges;
    std::vector<CoreRange> termination_master_core_ranges;

    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    const auto mux_connection_valid = [&backward_coord, &forward_coord](const uint32_t dir) {
        return (dir && backward_coord.has_value()) || (!dir && forward_coord.has_value());
    };

    // collect cores
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];

            if (mux_connection_valid(dir)) {
                mux_core_ranges.emplace_back(mux_core);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];

                if (worker == 0) {
                    termination_master_core_ranges.emplace_back(worker_core);
                }

                if (dir) {
                    sender_forward_core_ranges.emplace(worker_core);
                } else {
                    sender_backward_core_ranges.emplace(worker_core);
                }
                sender_worker_core_ranges.emplace_back(worker_core);
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;

    // scatter-write currently supports 4 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 4;

    // for bfloat8_b, tile_num_per_link=6, we would need to send 2 packages, but they can be of size 3 instead of 4
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t cb_num_pages = 3 * num_tiles_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_sender_config);

    bool input_is_sharded = input_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        reader_compute_defines["OUTPUT_IS_SHARDED"] = "1";
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    // KERNEL CREATION
    /* All gather fusion */
    if (fuse_op) {
        auto sender_workers_forward = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
        auto sender_workers_backward = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);
        fused_op_signaler_forward->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
        fused_op_signaler_backward->init_all_gather(
            program, mesh_device, sender_backward_core_ranges, sender_workers_backward);
        fused_op_signaler_sender_workers->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
    }

    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;

    auto map_nd_to_4d = [&]() {
        // Here we do a couple of tricks so that the kernels can handle ND tensors
        // implicitly reshape lower dims so it is treated as 4D
        uint32_t batch_head_size =
            std::accumulate(input_tensor_shape.cbegin(), input_tensor_shape.cend() - 2, 1, std::multiplies<uint32_t>());

        auto [normalized_dim, rank_diff] = composite_common::normalize_dim_4d(dim, input_tensor_shape.rank());

        // if the gather dim is 4D normalized to 0,2,3 we can proceed as if nothing has changed
        // if not we have to roll up the lower dims from the gather dim up to 1 into C and gather on 1.
        uint32_t c_includes_dim;
        if (rank_diff >= 1 && dim <= rank_diff) {
            // gather dim to rank-3 accumulated into C
            c_includes_dim = dim;
            normalized_dim = 1;
        } else {
            // C will be 4D normalized dim 1
            c_includes_dim = 1 + rank_diff;
        }

        uint32_t input_tensor_C = std::accumulate(
            input_tensor_shape.view().rbegin() + 2,
            input_tensor_shape.view().rend() - c_includes_dim,
            1,
            std::multiplies<uint32_t>());

        uint32_t output_tensor_C = std::accumulate(
            output_tensor_shape.view().rbegin() + 2,
            output_tensor_shape.view().rend() - c_includes_dim,
            1,
            std::multiplies<uint32_t>());

        return std::make_tuple(normalized_dim, batch_head_size, input_tensor_C, output_tensor_C);
    };

    auto map_2d_to_4d = [&]() {
        const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, input_tensor_shape.rank()));
        constexpr uint32_t input_tensor_C = 1, output_tensor_C = 1, batch_head_size = 1;

        return std::make_tuple(normalized_dim, batch_head_size, input_tensor_C, output_tensor_C);
    };

    const auto [normalized_dim, batch_head_size, input_tensor_C, output_tensor_C] =
        (input_tensor_shape.rank() == 2) ? map_2d_to_4d() : map_nd_to_4d();

    uint32_t single_batch_head_num_pages = input_tensor_num_pages / batch_head_size;
    TT_FATAL(!(input_tensor_shape[-1] % TILE_WIDTH), "Input tensor width must be a multiple of TILE_WIDTH");
    TT_FATAL(!(output_tensor_shape[-1] % TILE_WIDTH), "Output tensor width must be a multiple of TILE_WIDTH");
    uint32_t TILE_WIDTH = 32;

    uint32_t input_tensor_Wt = input_tensor_shape[-1] / TILE_WIDTH;
    uint32_t input_tensor_Ht = input_tensor_shape[-2] / TILE_WIDTH;

    uint32_t output_tensor_Wt = output_tensor_shape[-1] / TILE_WIDTH;
    uint32_t output_tensor_Ht = output_tensor_shape[-2] / TILE_WIDTH;

    auto num_full_size_channels = num_workers_per_direction;
    auto num_header_only_channels = 0;
    size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // Create Reader Kernels
    std::vector<uint32_t> sender_reader_compile_args = {
        ring_size,                        // ring_size
        ring_index,                       // my_chip_id
        sender_cb_index,                  // cb_forward_id
        num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
        page_size,                        // page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        normalized_dim,                   // gather_dim
        batch_head_size,                  // input_batch_head_count (product of the first two dims)
        input_tensor_Wt,                  // input_tensor_Wt
        input_tensor_Ht,                  // input_tensor_Ht
        input_tensor_C,                   // input_tensor_C
        output_tensor_Wt,                 // output_tensor_Wt
        output_tensor_Ht,                 // output_tensor_Ht
        output_tensor_C,                  // output_tensor_C
        fuse_op,                          // fuse_op
        reverse_order,                    // reverse
    };
    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "minimal_default_reader.cpp",
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Create Writer kernels
    std::vector<uint32_t> sender_writer_compile_args = {
        ring_size,                        // ring_size
        ring_index,                       // my_chip_id
        sender_cb_index,                  // cb_output_id
        num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
        page_size,                        // page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        static_cast<uint32_t>(topology),  // topology
        normalized_dim,                   // gather_dim
        batch_head_size,                  // input_batch_head_count (product of the first two dims)
        input_tensor_Wt,                  // input_tensor_Wt
        input_tensor_Ht,                  // input_tensor_Ht
        input_tensor_C,                   // input_tensor_C
        output_tensor_Wt,                 // output_tensor_Wt
        output_tensor_Ht,                 // output_tensor_Ht
        output_tensor_C,                  // output_tensor_C
        fuse_op,                          // fuse_op
        reverse_order,                    // reverse
    };
    fabric_mux_connection_ct_args_2(
        num_workers_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        sender_writer_compile_args);

    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), barrier_mcast_forward_args.begin(), barrier_mcast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), barrier_mcast_backward_args.begin(), barrier_mcast_backward_args.end());

    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "minimal_default_writer.cpp",
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // create mux kernel
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            CoreCoord mux_virtual_core = {0, 0};
            if (mux_connection_valid(dir)) {
                auto mux_logical_core = *((mux_core_iter++)->begin());
                mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                if (dir) {  // forward
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                CoreCoord supplemental_core = all_cores
                    [(link * num_cores_per_link) +
                     ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction)) +
                     num_mux_cores_per_direction_per_link + worker];
                CoreCoord opposite_core_coord = mesh_device->worker_core_from_logical_core(supplemental_core);

                uint32_t global_worker_id = (link * num_workers_per_direction) + worker;
                uint32_t global_worker_count = num_links * num_workers_per_direction;
                uint32_t base_pages_per_worker = single_batch_head_num_pages / global_worker_count;
                uint32_t remainder = single_batch_head_num_pages % global_worker_count;
                uint32_t input_tile_id_start =
                    (global_worker_id * base_pages_per_worker) + std::min(global_worker_id, remainder);
                uint32_t input_tile_id_end =
                    ((global_worker_id + 1) * base_pages_per_worker) + std::min(global_worker_id + 1, remainder);

                // Heuristic is based on a sweep of large shapes. This will be used when total chunks per worker is
                // larger than 160. Doing it less frequently adds performance cost to many shapes. Sweep test:
                // tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_T3K.py
                constexpr uint32_t HEURISTIC_MAX_CHUNKS_PER_SYNC = 160;
                uint32_t chunks_per_sync_val = chunks_per_sync.value_or(std::min(
                    std::max((input_tile_id_end - input_tile_id_start) / num_tiles_to_write_per_packet, (uint32_t)1),
                    HEURISTIC_MAX_CHUNKS_PER_SYNC));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                uint32_t start_pages_read_in_row = input_tile_id_start % input_tensor_Wt;
                uint32_t start_row_offset = input_tile_id_start / input_tensor_Wt * output_tensor_Wt;

                uint32_t self_write_done_semaphore;
                if (fuse_op) {
                    self_write_done_semaphore = CreateSemaphore(program, {core}, 0);
                }

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),   // input_tensor_address
                    output_tensor.buffer()->address(),  // output_tensor_address
                    semaphore.at(dir).address(),        // out_ready_sem
                    dir,                                // direction RT ARG
                    input_tile_id_start,                // input_tile_id_start RT ARG
                    input_tile_id_end,                  // input_tile_id_end RT ARG
                    start_pages_read_in_row,            // start_pages_read_in_row RT ARG
                    start_row_offset,                   // start_row_offset RT ARG
                    chunks_per_sync_val,                // chunks_per_sync RT ARG
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    reader_rt_args.push_back(self_write_done_semaphore);
                    if (dir) {
                        fused_op_signaler_forward->push_all_gather_fused_op_rt_args(
                            reader_rt_args,
                            num_workers_per_direction * num_links,
                            worker + (link * num_workers_per_direction),
                            1);
                    } else {
                        fused_op_signaler_backward->push_all_gather_fused_op_rt_args(
                            reader_rt_args,
                            num_workers_per_direction * num_links,
                            worker + (link * num_workers_per_direction),
                            0);
                    }
                }
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                std::vector<uint32_t> writer_rt_args = {
                    output_tensor.buffer()->address(),                           // output_tensor_address
                    virtual_core.x,                                              // out_ready_sem_noc0_x
                    virtual_core.y,                                              // out_ready_sem_noc0_y
                    semaphore.at(dir).address(),                                 // out_ready_sem
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use synchronize barrier semaphore
                    barrier_semaphore.has_value()                                // synchronize barrier semaphore
                        ? barrier_semaphore.value().address()
                        : 0,
                    opposite_core_coord.x,    // opposite_core_sem_noc0_x
                    opposite_core_coord.y,    // opposite_core_sem_noc0_y
                    dir,                      // direction
                    input_tile_id_start,      // input_tile_id_start
                    input_tile_id_end,        // input_tile_id_end
                    start_pages_read_in_row,  // start_pages_read_in_row
                    start_row_offset,         // start_row_offset
                    chunks_per_sync_val};     // chunks_per_sync

                fabric_mux_connection_rt_args_2(
                    mux_connection_valid(dir),
                    worker == 0,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core,
                    worker,
                    core,
                    mux_kernel_config,
                    program,
                    termination_master_virtual_core,
                    writer_rt_args);
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }
                if (fuse_op) {
                    writer_rt_args.push_back(self_write_done_semaphore);
                    fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                        writer_rt_args,
                        num_workers_per_direction * num_links,
                        worker + (link * num_workers_per_direction),
                        1);
                }

                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);
            }
        }
    }

    // Return the program artifacts
    return {
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link};
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_default_helper_2(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset = CoreCoord(0, 0),
    const bool reverse_order = false,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt) {
    // Call the builder to create the program and get artifacts
    auto
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link] =
            build_all_gather_async_minimal_default_program_artifacts_2(
                program,
                input_tensor,
                sender_device_coord,
                forward_coord,
                backward_coord,
                output_tensor,
                dim,
                num_links,
                ring_size,
                ring_index,
                topology,
                semaphore,
                barrier_semaphore,
                using_persistent_buffers,
                sub_device_id,
                fused_op_signaler,
                chunks_per_sync,
                num_workers_per_direction_opt,
                num_buffers_per_channel,
                core_grid_offset,
                reverse_order,
                sub_core_grid);

    // Create the callback using the artifacts returned by the builder
    auto override_runtime_arguments_callback =
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_links,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto barrier_semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->barrier_semaphore;
            auto semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore;

            all_gather_async_minimal_default_helper_override_runtime_arguments(
                program,
                reader_kernel_ids,
                writer_kernel_ids,
                all_cores,
                num_links,
                num_directions_per_link,
                num_workers_per_direction,
                num_mux_cores_per_direction_per_link,
                num_cores_per_link,
                barrier_semaphore,
                semaphore,
                input,
                output);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn

namespace ttnn::operations::experimental::ccl::all_gather_matmul_async::program {

using Tensors = std::vector<Tensor>;

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
AllGatherMatmulAsyncMeshWorkloadFactory::cached_program_t AllGatherMatmulAsyncMeshWorkloadFactory::create_at(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    IDevice* target_device,
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<Tensor>& bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out

) {
    tt::tt_metal::Program program{};

    ////////////// Params for fused op signalers //////////////
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input_tensor, all_gather_output_tensor, dim, ring_index);
    bool is_clockwise_direction = true;
    const uint32_t num_transfers = 4;
    const uint32_t weight_tensor_width = weight_tensor.padded_shape()[3] / 32;

    ////////////////////////////////////////////////////////

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(ttnn::experimental::ccl::MatmulFusedOpSignalerType::ALL_GATHER);
    matmul_fused_op_signaler->init_all_gather(
        num_transfers,
        ring_size,
        ring_index,
        tensor_slicer.num_cols,
        tensor_slicer.output_page_offset,
        is_clockwise_direction,
        tensor_slicer.num_cols *
            weight_tensor_width /* weight_output_page_offset: stride across a tensor slice in the weight_tensor */
    );

    // Matmul
    std::optional<tt::tt_metal::operation::ProgramWithCallbacks>
        matmul_program_with_callbacks;  // TODO: migrate to not use the old program_with_callbacks in matmul old program
                                        // factories
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<Tensors>>
        matmul_override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                     // override_runtime_arguments_callback in matmul old program
                                                     // factories

    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_2d_optimized_helper(
                    program,
                    all_gather_output_tensor,
                    weight_tensor,
                    bias,
                    matmul_output_tensor,
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler);  // TODO: migrate to not use the old program_with_callbacks in matmul old
                                                // program factories
                matmul_override_runtime_arguments_callback =
                    matmul_program_with_callbacks
                        ->override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                                // override_runtime_arguments_callback in matmul old
                                                                // program factories
            } else if (std::is_same_v<
                           ProgramConfigType,
                           operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_1d_optimized_helper(
                    program,
                    all_gather_output_tensor,
                    {weight_tensor},
                    bias,
                    {matmul_output_tensor},
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler,
                    std::nullopt,
                    std::nullopt);  // TODO: migrate to not use the old program_with_callbacks in matmul old program
                                    // factories
                matmul_override_runtime_arguments_callback =
                    matmul_program_with_callbacks
                        ->override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                                // override_runtime_arguments_callback in matmul old
                                                                // program factories
            } else {
                TT_THROW("Unsupported MatmulProgramConfig type. Needs to be 1D or 2D Multicast.");
            }
        },
        program_config);

    if (!matmul_program_with_callbacks.has_value()) {
        TT_THROW("Matmul program with callbacks not created");  // TODO: migrate to not use the old
                                                                // matmul_program_with_callbacks in matmul old program
                                                                // factories
    }

    // Create the all gather fused op signaler
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    tt::tt_metal::operation::ProgramWithCallbacks
        program_with_callbacks =  // TODO: migrate to not use the old program_with_callbacks in all_gather_async old
                                  // program factories
        ttnn::all_gather_async_minimal_default_helper_2(
            matmul_program_with_callbacks->program,
            input_tensor,
            target_device_coord,
            forward_coord,
            backward_coord,
            all_gather_output_tensor,
            dim,
            num_links,
            ring_size,
            ring_index,
            topology,
            semaphore,
            barrier_semaphore,
            using_persistent_buffers,
            sub_device_id,
            all_gather_fused_op_signaler,
            chunks_per_sync,
            num_workers_per_direction_opt,
            num_buffers_per_channel,
            core_grid_offset,
            false);  // reverse_order = false by default
    const auto all_gather_override_runtime_arguments_callback =
        program_with_callbacks.override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                                     // override_runtime_arguments_callback in
                                                                     // all_gather_async old program factories

    return cached_program_t(
        {std::move(program_with_callbacks.program),
         shared_variables_t{
             .all_gather_override_runtime_arguments_callback =
                 all_gather_override_runtime_arguments_callback,  // TODO: migrate to not use the old
                                                                  // override_runtime_arguments_callback in matmul and
                                                                  // all_gather_async old program factories
             // TODO: migrate to not use the old
             // override_runtime_arguments_callback in matmul program factories
             .matmul_override_runtime_arguments_callback = matmul_override_runtime_arguments_callback}});
}

AllGatherMatmulAsyncMeshWorkloadFactory::cached_mesh_workload_t
AllGatherMatmulAsyncMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord : tensor_coords.coords()) {
        const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
        auto* mesh_device = tensor_args.input_tensor.device();
        IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coord) : tensor_args.input_tensor.device();

        uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coord, operation_attributes.all_gather_async_attributes.cluster_axis);

        std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coord,
            1,
            operation_attributes.all_gather_async_attributes.topology,
            operation_attributes.all_gather_async_attributes.cluster_axis);

        std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coord,
            -1,
            operation_attributes.all_gather_async_attributes.topology,
            operation_attributes.all_gather_async_attributes.cluster_axis);

        auto cached_program = create_at(
            tensor_args.input_tensor,
            tensor_return_value[0],
            tensor_args.weight_tensor,
            tensor_return_value[1],

            target_device,
            mesh_coord,
            forward_coord,
            backward_coord,
            operation_attributes.all_gather_async_attributes.dim,
            operation_attributes.all_gather_async_attributes.num_links,
            operation_attributes.all_gather_async_attributes.ring_size,
            device_index,
            operation_attributes.all_gather_async_attributes.topology,
            operation_attributes.all_gather_async_attributes.semaphore,
            operation_attributes.all_gather_async_attributes.barrier_semaphore,
            operation_attributes.all_gather_async_attributes.using_persistent_buffers,
            operation_attributes.all_gather_async_attributes.sub_device_id,
            operation_attributes.all_gather_async_attributes.chunks_per_sync,
            operation_attributes.all_gather_async_attributes.num_workers_per_link,
            operation_attributes.all_gather_async_attributes.num_buffers_per_channel,
            operation_attributes.all_gather_core_grid_offset,

            tensor_args.bias,
            operation_attributes.matmul.bcast_batch.value(),
            operation_attributes.matmul.compute_kernel_config.value(),
            operation_attributes.matmul.program_config.value(),
            operation_attributes.matmul.untilize_out);

        workload.add_program(single_coord_range, std::move(cached_program.program));
        shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void AllGatherMatmulAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Fuse the override runtime arguments callbacks
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        if (shared_vars.matmul_override_runtime_arguments_callback.has_value()) {
            shared_vars.matmul_override_runtime_arguments_callback
                .value()(  // TODO: migrate to not use the old override_runtime_arguments_callback in matmul old program
                           // factories
                    &operation_attributes,
                    program,
                    {tensor_return_value[0], tensor_args.weight_tensor}, /* all gather output tensor, weight tensor */
                    {tensor_args.bias},
                    {tensor_return_value[1]} /* matmul output tensor */
                );
        }

        if (shared_vars.all_gather_override_runtime_arguments_callback.has_value()) {
            shared_vars.all_gather_override_runtime_arguments_callback
                .value()(  // TODO: migrate to not use the old override_runtime_arguments_callback in all_gather_async
                           // old program factories
                    &operation_attributes,
                    program,
                    {tensor_args.input_tensor}, /* input tensor */
                    {tensor_args.bias},
                    {tensor_return_value[0]} /* all gather output tensor */
                );
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl::all_gather_matmul_async::program
