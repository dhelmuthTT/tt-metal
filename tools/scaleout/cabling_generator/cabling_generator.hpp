// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt_stl/strong_type.hpp>
#include <board/board.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::scaleout_tools::cabling_generator::proto {
class ClusterDescriptor;
class GraphTemplate;
class ChildInstance;
}  // namespace tt::scaleout_tools::cabling_generator::proto

namespace tt::scaleout_tools {
enum class NodeType;
struct MergeValidationResult;
}  // namespace tt::scaleout_tools

namespace tt::scaleout_tools::fsd::proto {
    class FactorySystemDescriptor;
}

namespace tt::scaleout_tools {

// Strong types to prevent mixing with other uint32_t values
using HostId = ttsl::StrongType<uint32_t, struct HostIdTag>;
using TrayId = ttsl::StrongType<uint32_t, struct TrayIdTag>;

struct Host {
    std::string hostname;
    std::string hall;
    std::string aisle;
    uint32_t rack = 0;
    uint32_t shelf_u = 0;
    std::string motherboard;
    std::string node_type;
};

struct LogicalChannelEndpoint {
    HostId host_id{0};
    TrayId tray_id{0};
    AsicChannel asic_channel;

    auto operator<=>(const LogicalChannelEndpoint& other) const = default;
};

struct PhysicalChannelEndpoint {
    std::string hostname;
    TrayId tray_id{0};
    AsicChannel asic_channel;

    auto operator<=>(const PhysicalChannelEndpoint& other) const = default;
};

struct PhysicalPortEndpoint {
    std::string hostname;
    std::string aisle;
    uint32_t rack = 0;
    uint32_t shelf_u = 0;
    TrayId tray_id{0};
    PortType port_type = PortType::TRACE;
    PortId port_id{0};

    auto operator<=>(const PhysicalPortEndpoint& other) const = default;
};

// Overload operator<< for readable test output
std::ostream& operator<<(std::ostream& os, const PhysicalChannelEndpoint& conn);
std::ostream& operator<<(std::ostream& os, const PhysicalPortEndpoint& conn);

using LogicalChannelConnection = std::pair<LogicalChannelEndpoint, LogicalChannelEndpoint>;
using PhysicalChannelConnection = std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint>;

struct Node {
    std::string motherboard;
    std::map<TrayId, Board> boards;
    HostId host_id{0};
    // Board-to-board connections within this node: PortType -> [(tray_id, port_id) <-> (tray_id, port_id)]
    using PortEndpoint = std::pair<TrayId, PortId>;
    using PortConnection = std::pair<PortEndpoint, PortEndpoint>;
    std::unordered_map<PortType, std::vector<PortConnection>> inter_board_connections;
};

// Port connection types for flat storage
using PortEndpoint = std::tuple<HostId, TrayId, PortId>;  // host_id, tray_id, port_id
using PortConnection = std::pair<PortEndpoint, PortEndpoint>;

enum class CableLength { CABLE_0P5, CABLE_1, CABLE_2P5, CABLE_3, CABLE_5, UNKNOWN };

CableLength calc_cable_length(
    const Host& host1, int tray_id1, const Host& host2, int tray_id2, const std::string& node_type);

class CablingGenerator {
public:
    // Constructor with full deployment descriptor (includes physical location info)
    // cluster_descriptor_path can be a single file or a directory containing multiple .textproto files
    CablingGenerator(const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path);

    // Constructor with just hostnames (no physical location info)
    // cluster_descriptor_path can be a single file or a directory containing multiple .textproto files
    CablingGenerator(const std::string& cluster_descriptor_path, const std::vector<std::string>& hostnames);

    CablingGenerator() = default;

    // Merge another CablingGenerator into this one
    // Validates host_id uniqueness and merges all structures
    // source_file is optional, used for error messages
    void merge(const CablingGenerator& other, const std::string& source_file = "");

    // Getters for all data
    const std::vector<Host>& get_deployment_hosts() const;
    const std::vector<LogicalChannelConnection>& get_chip_connections() const;

    // Method to emit factory system descriptor
    void emit_factory_system_descriptor(const std::string& output_path) const;

    // Method to generate factory system descriptor as protobuf object
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor generate_factory_system_descriptor() const;

    // Method to emit cabling guide CSV
    void emit_cabling_guide_csv(const std::string& output_path, bool loc_info = true) const;

    // Utility functions for directory and file handling
    static bool is_directory(const std::string& path);
    static std::vector<std::string> find_descriptor_files(const std::string& directory_path);

    // Validation functions for descriptor merging
    static MergeValidationResult validate_host_consistency(const std::vector<std::string>& descriptor_paths);
    static void validate_structure_identity(
        const cabling_generator::proto::ClusterDescriptor& desc1,
        const std::string& file1,
        const cabling_generator::proto::ClusterDescriptor& desc2,
        const std::string& file2,
        MergeValidationResult& result);

private:
    // Validate that each host_id is assigned to exactly one node
    void validate_host_id_uniqueness();

    // Collect all host_id assignments (flat structure, no paths needed)
    void collect_host_assignments(std::unordered_map<HostId, std::string>& host_to_node_path);

    // Utility function to generate logical chip connections from cluster hierarchy
    void generate_logical_chip_connections();

    void generate_connections_from_flat_structure();

    void populate_host_id_to_node();

    void get_all_connections_of_type(
        const std::vector<PortType>& port_types, std::vector<PortConnection>& conn_list) const;

private:
    // Helper functions for validation
    static std::set<uint32_t> extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor);
    static void validate_node_descriptors_identity(
        const cabling_generator::proto::ClusterDescriptor& desc1,
        const std::string& file1,
        const cabling_generator::proto::ClusterDescriptor& desc2,
        const std::string& file2,
        MergeValidationResult& result);
    static void validate_graph_template_children_identity(
        const cabling_generator::proto::GraphTemplate& tmpl1,
        const cabling_generator::proto::GraphTemplate& tmpl2,
        const std::string& template_name,
        const std::string& file1,
        const std::string& file2,
        MergeValidationResult& result);
    static void validate_child_identity(
        const cabling_generator::proto::ChildInstance& child1,
        const cabling_generator::proto::ChildInstance& child2,
        const std::string& template_name,
        const std::string& file1,
        const std::string& file2,
        MergeValidationResult& result);

    // Caches for optimization
    std::unordered_map<std::string, Node> node_templates_;  // Templates with host_id=0

    // Flat storage for all nodes (by name for merge validation, by host_id for lookup)
    std::map<std::string, Node> nodes_;        // All nodes in the cluster (flat)
    std::map<HostId, Node*> host_id_to_node_;  // Global lookup map for HostId -> Node reference

    // Flat storage for all port connections
    std::unordered_map<PortType, std::vector<PortConnection>> port_connections_;

    // Guaranteed to be sorted
    std::vector<LogicalChannelConnection> chip_connections_;
    std::vector<Host> deployment_hosts_;

    // Helper to add a connection and update lookup structures
    void add_port_connection(
        PortType port_type,
        const PortConnection& conn,
        std::map<PortEndpoint, PortEndpoint>& endpoint_to_dest,
        std::set<std::pair<PortEndpoint, PortEndpoint>>& connection_pairs);
};

}  // namespace tt::scaleout_tools

// Hash specializations
namespace std {
template <>
struct hash<tt::scaleout_tools::LogicalChannelEndpoint>;

template <>
struct hash<tt::scaleout_tools::PhysicalChannelEndpoint>;

template <>
struct hash<tt::scaleout_tools::PhysicalPortEndpoint>;

}  // namespace std
