// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cabling_generator.hpp"
#include "descriptor_merger.hpp"  // Still needed for MergeValidationResult
#include "protobuf_utils.hpp"

#include <board/board.hpp>
#include <connector/connector.hpp>
#include <node/node_types.hpp>
#include <node/node.hpp>  // For Topology enum

#include <algorithm>
#include <concepts>
#include <enchantum/enchantum.hpp>
#include <filesystem>
#include <fstream>
#include <fmt/base.h>
#include <type_traits>
#include <google/protobuf/text_format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>

// Add protobuf includes
#include "protobuf/cluster_config.pb.h"
#include "protobuf/deployment.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/node_config.pb.h"

namespace tt::scaleout_tools {

namespace {

// Find node descriptor by name - search inline first, then fallback to file
tt::scaleout_tools::cabling_generator::proto::NodeDescriptor find_node_descriptor(
    const std::string& node_descriptor_name,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor) {
    // First, search in inline node descriptors
    auto it = cluster_descriptor.node_descriptors().find(node_descriptor_name);
    if (it != cluster_descriptor.node_descriptors().end()) {
        return it->second;
    }

    auto node_type = get_node_type_from_string(node_descriptor_name);
    return create_node_descriptor(node_type);
}

void create_port_connection(
    Board& board_a,
    Board& board_b,
    PortType port_type,
    HostId host_a_id,
    HostId host_b_id,
    TrayId board_a_id,
    TrayId board_b_id,
    PortId port_a_id,
    PortId port_b_id) {
    const auto& available_a = board_a.get_available_port_ids(port_type);
    const auto& available_b = board_b.get_available_port_ids(port_type);

    if (std::find(available_a.begin(), available_a.end(), port_a_id) == available_a.end()) {
        throw std::runtime_error(fmt::format(
            "{} Port {} not available on board {} in host {}",
            enchantum::to_string(port_type),
            *port_a_id,
            *board_a_id,
            *host_a_id));
    }
    if (std::find(available_b.begin(), available_b.end(), port_b_id) == available_b.end()) {
        throw std::runtime_error(fmt::format(
            "{} Port {} not available on board {} in host {}",
            enchantum::to_string(port_type),
            *port_b_id,
            *board_b_id,
            *host_b_id));
    }

    if (board_a.get_arch() != board_b.get_arch()) {
        throw std::runtime_error("Trying to connect boards with different architectures");
    }

    board_a.mark_port_used(port_type, port_a_id);
    board_b.mark_port_used(port_type, port_b_id);
}
// Build node from descriptor with port connections and validation
Node build_node(
    const std::string& node_descriptor_name,
    HostId host_id,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    std::unordered_map<std::string, Node>& node_templates) {
    const std::string& node_type = node_descriptor_name;
    auto it = node_templates.find(node_type);
    if (it != node_templates.end()) {
        Node node = it->second;  // Copy template
        node.host_id = host_id;
        return node;
    }

    // Build new node template (with host_id=0)
    Node template_node;

    auto node_descriptor = find_node_descriptor(node_descriptor_name, cluster_descriptor);
    if (node_descriptor.motherboard().empty()) {
        throw std::runtime_error("Node descriptor " + node_descriptor_name + " missing motherboard");
    }
    template_node.motherboard = node_descriptor.motherboard();

    // Create boards with internal connections marked (using cached boards)
    for (const auto& board_item : node_descriptor.boards().board()) {
        TrayId tray_id = TrayId(board_item.tray_id());
        auto board_type = get_board_type_from_string(board_item.board_type());
        template_node.boards.emplace(tray_id, create_board(board_type));
    }

    // Add inter-board connections and validate/mark ports
    for (const auto& [port_type_str, port_connections] : node_descriptor.port_type_connections()) {
        auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
        if (!port_type.has_value()) {
            throw std::runtime_error("Invalid port type: " + port_type_str);
        }

        for (const auto& conn : port_connections.connections()) {
            TrayId board_a_id = TrayId(conn.port_a().tray_id());
            PortId port_a_id = PortId(conn.port_a().port_id());
            TrayId board_b_id = TrayId(conn.port_b().tray_id());
            PortId port_b_id = PortId(conn.port_b().port_id());

            create_port_connection(
                template_node.boards.at(board_a_id),
                template_node.boards.at(board_b_id),
                *port_type,
                host_id,
                host_id,
                board_a_id,
                board_b_id,
                port_a_id,
                port_b_id);

            // Store connection
            template_node.inter_board_connections[*port_type].emplace_back(
                std::make_pair(board_a_id, port_a_id), std::make_pair(board_b_id, port_b_id));
        }
    }

    // Cache the template
    node_templates[node_type] = template_node;

    // Create instance with actual host_id
    Node node = template_node;
    node.host_id = host_id;
    return node;
}

// Resolve path from proto data to HostId
HostId resolve_path_from_proto(
    const google::protobuf::RepeatedPtrField<std::string>& path,
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    uint32_t index = 0) {
    if (index == path.size() - 1) {
        // Direct node reference - look up in child_mappings
        const std::string& node_name = path[index];
        const auto& child_mapping = graph_instance.child_mappings().at(node_name);

        if (child_mapping.mapping_case() == tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
            return HostId(child_mapping.host_id());
        } else {
            throw std::runtime_error("Node " + node_name + " is not a leaf node");
        }
    } else {
        // Multi-level path - descend into subgraph
        const std::string& subgraph_name = path[index];
        const auto& child_mapping = graph_instance.child_mappings().at(subgraph_name);

        if (child_mapping.mapping_case() == tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
            return resolve_path_from_proto(path, child_mapping.sub_instance(), cluster_descriptor, index + 1);
        } else {
            throw std::runtime_error("Subgraph " + subgraph_name + " is not a graph instance");
        }
    }
}

// Builds a resolved graph instance from a graph instance and deployment descriptor.
// Recursively build flat structure from protobuf graph instance
// deployment_descriptor is optional - if nullptr, no validation is performed.
void build_graph_instance_impl(
    const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
    const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor* deployment_descriptor,
    const std::string& node_name_prefix,
    std::unordered_map<std::string, Node>& node_templates,
    std::map<std::string, Node>& nodes,
    std::unordered_map<PortType, std::vector<PortConnection>>& port_connections) {
    // Get the template definition
    const auto& template_def = cluster_descriptor.graph_templates().at(graph_instance.template_name());

    // Build children based on template + instance mapping
    for (const auto& child_def : template_def.children()) {
        const std::string& child_name = child_def.name();
        const auto& child_mapping = graph_instance.child_mappings().at(child_name);
        std::string full_node_name = node_name_prefix.empty() ? child_name : node_name_prefix + "/" + child_name;

        if (child_def.has_node_ref()) {
            // Leaf node - create node
            if (child_mapping.mapping_case() != tt::scaleout_tools::cabling_generator::proto::ChildMapping::kHostId) {
                throw std::runtime_error("Node child must have host_id mapping: " + child_name);
            }

            HostId host_id = HostId(child_mapping.host_id());
            const std::string& node_descriptor_name = child_def.node_ref().node_descriptor();

            // Validate deployment node type if deployment descriptor is provided
            if (deployment_descriptor != nullptr) {
                if (*host_id < deployment_descriptor->hosts().size()) {
                    const auto& deployment_host = deployment_descriptor->hosts()[*host_id];
                    if (!deployment_host.node_type().empty() && deployment_host.node_type() != node_descriptor_name) {
                        throw std::runtime_error(fmt::format(
                            "Node type mismatch for host {} (host_id {}): deployment specifies '{}' but cluster "
                            "configuration expects '{}'",
                            deployment_host.host(),
                            *host_id,
                            deployment_host.node_type(),
                            node_descriptor_name));
                    }
                } else {
                    throw std::runtime_error(fmt::format("Host ID {} not found in deployment", *host_id));
                }
            }

            // Find node descriptor and build node, store in flat structure
            nodes[full_node_name] = build_node(node_descriptor_name, host_id, cluster_descriptor, node_templates);

        } else if (child_def.has_graph_ref()) {
            // Non-leaf node - recursively process subgraph
            if (child_mapping.mapping_case() !=
                tt::scaleout_tools::cabling_generator::proto::ChildMapping::kSubInstance) {
                throw std::runtime_error("Graph child must have sub_instance mapping: " + child_name);
            }

            build_graph_instance_impl(
                child_mapping.sub_instance(),
                cluster_descriptor,
                deployment_descriptor,
                full_node_name,
                node_templates,
                nodes,
                port_connections);
        }
    }

    // Process internal connections within this graph instance
    for (const auto& [port_type_str, port_connections_proto] : template_def.internal_connections()) {
        auto port_type = enchantum::cast<PortType>(port_type_str, ttsl::ascii_caseless_comp);
        if (!port_type.has_value()) {
            throw std::runtime_error("Invalid port type: " + port_type_str);
        }

        for (const auto& conn : port_connections_proto.connections()) {
            const auto& path_a = conn.port_a().path();
            const auto& path_b = conn.port_b().path();
            TrayId board_a_id = TrayId(conn.port_a().tray_id());
            PortId port_a_id = PortId(conn.port_a().port_id());

            TrayId board_b_id = TrayId(conn.port_b().tray_id());
            PortId port_b_id = PortId(conn.port_b().port_id());

            // Resolve paths to HostId using proto data
            HostId host_a_id = resolve_path_from_proto(path_a, graph_instance, cluster_descriptor);
            HostId host_b_id = resolve_path_from_proto(path_b, graph_instance, cluster_descriptor);

            // Store connection with resolved HostId
            PortConnection port_conn(
                std::make_tuple(host_a_id, board_a_id, port_a_id), std::make_tuple(host_b_id, board_b_id, port_b_id));

            // Add to flat structure
            port_connections[*port_type].push_back(port_conn);
        }
    }
}

void populate_deployment_hosts(
    const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
    const std::unordered_map<std::string, Node>& node_templates,
    std::vector<Host>& deployment_hosts) {
    // Store deployment hosts
    deployment_hosts.reserve(deployment_descriptor.hosts().size());
    for (const auto& proto_host : deployment_descriptor.hosts()) {
        deployment_hosts.emplace_back(Host{
            .hostname = proto_host.host(),
            .hall = proto_host.hall(),
            .aisle = proto_host.aisle(),
            .rack = proto_host.rack(),
            .shelf_u = proto_host.shelf_u(),
            .motherboard = node_templates.at(proto_host.node_type()).motherboard,
            .node_type = proto_host.node_type()});
    }
}

void populate_deployment_hosts_from_hostnames(
    const std::vector<std::string>& hostnames,
    const std::map<HostId, Node*>& host_id_to_node,
    std::vector<Host>& deployment_hosts) {
    // Store deployment hosts with just hostname and motherboard (no physical location info)
    deployment_hosts.reserve(hostnames.size());
    for (size_t i = 0; i < hostnames.size(); ++i) {
        HostId host_id = HostId(i);
        auto it = host_id_to_node.find(host_id);
        if (it == host_id_to_node.end()) {
            throw std::runtime_error(fmt::format("Host ID {} not found in cluster configuration", i));
        }
        deployment_hosts.emplace_back(Host{
            .hostname = hostnames[i],
            .hall = "",
            .aisle = "",
            .rack = 0,
            .shelf_u = 0,
            .motherboard = it->second->motherboard});
    }
}

// Helper to build from directory by merging multiple files
// Note: Each file is built via the constructor (which calls build_graph_instance_impl),
// then merged into the accumulated result. This ensures proper validation and processing
// of each file before merging.
template <typename DeploymentArg>
static CablingGenerator build_from_directory(const std::string& dir_path, const DeploymentArg& deployment_arg) {
    auto descriptor_files = CablingGenerator::find_descriptor_files(dir_path);
    if (descriptor_files.empty()) {
        throw std::runtime_error("No .textproto files found in directory: " + dir_path);
    }

    log_info(
        tt::LogDistributed, "Found {} cabling descriptor files in directory: {}", descriptor_files.size(), dir_path);
    for (const auto& file : descriptor_files) {
        log_info(tt::LogDistributed, "  - {}", file);
    }

    CablingGenerator merged;
    for (const auto& file : descriptor_files) {
        CablingGenerator other(file, deployment_arg);
        merged.merge(other, file);
    }
    return merged;
}

}  // anonymous namespace

std::vector<std::string> CablingGenerator::find_descriptor_files(const std::string& directory_path) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".textproto") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Error reading directory " + directory_path + ": " + e.what());
    }

    // Sorting files so we dont have undeterministic order of files in the directory so it's easier to debug errors.
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        throw std::runtime_error("No .textproto files found in directory: " + directory_path);
    }
    return files;
}

// Validation helper functions
std::set<uint32_t> CablingGenerator::extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor) {
    std::set<uint32_t> host_ids;

    const auto extract_from_instance = [&](const auto& instance, auto& self) -> void {
        for (const auto& [child_name, mapping] : instance.child_mappings()) {
            if (mapping.has_host_id()) {
                host_ids.insert(mapping.host_id());
            } else if (mapping.has_sub_instance()) {
                self(mapping.sub_instance(), self);
            }
        }
    };

    if (descriptor.has_root_instance()) {
        extract_from_instance(descriptor.root_instance(), extract_from_instance);
    }
    return host_ids;
}

MergeValidationResult CablingGenerator::validate_host_consistency(const std::vector<std::string>& descriptor_paths) {
    MergeValidationResult result;
    if (descriptor_paths.size() < 2) {
        return result;
    }

    std::optional<size_t> first_host_count;

    for (const auto& path : descriptor_paths) {
        const auto host_ids = extract_host_ids(load_cluster_descriptor(path));
        if (host_ids.empty()) {
            continue;
        }

        // Calculate host count from highest host_id (assuming 0-based indexing)
        const size_t count = *host_ids.rbegin() + 1;
        if (!first_host_count) {
            first_host_count = count;
        } else if (count != *first_host_count) {
            result.add_warning(fmt::format(
                "Host count mismatch between descriptors: {} vs {} hosts (file: {})", *first_host_count, count, path));
            break;
        }
    }
    return result;
}

void CablingGenerator::validate_node_descriptors_identity(
    const cabling_generator::proto::ClusterDescriptor& desc1,
    const std::string& file1,
    const cabling_generator::proto::ClusterDescriptor& desc2,
    const std::string& file2,
    MergeValidationResult& result) {
    for (const auto& [name, node_desc1] : desc1.node_descriptors()) {
        if (desc2.node_descriptors().contains(name)) {
            const auto& node_desc2 = desc2.node_descriptors().at(name);
            std::string serialized1, serialized2;
            node_desc1.SerializeToString(&serialized1);
            node_desc2.SerializeToString(&serialized2);
            if (serialized1 != serialized2) {
                std::ostringstream oss;
                oss << "Node descriptor '" << name << "' differs between " << file1 << " and " << file2
                    << ". Node descriptors must be identical across files.";
                result.add_error(oss.str());
            }
        }
    }
}

void CablingGenerator::validate_child_identity(
    const cabling_generator::proto::ChildInstance& child1,
    const cabling_generator::proto::ChildInstance& child2,
    const std::string& template_name,
    const std::string& file1,
    const std::string& file2,
    MergeValidationResult& result) {
    if (child1.has_node_ref() != child2.has_node_ref()) {
        std::ostringstream oss;
        oss << "Graph template '" << template_name << "' child '" << child1.name()
            << "' has different reference types between " << file1 << " and " << file2
            << ". Children must be identical across files.";
        result.add_error(oss.str());
        return;
    }

    if (child1.has_node_ref()) {
        const std::string desc1 = child1.node_ref().node_descriptor();
        const std::string desc2 = child2.node_ref().node_descriptor();
        if (desc1 != desc2) {
            // Special case: allow torus variants to differ
            // XY_TORUS is a superset, so it can be combined with X_TORUS or Y_TORUS
            auto node_type1 = enchantum::cast<NodeType>(desc1, ttsl::ascii_caseless_comp);
            auto node_type2 = enchantum::cast<NodeType>(desc2, ttsl::ascii_caseless_comp);
            const bool is_torus_variant = node_type1.has_value() && node_type2.has_value() &&
                                          tt::scaleout_tools::is_torus_compatible(*node_type1, *node_type2);
            if (!is_torus_variant) {
                std::ostringstream oss;
                oss << "Graph template '" << template_name << "' child '" << child1.name()
                    << "' has different node_descriptor between " << file1 << " ('" << desc1 << "') and " << file2
                    << " ('" << desc2
                    << "'). Only torus variants (X_TORUS, Y_TORUS, XY_TORUS) differences are allowed.";
                result.add_error(oss.str());
            }
        }
    } else if (child1.has_graph_ref()) {
        if (child1.graph_ref().graph_template() != child2.graph_ref().graph_template()) {
            std::ostringstream oss;
            oss << "Graph template '" << template_name << "' child '" << child1.name()
                << "' has different graph_template reference between " << file1 << " ('"
                << child1.graph_ref().graph_template() << "') and " << file2 << " ('"
                << child2.graph_ref().graph_template() << "'). Children must be identical across files.";
            result.add_error(oss.str());
        }
    }
}

void CablingGenerator::validate_graph_template_children_identity(
    const cabling_generator::proto::GraphTemplate& tmpl1,
    const cabling_generator::proto::GraphTemplate& tmpl2,
    const std::string& template_name,
    const std::string& file1,
    const std::string& file2,
    MergeValidationResult& result) {
    if (tmpl1.children_size() != tmpl2.children_size()) {
        std::ostringstream oss;
        oss << "Graph template '" << template_name << "' has different number of children between " << file1 << " ("
            << tmpl1.children_size() << ") and " << file2 << " (" << tmpl2.children_size()
            << "). Children must be identical across files.";
        result.add_error(oss.str());
        return;
    }

    std::map<std::string, int> children1_indices;
    for (int i = 0; i < tmpl1.children_size(); ++i) {
        const auto& child = tmpl1.children(i);
        if (children1_indices.contains(child.name())) {
            std::ostringstream oss;
            oss << "Graph template '" << template_name << "' has duplicate child name '" << child.name() << "' in "
                << file1 << ". Child names must be unique.";
            result.add_error(oss.str());
            continue;
        }
        children1_indices[child.name()] = i;
    }

    for (int i = 0; i < tmpl2.children_size(); ++i) {
        const auto& child2 = tmpl2.children(i);
        const auto it = children1_indices.find(child2.name());
        if (it == children1_indices.end()) {
            std::ostringstream oss;
            oss << "Graph template '" << template_name << "' has child '" << child2.name() << "' in " << file2
                << " but not in " << file1 << ". Children must be identical across files.";
            result.add_error(oss.str());
            continue;
        }

        const auto& child1 = tmpl1.children(it->second);
        validate_child_identity(child1, child2, template_name, file1, file2, result);
    }
}

void CablingGenerator::validate_structure_identity(
    const cabling_generator::proto::ClusterDescriptor& desc1,
    const std::string& file1,
    const cabling_generator::proto::ClusterDescriptor& desc2,
    const std::string& file2,
    MergeValidationResult& result) {
    validate_node_descriptors_identity(desc1, file1, desc2, file2, result);

    for (const auto& [template_name, tmpl1] : desc1.graph_templates()) {
        if (!desc2.graph_templates().contains(template_name)) {
            continue;
        }

        const auto& tmpl2 = desc2.graph_templates().at(template_name);
        const bool has_children1 = tmpl1.children_size() > 0;
        const bool has_children2 = tmpl2.children_size() > 0;

        if (has_children1 && has_children2) {
            validate_graph_template_children_identity(tmpl1, tmpl2, template_name, file1, file2, result);
        }
    }
}

// Constructor with full deployment descriptor
// cluster_descriptor_path can be a single file or a directory
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path) {
    if (!std::filesystem::exists(cluster_descriptor_path)) {
        throw std::runtime_error("Path does not exist: " + cluster_descriptor_path);
    }
    if (std::filesystem::is_directory(cluster_descriptor_path)) {
        auto merged = build_from_directory(cluster_descriptor_path, deployment_descriptor_path);
        node_templates_ = std::move(merged.node_templates_);
        nodes_ = std::move(merged.nodes_);
        port_connections_ = std::move(merged.port_connections_);
        host_id_to_node_ = std::move(merged.host_id_to_node_);
        chip_connections_ = std::move(merged.chip_connections_);
        deployment_hosts_ = std::move(merged.deployment_hosts_);
    } else {
        auto cluster_descriptor = load_cluster_descriptor(cluster_descriptor_path);
        auto deployment_descriptor =
            load_descriptor_from_textproto<tt::scaleout_tools::deployment::proto::DeploymentDescriptor>(
                deployment_descriptor_path);
        build_graph_instance_impl(
            cluster_descriptor.root_instance(),
            cluster_descriptor,
            &deployment_descriptor,
            "",
            node_templates_,
            nodes_,
            port_connections_);
        validate_host_id_uniqueness();
        populate_host_id_to_node();
        generate_logical_chip_connections();
        populate_deployment_hosts(deployment_descriptor, node_templates_, deployment_hosts_);
    }
}

// Constructor with just hostnames (no physical location info)
// cluster_descriptor_path can be a single file or a directory
CablingGenerator::CablingGenerator(
    const std::string& cluster_descriptor_path, const std::vector<std::string>& hostnames) {
    if (!std::filesystem::exists(cluster_descriptor_path)) {
        throw std::runtime_error("Path does not exist: " + cluster_descriptor_path);
    }
    if (std::filesystem::is_directory(cluster_descriptor_path)) {
        auto merged = build_from_directory(cluster_descriptor_path, hostnames);
        node_templates_ = std::move(merged.node_templates_);
        nodes_ = std::move(merged.nodes_);
        port_connections_ = std::move(merged.port_connections_);
        host_id_to_node_ = std::move(merged.host_id_to_node_);
        chip_connections_ = std::move(merged.chip_connections_);
        deployment_hosts_ = std::move(merged.deployment_hosts_);
    } else {
        auto cluster_descriptor = load_cluster_descriptor(cluster_descriptor_path);
        build_graph_instance_impl(
            cluster_descriptor.root_instance(),
            cluster_descriptor,
            nullptr,
            "",
            node_templates_,
            nodes_,
            port_connections_);
        validate_host_id_uniqueness();
        populate_host_id_to_node();
        generate_logical_chip_connections();
        populate_deployment_hosts_from_hostnames(hostnames, host_id_to_node_, deployment_hosts_);
    }
}

// Helper to add a connection and update lookup structures
void CablingGenerator::add_port_connection(
    PortType port_type,
    const PortConnection& conn,
    std::map<PortEndpoint, PortEndpoint>& endpoint_to_dest,
    std::set<std::pair<PortEndpoint, PortEndpoint>>& connection_pairs) {
    port_connections_[port_type].push_back(conn);
    endpoint_to_dest[conn.first] = conn.second;
    endpoint_to_dest[conn.second] = conn.first;
    // Normalize for duplicate detection (smaller endpoint first)
    auto normalized =
        (conn.first < conn.second) ? std::make_pair(conn.first, conn.second) : std::make_pair(conn.second, conn.first);
    connection_pairs.insert(normalized);
}

// Helper to rebuild lookup structures from existing port_connections_
static void rebuild_lookup_structures(
    const std::unordered_map<PortType, std::vector<PortConnection>>& port_connections,
    std::map<PortEndpoint, PortEndpoint>& endpoint_to_dest,
    std::set<std::pair<PortEndpoint, PortEndpoint>>& connection_pairs) {
    endpoint_to_dest.clear();
    connection_pairs.clear();
    for (const auto& [port_type, connections] : port_connections) {
        for (const auto& conn : connections) {
            endpoint_to_dest[conn.first] = conn.second;
            endpoint_to_dest[conn.second] = conn.first;
            auto normalized = (conn.first < conn.second) ? std::make_pair(conn.first, conn.second)
                                                         : std::make_pair(conn.second, conn.first);
            connection_pairs.insert(normalized);
        }
    }
}

void CablingGenerator::merge(const CablingGenerator& other, const std::string& source_file) {
    // Validate node_templates_ are identical (must match exactly)
    for (const auto& [name, other_template] : other.node_templates_) {
        if (node_templates_.count(name)) {
            const auto& this_template = node_templates_.at(name);
            if (this_template.motherboard != other_template.motherboard) {
                throw std::runtime_error(fmt::format(
                    "Node template '{}' has conflicting motherboard: '{}' vs '{}' from {}",
                    name,
                    this_template.motherboard,
                    other_template.motherboard,
                    (source_file.empty() ? "merged descriptor" : source_file)));
            }
            if (this_template.boards.size() != other_template.boards.size()) {
                throw std::runtime_error(fmt::format(
                    "Node template '{}' has conflicting board count: {} vs {} from {}",
                    name,
                    this_template.boards.size(),
                    other_template.boards.size(),
                    (source_file.empty() ? "merged descriptor" : source_file)));
            }
            for (const auto& [tray_id, this_board] : this_template.boards) {
                if (!other_template.boards.count(tray_id)) {
                    throw std::runtime_error(fmt::format(
                        "Node template '{}' missing board at tray_id {} from {}",
                        name,
                        *tray_id,
                        (source_file.empty() ? "merged descriptor" : source_file)));
                }
                const auto& other_board = other_template.boards.at(tray_id);
                if (this_board.get_arch() != other_board.get_arch()) {
                    throw std::runtime_error(fmt::format(
                        "Node template '{}' board at tray_id {} has conflicting architecture from {}",
                        name,
                        *tray_id,
                        (source_file.empty() ? "merged descriptor" : source_file)));
                }
            }
            // inter_board_connections can differ (templates are just for structure)
        } else {
            // New template - add it
            node_templates_[name] = other_template;
        }
    }

    // Merge nodes - if same name exists, validate they're identical except for inter_board_connections
    for (const auto& [name, source_node] : other.nodes_) {
        if (nodes_.count(name)) {
            const auto& target_node = nodes_[name];
            // Validate host_id matches
            if (target_node.host_id != source_node.host_id) {
                throw std::runtime_error(fmt::format(
                    "Node '{}' has conflicting host_id: {} vs {} from {}",
                    name,
                    target_node.host_id.get(),
                    source_node.host_id.get(),
                    (source_file.empty() ? "merged descriptor" : source_file)));
            }
            // Validate motherboard matches
            if (target_node.motherboard != source_node.motherboard) {
                throw std::runtime_error(fmt::format(
                    "Node '{}' has conflicting motherboard: '{}' vs '{}' from {}",
                    name,
                    target_node.motherboard,
                    source_node.motherboard,
                    (source_file.empty() ? "merged descriptor" : source_file)));
            }
            // Validate boards match exactly
            if (target_node.boards.size() != source_node.boards.size()) {
                throw std::runtime_error(fmt::format(
                    "Node '{}' has conflicting board count: {} vs {} from {}",
                    name,
                    target_node.boards.size(),
                    source_node.boards.size(),
                    (source_file.empty() ? "merged descriptor" : source_file)));
            }
            for (const auto& [tray_id, target_board] : target_node.boards) {
                if (!source_node.boards.count(tray_id)) {
                    throw std::runtime_error(fmt::format(
                        "Node '{}' missing board at tray_id {} from {}",
                        name,
                        *tray_id,
                        (source_file.empty() ? "merged descriptor" : source_file)));
                }
                const auto& source_board = source_node.boards.at(tray_id);
                if (target_board.get_arch() != source_board.get_arch()) {
                    throw std::runtime_error(fmt::format(
                        "Node '{}' board at tray_id {} has conflicting architecture from {}",
                        name,
                        *tray_id,
                        (source_file.empty() ? "merged descriptor" : source_file)));
                }
            }
            // Only inter_board_connections can differ - merge them (append)
            for (const auto& [port_type, connections] : source_node.inter_board_connections) {
                auto& target_conns = nodes_[name].inter_board_connections[port_type];
                target_conns.insert(target_conns.end(), connections.begin(), connections.end());
            }
        } else {
            // New node - add it
            nodes_[name] = source_node;
        }
    }

    // Rebuild lookup structures from existing connections for conflict detection
    std::map<PortEndpoint, PortEndpoint> endpoint_to_dest;
    std::set<std::pair<PortEndpoint, PortEndpoint>> connection_pairs;
    rebuild_lookup_structures(port_connections_, endpoint_to_dest, connection_pairs);

    // Merge port connections
    for (const auto& [port_type, source_conns] : other.port_connections_) {
        for (const auto& conn : source_conns) {
            auto normalized = (conn.first < conn.second) ? std::make_pair(conn.first, conn.second)
                                                         : std::make_pair(conn.second, conn.first);

            if (connection_pairs.contains(normalized)) {
                // Duplicate - warn but allow
                log_warning(
                    tt::LogDistributed,
                    "Duplicate connection from {}",
                    (source_file.empty() ? "merged descriptor" : source_file));
            } else {
                // Check for conflicts (same endpoint, different destination)
                auto it_a = endpoint_to_dest.find(conn.first);
                if (it_a != endpoint_to_dest.end() && it_a->second != conn.second) {
                    throw std::runtime_error(
                        "Connection conflict from " + (source_file.empty() ? "merged descriptor" : source_file));
                }
                auto it_b = endpoint_to_dest.find(conn.second);
                if (it_b != endpoint_to_dest.end() && it_b->second != conn.first) {
                    throw std::runtime_error(
                        "Connection conflict from " + (source_file.empty() ? "merged descriptor" : source_file));
                }

                // No conflict, add connection (this updates lookup structures automatically)
                add_port_connection(port_type, conn, endpoint_to_dest, connection_pairs);
            }
        }
    }

    // Rebuild host_id_to_node_ from merged nodes
    populate_host_id_to_node();

    // Re-validate host_id uniqueness
    validate_host_id_uniqueness();

    // Regenerate chip_connections_ from merged flat structure
    generate_logical_chip_connections();

    // Merge deployment_hosts_ (append unique hostnames)
    std::set<std::string> existing_hostnames;
    for (const auto& host : deployment_hosts_) {
        existing_hostnames.insert(host.hostname);
    }
    for (const auto& other_host : other.deployment_hosts_) {
        if (!existing_hostnames.count(other_host.hostname)) {
            deployment_hosts_.push_back(other_host);
            existing_hostnames.insert(other_host.hostname);
        }
    }
}

// Getters for all data
const std::vector<Host>& CablingGenerator::get_deployment_hosts() const { return deployment_hosts_; }

const std::vector<LogicalChannelConnection>& CablingGenerator::get_chip_connections() const {
    return chip_connections_;
}

// Helper function to build factory system descriptor protobuf (shared between emit and generate_string methods)
static tt::scaleout_tools::fsd::proto::FactorySystemDescriptor build_factory_system_descriptor(
    const std::vector<Host>& deployment_hosts,
    const std::map<HostId, Node*>& host_id_to_node,
    const std::vector<LogicalChannelConnection>& chip_connections) {
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd;

    // Add host information from deployment hosts (indexed by host_id)
    for (size_t i = 0; i < deployment_hosts.size(); ++i) {
        const auto& deployment_host = deployment_hosts[i];
        auto* host = fsd.add_hosts();
        host->set_hostname(deployment_host.hostname);
        host->set_hall(deployment_host.hall);
        host->set_aisle(deployment_host.aisle);
        host->set_rack(deployment_host.rack);
        host->set_shelf_u(deployment_host.shelf_u);
        host->set_motherboard(deployment_host.motherboard);
    }

    // Add board types
    for (const auto& [host_id, node] : host_id_to_node) {
        for (const auto& [tray_id, board] : node->boards) {
            auto* board_location = fsd.mutable_board_types()->add_board_locations();
            board_location->set_host_id(*host_id);
            board_location->set_tray_id(*tray_id);
            board_location->set_board_type(enchantum::to_string(board.get_board_type()).data());
        }
    }

    // Add ASIC connections from chip_connections
    for (const auto& [start, end] : chip_connections) {
        auto* connection = fsd.mutable_eth_connections()->add_connection();

        auto* endpoint_a = connection->mutable_endpoint_a();
        endpoint_a->set_host_id(*start.host_id);
        endpoint_a->set_tray_id(*start.tray_id);
        endpoint_a->set_asic_location(start.asic_channel.asic_location);
        endpoint_a->set_chan_id(*start.asic_channel.channel_id);

        auto* endpoint_b = connection->mutable_endpoint_b();
        endpoint_b->set_host_id(*end.host_id);
        endpoint_b->set_tray_id(*end.tray_id);
        endpoint_b->set_asic_location(end.asic_channel.asic_location);
        endpoint_b->set_chan_id(*end.asic_channel.channel_id);
    }

    return fsd;
}

// Method to emit textproto factory system descriptor
void CablingGenerator::emit_factory_system_descriptor(const std::string& output_path) const {
    auto fsd = build_factory_system_descriptor(deployment_hosts_, host_id_to_node_, chip_connections_);

    // Create parent directory if it doesn't exist
    std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    std::string output_string;
    google::protobuf::TextFormat::Printer printer;
    printer.SetUseShortRepeatedPrimitives(true);
    printer.SetUseUtf8StringEscaping(true);
    printer.SetSingleLineMode(false);
    printer.SetPrintMessageFieldsInIndexOrder(true);

    if (!printer.PrintToString(fsd, &output_string)) {
        throw std::runtime_error("Failed to write textproto to file: " + output_path);
    }

    output_file << output_string;
    output_file.close();
}

// Method to generate factory system descriptor as protobuf object (uses shared helper)
tt::scaleout_tools::fsd::proto::FactorySystemDescriptor CablingGenerator::generate_factory_system_descriptor() const {
    return build_factory_system_descriptor(deployment_hosts_, host_id_to_node_, chip_connections_);
}

void CablingGenerator::emit_cabling_guide_csv(const std::string& output_path, bool loc_info) const {
    // Create parent directory if it doesn't exist
    std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    const std::unordered_map<CableLength, std::string> cable_length_str = {
        {CableLength::CABLE_0P5, "0.5m"},
        {CableLength::CABLE_1, "1m"},
        {CableLength::CABLE_2P5, "2.5m"},
        {CableLength::CABLE_3, "3m"},
        {CableLength::CABLE_5, "5m"},
        {CableLength::UNKNOWN, "UNKNOWN"}};

    const std::unordered_map<tt::ARCH, std::string> speed_str = {
        //TODO: BLACKHOLE cable speed 200G in early stages/validation, but should be able to support 800G in the future.
        {tt::ARCH::WORMHOLE_B0, "400G"}, {tt::ARCH::BLACKHOLE, "400G"}, {tt::ARCH::Invalid, "UNKNOWN"}};

    // Unknown for lengths unable to be calculated (longer than avaiable cables, cross-aisle/hall, etc.)

    // Vector of (Host,Tray,Port) Connection Pairs
    std::vector<std::pair<std::tuple<HostId, TrayId, PortId>, std::tuple<HostId, TrayId, PortId>>> conn_list;

    get_all_connections_of_type({PortType::QSFP_DD}, conn_list);
    output_file.fill('0');
    if (loc_info) {
        output_file << "Source,,,,,,,,,Destination,,,,,,,,,Cable Length,Cable Type" << std::endl;
        output_file << "Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type,Hostname,Hall,Aisle,Rack,Shelf "
                       "U,Tray,Port,Label,Node Type,,"
                    << std::endl;
    } else {
        output_file << "Source,,,,Destination,,," << std::endl;
        output_file << "Hostname,Tray,Port,Node Type,Hostname,Tray,Port,Node Type" << std::endl;
    }
    for (const auto& [start, end] : conn_list) {
        auto host_id1 = std::get<0>(start).get();
        auto tray_id1 = std::get<1>(start).get();
        auto port_id1 = std::get<2>(start).get();

        auto host_id2 = std::get<0>(end).get();
        auto tray_id2 = std::get<1>(end).get();
        auto port_id2 = std::get<2>(end).get();

        const auto& host1 = deployment_hosts_[host_id1];
        const auto& host2 = deployment_hosts_[host_id2];

        // Create node_type strings with "_DEFAULT" suffix removed if present;
        //  all the default connections are enumerated
        const std::string suffix = "_DEFAULT";
        std::string host1_node_type = host1.node_type;
        if (host1_node_type.ends_with(suffix)) {
            host1_node_type = host1_node_type.substr(0, host1_node_type.size() - suffix.size());
        }
        std::string host2_node_type = host2.node_type;
        if (host2_node_type.ends_with(suffix)) {
            host2_node_type = host2_node_type.substr(0, host2_node_type.size() - suffix.size());
        }

        // Get arch from node
        // Assume arch for start and end are the same
        // This is validated in create_port_connection

        // TODO: Determine better heuristic/specification for cable length and type
        // auto arch = host_id_to_node_.at(std::get<0>(start))->boards.at(std::get<1>(start)).get_arch();
        // CableLength cable_l = calc_cable_length(host1, tray_id1, host2, tray_id2, host1_node_type);

        if (loc_info) {
            output_file << host1.hostname << ",";
            output_file << host1.hall << "," << host1.aisle << "," << std::setw(2) << host1.rack << ",U" << std::setw(2)
                        << host1.shelf_u << "," << tray_id1 << "," << port_id1 << ",";

            output_file << host1.hall << host1.aisle << std::setw(2) << host1.rack << "U" << std::setw(2)
                        << host1.shelf_u << "-" << tray_id1 << "-" << port_id1 << "," << host1_node_type << ",";

            output_file << host2.hostname << ",";
            output_file << host2.hall << "," << host2.aisle << "," << std::setw(2) << host2.rack << ",U" << std::setw(2)
                        << host2.shelf_u << "," << tray_id2 << "," << port_id2 << ",";
            output_file << host2.hall << host2.aisle << std::setw(2) << host2.rack << "U" << std::setw(2)
                        << host2.shelf_u << "-" << tray_id2 << "-" << port_id2 << "," << host2_node_type << ",";

            output_file << ",";        // Length blank, leaving up to technician
            output_file << std::endl;  // Type blank, leaving up to technician
        } else {
            output_file << host1.hostname << "," << tray_id1 << "," << port_id1 << "," << host1_node_type << ",";
            output_file << host2.hostname << "," << tray_id2 << "," << port_id2 << "," << host2_node_type << std::endl;
        }
    }

    output_file.close();
}

// Validate that each host_id is assigned to exactly one node
void CablingGenerator::validate_host_id_uniqueness() {
    std::unordered_map<HostId, std::string> host_to_node_path;
    collect_host_assignments(host_to_node_path);
}

// Collect all host_id assignments (flat structure)
void CablingGenerator::collect_host_assignments(std::unordered_map<HostId, std::string>& host_to_node_path) {
    for (const auto& [node_name, node] : nodes_) {
        HostId host_id = node.host_id;
        if (host_to_node_path.count(host_id)) {
            throw std::runtime_error(fmt::format(
                "Host ID {} is assigned to multiple nodes: '{}' and '{}'",
                *host_id,
                host_to_node_path[host_id],
                node_name));
        }
        host_to_node_path[host_id] = node_name;
    }
}

// Utility function to generate logical chip connections from flat structure
void CablingGenerator::generate_logical_chip_connections() {
    chip_connections_.clear();
    generate_connections_from_flat_structure();
    std::sort(chip_connections_.begin(), chip_connections_.end());
}

void CablingGenerator::generate_connections_from_flat_structure() {
    // Lambda to add chip connections between two ports
    auto add_chip_connection = [&](PortType port_type,
                                   const Board& start_board,
                                   const Board& end_board,
                                   HostId start_host_id,
                                   TrayId start_tray_id,
                                   PortId start_port_id,
                                   HostId end_host_id,
                                   TrayId end_tray_id,
                                   PortId end_port_id) {
        const auto& start_channels = start_board.get_port_channels(port_type, start_port_id);
        const auto& end_channels = end_board.get_port_channels(port_type, end_port_id);
        auto asic_channel_pairs =
            tt::scaleout_tools::get_asic_channel_connections(port_type, start_channels, end_channels);
        for (const auto& [start_channel, end_channel] : asic_channel_pairs) {
            chip_connections_.emplace_back(
                LogicalChannelEndpoint{
                    .host_id = start_host_id, .tray_id = start_tray_id, .asic_channel = start_channel},
                LogicalChannelEndpoint{.host_id = end_host_id, .tray_id = end_tray_id, .asic_channel = end_channel});
        }
    };

    // Process all nodes
    for (const auto& [node_name, node] : nodes_) {
        HostId host_id = node.host_id;

        // Add internal board connections
        for (const auto& [tray_id, board] : node.boards) {
            for (const auto& [port_type, connections] : board.get_internal_connections()) {
                for (const auto& [port_a_id, port_b_id] : connections) {
                    add_chip_connection(
                        port_type, board, board, host_id, tray_id, port_a_id, host_id, tray_id, port_b_id);
                }
            }
        }

        // Add inter-board connections within node
        for (const auto& [port_type, connections] : node.inter_board_connections) {
            for (const auto& [board_a, board_b] : connections) {
                TrayId board_a_id = board_a.first;
                PortId port_a_id = board_a.second;
                TrayId board_b_id = board_b.first;
                PortId port_b_id = board_b.second;

                const auto& board_a_ref = node.boards.at(board_a_id);
                const auto& board_b_ref = node.boards.at(board_b_id);
                add_chip_connection(
                    port_type,
                    board_a_ref,
                    board_b_ref,
                    host_id,
                    board_a_id,
                    port_a_id,
                    host_id,
                    board_b_id,
                    port_b_id);
            }
        }
    }

    // Process all port connections (inter-node connections)
    for (const auto& [port_type, connections] : port_connections_) {
        for (const auto& [conn_a, conn_b] : connections) {
            auto [host_a_id, tray_a_id, port_a_id] = conn_a;
            auto [host_b_id, tray_b_id, port_b_id] = conn_b;

            // Look up nodes using HostId
            Node* node_a = host_id_to_node_.at(host_a_id);
            Node* node_b = host_id_to_node_.at(host_b_id);

            auto& board_a_ref = node_a->boards.at(tray_a_id);
            auto& board_b_ref = node_b->boards.at(tray_b_id);
            create_port_connection(
                board_a_ref, board_b_ref, port_type, host_a_id, host_b_id, tray_a_id, tray_b_id, port_a_id, port_b_id);
            add_chip_connection(
                port_type, board_a_ref, board_b_ref, host_a_id, tray_a_id, port_a_id, host_b_id, tray_b_id, port_b_id);
        }
    }
}

void CablingGenerator::populate_host_id_to_node() {
    host_id_to_node_.clear();
    for (auto& [node_name, node] : nodes_) {
        host_id_to_node_[node.host_id] = &node;
    }
}

void CablingGenerator::get_all_connections_of_type(
    const std::vector<PortType>& port_types, std::vector<PortConnection>& conn_list) const {
    for (auto port_type : port_types) {
        // Add port connections (inter-node connections)
        auto port_conns = port_connections_.find(port_type);
        if (port_conns != port_connections_.end()) {
            conn_list.insert(conn_list.end(), port_conns->second.begin(), port_conns->second.end());
        }

        // Add inter-board connections from all nodes
        for (const auto& [node_name, node] : nodes_) {
            auto inter_board_connections = node.inter_board_connections.find(port_type);
            if (inter_board_connections == node.inter_board_connections.end()) {
                continue;
            }
            conn_list.reserve(conn_list.size() + inter_board_connections->second.size());
            for (const auto& [start, end] : inter_board_connections->second) {
                PortEndpoint s_tuple = std::make_tuple(node.host_id, start.first, start.second);
                PortEndpoint e_tuple = std::make_tuple(node.host_id, end.first, end.second);
                conn_list.push_back(std::make_pair(s_tuple, e_tuple));
            }
        }
    }
}

CableLength calc_cable_length(
    const Host& host1, int tray_id1, const Host& host2, int tray_id2, const std::string& node_type) {
    if (host1.hall != host2.hall) {
        return CableLength::UNKNOWN;
    } else if (host1.aisle != host2.aisle) {
        return CableLength::UNKNOWN;
    }


    int tray_id_0 = tray_id1;
    int tray_id_1 = tray_id2;
    int rack_0 = host1.rack;
    int rack_1 = host2.rack;

    double tray_u_est_0 = host1.shelf_u;
    double tray_u_est_1 = host2.shelf_u;
    if (node_type.find("GALAXY") != std::string::npos) {
        // 1.25 U per tray, 1 U at bottom of 6U shelf, BH_GALAXY has 8U shelves
        tray_u_est_0 += (((4 - tray_id_0) * 1.25) + 1);
        tray_u_est_1 += (((4 - tray_id_1) * 1.25) + 1);
    }


    double standard_rack_w = 600.0;    // mm
    double standard_rack_u_h = 44.45;  // mm

    double rack_distance = std::abs(rack_0 - rack_1) * standard_rack_w;
    double u_distance = std::abs(tray_u_est_0 - tray_u_est_1) * standard_rack_u_h;

    double cable_length = std::sqrt((rack_distance * rack_distance) + (u_distance * u_distance)) + 150;  // 150mm slack

    if (cable_length <= 500.0) {
        return CableLength::CABLE_0P5;
    } else if (cable_length <= 1000.0) {
        return CableLength::CABLE_1;
    } else if (cable_length <= 2500.0) {
        return CableLength::CABLE_2P5;
    } else if (cable_length <= 3000.0) {
        return CableLength::CABLE_3;
    } else if (cable_length <= 5000.0) {
        return CableLength::CABLE_5;
    } else {
        return CableLength::UNKNOWN;
    }
}

// Overload operator<< for readable test output
std::ostream& operator<<(std::ostream& os, const PhysicalChannelEndpoint& conn) {
    os << "PhysicalChannelEndpoint{hostname='" << conn.hostname << "', tray_id=" << *conn.tray_id
       << ", asic_channel=" << conn.asic_channel << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const PhysicalPortEndpoint& conn) {
    os << "PhysicalPortEndpoint{hostname='" << conn.hostname << "', aisle='" << conn.aisle << "', rack=" << conn.rack
       << ", shelf_u=" << conn.shelf_u << ", tray_id=" << *conn.tray_id
       << ", port_type=" << enchantum::to_string(conn.port_type) << ", port_id=" << *conn.port_id << "}";
    return os;
}

}  // namespace tt::scaleout_tools

// Hash specializations
namespace std {
template <>
struct hash<tt::scaleout_tools::LogicalChannelEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::LogicalChannelEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(conn.host_id, conn.tray_id, conn.asic_channel);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalChannelEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalChannelEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(conn.hostname, conn.tray_id, conn.asic_channel);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalPortEndpoint> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalPortEndpoint& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.aisle, conn.rack, conn.shelf_u, *conn.tray_id, conn.port_type, *conn.port_id);
    }
};

}  // namespace std
