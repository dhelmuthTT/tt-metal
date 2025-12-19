// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

class DescriptorMergerTest : public ::testing::Test {
protected:
    // Helper to create a test directory (cleans up any existing files)
    static std::string create_test_dir(const std::string& test_name) {
        const std::string dir = "generated/tests/" + test_name + "/";
        // Remove directory if it exists to clean up previous test runs
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
        }
        std::filesystem::create_directories(dir);
        return dir;
    }

    // Helper to create a vector of hostnames (host0, host1, ..., hostN-1)
    static std::vector<std::string> create_host_vector(int count) {
        std::vector<std::string> hostnames;
        hostnames.reserve(count);
        for (int i = 0; i < count; ++i) {
            hostnames.push_back("host" + std::to_string(i));
        }
        return hostnames;
    }

    // Helper to get host count from a descriptor file
    static int get_host_count(const std::string& descriptor_path) {
        auto desc = load_descriptor(descriptor_path);
        return desc.root_instance().child_mappings().size();
    }

    // Helper to create a torus descriptor file for testing merging
    static void create_torus_descriptor(
        const std::string& path,
        const std::string& template_name,
        const std::string& node_name,
        const std::string& node_type,
        uint32_t host_id = 0) {
        write_textproto(
            path,
            "graph_templates {\n"
            "  key: \"" +
                template_name +
                "\"\n"
                "  value {\n"
                "    children {\n"
                "      name: \"" +
                node_name +
                "\"\n"
                "      node_ref { node_descriptor: \"" +
                node_type +
                "\" }\n"
                "    }\n"
                "  }\n"
                "}\n"
                "\n"
                "root_instance {\n"
                "  template_name: \"" +
                template_name +
                "\"\n"
                "  child_mappings {\n"
                "    key: \"" +
                node_name +
                "\"\n"
                "    value { host_id: " +
                std::to_string(host_id) +
                " }\n"
                "  }\n"
                "}\n");
    }

    // Helper to write a string to a textproto file
    static void write_textproto(const std::string& path, const std::string& content) { std::ofstream(path) << content; }

    // Helper to write a protobuf message to a textproto file
    template <typename T>
    static void write_proto_to_textproto(const std::string& path, const T& proto) {
        std::string proto_str;
        google::protobuf::TextFormat::PrintToString(proto, &proto_str);
        write_textproto(path, proto_str);
    }

    // Helper to create a simple single-node descriptor
    static void create_simple_descriptor(
        const std::string& path,
        const std::string& template_name,
        const std::string& node_name,
        const std::string& node_descriptor,
        uint32_t host_id = 0) {
        write_textproto(
            path,
            "graph_templates {\n"
            "  key: \"" +
                template_name +
                "\"\n"
                "  value {\n"
                "    children {\n"
                "      name: \"" +
                node_name +
                "\"\n"
                "      node_ref { node_descriptor: \"" +
                node_descriptor +
                "\" }\n"
                "    }\n"
                "  }\n"
                "}\n"
                "root_instance {\n"
                "  template_name: \"" +
                template_name +
                "\"\n"
                "  child_mappings { key: \"" +
                node_name + "\" value { host_id: " + std::to_string(host_id) +
                " } }\n"
                "}\n");
    }

    // Helper to create a two-node descriptor with a connection
    static void create_two_node_descriptor_with_connection(
        const std::string& path,
        const std::string& template_name,
        const std::string& node1_descriptor,
        const std::string& node2_descriptor,
        const std::string& conn_path_b,  // "node2", "node3", etc.
        uint32_t tray_a = 1,
        uint32_t port_a = 1,
        uint32_t tray_b = 1,
        uint32_t port_b = 1) {
        write_textproto(
            path,
            "graph_templates {\n"
            "  key: \"" +
                template_name +
                "\"\n"
                "  value {\n"
                "    children { name: \"node1\" node_ref { node_descriptor: \"" +
                node1_descriptor +
                "\" } }\n"
                "    children { name: \"node2\" node_ref { node_descriptor: \"" +
                node2_descriptor +
                "\" } }\n"
                "    internal_connections {\n"
                "      key: \"QSFP_DD\"\n"
                "      value {\n"
                "        connections {\n"
                "          port_a { path: [\"node1\"] tray_id: " +
                std::to_string(tray_a) + " port_id: " + std::to_string(port_a) +
                " }\n"
                "          port_b { path: [\"" +
                conn_path_b + "\"] tray_id: " + std::to_string(tray_b) + " port_id: " + std::to_string(port_b) +
                " }\n"
                "        }\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n"
                "root_instance {\n"
                "  template_name: \"" +
                template_name +
                "\"\n"
                "  child_mappings { key: \"node1\" value { host_id: 0 } }\n"
                "  child_mappings { key: \"node2\" value { host_id: 1 } }\n"
                "}\n");
    }

    // Helper to create a multi-node descriptor with a connection
    static void create_multi_node_descriptor_with_connection(
        const std::string& path,
        const std::string& template_name,
        const std::vector<std::string>& node_names,
        const std::string& node_descriptor,
        const std::string& conn_from,
        const std::string& conn_to,
        uint32_t tray_a = 1,
        uint32_t port_a = 1,
        uint32_t tray_b = 1,
        uint32_t port_b = 1) {
        std::string content = "graph_templates {\n  key: \"" + template_name + "\"\n  value {\n";

        // Add children
        for (const auto& node_name : node_names) {
            content += "    children { name: \"" + node_name + "\" node_ref { node_descriptor: \"" + node_descriptor +
                       "\" } }\n";
        }

        // Add connection
        content +=
            "    internal_connections {\n"
            "      key: \"QSFP_DD\"\n"
            "      value {\n"
            "        connections {\n"
            "          port_a { path: [\"" +
            conn_from + "\"] tray_id: " + std::to_string(tray_a) + " port_id: " + std::to_string(port_a) +
            " }\n"
            "          port_b { path: [\"" +
            conn_to + "\"] tray_id: " + std::to_string(tray_b) + " port_id: " + std::to_string(port_b) +
            " }\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n"
            "root_instance {\n"
            "  template_name: \"" +
            template_name + "\"\n";

        // Add child mappings
        for (size_t i = 0; i < node_names.size(); ++i) {
            content +=
                "  child_mappings { key: \"" + node_names[i] + "\" value { host_id: " + std::to_string(i) + " } }\n";
        }

        content += "}\n";
        write_textproto(path, content);
    }

    // Helper to create a descriptor with internal connections (graph-level connections between nodes)
    static void create_descriptor_with_internal_connections(
        const std::string& path,
        const std::string& template_name,
        const std::vector<std::string>& node_names,
        const std::vector<std::pair<
            std::pair<std::string, std::pair<uint32_t, uint32_t>>,
            std::pair<std::string, std::pair<uint32_t, uint32_t>>>>& connections,
        const std::vector<uint32_t>& host_ids) {
        std::string content = "graph_templates {\n  key: \"" + template_name + "\"\n";
        content += "  value {\n";
        for (const auto& node_name : node_names) {
            content += "    children { name: \"" + node_name + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        if (!connections.empty()) {
            content += "    internal_connections {\n      key: \"QSFP_DD\"\n      value {\n";
            for (const auto& [node_a_port, node_b_port] : connections) {
                const auto& [node_a, port_a] = node_a_port;
                const auto& [node_b, port_b] = node_b_port;
                content += "        connections {\n";
                content += "          port_a { path: [\"" + node_a + "\"] tray_id: " + std::to_string(port_a.first) +
                           " port_id: " + std::to_string(port_a.second) + " }\n";
                content += "          port_b { path: [\"" + node_b + "\"] tray_id: " + std::to_string(port_b.first) +
                           " port_id: " + std::to_string(port_b.second) + " }\n";
                content += "        }\n";
            }
            content += "      }\n    }\n";
        }
        content += "  }\n}\n";
        content += "root_instance {\n  template_name: \"" + template_name + "\"\n";
        for (size_t i = 0; i < node_names.size(); ++i) {
            content += "  child_mappings { key: \"" + node_names[i] +
                       "\" value { host_id: " + std::to_string(host_ids[i]) + " } }\n";
        }
        content += "}\n";
        write_textproto(path, content);
    }

    // Helper to load a ClusterDescriptor from file
    static cabling_generator::proto::ClusterDescriptor load_descriptor(const std::string& path) {
        std::ifstream file(path);
        EXPECT_TRUE(file.is_open()) << "Failed to open " << path;
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cabling_generator::proto::ClusterDescriptor desc;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc)) << "Failed to parse " << path;
        return desc;
    }

    static std::vector<std::string> split_descriptor(
        const std::string& source_path,
        const std::string& output_dir,
        const std::string& template_name = "",
        int num_splits = 2) {
        EXPECT_GT(num_splits, 1) << "num_splits must be greater than 1";

        std::ifstream file(source_path);
        EXPECT_TRUE(file.is_open()) << "Failed to open " << source_path;
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        cabling_generator::proto::ClusterDescriptor original_desc;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &original_desc))
            << "Failed to parse " << source_path;

        const std::string target_template =
            template_name.empty() ? original_desc.graph_templates().begin()->first : template_name;

        EXPECT_TRUE(original_desc.graph_templates().contains(target_template))
            << "Template '" << target_template << "' not found";

        // Create empty parts (don't copy everything)
        std::vector<cabling_generator::proto::ClusterDescriptor> parts(num_splits);

        // Split graph_templates - ensure all files have complete structure EXCEPT internal_connections
        // All children must be in all parts (complete structure required for merge)
        // Only internal_connections should be split/incomplete in each part
        for (const auto& [tmpl_name, tmpl] : original_desc.graph_templates()) {
            // All children must be in all parts (complete structure required for merge)
            for (int i = 0; i < num_splits; i++) {
                for (const auto& child : tmpl.children()) {
                    *(*parts[i].mutable_graph_templates())[tmpl_name].add_children() = child;
                }
            }

            // Split internal_connections across parts using round-robin distribution
            // Each connection goes to exactly one part, distributed evenly
            for (const auto& [port_type, port_conns] : tmpl.internal_connections()) {
                int connection_idx = 0;
                for (const auto& conn : port_conns.connections()) {
                    // Round-robin: connection i goes to part (i % num_splits)
                    int target_part = connection_idx % num_splits;
                    auto* target_internal_conns =
                        (*parts[target_part].mutable_graph_templates())[tmpl_name].mutable_internal_connections();
                    *(*target_internal_conns)[port_type].add_connections() = conn;
                    connection_idx++;
                }
            }
        }

        // Split inline node_descriptors if any
        for (const auto& [node_name, node_desc] : original_desc.node_descriptors()) {
            // Since we don't have a natural way to split node_descriptors, just duplicate them in all parts
            // (They're typically small and not the main source of duplication)
            for (int i = 0; i < num_splits; i++) {
                (*parts[i].mutable_node_descriptors())[node_name] = node_desc;
            }
        }

        // Split root_instance child_mappings - all mappings must be in all parts (complete structure)
        if (original_desc.has_root_instance()) {
            // All mappings must be in all parts (complete structure required for merge)
            for (int i = 0; i < num_splits; i++) {
                auto* root = parts[i].mutable_root_instance();
                root->set_template_name(original_desc.root_instance().template_name());
                for (const auto& [key, mapping] : original_desc.root_instance().child_mappings()) {
                    (*root->mutable_child_mappings())[key] = mapping;
                }
            }
        }

        std::filesystem::create_directories(output_dir);

        std::vector<std::string> paths;
        for (int i = 0; i < num_splits; i++) {
            const std::string path = output_dir + "/part" + std::to_string(i + 1) + ".textproto";
            write_proto_to_textproto(path, parts[i]);
            paths.push_back(path);
        }

        return paths;
    }
};

TEST_F(DescriptorMergerTest, FindDescriptorFilesInDirectory) {
    // Use an existing directory with descriptors
    const auto files = CablingGenerator::find_descriptor_files("tools/tests/scaleout/cabling_descriptors");

    EXPECT_GT(files.size(), 0);
    for (const auto& file : files) {
        EXPECT_TRUE(file.ends_with(".textproto"));
    }

    EXPECT_TRUE(std::is_sorted(files.begin(), files.end()));
}

TEST_F(DescriptorMergerTest, FindDescriptorFilesEmptyDirectory) {
    const std::string empty_dir = create_test_dir("empty_merge_test_dir");
    EXPECT_THROW(CablingGenerator::find_descriptor_files(empty_dir), std::runtime_error);
}

TEST_F(DescriptorMergerTest, FindDescriptorFilesNonexistentDirectory) {
    EXPECT_THROW(CablingGenerator::find_descriptor_files("nonexistent_directory_12345"), std::runtime_error);
}

TEST_F(DescriptorMergerTest, EmptyPathsThrows) {
    // Empty directory should throw
    const std::string empty_dir = create_test_dir("empty_test");
    EXPECT_THROW(CablingGenerator(empty_dir, std::vector<std::string>{}), std::runtime_error);
}

TEST_F(DescriptorMergerTest, NonexistentFileThrows) {
    EXPECT_THROW(CablingGenerator("nonexistent_file.textproto", std::vector<std::string>{"host0"}), std::runtime_error);
}

TEST_F(DescriptorMergerTest, SplitAndMerge8x16WhGalaxyXyTorusSuperpod) {
    // Test splitting the 8x16 WH_GALAXY_XY_TORUS superpod descriptor and merging it back
    const std::string source_path =
        "tools/tests/scaleout/cabling_descriptors/8x16_wh_galaxy_xy_torus_superpod.textproto";

    // Create hostnames based on descriptor
    const auto hostnames = create_host_vector(get_host_count(source_path));

    // Test with different split counts from 2 to 16
    for (int num_splits = 2; num_splits <= 16; ++num_splits) {
        // Create unique test directory for this iteration
        const std::string test_dir = create_test_dir("split_8x16_test_" + std::to_string(num_splits));
        const std::string split_dir = test_dir + "split/";

        // Split into num_splits parts
        auto split_paths = split_descriptor(source_path, split_dir, "", num_splits);
        EXPECT_EQ(split_paths.size(), num_splits) << "Failed to split into " << num_splits << " parts";

        // Test that each split file can be loaded individually first
        for (const auto& split_path : split_paths) {
            EXPECT_NO_THROW({ CablingGenerator gen(split_path, hostnames); })
                << "Failed to load split file: " << split_path << " (num_splits=" << num_splits << ")";
        }

        // Create CablingGenerator from original file
        CablingGenerator original_gen(source_path, hostnames);

        // Create CablingGenerator from merged split files
        CablingGenerator merged_gen(split_dir, hostnames);

        // Verify that the merged CablingGenerator equals the original
        EXPECT_EQ(original_gen, merged_gen)
            << "Merged CablingGenerator does not match original - split/merge failed (num_splits=" << num_splits << ")";
    }
}

TEST_F(DescriptorMergerTest, SplitAndMerge5WhGalaxyYTorusSuperpod) {
    // Test splitting the 5 WH_GALAXY_Y_TORUS superpod descriptor and merging it back
    const std::string source_path = "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto";

    // Create hostnames based on descriptor
    const auto hostnames = create_host_vector(get_host_count(source_path));

    // Create CablingGenerator from original file once
    CablingGenerator original_gen(source_path, hostnames);

    for (int num_splits = 2; num_splits <= 5; num_splits++) {
        // Create a fresh test directory for each split count to avoid file conflicts
        const std::string test_dir = create_test_dir("split_5_test_" + std::to_string(num_splits));
        const std::string split_dir = test_dir + "split/";

        auto split_paths = split_descriptor(source_path, split_dir, "", num_splits);
        EXPECT_EQ(split_paths.size(), num_splits);

        // Test that each split file can be loaded individually first
        for (const auto& split_path : split_paths) {
            EXPECT_NO_THROW({ CablingGenerator gen(split_path, hostnames); })
                << "Failed to load split file: " << split_path;
        }

        // Create CablingGenerator from merged split files
        CablingGenerator merged_gen(split_dir, hostnames);

        // Verify that the merged CablingGenerator equals the original
        EXPECT_EQ(original_gen, merged_gen)
            << "Merged CablingGenerator does not match original for " << num_splits << " splits";
    }
}

TEST_F(DescriptorMergerTest, MergeTorusCompatibleDescriptors) {
    // Test that merging files with torus-compatible node types works correctly
    // This validates that BH_GALAXY_XY_TORUS node templates can be processed
    const std::string test_dir = create_test_dir("torus_compatible_test");

    create_simple_descriptor(test_dir + "file1.textproto", "test_cluster", "node1", "BH_GALAXY_XY_TORUS", 0);

    // Should succeed - torus node types are fully supported
    EXPECT_NO_THROW({
        CablingGenerator gen(test_dir, create_host_vector(1));
        auto fsd = gen.generate_factory_system_descriptor();
        EXPECT_EQ(fsd.hosts().size(), 1);
    }) << "Torus-compatible descriptors should merge successfully";
}

TEST_F(DescriptorMergerTest, MergeXTorusAndYTorusIntoXYTorus) {
    // Test that X_TORUS and Y_TORUS node types can merge into a combined configuration
    // Both X and Y torus have the same Wormhole architecture and compatible topology
    // Their inter_board_connections should merge successfully
    const std::string test_dir = create_test_dir("xy_torus_merge_test");

    const std::string merge_dir = test_dir + "merge/";
    std::filesystem::create_directories(merge_dir);
    create_simple_descriptor(merge_dir + "x_torus.textproto", "test_cluster", "node1", "WH_GALAXY_X_TORUS");
    create_simple_descriptor(merge_dir + "y_torus.textproto", "test_cluster", "node1", "WH_GALAXY_Y_TORUS");

    EXPECT_NO_THROW({ CablingGenerator merged(merge_dir, create_host_vector(1)); })
        << "X and Y torus should merge successfully (same architecture, both torus)";
}

TEST_F(DescriptorMergerTest, RejectMismatchedNodeTypes) {
    // Test that merging files with incompatible node types fails
    // Y_TORUS and N300_T3K_NODE have different board structures and cannot merge
    const std::string test_dir = create_test_dir("node_type_mismatch_test");

    create_simple_descriptor(test_dir + "file1.textproto", "test_cluster", "node1", "WH_GALAXY_Y_TORUS");
    create_simple_descriptor(test_dir + "file2.textproto", "test_cluster", "node1", "N300_T3K_NODE");

    EXPECT_THROW(
        {
            try {
                CablingGenerator gen(test_dir, create_host_vector(1));
                FAIL() << "Expected exception for incompatible node types";
            } catch (const std::runtime_error& e) {
                const std::string error_msg = e.what();
                // Verify error message mentions structural mismatch
                EXPECT_TRUE(
                    error_msg.find("motherboard") != std::string::npos ||
                    error_msg.find("board") != std::string::npos || error_msg.find("node") != std::string::npos)
                    << "Error: " << error_msg;
                throw;
            }
        },
        std::runtime_error)
        << "Incompatible node types should not merge";
}

TEST_F(DescriptorMergerTest, AllowCrossDescriptorConnectionsOnDifferentPorts) {
    // Test that the same node can connect to different nodes across multiple descriptors
    // as long as different ports are used. This is valid because:
    // 1. Each port is only used once (no duplicate connections within a descriptor)
    // 2. Physical port exhaustion is checked at FSD generation time, not merge time
    // 3. This allows splitting a fully-connected graph across multiple descriptor files
    const std::string test_dir = create_test_dir("cross_descriptor_connections");

    // Use WH_GALAXY (MESH) which has no internal QSFP connections, so all ports 1-6 are available
    // File 1: node1 port 1 -> node2, node2 port 2 -> node3
    write_textproto(test_dir + "file1.textproto", R"(
graph_templates {
  key: "test_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node3" node_ref { node_descriptor: "WH_GALAXY" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 1 }
          port_b { path: ["node2"] tray_id: 1 port_id: 1 }
        }
        connections {
          port_a { path: ["node2"] tray_id: 1 port_id: 2 }
          port_b { path: ["node3"] tray_id: 1 port_id: 2 }
        }
      }
    }
  }
}
root_instance {
  template_name: "test_cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
  child_mappings { key: "node3" value { host_id: 2 } }
}
)");

    // File 2: node1 port 3 (DIFFERENT port) -> node3
    write_textproto(test_dir + "file2.textproto", R"(
graph_templates {
  key: "test_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node3" node_ref { node_descriptor: "WH_GALAXY" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 3 }
          port_b { path: ["node3"] tray_id: 1 port_id: 3 }
        }
      }
    }
  }
}
root_instance {
  template_name: "test_cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
  child_mappings { key: "node3" value { host_id: 2 } }
}
)");

    // This should succeed - each port is used once, connections are valid
    EXPECT_NO_THROW({ CablingGenerator gen(test_dir, create_host_vector(3)); })
        << "Cross-descriptor connections on different ports should be allowed";
}

TEST_F(DescriptorMergerTest, MergeXTorusAndYTorusDescriptors) {
    // Test merging X_TORUS and Y_TORUS descriptors with the same template name
    // Both files define the same node with compatible but different torus types
    // The merge should combine their inter_board_connections (same architecture: Wormhole)
    const std::string test_dir = create_test_dir("x_plus_y_torus_merge");

    create_torus_descriptor(test_dir + "x_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(test_dir + "y_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_Y_TORUS", 0);

    EXPECT_NO_THROW({ CablingGenerator merged_gen(test_dir, create_host_vector(1)); })
        << "Merging X + Y torus (same architecture) should succeed";
}

TEST_F(DescriptorMergerTest, RejectMergingWormholeAndBlackholeTorusTogether) {
    // Test that merging WH torus and BH torus fails due to different architectures
    // Even though both are torus types, they have incompatible hardware architectures
    const std::string test_dir = create_test_dir("wh_bh_torus_mismatch");

    create_torus_descriptor(test_dir + "wh_torus.textproto", "mixed_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(test_dir + "bh_torus.textproto", "mixed_torus", "node1", "BH_GALAXY_X_TORUS", 0);

    EXPECT_THROW(
        {
            try {
                CablingGenerator merged_gen(test_dir, create_host_vector(1));
                FAIL() << "Expected exception for different architectures";
            } catch (const std::runtime_error& e) {
                const std::string error_msg = e.what();
                // Should mention structural mismatch (different node descriptor names)
                EXPECT_TRUE(
                    error_msg.find("structural") != std::string::npos ||
                    error_msg.find("mismatch") != std::string::npos || error_msg.find("board") != std::string::npos)
                    << "Error: " << error_msg;
                throw;
            }
        },
        std::runtime_error)
        << "Merging WH and BH torus should fail (different architectures)";
}

TEST_F(DescriptorMergerTest, MergeTwoIdenticalXTorusDescriptors) {
    // Test merging two identical X_TORUS descriptors
    // Both have the same torus type and architecture - should merge successfully
    // (duplicate connections will be deduplicated during merge)
    const std::string test_dir = create_test_dir("two_x_torus_merge");

    create_torus_descriptor(test_dir + "x_torus1.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(test_dir + "x_torus2.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);

    EXPECT_NO_THROW({ CablingGenerator merged_gen(test_dir, create_host_vector(1)); })
        << "Merging two identical X torus descriptors should succeed";
}

TEST_F(DescriptorMergerTest, MergeXYTorusWithXTorusDescriptors) {
    // Test merging XY_TORUS with X_TORUS descriptors
    // XY_TORUS already contains X-direction connections, X_TORUS adds more
    // Both are torus types with the same architecture (Wormhole) - should merge
    const std::string test_dir = create_test_dir("xy_plus_x_torus_merge");

    create_torus_descriptor(test_dir + "xy_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_XY_TORUS", 0);
    create_torus_descriptor(test_dir + "x_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);

    EXPECT_NO_THROW({ CablingGenerator merged_gen(test_dir, create_host_vector(1)); })
        << "Merging XY + X torus (same architecture) should succeed";
}

}  // namespace tt::scaleout_tools
