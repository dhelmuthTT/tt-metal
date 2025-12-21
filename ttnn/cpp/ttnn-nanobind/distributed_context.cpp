// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_context.hpp"

#include <memory>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/distributed_context.hpp>

namespace ttnn::distributed_context {

namespace nb = nanobind;

void bind_distributed_context_api(nb::module_& mod) {
    using namespace tt::tt_metal::distributed::multihost;

    // Bind strong types for Rank and Size
    nb::class_<Rank>(mod, "Rank", "Rank of a process in the distributed context")
        .def("__int__", [](const Rank& r) { return static_cast<int>(*r); })
        .def("__repr__", [](const Rank& r) { return "Rank(" + std::to_string(*r) + ")"; })
        .def("__str__", [](const Rank& r) { return std::to_string(*r); })
        .def("__eq__", [](const Rank& a, const Rank& b) { return a == b; })
        .def("__ne__", [](const Rank& a, const Rank& b) { return a != b; })
        .def("__lt__", [](const Rank& a, const Rank& b) { return a < b; })
        .def("__le__", [](const Rank& a, const Rank& b) { return a <= b; })
        .def("__gt__", [](const Rank& a, const Rank& b) { return a > b; })
        .def("__ge__", [](const Rank& a, const Rank& b) { return a >= b; });

    nb::class_<Size>(mod, "Size", "Size (number of processes) in the distributed context")
        .def("__int__", [](const Size& s) { return static_cast<int>(*s); })
        .def("__repr__", [](const Size& s) { return "Size(" + std::to_string(*s) + ")"; })
        .def("__str__", [](const Size& s) { return std::to_string(*s); })
        .def("__eq__", [](const Size& a, const Size& b) { return a == b; })
        .def("__ne__", [](const Size& a, const Size& b) { return a != b; })
        .def("__lt__", [](const Size& a, const Size& b) { return a < b; })
        .def("__le__", [](const Size& a, const Size& b) { return a <= b; })
        .def("__gt__", [](const Size& a, const Size& b) { return a > b; })
        .def("__ge__", [](const Size& a, const Size& b) { return a >= b; });

    // Initialize the distributed context
    mod.def(
        "init_distributed_context",
        []() {
            // In Python context, we typically don't have argc/argv in the same way
            // This is a simplified initialization that works with default MPI settings
            static char prog_name[] = "python";
            static char* argv[] = {prog_name, nullptr};
            DistributedContext::create(1, argv);
        },
        R"doc(
            Initialize the distributed context with default settings.

            This is a convenience function for Python that initializes the distributed
            context without requiring command-line arguments.

            Example:
                >>> import ttnn
                >>> ttnn.distributed_context.init_distributed_context()
                >>> rank = ttnn.distributed_context.get_rank()
                >>> size = ttnn.distributed_context.get_size()
                >>> print(f"Rank {int(rank)} of {int(size)}")
        )doc");

    // Check if distributed context is initialized
    mod.def(
        "is_initialized",
        &DistributedContext::is_initialized,
        R"doc(
            Check if the distributed context has been initialized.

            Returns:
                bool: True if initialized, False otherwise.

            Example:
                >>> import ttnn
                >>> if ttnn.distributed_context.is_initialized():
                >>>     rank = ttnn.distributed_context.get_rank()
        )doc");

    // Get the rank of the current process
    mod.def(
        "get_rank",
        []() -> Rank {
            if (!DistributedContext::is_initialized()) {
                throw std::runtime_error("Distributed context not initialized. Call init_distributed_context() first.");
            }
            return DistributedContext::get_current_world()->rank();
        },
        R"doc(
            Get the rank of the current process.

            Returns:
                Rank: The rank of the current process (0-indexed).

            Raises:
                RuntimeError: If the distributed context has not been initialized.

            Example:
                >>> import ttnn
                >>> rank = ttnn.distributed_context.get_rank()
                >>> print(f"This is rank {rank}")
                >>> # Convert to int if needed
                >>> rank_int = int(rank)
        )doc");

    // Get the total number of processes
    mod.def(
        "get_size",
        []() -> Size {
            if (!DistributedContext::is_initialized()) {
                throw std::runtime_error("Distributed context not initialized. Call init_distributed_context() first.");
            }
            return DistributedContext::get_current_world()->size();
        },
        R"doc(
            Get the total number of processes.

            Returns:
                Size: The total number of processes.

            Raises:
                RuntimeError: If the distributed context has not been initialized.

            Example:
                >>> import ttnn
                >>> size = ttnn.distributed_context.get_size()
                >>> print(f"Total processes: {size}")
                >>> # Convert to int if needed
                >>> size_int = int(size)
        )doc");

    // Synchronize all processes
    mod.def(
        "barrier",
        []() {
            if (!DistributedContext::is_initialized()) {
                throw std::runtime_error("Distributed context not initialized. Call init_distributed_context() first.");
            }
            DistributedContext::get_current_world()->barrier();
        },
        R"doc(
            Synchronize all processes.

            This function blocks until all processes have reached this point.

            Raises:
                RuntimeError: If the distributed context has not been initialized.

            Example:
                >>> import ttnn
                >>> # Do some work...
                >>> ttnn.distributed_context.barrier()
                >>> # All processes continue from here
        )doc");
}

}  // namespace ttnn::distributed_context
