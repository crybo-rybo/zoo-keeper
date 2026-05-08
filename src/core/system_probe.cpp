/**
 * @file system_probe.cpp
 * @brief Host-system capability probing (RAM, CPU, GPU/VRAM).
 */

#include "zoo/core/system_probe.hpp"
#include "core/backend_init.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <cstdint>
#include <thread>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <charconv>
#include <fstream>
#include <string>
#include <string_view>
#endif

namespace zoo::core {

namespace {

#if defined(__APPLE__)

void probe_ram_bytes(uint64_t& total_out, uint64_t& available_out) {
    total_out = 0;
    available_out = 0;

    size_t size = sizeof(total_out);
    if (sysctlbyname("hw.memsize", &total_out, &size, nullptr, 0) != 0) {
        total_out = 0;
    }

    vm_size_t page_size = 0;
    mach_port_t port = mach_host_self();
    if (host_page_size(port, &page_size) != KERN_SUCCESS) {
        return;
    }
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(port, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) !=
        KERN_SUCCESS) {
        return;
    }
    // Available = free + inactive + purgeable. macOS's "Memory Available"
    // (Activity Monitor) treats purgeable pages as reclaimable, and ignoring
    // them under-reports headroom on systems with cache-heavy workloads.
    const uint64_t free_pages = static_cast<uint64_t>(stats.free_count) +
                                static_cast<uint64_t>(stats.inactive_count) +
                                static_cast<uint64_t>(stats.purgeable_count);
    available_out = free_pages * static_cast<uint64_t>(page_size);
}

#elif defined(__linux__)

// Parses a `key: <num> kB` line from /proc/meminfo and returns bytes, or 0 on
// any parse failure. Non-throwing: uses std::from_chars instead of std::stoull.
uint64_t parse_meminfo_kb_line(std::string_view line) {
    const auto colon = line.find(':');
    if (colon == std::string_view::npos) {
        return 0;
    }
    auto rest = line.substr(colon + 1);
    const auto digit_start = rest.find_first_not_of(" \t");
    if (digit_start == std::string_view::npos) {
        return 0;
    }
    rest.remove_prefix(digit_start);
    uint64_t kb = 0;
    const auto* first = rest.data();
    const auto* last = rest.data() + rest.size();
    if (std::from_chars(first, last, kb).ec != std::errc{}) {
        return 0;
    }
    return kb * 1024ULL;
}

void probe_ram_bytes(uint64_t& total_out, uint64_t& available_out) {
    total_out = 0;
    available_out = 0;
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        const std::string_view view(line);
        if (view.starts_with("MemTotal:")) {
            total_out = parse_meminfo_kb_line(view);
        } else if (view.starts_with("MemAvailable:")) {
            available_out = parse_meminfo_kb_line(view);
        }
    }
}

#else

void probe_ram_bytes(uint64_t& total_out, uint64_t& available_out) {
    total_out = 0;
    available_out = 0;
}

#endif

std::vector<GpuInfo> enumerate_gpus() {
    std::vector<GpuInfo> result;
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        GpuInfo gpu;
        if (props.name != nullptr) {
            gpu.name = props.name;
        }
        gpu.free_vram_bytes = static_cast<uint64_t>(props.memory_free);
        gpu.total_vram_bytes = static_cast<uint64_t>(props.memory_total);
        result.push_back(std::move(gpu));
    }
    return result;
}

} // namespace

Expected<SystemInfo> SystemProbe::probe() {
    ensure_backend_initialized();

    SystemInfo info;
    probe_ram_bytes(info.total_ram_bytes, info.available_ram_bytes);
    info.logical_cpu_count = std::thread::hardware_concurrency();
    info.gpu_offload_supported = llama_supports_gpu_offload();
    if (info.gpu_offload_supported) {
        info.gpus = enumerate_gpus();
    }
    return info;
}

} // namespace zoo::core
