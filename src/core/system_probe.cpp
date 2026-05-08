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
#include <fstream>
#include <string>
#endif

namespace zoo::core {

namespace {

#if defined(__APPLE__)

uint64_t probe_total_ram_bytes() {
    uint64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname("hw.memsize", &value, &size, nullptr, 0) != 0) {
        return 0;
    }
    return value;
}

uint64_t probe_available_ram_bytes() {
    vm_size_t page_size = 0;
    mach_port_t port = mach_host_self();
    if (host_page_size(port, &page_size) != KERN_SUCCESS) {
        return 0;
    }
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(port, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    const uint64_t free_pages =
        static_cast<uint64_t>(stats.free_count) + static_cast<uint64_t>(stats.inactive_count);
    return free_pages * static_cast<uint64_t>(page_size);
}

#elif defined(__linux__)

uint64_t parse_meminfo_kb(const std::string& key) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0;
    }
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.compare(0, key.size(), key) == 0) {
            const auto colon = line.find(':');
            if (colon == std::string::npos) {
                return 0;
            }
            uint64_t value_kb = 0;
            try {
                value_kb = std::stoull(line.substr(colon + 1));
            } catch (...) {
                return 0;
            }
            return value_kb * 1024ULL;
        }
    }
    return 0;
}

uint64_t probe_total_ram_bytes() {
    return parse_meminfo_kb("MemTotal");
}

uint64_t probe_available_ram_bytes() {
    return parse_meminfo_kb("MemAvailable");
}

#else

uint64_t probe_total_ram_bytes() {
    return 0;
}

uint64_t probe_available_ram_bytes() {
    return 0;
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
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        GpuInfo gpu;
        if (const char* name = ggml_backend_dev_name(dev)) {
            gpu.name = name;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        gpu.free_vram_bytes = static_cast<uint64_t>(free_bytes);
        gpu.total_vram_bytes = static_cast<uint64_t>(total_bytes);
        result.push_back(std::move(gpu));
    }
    return result;
}

} // namespace

Expected<SystemInfo> SystemProbe::probe() {
    ensure_backend_initialized();

    SystemInfo info;
    info.total_ram_bytes = probe_total_ram_bytes();
    info.available_ram_bytes = probe_available_ram_bytes();
    info.logical_cpu_count = std::thread::hardware_concurrency();
    info.gpu_offload_supported = llama_supports_gpu_offload();
    if (info.gpu_offload_supported) {
        info.gpus = enumerate_gpus();
    }
    return info;
}

} // namespace zoo::core
