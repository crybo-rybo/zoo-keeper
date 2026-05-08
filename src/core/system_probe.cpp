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
    const uint64_t free_pages =
        static_cast<uint64_t>(stats.free_count) + static_cast<uint64_t>(stats.inactive_count);
    available_out = free_pages * static_cast<uint64_t>(page_size);
}

#elif defined(__linux__)

void probe_ram_bytes(uint64_t& total_out, uint64_t& available_out) {
    total_out = 0;
    available_out = 0;
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return;
    }
    std::string line;
    while (std::getline(meminfo, line) && (total_out == 0 || available_out == 0)) {
        uint64_t* target = nullptr;
        if (total_out == 0 && line.compare(0, 9, "MemTotal:") == 0) {
            target = &total_out;
        } else if (available_out == 0 && line.compare(0, 13, "MemAvailable:") == 0) {
            target = &available_out;
        }
        if (target == nullptr) {
            continue;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        try {
            *target = std::stoull(line.substr(colon + 1)) * 1024ULL;
        } catch (...) {
            // leave field at 0
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
