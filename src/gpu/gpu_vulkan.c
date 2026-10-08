/* Vulkan 1.1 backend for the GPU section (see gpu_backend.h). Timing uses
 * timestamp queries around the dispatches (timestampPeriod, masked to
 * timestampValidBits); host wall time is the fallback when the queue has no
 * timestamps. */
#include "gpu/gpu_backend.h"
#include "core/report.h"
#include "core/timer.h"

#include "bandwidth.h"
#include "chase.h"
#include "fma_f16.h"
#include "fma_f32.h"
#include "int32.h"
#include "texture.h"
#include "tgmem.h"

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VK_KHR_portability_enumeration
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif
#define PORTABILITY_SUBSET_NAME "VK_KHR_portability_subset"

#define VK(call)                                    \
    do {                                            \
        VkResult vr_ = (call);                      \
        if (vr_ != VK_SUCCESS) {                    \
            s = vk_status(vr_);                     \
            goto fail;                              \
        }                                           \
    } while (0)

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *map;     /* non-NULL when host-visible (mapped persistently) */
    u64            size;
} gbuf;

typedef struct {
    u32 n;
    u32 a;
    u32 b;
    u32 c;
} prm;

struct sb_gpu_dev {
    VkInstance                       inst;
    VkPhysicalDevice                 phys;
    VkDevice                         dev;
    VkQueue                          queue;
    u32                              qfam;
    VkCommandPool                    pool;
    VkCommandBuffer                  cmd;
    VkFence                          fence;
    VkQueryPool                      qpool;
    VkDescriptorPool                 dpool;
    VkPhysicalDeviceProperties       props;
    VkPhysicalDeviceMemoryProperties mem;
    f64                              ts_period;
    u64                              ts_mask;
    bool                             fp16;

    /* prepared test */
    sb_gpu_kernel_e                  kernel;
    u32                              threads;
    u32                              tex_dim;
    VkShaderModule                   shader;
    VkDescriptorSetLayout            dsl;
    VkPipelineLayout                 layout;
    VkPipeline                       pipe;
    VkDescriptorSet                  set;
    gbuf                             b0;
    gbuf                             b1;
    VkImage                          img;
    VkDeviceMemory                   img_mem;
    VkImageView                      view;
    VkSampler                        sampler;
};

static sb_status_e vk_status(VkResult r) {
    switch (r) {
    case VK_SUCCESS:                       return SB_OK;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_TOO_MANY_OBJECTS:        return SB_ERR_NOMEM;
    case VK_ERROR_DEVICE_LOST:             return SB_ERR_IO;
    case VK_TIMEOUT:                       return SB_ERR_TIMEOUT;
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_EXTENSION_NOT_PRESENT:
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_FORMAT_NOT_SUPPORTED:    return SB_ERR_UNSUPPORTED;
    default:                               return SB_ERR_SYS;
    }
}

static u32 f32_bits(f32 f) {
    u32 u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/* --- Device selection --- */

static const char *type_name(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
    default:                                     return "other";
    }
}

static u32 type_score(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1;
    default:                                     return 0;
    }
}

static bool has_ext(const VkExtensionProperties *ext, u32 n, const char *name) {
    for (u32 i = 0; i < n; i++) {
        if (strcmp(ext[i].extensionName, name) == 0) { return true; }
    }
    return false;
}

static sb_status_e create_instance(sb_gpu_dev *d) {
    u32 n = 0;
    if (vkEnumerateInstanceExtensionProperties(NULL, &n, NULL) != VK_SUCCESS) { return SB_ERR_UNSUPPORTED; }
    VkExtensionProperties *ext = SB_MALLOC((size_t)(n ? n : 1) * sizeof(*ext));
    if (ext == NULL) { return SB_ERR_NOMEM; }
    if (vkEnumerateInstanceExtensionProperties(NULL, &n, ext) != VK_SUCCESS) { n = 0; }
    /* Portability drivers (MoltenVK) are listed only when the app opts in. */
    bool portability = has_ext(ext, n, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    SB_FREE(ext);

    const char *names[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    VkApplicationInfo app = {
        .sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "bench",
        .apiVersion       = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags                   = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0,
        .pApplicationInfo        = &app,
        .enabledExtensionCount   = portability ? 1 : 0,
        .ppEnabledExtensionNames = portability ? names : NULL,
    };
    return vkCreateInstance(&ici, NULL, &d->inst) == VK_SUCCESS ? SB_OK : SB_ERR_UNSUPPORTED;
}

/* Picks SB_GPU_DEVICE (index or name substring) if set, else the best type
 * (discrete > integrated > virtual > cpu), first in enumeration order on ties.
 * Writes the device list and the reason into `caps->notes`. */
static sb_status_e pick_device(sb_gpu_dev *d, sb_gpu_caps *caps) {
    u32 n = 0;
    if (vkEnumeratePhysicalDevices(d->inst, &n, NULL) != VK_SUCCESS || n == 0) { return SB_ERR_NOTFOUND; }
    VkPhysicalDevice *devs = SB_MALLOC((size_t)n * sizeof(*devs));
    if (devs == NULL) { return SB_ERR_NOMEM; }
    if (vkEnumeratePhysicalDevices(d->inst, &n, devs) != VK_SUCCESS || n == 0) {
        SB_FREE(devs);
        return SB_ERR_NOTFOUND;
    }

    const char *env  = getenv("SB_GPU_DEVICE");
    char       *end  = NULL;
    u64         eidx = env != NULL ? strtoull(env, &end, 10) : 0;
    bool        by_index = env != NULL && env[0] != '\0' && end != env && *end == '\0';
    i64         pick = -1;
    u32         best = 0;
    size_t      off  = 0;
    off += (size_t)snprintf(caps->notes[0], sizeof(caps->notes[0]), "Vulkan devices:");
    for (u32 i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (off < sizeof(caps->notes[0])) {
            off += (size_t)snprintf(caps->notes[0] + off, sizeof(caps->notes[0]) - off, "%s %u = %s (%s)",
                                    i ? "," : "", i, p.deviceName, type_name(p.deviceType));
        }
        if (env != NULL && env[0] != '\0') {
            bool match = by_index ? eidx == i : strstr(p.deviceName, env) != NULL;
            if (match && pick < 0) { pick = i; }
        } else if (p.apiVersion >= VK_API_VERSION_1_1 && (pick < 0 || type_score(p.deviceType) > best)) {
            pick = i;
            best = type_score(p.deviceType);
        }
    }
    if (pick >= 0) { d->phys = devs[pick]; }
    SB_FREE(devs);
    if (env != NULL && env[0] != '\0') {
        snprintf(caps->notes[1], sizeof(caps->notes[1]), "Selected by SB_GPU_DEVICE=%s", env);
    } else {
        snprintf(caps->notes[1], sizeof(caps->notes[1]),
                 "Selected: discrete > integrated > virtual > cpu (SB_GPU_DEVICE=<index|name> overrides)");
    }
    return pick >= 0 ? SB_OK : SB_ERR_NOTFOUND;
}

static u64 max_alloc(const sb_gpu_dev *d) {
    VkPhysicalDeviceMaintenance3Properties m3 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &m3,
    };
    vkGetPhysicalDeviceProperties2(d->phys, &p2);
    return m3.maxMemoryAllocationSize;
}

static bool has_host_only_memory(const sb_gpu_dev *d) {
    for (u32 i = 0; i < d->mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = d->mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { return true; }
    }
    return false;
}

static sb_status_e create_device(sb_gpu_dev *d) {
    sb_status_e s = SB_OK;
    u32 nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &nq, NULL);
    VkQueueFamilyProperties *qf  = SB_MALLOC((size_t)(nq ? nq : 1) * sizeof(*qf));
    VkExtensionProperties   *ext = NULL;
    if (qf == NULL) { return SB_ERR_NOMEM; }
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &nq, qf);
    d->qfam = UINT32_MAX;
    u32 valid_bits = 0;
    for (u32 i = 0; i < nq; i++) {
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].queueCount > 0) {
            d->qfam    = i;
            valid_bits = qf[i].timestampValidBits;
            break;
        }
    }
    SB_FREE(qf);
    if (d->qfam == UINT32_MAX) { return SB_ERR_UNSUPPORTED; }
    if (valid_bits > 0 && d->props.limits.timestampPeriod > 0) {
        d->ts_period = (f64)d->props.limits.timestampPeriod;
        d->ts_mask   = valid_bits >= 64 ? UINT64_MAX : (1ULL << valid_bits) - 1;
    }

    u32 next = 0;
    if (vkEnumerateDeviceExtensionProperties(d->phys, NULL, &next, NULL) != VK_SUCCESS) { return SB_ERR_SYS; }
    ext = SB_MALLOC((size_t)(next ? next : 1) * sizeof(*ext));
    if (ext == NULL) { return SB_ERR_NOMEM; }
    VK(vkEnumerateDeviceExtensionProperties(d->phys, NULL, &next, ext));

    /* FP16 arithmetic only: the shader converts to/from 32-bit buffer data,
     * so 16-bit storage is not required. */
    const char *names[2];
    u32         nnames = 0;
    VkPhysicalDeviceShaderFloat16Int8Features f16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    };
    if (has_ext(ext, next, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 f2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &f16,
        };
        vkGetPhysicalDeviceFeatures2(d->phys, &f2);
        d->fp16 = f16.shaderFloat16 == VK_TRUE;
        f16.shaderInt8 = VK_FALSE;
        if (d->fp16) { names[nnames++] = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME; }
    }
    if (has_ext(ext, next, PORTABILITY_SUBSET_NAME)) { names[nnames++] = PORTABILITY_SUBSET_NAME; }

    f32 prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = d->qfam,
        .queueCount       = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = d->fp16 ? &f16 : NULL,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &qci,
        .enabledExtensionCount   = nnames,
        .ppEnabledExtensionNames = nnames ? names : NULL,
    };
    VK(vkCreateDevice(d->phys, &dci, NULL, &d->dev));
    vkGetDeviceQueue(d->dev, d->qfam, 0, &d->queue);

    VkCommandPoolCreateInfo cpci = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = d->qfam,
    };
    VK(vkCreateCommandPool(d->dev, &cpci, NULL, &d->pool));
    VkCommandBufferAllocateInfo cbai = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = d->pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK(vkAllocateCommandBuffers(d->dev, &cbai, &d->cmd));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK(vkCreateFence(d->dev, &fci, NULL, &d->fence));
    if (d->ts_mask != 0) {
        VkQueryPoolCreateInfo qpci = {
            .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType  = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        VK(vkCreateQueryPool(d->dev, &qpci, NULL, &d->qpool));
    }
    VkDescriptorPoolSize sizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 2 },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1 },
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = (u32)SB_ARRAY_LEN(sizes),
        .pPoolSizes    = sizes,
    };
    VK(vkCreateDescriptorPool(d->dev, &dpci, NULL, &d->dpool));
    SB_FREE(ext);
    return SB_OK;
fail:
    SB_FREE(ext);
    return s;
}

static sb_gpu_dev g_dev;

sb_status_e sb_gpu_open(sb_gpu_dev **out, sb_gpu_caps *caps) {
    sb_gpu_dev *d = &g_dev;
    memset(d, 0, sizeof(*d));
    memset(caps, 0, sizeof(*caps));
    sb_status_e s = create_instance(d);
    if (s == SB_OK) { s = pick_device(d, caps); }
    if (s == SB_OK) {
        vkGetPhysicalDeviceProperties(d->phys, &d->props);
        vkGetPhysicalDeviceMemoryProperties(d->phys, &d->mem);
        if (d->props.apiVersion < VK_API_VERSION_1_1) { s = SB_ERR_UNSUPPORTED; }
    }
    if (s == SB_OK) { s = create_device(d); }
    if (s != SB_OK) {
        if (caps->notes[0][0] != '\0') { sb_report_info("%s", caps->notes[0]); }
        sb_gpu_close(d);
        return s;
    }

    VkPhysicalDeviceLimits *lim = &d->props.limits;
    snprintf(caps->name, sizeof(caps->name), "%s", d->props.deviceName);
    snprintf(caps->api, sizeof(caps->api), "Vulkan %u.%u.%u, %s", VK_VERSION_MAJOR(d->props.apiVersion),
             VK_VERSION_MINOR(d->props.apiVersion), VK_VERSION_PATCH(d->props.apiVersion),
             type_name(d->props.deviceType));
    caps->is_cpu    = d->props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    caps->unified   = d->props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || !has_host_only_memory(d);
    caps->fp16      = d->fp16;
    caps->gpu_timer = d->ts_mask != 0;
    u64 mb = lim->maxStorageBufferRange;
    u64 ma = max_alloc(d);
    if (ma != 0 && ma < mb) { mb = ma; }
    caps->max_buffer  = mb;
    caps->max_tex_dim = lim->maxImageDimension2D;
    if (caps->unified) {
        snprintf(caps->notes[2], sizeof(caps->notes[2]),
                 "Host-visible memory is the same DRAM as device-local on this device");
    }
    *out = d;
    return SB_OK;
}

void sb_gpu_close(sb_gpu_dev *d) {
    if (d == NULL) { return; }
    if (d->dev != VK_NULL_HANDLE) {
        sb_gpu_release(d);
        if (d->dpool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(d->dev, d->dpool, NULL); }
        if (d->qpool != VK_NULL_HANDLE) { vkDestroyQueryPool(d->dev, d->qpool, NULL); }
        if (d->fence != VK_NULL_HANDLE) { vkDestroyFence(d->dev, d->fence, NULL); }
        if (d->pool != VK_NULL_HANDLE)  { vkDestroyCommandPool(d->dev, d->pool, NULL); }
        vkDestroyDevice(d->dev, NULL);
    }
    if (d->inst != VK_NULL_HANDLE) { vkDestroyInstance(d->inst, NULL); }
    memset(d, 0, sizeof(*d));
}

/* --- Memory --- */

/* First memory type allowed by `bits` with all of `want` and none of `avoid`;
 * falls back to ignoring `avoid`. */
static u32 find_type(const sb_gpu_dev *d, u32 bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (u32 pass = 0; pass < 2; pass++) {
        for (u32 i = 0; i < d->mem.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = d->mem.memoryTypes[i].propertyFlags;
            if (!(bits & (1U << i)) || (f & want) != want) { continue; }
            if (pass == 0 && (f & avoid) != 0) { continue; }
            return i;
        }
    }
    return UINT32_MAX;
}

static void buf_free(sb_gpu_dev *d, gbuf *b) {
    if (b->buf != VK_NULL_HANDLE) { vkDestroyBuffer(d->dev, b->buf, NULL); }
    if (b->mem != VK_NULL_HANDLE) { vkFreeMemory(d->dev, b->mem, NULL); }
    memset(b, 0, sizeof(*b));
}

static sb_status_e buf_create(sb_gpu_dev *d, u64 size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want,
                              VkMemoryPropertyFlags avoid, gbuf *out) {
    sb_status_e s = SB_OK;
    memset(out, 0, sizeof(*out));
    out->size = size;
    VkBufferCreateInfo bci = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK(vkCreateBuffer(d->dev, &bci, NULL, &out->buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d->dev, out->buf, &req);
    u32 type = find_type(d, req.memoryTypeBits, want, avoid);
    if (type == UINT32_MAX) {
        s = SB_ERR_UNSUPPORTED;
        goto fail;
    }
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = req.size,
        .memoryTypeIndex = type,
    };
    VK(vkAllocateMemory(d->dev, &mai, NULL, &out->mem));
    VK(vkBindBufferMemory(d->dev, out->buf, out->mem, 0));
    VkMemoryPropertyFlags f = d->mem.memoryTypes[type].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VK(vkMapMemory(d->dev, out->mem, 0, VK_WHOLE_SIZE, 0, &out->map));
    }
    return SB_OK;
fail:
    buf_free(d, out);
    return s;
}

static sb_status_e cmd_begin(sb_gpu_dev *d) {
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkResetCommandBuffer(d->cmd, 0) != VK_SUCCESS) { return SB_ERR_SYS; }
    return vkBeginCommandBuffer(d->cmd, &bi) == VK_SUCCESS ? SB_OK : SB_ERR_SYS;
}

static sb_status_e cmd_submit_wait(sb_gpu_dev *d) {
    sb_status_e s = SB_OK;
    VK(vkEndCommandBuffer(d->cmd));
    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &d->cmd,
    };
    VK(vkQueueSubmit(d->queue, 1, &si, d->fence));
    VK(vkWaitForFences(d->dev, 1, &d->fence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(d->dev, 1, &d->fence));
    return SB_OK;
fail:
    return s;
}

/* Global barrier: earlier writes of stage `src` (TRANSFER or COMPUTE_SHADER)
 * -> later compute shader reads and writes. */
static void barrier(VkCommandBuffer cmd, VkPipelineStageFlags src) {
    VkMemoryBarrier mb = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = src == VK_PIPELINE_STAGE_TRANSFER_BIT ? VK_ACCESS_TRANSFER_WRITE_BIT
                                                               : VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, src, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* Device-local buffer filled from `data` (direct when mappable, else staging),
 * or with the 32-bit `pattern` when data is NULL. */
static sb_status_e buf_device(sb_gpu_dev *d, u64 size, const void *data, u32 pattern, gbuf *out) {
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    sb_status_e s = buf_create(d, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, out);
    if (s != SB_OK) { return s; }
    gbuf staging = {0};
    if (data != NULL && out->map == NULL) {
        s = buf_create(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &staging);
        if (s != SB_OK) { return s; }
        memcpy(staging.map, data, (size_t)size);
    } else if (data != NULL) {
        memcpy(out->map, data, (size_t)size);
        return SB_OK;
    }
    s = cmd_begin(d);
    if (s == SB_OK) {
        if (data != NULL) {
            VkBufferCopy region = { .size = size };
            vkCmdCopyBuffer(d->cmd, staging.buf, out->buf, 1, &region);
        } else {
            vkCmdFillBuffer(d->cmd, out->buf, 0, VK_WHOLE_SIZE, pattern);
        }
        barrier(d->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT);
        s = cmd_submit_wait(d);
    }
    buf_free(d, &staging);
    return s;
}

/* RGBA8 texture of incompressible bytes (drivers may compress images losslessly),
 * device-local, uploaded through a staging buffer. */
static sb_status_e make_texture(sb_gpu_dev *d, u32 dim) {
    sb_status_e s     = SB_OK;
    u64         bytes = (u64)dim * dim * 4;
    gbuf        staging = {0};
    VkImageCreateInfo ici = {
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = VK_FORMAT_R8G8B8A8_UNORM,
        .extent        = { dim, dim, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK(vkCreateImage(d->dev, &ici, NULL, &d->img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(d->dev, d->img, &req);
    u32 type = find_type(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (type == UINT32_MAX) {
        s = SB_ERR_UNSUPPORTED;
        goto fail;
    }
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = req.size,
        .memoryTypeIndex = type,
    };
    VK(vkAllocateMemory(d->dev, &mai, NULL, &d->img_mem));
    VK(vkBindImageMemory(d->dev, d->img, d->img_mem, 0));

    s = buf_create(d, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &staging);
    if (s != SB_OK) { goto fail; }
    u32 *p = staging.map;
    u64  x = 0x243F6A8885A308D3ULL;
    for (u64 i = 0; i < bytes / 4; i++) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        p[i] = (u32)(x >> 32);
    }

    s = cmd_begin(d);
    if (s != SB_OK) { goto fail; }
    VkImageMemoryBarrier ib = {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask       = 0,
        .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = d->img,
        .subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0,
                         NULL, 1, &ib);
    VkBufferImageCopy region = {
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent      = { dim, dim, 1 },
    };
    vkCmdCopyBufferToImage(d->cmd, staging.buf, d->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(d->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                         0, NULL, 1, &ib);
    s = cmd_submit_wait(d);
    if (s != SB_OK) { goto fail; }
    buf_free(d, &staging);

    VkImageViewCreateInfo vci = {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = d->img,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    VK(vkCreateImageView(d->dev, &vci, NULL, &d->view));
    VkSamplerCreateInfo sci = {
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod       = 0.0f,
    };
    VK(vkCreateSampler(d->dev, &sci, NULL, &d->sampler));
    return SB_OK;
fail:
    buf_free(d, &staging);
    return s;
}

/* --- Tests --- */

void sb_gpu_release(sb_gpu_dev *d) {
    if (d->dev == VK_NULL_HANDLE) { return; }
    vkDeviceWaitIdle(d->dev);
    if (d->pipe != VK_NULL_HANDLE)    { vkDestroyPipeline(d->dev, d->pipe, NULL); }
    if (d->layout != VK_NULL_HANDLE)  { vkDestroyPipelineLayout(d->dev, d->layout, NULL); }
    if (d->dsl != VK_NULL_HANDLE)     { vkDestroyDescriptorSetLayout(d->dev, d->dsl, NULL); }
    if (d->shader != VK_NULL_HANDLE)  { vkDestroyShaderModule(d->dev, d->shader, NULL); }
    if (d->sampler != VK_NULL_HANDLE) { vkDestroySampler(d->dev, d->sampler, NULL); }
    if (d->view != VK_NULL_HANDLE)    { vkDestroyImageView(d->dev, d->view, NULL); }
    if (d->img != VK_NULL_HANDLE)     { vkDestroyImage(d->dev, d->img, NULL); }
    if (d->img_mem != VK_NULL_HANDLE) { vkFreeMemory(d->dev, d->img_mem, NULL); }
    if (d->dpool != VK_NULL_HANDLE)   { vkResetDescriptorPool(d->dev, d->dpool, 0); }
    buf_free(d, &d->b0);
    buf_free(d, &d->b1);
    d->pipe    = VK_NULL_HANDLE;
    d->layout  = VK_NULL_HANDLE;
    d->dsl     = VK_NULL_HANDLE;
    d->shader  = VK_NULL_HANDLE;
    d->sampler = VK_NULL_HANDLE;
    d->view    = VK_NULL_HANDLE;
    d->img     = VK_NULL_HANDLE;
    d->img_mem = VK_NULL_HANDLE;
    d->set     = VK_NULL_HANDLE;
}

static sb_status_e create_pipeline(sb_gpu_dev *d, const u32 *code, size_t size, bool image) {
    sb_status_e s = SB_OK;
    VkShaderModuleCreateInfo smci = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode    = code,
    };
    VK(vkCreateShaderModule(d->dev, &smci, NULL, &d->shader));
    VkDescriptorSetLayoutBinding bind[2] = {
        {
            .binding         = 0,
            .descriptorType  = image ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    u32 nbind = d->b1.buf != VK_NULL_HANDLE || image ? 2 : 1;
    VkDescriptorSetLayoutCreateInfo dlci = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = nbind,
        .pBindings    = bind,
    };
    VK(vkCreateDescriptorSetLayout(d->dev, &dlci, NULL, &d->dsl));
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .size       = sizeof(prm),
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &d->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcr,
    };
    VK(vkCreatePipelineLayout(d->dev, &plci, NULL, &d->layout));
    VkComputePipelineCreateInfo cpci = {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = d->shader,
            .pName  = "main",
        },
        .layout = d->layout,
    };
    VK(vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &d->pipe));

    VkDescriptorSetAllocateInfo dsai = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = d->dpool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &d->dsl,
    };
    VK(vkAllocateDescriptorSets(d->dev, &dsai, &d->set));
    VkDescriptorImageInfo  dii    = { .sampler = d->sampler, .imageView = d->view,
                                      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo dbi[2] = {
        { .buffer = d->b0.buf, .range = VK_WHOLE_SIZE },
        { .buffer = image ? d->b0.buf : d->b1.buf, .range = VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[2] = {
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = d->set,
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = bind[0].descriptorType,
            .pImageInfo      = image ? &dii : NULL,
            .pBufferInfo     = image ? NULL : &dbi[0],
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = d->set,
            .dstBinding      = 1,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &dbi[1],
        },
    };
    vkUpdateDescriptorSets(d->dev, nbind, w, 0, NULL);
    return SB_OK;
fail:
    return s;
}

static sb_status_e prepare_buffers(sb_gpu_dev *d, const sb_gpu_test *t) {
    sb_status_e s = SB_OK;
    switch (t->kernel) {
    case SB_GPU_K_FP32:
    case SB_GPU_K_FP16:
    case SB_GPU_K_INT32: {
        u32 *init = SB_MALLOC((size_t)t->threads * sizeof(u32));
        if (init == NULL) { return SB_ERR_NOMEM; }
        for (u32 i = 0; i < t->threads; i++) {
            init[i] = t->kernel == SB_GPU_K_INT32 ? i * 2654435761U
                                                  : f32_bits(0.5f + (f32)(i & 255) * (1.0f / 512.0f));
        }
        s = buf_device(d, (u64)t->threads * 4, init, 0, &d->b0);
        SB_FREE(init);
        return s;
    }
    case SB_GPU_K_BW_DEVICE:
        s = buf_device(d, t->bytes, NULL, 0x3F3F3F3FU, &d->b0);
        if (s == SB_OK) { s = buf_device(d, (u64)t->threads * 4, NULL, 0, &d->b1); }
        return s;
    case SB_GPU_K_BW_HOST:
        s = buf_create(d, t->bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &d->b0);
        if (s != SB_OK) { return s; }
        if (d->b0.map == NULL) { return SB_ERR_UNSUPPORTED; }
        memset(d->b0.map, 0x3F, (size_t)t->bytes);
        return buf_device(d, (u64)t->threads * 4, NULL, 0, &d->b1);
    case SB_GPU_K_LATENCY:
        s = buf_device(d, t->bytes, t->chain, 0, &d->b0);
        if (s == SB_OK) { s = buf_device(d, 4, NULL, 0, &d->b1); }  /* chase position, starts at 0 */
        return s;
    case SB_GPU_K_TGMEM:
        if (d->props.limits.maxComputeSharedMemorySize < (SB_GPU_TG_N + SB_GPU_TG_LOADS * SB_GPU_TG_STEP) * 16) {
            return SB_ERR_UNSUPPORTED;
        }
        return buf_device(d, (u64)t->threads * 16, NULL, 0, &d->b0);
    case SB_GPU_K_TEX_CACHED:
    case SB_GPU_K_TEX_STREAM:
        if (t->tex_dim > d->props.limits.maxImageDimension2D) { return SB_ERR_UNSUPPORTED; }
        s = make_texture(d, t->tex_dim);
        if (s == SB_OK) { s = buf_device(d, (u64)t->threads * 4, NULL, 0, &d->b0); }
        return s;
    }
    return SB_ERR_INVALID;
}

sb_status_e sb_gpu_prepare(sb_gpu_dev *d, const sb_gpu_test *t) {
    sb_gpu_release(d);
    if (t->kernel != SB_GPU_K_LATENCY && d->props.limits.maxComputeWorkGroupSize[0] < SB_GPU_TG) {
        return SB_ERR_UNSUPPORTED;
    }
    if (t->kernel != SB_GPU_K_LATENCY && t->threads / SB_GPU_TG > d->props.limits.maxComputeWorkGroupCount[0]) {
        return SB_ERR_UNSUPPORTED;
    }
    if ((t->kernel == SB_GPU_K_BW_DEVICE || t->kernel == SB_GPU_K_BW_HOST || t->kernel == SB_GPU_K_LATENCY) &&
        t->bytes > d->props.limits.maxStorageBufferRange) {
        return SB_ERR_UNSUPPORTED;
    }
    d->kernel  = t->kernel;
    d->threads = t->threads;
    d->tex_dim = t->tex_dim;
    sb_status_e s = prepare_buffers(d, t);
    if (s != SB_OK) { return s; }

    const u32 *code = NULL;
    size_t     size = 0;
    switch (t->kernel) {
    case SB_GPU_K_FP32:       code = fma_f32_spv;   size = fma_f32_spv_size;   break;
    case SB_GPU_K_FP16:       code = fma_f16_spv;   size = fma_f16_spv_size;   break;
    case SB_GPU_K_INT32:      code = int32_spv;     size = int32_spv_size;     break;
    case SB_GPU_K_BW_DEVICE:
    case SB_GPU_K_BW_HOST:    code = bandwidth_spv; size = bandwidth_spv_size; break;
    case SB_GPU_K_LATENCY:    code = chase_spv;     size = chase_spv_size;     break;
    case SB_GPU_K_TGMEM:      code = tgmem_spv;     size = tgmem_spv_size;     break;
    case SB_GPU_K_TEX_CACHED:
    case SB_GPU_K_TEX_STREAM: code = texture_spv;   size = texture_spv_size;   break;
    }
    bool image = t->kernel == SB_GPU_K_TEX_CACHED || t->kernel == SB_GPU_K_TEX_STREAM;
    return create_pipeline(d, code, size, image);
}

sb_status_e sb_gpu_run(sb_gpu_dev *d, u32 n, u32 ndispatch, f64 *out_ns) {
    prm p = { .n = n, .a = d->tex_dim };
    if (d->kernel == SB_GPU_K_FP16) {
        p.b = f32_bits(0.999f);
        p.c = f32_bits(0.001f);
    } else if (d->kernel == SB_GPU_K_INT32) {
        p.b = 0x4F6CDD1DU;
        p.c = 0x7F4A7C15U;
    } else {
        p.b = f32_bits(0.9999f);
        p.c = f32_bits(0.0001f);
    }
    u32 groups = d->kernel == SB_GPU_K_LATENCY ? 1 : d->threads / SB_GPU_TG;

    sb_status_e s = cmd_begin(d);
    if (s != SB_OK) { return s; }
    if (d->qpool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(d->cmd, d->qpool, 0, 2);
    }
    /* Previous runs' shader writes (same buffers) before this run's accesses. */
    barrier(d->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipe);
    vkCmdBindDescriptorSets(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->layout, 0, 1, &d->set, 0, NULL);
    vkCmdPushConstants(d->cmd, d->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
    if (d->qpool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(d->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, d->qpool, 0);
    }
    for (u32 i = 0; i < ndispatch; i++) {
        if (i > 0) { barrier(d->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT); }
        vkCmdDispatch(d->cmd, groups, 1, 1);
    }
    if (d->qpool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(d->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, d->qpool, 1);
    }
    u64 t0 = sb_timer_now_ns();
    s = cmd_submit_wait(d);
    u64 t1 = sb_timer_now_ns();
    if (s != SB_OK) { return s; }

    f64 ns = (f64)(t1 - t0);
    if (d->qpool != VK_NULL_HANDLE) {
        u64 ts[2] = {0, 0};
        VkResult r = vkGetQueryPoolResults(d->dev, d->qpool, 0, 2, sizeof(ts), ts, sizeof(u64),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (r != VK_SUCCESS) { return vk_status(r); }
        ns = (f64)((ts[1] - ts[0]) & d->ts_mask) * d->ts_period;
    }
    if (ns <= 0) { return SB_ERR_RANGE; }
    *out_ns = ns;
    return SB_OK;
}
