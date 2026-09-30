#include "bench.h"
#include "timer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__

#include <vulkan/vulkan.h>

/* Embedded SPIR-V shaders */
#include "fma_bench.h"
#include "int_bench.h"
#include "bw_bench.h"
#include "latency_chase.h"
#include "fma_fp16_bench.h"
#include "shared_mem.h"
#include "tex_sample.h"

#define VK_CHECK(call) do { \
    VkResult status = (call); \
    if (status != VK_SUCCESS) { \
        fprintf(stderr, "  %s failed (%d)\n", #call, status); \
        goto cleanup; \
    } \
} while (0)

/* Make earlier uploads and shader writes visible to the next dispatch. */
static void compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
}

#define NUM_THREADS    (1U << 20)           /* 1M threads for compute */
#define WG_SIZE        64U
#define BW_TOTAL_BYTES (256ULL * 1024 * 1024)
#define LAT_BUF_SIZE   (64ULL * 1024 * 1024)
#define LAT_STRIDE     16                   /* elements (64 bytes) per node */

struct vk_ctx {
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    uint32_t         qfamily;
    VkQueue          queue;
    VkCommandPool    cmdpool;
    VkDescriptorPool ds_pool;
    char             device_name[256];
    uint32_t         api_version;
    int              has_fp16;
};

static int vk_init(struct vk_ctx *vk) {
    memset(vk, 0, sizeof(*vk));

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "bench",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (vkCreateInstance(&ici, NULL, &vk->instance) != VK_SUCCESS) {
        fprintf(stderr, "  vkCreateInstance failed\n");
        return -1;
    }

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(vk->instance, &count, NULL) != VK_SUCCESS || count == 0) {
        fprintf(stderr, "  No Vulkan physical device found\n");
        return -1;
    }
    VkPhysicalDevice *devices = malloc(count * sizeof(*devices));
    if (!devices) return -1;
    VkResult enumerated = vkEnumeratePhysicalDevices(vk->instance, &count, devices);
    if (enumerated != VK_SUCCESS || count == 0) { free(devices); return -1; }
    /* Prefer a hardware GPU if a software Vulkan implementation is also installed. */
    vk->phys = devices[0];
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties candidate;
        vkGetPhysicalDeviceProperties(devices[i], &candidate);
        if (candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            vk->phys = devices[i];
            break;
        }
    }
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk->phys, &props);
    if (props.apiVersion < VK_API_VERSION_1_1) {
        fprintf(stderr, "  Vulkan 1.1 or newer is required\n");
        return -1;
    }
    snprintf(vk->device_name, sizeof(vk->device_name), "%s", props.deviceName);
    vk->api_version = props.apiVersion;

    uint32_t qfcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->phys, &qfcount, NULL);
    VkQueueFamilyProperties *qf = malloc(qfcount * sizeof(*qf));
    if (!qf || qfcount == 0) { free(qf); return -1; }
    vkGetPhysicalDeviceQueueFamilyProperties(vk->phys, &qfcount, qf);

    vk->qfamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfcount; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            vk->qfamily = i;
            break;
        }
    }
    free(qf);
    if (vk->qfamily == UINT32_MAX) {
        fprintf(stderr, "  No compute queue family\n");
        return -1;
    }

    /* Check for VK_KHR_shader_float16_int8 extension */
    uint32_t ext_count = 0;
    if (vkEnumerateDeviceExtensionProperties(vk->phys, NULL, &ext_count, NULL) != VK_SUCCESS)
        return -1;
    VkExtensionProperties *exts = malloc(ext_count * sizeof(*exts));
    if (!exts && ext_count) return -1;
    if (vkEnumerateDeviceExtensionProperties(vk->phys, NULL, &ext_count, exts) != VK_SUCCESS) {
        free(exts);
        return -1;
    }
    int has_fp16_ext = 0;
    for (uint32_t i = 0; i < ext_count; i++) {
        if (strcmp(exts[i].extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) {
            has_fp16_ext = 1;
            break;
        }
    }
    free(exts);

    VkPhysicalDeviceShaderFloat16Int8Features fp16_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    };
    VkPhysicalDevice16BitStorageFeatures storage16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    };
    fp16_features.pNext = &storage16;
    if (has_fp16_ext) {
        VkPhysicalDeviceFeatures2 features2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &fp16_features,
        };
        vkGetPhysicalDeviceFeatures2(vk->phys, &features2);
        vk->has_fp16 = fp16_features.shaderFloat16 && storage16.storageBuffer16BitAccess;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->qfamily,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };

    /* Enable FP16 if available */
    fp16_features.shaderFloat16 = vk->has_fp16 ? VK_TRUE : VK_FALSE;
    fp16_features.shaderInt8 = VK_FALSE;
    storage16.storageBuffer16BitAccess = vk->has_fp16 ? VK_TRUE : VK_FALSE;
    storage16.uniformAndStorageBuffer16BitAccess = VK_FALSE;
    storage16.storagePushConstant16 = VK_FALSE;
    storage16.storageInputOutput16 = VK_FALSE;
    const char *fp16_ext_name = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;

    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = vk->has_fp16 ? 1 : 0,
        .ppEnabledExtensionNames = vk->has_fp16 ? &fp16_ext_name : NULL,
        .pNext = vk->has_fp16 ? &fp16_features : NULL,
    };
    if (vkCreateDevice(vk->phys, &dci, NULL, &vk->device) != VK_SUCCESS) {
        fprintf(stderr, "  vkCreateDevice failed\n");
        return -1;
    }
    vkGetDeviceQueue(vk->device, vk->qfamily, 0, &vk->queue);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = vk->qfamily,
    };
    if (vkCreateCommandPool(vk->device, &cpci, NULL, &vk->cmdpool) != VK_SUCCESS) {
        fprintf(stderr, "  vkCreateCommandPool failed\n");
        return -1;
    }

    VkDescriptorPoolSize dps[] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 16 },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 4 },
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 16,
        .poolSizeCount = 2,
        .pPoolSizes = dps,
    };
    if (vkCreateDescriptorPool(vk->device, &dpci, NULL, &vk->ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "  vkCreateDescriptorPool failed\n");
        return -1;
    }

    return 0;
}

static void vk_destroy(struct vk_ctx *vk) {
    if (vk->ds_pool)  vkDestroyDescriptorPool(vk->device, vk->ds_pool, NULL);
    if (vk->cmdpool)  vkDestroyCommandPool(vk->device, vk->cmdpool, NULL);
    if (vk->device)   vkDestroyDevice(vk->device, NULL);
    if (vk->instance) vkDestroyInstance(vk->instance, NULL);
}

static uint32_t find_mem_type(struct vk_ctx *vk, uint32_t type_filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(vk->phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1U << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

/* Upload host data to a device-local buffer via staging copy. */
static int upload_to_device(struct vk_ctx *vk, VkBuffer dst,
                            const void *src_data, size_t size)
{
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    int ret = -1;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    if (vkCreateBuffer(vk->device, &bci, NULL, &staging) != VK_SUCCESS) goto cleanup;

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(vk->device, staging, &mreq);
    uint32_t mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size, .memoryTypeIndex = mtype,
    };
    if (vkAllocateMemory(vk->device, &mai, NULL, &staging_mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, staging, staging_mem, 0));

    void *ptr;
    VK_CHECK(vkMapMemory(vk->device, staging_mem, 0, size, 0, &ptr));
    memcpy(ptr, src_data, size);
    vkUnmapMemory(vk->device, staging_mem);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(vk->device, &cbai, &cmd) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    VkBufferCopy region = { .size = size };
    vkCmdCopyBuffer(cmd, staging, dst, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    ret = 0;

cleanup:
    vkDeviceWaitIdle(vk->device);
    if (fence)       vkDestroyFence(vk->device, fence, NULL);
    if (cmd)         vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &cmd);
    if (staging_mem) vkFreeMemory(vk->device, staging_mem, NULL);
    if (staging)     vkDestroyBuffer(vk->device, staging, NULL);
    return ret;
}

/* Fill a uint32_t buffer with shuffled pointer-chase indices.
   Nodes are spaced `stride` elements apart. Each node stores the
   index of the next node, forming a single cycle through all nodes. */
static int fill_chase_indices(uint32_t *buf, size_t total_elements, size_t stride) {
    size_t count = total_elements / stride;

    memset(buf, 0, total_elements * sizeof(uint32_t));

    size_t *indices = malloc(count * sizeof(size_t));
    if (!indices) return -1;
    for (size_t i = 0; i < count; i++) indices[i] = i;

    shuffle_indices(indices, count);

    /* Link nodes into a cycle */
    for (size_t i = 0; i < count - 1; i++)
        buf[indices[i] * stride] = (uint32_t)(indices[i + 1] * stride);
    buf[indices[count - 1] * stride] = (uint32_t)(indices[0] * stride);

    free(indices);
    return 0;
}

/* --- Compute benchmark (ALU-bound, unchanged) --- */

static double run_compute_bench(struct vk_ctx *vk,
                                const uint32_t *spv, uint32_t spv_size,
                                size_t buf_bytes, size_t num_threads,
                                double flops_per_thread)
{
    int ok = 0;
    double result = -1.0;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size, .pCode = spv,
    };
    if (vkCreateShaderModule(vk->device, &smci, NULL, &shader) != VK_SUCCESS)
        goto cleanup;

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    if (vkCreateDescriptorSetLayout(vk->device, &dslci, NULL, &dsl) != VK_SUCCESS)
        goto cleanup;

    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
    };
    if (vkCreatePipelineLayout(vk->device, &plci, NULL, &pl) != VK_SUCCESS)
        goto cleanup;

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pl,
    };
    if (vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
        goto cleanup;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buf_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    if (vkCreateBuffer(vk->device, &bci, NULL, &buf) != VK_SUCCESS) goto cleanup;

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(vk->device, buf, &mreq);
    uint32_t mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size, .memoryTypeIndex = mtype,
    };
    if (vkAllocateMemory(vk->device, &mai, NULL, &mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, buf, mem, 0));

    void *ptr;
    VK_CHECK(vkMapMemory(vk->device, mem, 0, buf_bytes, 0, &ptr));
    if (spv == fma_fp16_bench_spv) {
        uint16_t *fp16 = ptr;
        for (size_t i = 0; i < num_threads; i++) fp16[i] = 0x3c00;
    } else if (spv == int_bench_spv) {
        uint32_t *ints = ptr;
        for (size_t i = 0; i < num_threads; i++) ints[i] = (uint32_t)i + 1;
    } else {
        float *fp = ptr;
        for (size_t i = 0; i < num_threads; i++) fp[i] = 1.0f + (float)(i % 256) * 0.0001f;
    }
    vkUnmapMemory(vk->device, mem);

    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->ds_pool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    if (vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) goto cleanup;

    VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wds = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(vk->device, 1, &wds, 0, NULL);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(vk->device, &cbai, &cmd) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
    };

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };

    /* Warmup */
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + WG_SIZE - 1) / WG_SIZE, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(vk->device, 1, &fence));

    /* Calibrate: time a single pass */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + WG_SIZE - 1) / WG_SIZE, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t passes = 1;
    {
        uint64_t tc0 = timer_ns();
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        uint64_t tc1 = timer_ns();
        uint64_t one_pass = tc1 - tc0;
        if (one_pass > 0) passes = 200000000ULL / one_pass + 1;
        if (passes < 2) passes = 2;
        if (passes > 1000) passes = 1000;
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }

    /* Timed run */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + WG_SIZE - 1) / WG_SIZE, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t t0 = timer_ns();
    for (uint64_t p = 0; p < passes; p++) {
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    if (elapsed_s > 0)
        result = flops_per_thread * (double)num_threads * passes / elapsed_s / 1e9;
    ok = 1;

cleanup:
    vkDeviceWaitIdle(vk->device);
    if (fence)  vkDestroyFence(vk->device, fence, NULL);
    if (cmd)    vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &cmd);
    if (mem)    vkFreeMemory(vk->device, mem, NULL);
    if (buf)    vkDestroyBuffer(vk->device, buf, NULL);
    if (pipe)   vkDestroyPipeline(vk->device, pipe, NULL);
    if (pl)     vkDestroyPipelineLayout(vk->device, pl, NULL);
    if (ds)     vkFreeDescriptorSets(vk->device, vk->ds_pool, 1, &ds);
    if (dsl)    vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    if (shader) vkDestroyShaderModule(vk->device, shader, NULL);
    return ok ? result : -1.0;
}

/* --- Bandwidth benchmark (parameterized by memory type) --- */

static double run_bw_bench(struct vk_ctx *vk, VkMemoryPropertyFlags mem_flags) {
    int ok = 0;
    double result = -1.0;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    int host_visible = (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = bw_bench_spv_size, .pCode = bw_bench_spv,
    };
    if (vkCreateShaderModule(vk->device, &smci, NULL, &shader) != VK_SUCCESS)
        goto cleanup;

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    if (vkCreateDescriptorSetLayout(vk->device, &dslci, NULL, &dsl) != VK_SUCCESS)
        goto cleanup;

    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
    };
    if (vkCreatePipelineLayout(vk->device, &plci, NULL, &pl) != VK_SUCCESS)
        goto cleanup;

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pl,
    };
    if (vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
        goto cleanup;

    /* 256 vec4s per thread = 4096 bytes/thread */
    size_t bytes_per_thread = 256 * 16;
    size_t num_threads = BW_TOTAL_BYTES / bytes_per_thread;
    size_t buf_size = num_threads * bytes_per_thread;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buf_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    if (vkCreateBuffer(vk->device, &bci, NULL, &buf) != VK_SUCCESS) goto cleanup;

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(vk->device, buf, &mreq);
    uint32_t mtype = find_mem_type(vk, mreq.memoryTypeBits, mem_flags);
    if (mtype == UINT32_MAX) goto cleanup;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size, .memoryTypeIndex = mtype,
    };
    if (vkAllocateMemory(vk->device, &mai, NULL, &mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, buf, mem, 0));

    /* Fill buffer */
    if (host_visible) {
        void *ptr;
        VK_CHECK(vkMapMemory(vk->device, mem, 0, buf_size, 0, &ptr));
        float *fp = (float *)ptr;
        for (size_t i = 0; i < buf_size / sizeof(float); i++) fp[i] = 1.0f;
        vkUnmapMemory(vk->device, mem);
    } else {
        float *tmp = malloc(buf_size);
        if (!tmp) goto cleanup;
        for (size_t i = 0; i < buf_size / sizeof(float); i++) tmp[i] = 1.0f;
        if (upload_to_device(vk, buf, tmp, buf_size) != 0) { free(tmp); goto cleanup; }
        free(tmp);
    }

    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->ds_pool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    if (vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) goto cleanup;

    VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wds = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(vk->device, 1, &wds, 0, NULL);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(vk->device, &cbai, &cmd) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
    };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };

    /* Record dispatch */
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + 255) / 256, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    /* Warmup */
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(vk->device, 1, &fence));

    /* Calibrate */
    uint64_t passes = 1;
    {
        uint64_t tc0 = timer_ns();
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        uint64_t tc1 = timer_ns();
        uint64_t one_pass = tc1 - tc0;
        if (one_pass > 0) passes = 200000000ULL / one_pass + 1;
        if (passes < 2) passes = 2;
        if (passes > 1000) passes = 1000;
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }

    /* Timed run */
    uint64_t t0 = timer_ns();
    for (uint64_t p = 0; p < passes; p++) {
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    if (elapsed_s > 0)
        result = (double)buf_size * passes / elapsed_s / 1e9;
    ok = 1;

cleanup:
    vkDeviceWaitIdle(vk->device);
    if (fence)  vkDestroyFence(vk->device, fence, NULL);
    if (cmd)    vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &cmd);
    if (mem)    vkFreeMemory(vk->device, mem, NULL);
    if (buf)    vkDestroyBuffer(vk->device, buf, NULL);
    if (pipe)   vkDestroyPipeline(vk->device, pipe, NULL);
    if (pl)     vkDestroyPipelineLayout(vk->device, pl, NULL);
    if (ds)     vkFreeDescriptorSets(vk->device, vk->ds_pool, 1, &ds);
    if (dsl)    vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    if (shader) vkDestroyShaderModule(vk->device, shader, NULL);
    return ok ? result : -1.0;
}

/* --- Latency benchmark (pointer chase, single thread) --- */

static double run_latency_bench(struct vk_ctx *vk, VkMemoryPropertyFlags mem_flags) {
    int ok = 0;
    double result = -1.0;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkBuffer data_buf = VK_NULL_HANDLE, result_buf = VK_NULL_HANDLE;
    VkDeviceMemory data_mem = VK_NULL_HANDLE, result_mem = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    int host_visible = (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    size_t total_elements = LAT_BUF_SIZE / sizeof(uint32_t);
    size_t data_bytes = LAT_BUF_SIZE;

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = latency_chase_spv_size, .pCode = latency_chase_spv,
    };
    if (vkCreateShaderModule(vk->device, &smci, NULL, &shader) != VK_SUCCESS)
        goto cleanup;

    /* Two bindings: data buffer + result buffer */
    VkDescriptorSetLayoutBinding bindings[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(vk->device, &dslci, NULL, &dsl) != VK_SUCCESS)
        goto cleanup;

    /* Pipeline layout with push constant for num_chases */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
    };
    if (vkCreatePipelineLayout(vk->device, &plci, NULL, &pl) != VK_SUCCESS)
        goto cleanup;

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pl,
    };
    if (vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
        goto cleanup;

    /* Data buffer (chase indices) */
    VkBufferCreateInfo data_bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = data_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    if (vkCreateBuffer(vk->device, &data_bci, NULL, &data_buf) != VK_SUCCESS) goto cleanup;

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(vk->device, data_buf, &mreq);
    uint32_t mtype = find_mem_type(vk, mreq.memoryTypeBits, mem_flags);
    if (mtype == UINT32_MAX) goto cleanup;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size, .memoryTypeIndex = mtype,
    };
    if (vkAllocateMemory(vk->device, &mai, NULL, &data_mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, data_buf, data_mem, 0));

    /* Fill with shuffled chase indices */
    {
        uint32_t *tmp = malloc(data_bytes);
        if (!tmp || fill_chase_indices(tmp, total_elements, LAT_STRIDE) < 0) {
            free(tmp);
            goto cleanup;
        }
        if (host_visible) {
            void *ptr;
            if (vkMapMemory(vk->device, data_mem, 0, data_bytes, 0, &ptr) != VK_SUCCESS) {
                free(tmp);
                goto cleanup;
            }
            memcpy(ptr, tmp, data_bytes);
            vkUnmapMemory(vk->device, data_mem);
        } else {
            if (upload_to_device(vk, data_buf, tmp, data_bytes) != 0) { free(tmp); goto cleanup; }
        }
        free(tmp);
    }

    /* Result buffer (small, always host-visible for readback) */
    VkBufferCreateInfo res_bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(uint32_t),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    if (vkCreateBuffer(vk->device, &res_bci, NULL, &result_buf) != VK_SUCCESS) goto cleanup;

    vkGetBufferMemoryRequirements(vk->device, result_buf, &mreq);
    mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    mai.allocationSize = mreq.size;
    mai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(vk->device, &mai, NULL, &result_mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, result_buf, result_mem, 0));

    /* Descriptor set */
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->ds_pool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    if (vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) goto cleanup;

    VkDescriptorBufferInfo dbis[2] = {
        { .buffer = data_buf, .offset = 0, .range = VK_WHOLE_SIZE },
        { .buffer = result_buf, .offset = 0, .range = VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
          .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbis[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
          .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbis[1] },
    };
    vkUpdateDescriptorSets(vk->device, 2, writes, 0, NULL);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(vk->device, &cbai, &cmd) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
    };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };

    /* Warmup with small chase count */
    uint32_t warmup_chases = 1000;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &warmup_chases);
    vkCmdDispatch(cmd, 1, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(vk->device, 1, &fence));

    /* Calibrate: time 10000 chases */
    uint32_t cal_chases = 10000;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &cal_chases);
    vkCmdDispatch(cmd, 1, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t tc0 = timer_ns();
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    uint64_t tc1 = timer_ns();
    VK_CHECK(vkResetFences(vk->device, 1, &fence));

    /* Scale to ~500ms for accurate measurement */
    uint64_t cal_ns = tc1 - tc0;
    uint32_t num_chases = cal_chases;
    if (cal_ns > 0) {
        num_chases = (uint32_t)((double)cal_chases * 500000000.0 / (double)cal_ns);
        if (num_chases < 10000) num_chases = 10000;
        if (num_chases > 100000000) num_chases = 100000000;
    }

    /* Timed run */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &num_chases);
    vkCmdDispatch(cmd, 1, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t t0 = timer_ns();
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    uint64_t t1 = timer_ns();

    double elapsed_ns = (double)(t1 - t0);
    if (num_chases > 0)
        result = elapsed_ns / (double)num_chases;
    ok = 1;

cleanup:
    vkDeviceWaitIdle(vk->device);
    if (fence)      vkDestroyFence(vk->device, fence, NULL);
    if (cmd)        vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &cmd);
    if (result_mem) vkFreeMemory(vk->device, result_mem, NULL);
    if (result_buf) vkDestroyBuffer(vk->device, result_buf, NULL);
    if (data_mem)   vkFreeMemory(vk->device, data_mem, NULL);
    if (data_buf)   vkDestroyBuffer(vk->device, data_buf, NULL);
    if (pipe)       vkDestroyPipeline(vk->device, pipe, NULL);
    if (pl)         vkDestroyPipelineLayout(vk->device, pl, NULL);
    if (ds)     vkFreeDescriptorSets(vk->device, vk->ds_pool, 1, &ds);
    if (dsl)        vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    if (shader)     vkDestroyShaderModule(vk->device, shader, NULL);
    return ok ? result : -1.0;
}

/* --- Shared memory benchmark (uses different workgroup size) --- */

static double run_shared_mem_bench(struct vk_ctx *vk) {
    int ok = 0;
    double result = -1.0;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    size_t num_threads = 1U << 20; /* 1M threads */
    size_t buf_bytes = num_threads * sizeof(float);
    uint32_t wg_size = 256;

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shared_mem_spv_size, .pCode = shared_mem_spv,
    };
    if (vkCreateShaderModule(vk->device, &smci, NULL, &shader) != VK_SUCCESS)
        goto cleanup;

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    if (vkCreateDescriptorSetLayout(vk->device, &dslci, NULL, &dsl) != VK_SUCCESS)
        goto cleanup;

    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
    };
    if (vkCreatePipelineLayout(vk->device, &plci, NULL, &pl) != VK_SUCCESS)
        goto cleanup;

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pl,
    };
    if (vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
        goto cleanup;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buf_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    if (vkCreateBuffer(vk->device, &bci, NULL, &buf) != VK_SUCCESS) goto cleanup;

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(vk->device, buf, &mreq);
    uint32_t mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size, .memoryTypeIndex = mtype,
    };
    if (vkAllocateMemory(vk->device, &mai, NULL, &mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, buf, mem, 0));

    void *ptr;
    VK_CHECK(vkMapMemory(vk->device, mem, 0, buf_bytes, 0, &ptr));
    float *fp = (float *)ptr;
    for (size_t i = 0; i < buf_bytes / sizeof(float); i++)
        fp[i] = 1.0f;
    vkUnmapMemory(vk->device, mem);

    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->ds_pool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    if (vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) goto cleanup;

    VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wds = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(vk->device, 1, &wds, 0, NULL);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(vk->device, &cbai, &cmd) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
    };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };

    /* Warmup */
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + wg_size - 1) / wg_size, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(vk->device, 1, &fence));

    /* Calibrate */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + wg_size - 1) / wg_size, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t passes = 1;
    {
        uint64_t tc0 = timer_ns();
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        uint64_t tc1 = timer_ns();
        uint64_t one_pass = tc1 - tc0;
        if (one_pass > 0) passes = 200000000ULL / one_pass + 1;
        if (passes < 2) passes = 2;
        if (passes > 1000) passes = 1000;
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }

    /* Timed run: each thread reads 256 floats from shared mem = 1024 bytes */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + wg_size - 1) / wg_size, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t t0 = timer_ns();
    for (uint64_t p = 0; p < passes; p++) {
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    /* Each thread: reads 256 floats (1024B) + writes 1 float (4B) from/to global,
       but the interesting metric is shared memory throughput:
       256 reads * 4 bytes = 1024 bytes per thread */
    if (elapsed_s > 0)
        result = (double)num_threads * 256.0 * sizeof(float) * passes / elapsed_s / 1e9;
    ok = 1;

cleanup:
    vkDeviceWaitIdle(vk->device);
    if (fence)  vkDestroyFence(vk->device, fence, NULL);
    if (cmd)    vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &cmd);
    if (mem)    vkFreeMemory(vk->device, mem, NULL);
    if (buf)    vkDestroyBuffer(vk->device, buf, NULL);
    if (pipe)   vkDestroyPipeline(vk->device, pipe, NULL);
    if (pl)     vkDestroyPipelineLayout(vk->device, pl, NULL);
    if (ds)     vkFreeDescriptorSets(vk->device, vk->ds_pool, 1, &ds);
    if (dsl)    vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    if (shader) vkDestroyShaderModule(vk->device, shader, NULL);
    return ok ? result : -1.0;
}

/* --- Texture sampling benchmark --- */

static double run_texture_bench(struct vk_ctx *vk) {
    int ok = 0;
    double result = -1.0;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory img_mem = VK_NULL_HANDLE;
    VkImageView img_view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer out_buf = VK_NULL_HANDLE;
    VkDeviceMemory out_mem = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
    VkFence tmp_fence = VK_NULL_HANDLE;

    uint32_t tex_w = 1024, tex_h = 1024;
    size_t num_threads = 1U << 20; /* 1M threads, each does 4096 samples */
    uint32_t wg_size = 64;

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = tex_sample_spv_size, .pCode = tex_sample_spv,
    };
    if (vkCreateShaderModule(vk->device, &smci, NULL, &shader) != VK_SUCCESS)
        goto cleanup;

    /* Descriptor layout: binding 0 = sampler2D, binding 1 = storage buffer */
    VkDescriptorSetLayoutBinding bindings[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(vk->device, &dslci, NULL, &dsl) != VK_SUCCESS)
        goto cleanup;

    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
    };
    if (vkCreatePipelineLayout(vk->device, &plci, NULL, &pl) != VK_SUCCESS)
        goto cleanup;

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pl,
    };
    if (vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
        goto cleanup;

    /* Create 1024x1024 RGBA8 texture */
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = { tex_w, tex_h, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(vk->device, &ici, NULL, &image) != VK_SUCCESS) goto cleanup;

    VkMemoryRequirements mreq;
    vkGetImageMemoryRequirements(vk->device, image, &mreq);
    uint32_t mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size, .memoryTypeIndex = mtype,
    };
    if (vkAllocateMemory(vk->device, &mai, NULL, &img_mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindImageMemory(vk->device, image, img_mem, 0));

    /* Upload texture data via staging buffer */
    size_t tex_bytes = tex_w * tex_h * 4;
    VkBufferCreateInfo stg_bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = tex_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    if (vkCreateBuffer(vk->device, &stg_bci, NULL, &staging) != VK_SUCCESS) goto cleanup;

    vkGetBufferMemoryRequirements(vk->device, staging, &mreq);
    mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    mai.allocationSize = mreq.size;
    mai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(vk->device, &mai, NULL, &staging_mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, staging, staging_mem, 0));

    {
        void *ptr;
        VK_CHECK(vkMapMemory(vk->device, staging_mem, 0, tex_bytes, 0, &ptr));
        uint8_t *p = (uint8_t *)ptr;
        for (size_t i = 0; i < tex_bytes; i++)
            p[i] = (uint8_t)(i & 0xFF);
        vkUnmapMemory(vk->device, staging_mem);
    }

    /* Copy staging -> image with layout transitions */
    {
        VkCommandBufferAllocateInfo cbai_tmp = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk->cmdpool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
        };
        if (vkAllocateCommandBuffers(vk->device, &cbai_tmp, &tmp_cmd) != VK_SUCCESS) goto cleanup;

        VkCommandBufferBeginInfo cbbi_tmp = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
        };
        VK_CHECK(vkBeginCommandBuffer(tmp_cmd, &cbbi_tmp));

        /* Transition to TRANSFER_DST */
        VkImageMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(tmp_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { tex_w, tex_h, 1 },
        };
        vkCmdCopyBufferToImage(tmp_cmd, staging, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        /* Transition to SHADER_READ_ONLY */
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(tmp_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

        VK_CHECK(vkEndCommandBuffer(tmp_cmd));

        VkFenceCreateInfo fci_tmp = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(vk->device, &fci_tmp, NULL, &tmp_fence));

        VkSubmitInfo si_tmp = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &tmp_cmd,
        };
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si_tmp, tmp_fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &tmp_fence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(vk->device, tmp_fence, NULL);
        tmp_fence = VK_NULL_HANDLE;
        vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &tmp_cmd);
        tmp_cmd = VK_NULL_HANDLE;
    }

    /* Image view */
    VkImageViewCreateInfo ivci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    if (vkCreateImageView(vk->device, &ivci, NULL, &img_view) != VK_SUCCESS) goto cleanup;

    /* Sampler (bilinear) */
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
    if (vkCreateSampler(vk->device, &sci, NULL, &sampler) != VK_SUCCESS) goto cleanup;

    /* Output buffer */
    size_t out_bytes = num_threads * sizeof(float);
    VkBufferCreateInfo out_bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = out_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    if (vkCreateBuffer(vk->device, &out_bci, NULL, &out_buf) != VK_SUCCESS) goto cleanup;

    vkGetBufferMemoryRequirements(vk->device, out_buf, &mreq);
    mtype = find_mem_type(vk, mreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) goto cleanup;

    mai.allocationSize = mreq.size;
    mai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(vk->device, &mai, NULL, &out_mem) != VK_SUCCESS) goto cleanup;
    VK_CHECK(vkBindBufferMemory(vk->device, out_buf, out_mem, 0));

    /* Descriptor set */
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->ds_pool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    if (vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) goto cleanup;

    VkDescriptorImageInfo dii = {
        .sampler = sampler, .imageView = img_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorBufferInfo dbi = { .buffer = out_buf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet writes[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
          .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
          .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi },
    };
    vkUpdateDescriptorSets(vk->device, 2, writes, 0, NULL);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(vk->device, &cbai, &cmd) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, /* Dispatch command buffers may be submitted repeatedly. */
    };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };

    /* Warmup */
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + wg_size - 1) / wg_size, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(vk->device, 1, &fence));

    /* Calibrate */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + wg_size - 1) / wg_size, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t passes = 1;
    {
        uint64_t tc0 = timer_ns();
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        uint64_t tc1 = timer_ns();
        uint64_t one_pass = tc1 - tc0;
        if (one_pass > 0) passes = 200000000ULL / one_pass + 1;
        if (passes < 2) passes = 2;
        if (passes > 500) passes = 500;
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }

    /* Timed run */
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    compute_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmd, (num_threads + wg_size - 1) / wg_size, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    uint64_t t0 = timer_ns();
    for (uint64_t p = 0; p < passes; p++) {
        VK_CHECK(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vk->device, 1, &fence));
    }
    uint64_t t1 = timer_ns();

    double elapsed_s = (double)(t1 - t0) / 1e9;
    /* Each thread does 4096 texture samples */
    double total_samples = (double)num_threads * 4096.0 * passes;
    if (elapsed_s > 0)
        result = total_samples / elapsed_s / 1e9;
    ok = 1;

cleanup:
    vkDeviceWaitIdle(vk->device);
    if (fence)       vkDestroyFence(vk->device, fence, NULL);
    if (cmd)         vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &cmd);
    if (out_mem)     vkFreeMemory(vk->device, out_mem, NULL);
    if (out_buf)     vkDestroyBuffer(vk->device, out_buf, NULL);
    if (sampler)     vkDestroySampler(vk->device, sampler, NULL);
    if (img_view)    vkDestroyImageView(vk->device, img_view, NULL);
    if (staging_mem) vkFreeMemory(vk->device, staging_mem, NULL);
    if (staging)     vkDestroyBuffer(vk->device, staging, NULL);
    if (img_mem)     vkFreeMemory(vk->device, img_mem, NULL);
    if (image)       vkDestroyImage(vk->device, image, NULL);
    if (pipe)        vkDestroyPipeline(vk->device, pipe, NULL);
    if (pl)          vkDestroyPipelineLayout(vk->device, pl, NULL);
    if (ds)     vkFreeDescriptorSets(vk->device, vk->ds_pool, 1, &ds);
    if (dsl)         vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    if (shader)      vkDestroyShaderModule(vk->device, shader, NULL);
    if (tmp_fence)   vkDestroyFence(vk->device, tmp_fence, NULL);
    if (tmp_cmd)     vkFreeCommandBuffers(vk->device, vk->cmdpool, 1, &tmp_cmd);
    return ok ? result : -1.0;
}

/* --- Entry point --- */

void bench_gpu(void) {
    struct vk_ctx vk;
    if (vk_init(&vk) != 0) {
        printf("=== GPU Compute Throughput (Vulkan) ===\n");
        printf("  Vulkan not available\n");
        vk_destroy(&vk);
        return;
    }

    printf("=== GPU Compute Throughput (Vulkan) ===\n");
    printf("  Device: %s (Vulkan %d.%d)\n", vk.device_name,
           VK_VERSION_MAJOR(vk.api_version), VK_VERSION_MINOR(vk.api_version));
    printf("%-20s %14s\n", "Test", "Throughput");
    printf("%-20s %14s\n", "----", "----------");

    double fp32 = run_compute_bench(&vk, fma_bench_spv, fma_bench_spv_size,
                                    NUM_THREADS * sizeof(float), NUM_THREADS,
                                    10.0 * 4096.0 * 2.0);
    if (fp32 > 0) printf("%-20s %10.2f GFLOPS\n", "FP32", fp32);
    else          printf("%-20s %14s\n", "FP32", "error");
    fflush(stdout);

    /* FP16 */
    if (vk.has_fp16) {
        double fp16 = run_compute_bench(&vk, fma_fp16_bench_spv, fma_fp16_bench_spv_size,
                                        NUM_THREADS * sizeof(uint16_t), NUM_THREADS,
                                        10.0 * 4096.0 * 2.0);
        if (fp16 > 0) printf("%-20s %10.2f GFLOPS\n", "FP16", fp16);
        else          printf("%-20s %14s\n", "FP16", "error");
    } else {
        printf("%-20s %14s\n", "FP16", "n/a (no ext)");
    }
    fflush(stdout);

    double int32 = run_compute_bench(&vk, int_bench_spv, int_bench_spv_size,
                                     NUM_THREADS * sizeof(uint32_t), NUM_THREADS,
                                     10.0 * 4096.0 * 2.0);
    if (int32 > 0) printf("%-20s %10.2f GINTOPS\n", "INT32", int32);
    else           printf("%-20s %14s\n", "INT32", "error");
    fflush(stdout);

    double vram_bw = run_bw_bench(&vk, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vram_bw > 0) printf("%-20s %10.2f GB/s\n", "VRAM BW", vram_bw);
    else             printf("%-20s %14s\n", "VRAM BW", "error");
    fflush(stdout);

    double vram_lat = run_latency_bench(&vk, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vram_lat > 0) printf("%-20s %10.2f ns\n", "VRAM Latency", vram_lat);
    else              printf("%-20s %14s\n", "VRAM Latency", "error");
    fflush(stdout);

    double host_bw = run_bw_bench(&vk,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (host_bw > 0) printf("%-20s %10.2f GB/s\n", "Host BW", host_bw);
    else             printf("%-20s %14s\n", "Host BW", "error");
    fflush(stdout);

    double host_lat = run_latency_bench(&vk,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (host_lat > 0) printf("%-20s %10.2f ns\n", "Host Latency", host_lat);
    else              printf("%-20s %14s\n", "Host Latency", "error");
    fflush(stdout);

    /* Shared memory bandwidth */
    double smem_bw = run_shared_mem_bench(&vk);
    if (smem_bw > 0) printf("%-20s %10.2f GB/s\n", "Shared Mem BW", smem_bw);
    else             printf("%-20s %14s\n", "Shared Mem BW", "error");
    fflush(stdout);

    /* Texture sampling */
    double tex = run_texture_bench(&vk);
    if (tex > 0) printf("%-20s %10.2f Gtexel/s\n", "Texture Sample", tex);
    else         printf("%-20s %14s\n", "Texture Sample", "error");
    fflush(stdout);

    vk_destroy(&vk);
}

#else /* not Linux */

void bench_gpu(void) {
    printf("=== GPU Compute Throughput (Vulkan) ===\n");
    printf("  Vulkan not available (non-Linux platform)\n");
}

#endif
