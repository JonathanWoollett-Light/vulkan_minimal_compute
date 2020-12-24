
#include <vulkan/vulkan.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

#include "lodepng.h" //Used for png encoding.

const int WIDTH = 3200; // Size of rendered mandelbrot set.
const int HEIGHT = 2400; // Size of renderered mandelbrot set.
const int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

/*
The application launches a compute shader that renders the mandelbrot set,
by rendering it into a storage buffer.
The storage buffer is then read from the GPU, and saved as .png. 
*/
class ComputeApplication {
private:
    // The pixels of the rendered mandelbrot set are in this format:
    struct Pixel {
        float r, g, b, a;
    };
    
    // In order to use Vulkan, you must create an instance.
    VkInstance instance;

    // The physical device is some device on the system that supports usage of Vulkan,
    //  typically a graphics card.
    VkPhysicalDevice physicalDevice;


    // The logical device (`VkDevice`), is our connection to the physical device.
    VkDevice device;

    // Pipeline contains sequence of shaders, for compute only 1 compute shader.
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    // A command buffer encapsulates a pipeline.
    // A command buffer submits a pipeline to the physical device.
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    // Descriptors define buffers and images (arrays and 2d arrays)
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    // The mandelbrot set will be rendered to this buffer.
    // `buffer` encapsulates `bufferMemory`.
    // `buffer` is an object, `bufferMemory` an array.
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
        
    uint32_t bufferSize; // size of `buffer` in bytes.

    // Command buffers are submitted to a queue.
    VkQueue queue;

    // Queues are grouped by capabilities, command buffers must be submited to a given queue in a given queue family.
    uint32_t queueFamilyIndex;

public:
    void run() {
        // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
        bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;

        // Initialize vulkan:
        createInstance();
        findPhysicalDevice();
        createDevice();
        createBuffer();
        createDescriptorSetLayout();
        createDescriptorSet();
        createComputePipeline();
        createCommandBuffer();

        // Finally, run the recorded command buffer.
        runCommandBuffer();

        // The former command rendered a mandelbrot set to a buffer.
        // Save that buffer as a png on disk.
        saveRenderedImage();

        // Clean up all vulkan resources.
        cleanup();
    }

    void saveRenderedImage() {
        void* mappedMemory = NULL;
        // Map the buffer memory, so that we can read from it on the CPU.
        vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedMemory);
        Pixel* pmappedMemory = (Pixel *)mappedMemory;

        // Get the color data from the buffer, and cast it to bytes.
        // We save the data to a vector.
        std::vector<unsigned char> image;
        image.reserve(WIDTH * HEIGHT * 4);
        for (int i = 0; i < WIDTH*HEIGHT; i += 1) {
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].r)));
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].g)));
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].b)));
            image.push_back((unsigned char)(255.0f * (pmappedMemory[i].a)));
        }
        // Done reading, so unmap.
        vkUnmapMemory(device, bufferMemory);

        // Now we save the acquired color data to a .png.
        unsigned error = lodepng::encode("mandelbrot.png", image, WIDTH, HEIGHT);
        if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

        return VK_FALSE;
     }

    // Initiates Vulkan instance
    void createInstance() {
        
        // Vulkan istance options.
        // `apiVersion` will likely be all that matters here.
        VkInstanceCreateInfo createInfo = {};
        {
            VkApplicationInfo applicationInfo = {};
            {
                applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
                applicationInfo.pApplicationName = "some app name"; // Optional
                applicationInfo.applicationVersion = 0; // Optional
                applicationInfo.pEngineName = "some engine name"; // Optional
                applicationInfo.engineVersion = 0; // Optional
                applicationInfo.apiVersion = VK_API_VERSION_1_0;
            }
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.flags = 0; // Optional
            createInfo.pApplicationInfo = &applicationInfo;
        }
    
        // Creates instance
        VK_CHECK_RESULT(vkCreateInstance(
            &createInfo,
            NULL,
            &instance)
        );
    }

    void findPhysicalDevice() {
        // Gets number of physical devices
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (deviceCount == 0) {
            throw std::runtime_error("No Vulkan compatible devices");
        }

        // Gets physical devices
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Picks 1st (would usualy do some checks and pick best)
        physicalDevice = devices[0];
    }

    // Returns an index of a queue family that supports compute
    uint32_t getComputeQueueFamilyIndex() {
        // Gets number of queue families
        uint32_t queueFamilyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

        // Gets queue families
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        // Picks 1st queue family which supports compute
        for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
            VkQueueFamilyProperties props = queueFamilies[i];
            // If queue family supports compute
            // `props.queueCount > 0` removed, since it seems rather pointless
            if (props.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return i;
            }
        }
        throw std::runtime_error("No compute queue family");
    }
    // Gets logical device
    void createDevice() {
        // Device info
        VkDeviceCreateInfo deviceCreateInfo = {};
        {   
            // Device queue info
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            {
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
                queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
                queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
                // `pQueuePriorities` relates to a more in-depth topic, look it up if you want, but probably best ignore it for now
                queueCreateInfo.pQueuePriorities = new float(1.0);
            }
            // Device features
            VkPhysicalDeviceFeatures deviceFeatures = {};

            deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            // When creating the logical device, we specify what queues it accesses.
            deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
            deviceCreateInfo.queueCreateInfoCount = 1;
            deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
        }

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

        // Get handle to queue 0 in `queueFamilyIndex` queue family
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

    // find memory type with desired properties.
    uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties;

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        /*
        How does this search work?
        See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description. 
        */
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((memoryTypeBits & (1 << i)) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }
        return -1;
    }

    void createBuffer() {
        // Buffer info
        VkBufferCreateInfo bufferCreateInfo = {};
        {
            bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size = bufferSize; // buffer size in bytes. 
            bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 
        }
        
        // Constructs buffer
        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer));

        // Buffers do not allocate memory upon construction, we must do it manually

        // Gets buffer memory info
        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
        
        // Sets buffer options
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size; // Bytes

        allocateInfo.memoryTypeIndex = findMemoryType(
            // Sets memory must supports all the operations our buffer memory supports
            memoryRequirements.memoryTypeBits,
            // Sets memory must have the properties:
            //  `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` Can more easily view
            //  `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` Can read from GPU to CPU
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        );

        // Allocates memory
        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));
        
        // Binds buffer to allocated memory
        VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));
    }

    // Create descriptor set layout
    void createDescriptorSetLayout() {
        
        // Descriptor set layout options
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        {
            // Descriptor layout options
            VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
            {
                // Binding to `layout(binding = 0)`
                descriptorSetLayoutBinding.binding = 0;
                // Defines this buffer as a storaghe buffer (i.e. it is accessible to our shader)
                descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                // TODO What does this do?
                descriptorSetLayoutBinding.descriptorCount = 1;
                // TODO Is accessible to compute shaders (bit unsure if I'm right here)
                descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCreateInfo.bindingCount = 1; // 1 `VkDescriptorSetLayoutBinding` in this descriptor set
            descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding; // Pointer to array of `VkDescriptorSetLayoutBinding`s
        }
        
        // Create the descriptor set layout. 
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    }
    // Creates descriptor set
    void createDescriptorSet() {
        // Creates descriptor pool
        // A pool implements a number of descriptors of each type 
        //  `VkDescriptorPoolSize` specifies for each descriptor type the number to hold
        // A descriptor set is initialised to contain all descriptors defined in a descriptor pool
        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        {
            // Descriptor type and number
            VkDescriptorPoolSize descriptorPoolSize = {};
            {
                descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Descriptor type
                descriptorPoolSize.descriptorCount = 1; // Number of descriptors
            }
            descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCreateInfo.maxSets = 1; // max number of sets that can be allocated from this pool
            descriptorPoolCreateInfo.poolSizeCount = 1; // length of `pPoolSizes`
            descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize; // pointer to array of `VkDescriptorPoolSize`
        }

        // create descriptor pool.
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

        // Specifies options for creation of multiple of descriptor sets
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        {
            descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool from which sets will be allocated
            descriptorSetAllocateInfo.descriptorSetCount = 1; // number of descriptor sets to implement (also length of `pSetLayouts`)
            descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout; // pointer to array of descriptor set layouts
        }
        

        // allocate descriptor set.
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

        /*
        Next, we need to connect our actual storage buffer with the descrptor. 
        We use vkUpdateDescriptorSets() to update the descriptor set.
        */

        // Binds descriptors from our descriptor sets to our buffers
        VkWriteDescriptorSet writeDescriptorSet = {};
        {
            // Binds descriptor to buffer
            VkDescriptorBufferInfo descriptorBufferInfo = {};
            {
                descriptorBufferInfo.buffer = buffer;
                descriptorBufferInfo.offset = 0;
                descriptorBufferInfo.range = bufferSize;
            }
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
            // TODO Wtf does this do?
            //  My best guess is that descriptor sets can have multile bindings to different sets of buffers.
            //  original comment said 'write to the first, and only binding.'
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.descriptorCount = 1; // update a single descriptor.
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
            writeDescriptorSet.pBufferInfo = &descriptorBufferInfo; // pointer to array of descriptor bindings
        }
        

        // perform the update of the descriptor set.
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
    }

    // Read file into array of bytes, and cast to uint32_t*, then return.
    // The data has been padded, so that it fits into an array uint32_t.
    uint32_t* readFile(uint32_t& length, const char* filename) {

        FILE* fp = fopen(filename, "rb");
        if (fp == NULL) {
            printf("Could not find or open file: %s\n", filename);
        }

        // get file size.
        fseek(fp, 0, SEEK_END);
        long filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        long filesizepadded = long(ceil(filesize / 4.0)) * 4;

        // read file contents.
        char *str = new char[filesizepadded];
        fread(str, filesize, sizeof(char), fp);
        fclose(fp);

        // data padding. 
        for (int i = filesize; i < filesizepadded; i++) {
            str[i] = 0;
        }

        length = filesizepadded;
        return (uint32_t *)str;
    }
    // Creates compute pipeline
    void createComputePipeline() {

        // Creates shader module (just a wrapper around our shader)
        VkShaderModuleCreateInfo createInfo = {};
        {
            //uint32_t* code = readFile(filelength, "shaders/comp.spv");
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            uint32_t filelength;
            createInfo.pCode = readFile(filelength, "shaders/comp.spv");
            createInfo.codeSize = filelength;
            //delete[] code;
        }

        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));

        // A compute pipeline is very simple compared to a graphics pipeline.
        // It only consists of a single stage with a compute shader.

        // The pipeline layout allows the pipeline to access descriptor sets. 
        // So we just specify the descriptor set layout we created earlier.
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        {
            pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount = 1; // 1 shader
            pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; // Descriptor set
        }
        
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

        // Set our pipeline options
        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        {
            // We specify the compute shader stage, and it's entry point(main).
            VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
            {
                shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT; // Shader type
                shaderStageCreateInfo.module = computeShaderModule; // Shader module
                shaderStageCreateInfo.pName = "main"; // Shader entry point
            }
            // We set our pipeline options
            pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineCreateInfo.stage = shaderStageCreateInfo; // Shader stage info
            pipelineCreateInfo.layout = pipelineLayout;
        }
        

        // Create compute pipeline
        VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &pipeline));
    }
    // Creates command buffer
    // Command buffers send commands to our physical device
    void createCommandBuffer() {
        // Command buffers must be created via a command pool
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        {
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            // Bit in-depth, would recommend ignoring for now
            commandPoolCreateInfo.flags = 0;
             // Queue family this command pool belongs to
            commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        }
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));

        // Allocates command buffer from command pool
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        {
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            // Pool to allocate from
            commandBufferAllocateInfo.commandPool = commandPool;
            // Primary command buffers are submitted directly,
            //  secondary command buffers are submited via primary command buffers.
            // A primary command buffer is what we need here.
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            // Allocates 1 command buffer. 
            commandBufferAllocateInfo.commandBufferCount = 1; 
        }
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.

        // Allocated command buffer options
        VkCommandBufferBeginInfo beginInfo = {};
        {
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            // Buffer only submitted once
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        }
        // Start recording commands
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        // Binds pipeline (our functions) and descriptor set (our data)
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

        // Sets invocations
        //
        // Each workgroup is some x number of invocations, we need y number of invocations,
        //  so we need to invoke some z number of workgroups such that z*x > y thus we do z = ceil(y/x).
        // We invoke some number of workgroups along each dimension.
        vkCmdDispatch(
            commandBuffer,
            (uint32_t) ceil(WIDTH / (float) WORKGROUP_SIZE),
            (uint32_t) ceil(HEIGHT / (float) WORKGROUP_SIZE),
            1
        );

        // End recording commands
        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
    }
    
    // Submits command buffer to queue for execution
    void runCommandBuffer() {
        VkSubmitInfo submitInfo = {};
        {
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            // submit 1 command buffer
            submitInfo.commandBufferCount = 1;
             // pointer to array of command buffers to submit
            submitInfo.pCommandBuffers = &commandBuffer;
        }

        // Creates fence (so we can await for command buffer to finish)
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        {
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.flags = 0; // TODO Add explanation of this
        }
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

        // Submit command buffer with fence
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

        // Wait for fence to signal (which it does when command buffer has finished)
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

        // Destructs fence
        vkDestroyFence(device, fence, NULL);
    }

    // Cleans up - Destructs everything
    void cleanup() {
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, buffer, NULL);	
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);	
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);		
    }
};

int main() {
    ComputeApplication app;

    try {
        app.run();
    }
    catch (const std::runtime_error& e) {
        printf("%s\n", e.what());
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
