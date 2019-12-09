#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <array>
#include <chrono>
#include <vector>
#include <algorithm>

#define GLM_FORCE_RADIANS
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <volk.h>


#define VK_CHECK(call) \
  do { \
    VkResult result_ = call; \
    assert(result_ == VK_SUCCESS); \
  }while(0)

#ifndef ARRAYSIZE
#define ARRAYSIZE(array) (sizeof(array) / sizeof((array)[0]))
#endif // ARRAYSIZE

VkInstance createInstance() {

  // SHORTCUT: In real Vulkan applications you should probably check if 1.1 is available via vkEnumerateInstanceVersion
  VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  createInfo.pApplicationInfo = &appInfo;

#ifdef _DEBUG
  const char* debugLayers[] = {

    "VK_LAYER_LUNARG_standard_validation"
  };

  createInfo.ppEnabledLayerNames = debugLayers;
  createInfo.enabledLayerCount = sizeof(debugLayers) / sizeof(debugLayers[0]);
#endif // _DEBUG

  const char* extensions[] = {

    VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_WIN32_KHR
    VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif

    VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
  };

  createInfo.ppEnabledExtensionNames = extensions;
  createInfo.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);

  VkInstance instance = 0;
  VK_CHECK(vkCreateInstance(&createInfo, 0, &instance));

  return instance;
}

VkBool32 debugReportCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object,
  size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void*pUserData) {
  
  const char* type =
    (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    ? "ERROR"
    : (flags & (VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT))
      ? "WARNING"
      : "INFO";

  char message[4096];
  snprintf(message, ARRAYSIZE(message), "%s: %s\n", type, pMessage);

  printf("%s", message);

#ifdef _WIN32
  OutputDebugString(message);
#endif // _WIN32


  if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    assert(!"Validation error encountered!");

  return VK_FALSE;
}

VkDebugReportCallbackEXT registerDebugCallback(VkInstance instance) {

  VkDebugReportCallbackCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT };
  createInfo.flags = VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT;
  createInfo.pfnCallback = debugReportCallback;

  VkDebugReportCallbackEXT callback = 0;
  VK_CHECK(vkCreateDebugReportCallbackEXT(instance, &createInfo, 0, &callback));

  return callback;
}

uint32_t getGraphicsFamilyIndex(VkPhysicalDevice physicalDevice) {

  uint32_t queueCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, 0);

  std::vector<VkQueueFamilyProperties> queues(queueCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queues.data());

  for (uint32_t i = 0; i < queueCount; ++i)
    if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
      return i;

  return VK_QUEUE_FAMILY_IGNORED;
}

bool supportsPresentation(VkPhysicalDevice physicalDevice, uint32_t familyIndex) {

#if defined(VK_USE_PLATFORM_WIN32_KHR)
  return vkGetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, familyIndex);
#else
  return true;
#endif
}

VkPhysicalDevice pickPhysicalDevice(VkPhysicalDevice* physicalDevices, uint32_t physicalDeviceCount) {

  VkPhysicalDevice discrete = 0;
  VkPhysicalDevice fallback = 0;

  for (uint32_t i = 0; i < physicalDeviceCount; ++i) {

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevices[i], &props);

    printf("GPU%d: %s\n", i, props.deviceName);

    uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevices[i]);
    if (familyIndex == VK_QUEUE_FAMILY_IGNORED)
      continue;

    if (!supportsPresentation(physicalDevices[i], familyIndex))
      continue;

    if (!discrete && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      discrete = physicalDevices[i];
    }

    if (!fallback) {
      fallback = physicalDevices[i];
    }
  }

  VkPhysicalDevice result = discrete ? discrete : fallback;

  if (result) {
  
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(result, &props);
    printf("Selected GPU: %s\n", props.deviceName);
  }
  else {

    printf("ERROR: No GPUs found\n");
  }
  return result;
}

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t familyIndex) {

  float queuePriorities[] = { 1.0f };

  VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  queueInfo.queueFamilyIndex = familyIndex; 
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = queuePriorities;

  const char* extensions[] = {

    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  VkDeviceCreateInfo createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  createInfo.queueCreateInfoCount = 1;
  createInfo.pQueueCreateInfos = &queueInfo;

  createInfo.ppEnabledExtensionNames = extensions;
  createInfo.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);
  
  VkDevice device = 0;
  VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, 0, &device));

  return device;
}

VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  VkWin32SurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
  createInfo.hinstance = GetModuleHandle(0);
  createInfo.hwnd = glfwGetWin32Window(window);

  VkSurfaceKHR surface = 0;
  VK_CHECK(vkCreateWin32SurfaceKHR(instance, &createInfo, 0, &surface));
  return surface;
#else
#error Unsupported platform
#endif // VK_USE_PLATFORM_WIN32_KHR
}

VkFormat getSwapchainFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {

  uint32_t formatCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, 0));

  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data()));

  if (formatCount == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
    return VK_FORMAT_R8G8B8A8_UNORM;

  for (uint32_t i = 0; i < formatCount; ++i)
    if (formats[i].format == VK_FORMAT_R8G8B8A8_UNORM || formats[i].format == VK_FORMAT_B8G8R8A8_UNORM)
      return formats[i].format;

  return formats[0].format;
}

VkSwapchainKHR createSwapchain(VkDevice device, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR surfaceCaps, uint32_t familyIndex, VkFormat format, uint32_t width, uint32_t height, VkSwapchainKHR oldSwapchain) {

  VkCompositeAlphaFlagBitsKHR surfaceComposite =
    (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
    ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
    : (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
    ? VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR
    : (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
    ? VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR
    : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;

  VkSwapchainCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
  createInfo.surface = surface;
  createInfo.minImageCount = std::max(2u, surfaceCaps.minImageCount);
  createInfo.imageFormat = format;
  createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  createInfo.imageExtent.width = width;
  createInfo.imageExtent.height = height;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  createInfo.queueFamilyIndexCount = 1;
  createInfo.pQueueFamilyIndices = &familyIndex;
  createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  createInfo.compositeAlpha = surfaceComposite;
  createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR; // TODO: Research presentation mode
  createInfo.oldSwapchain = oldSwapchain;

  VkSwapchainKHR swapchain = 0;
  VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, 0, &swapchain));

  return swapchain;
}

VkSemaphore createSemaphore(VkDevice device) {

  VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

  VkSemaphore semaphore = 0;
  VK_CHECK(vkCreateSemaphore(device, &createInfo, 0, &semaphore));

  return semaphore;
}

VkCommandPool createCommandPool(VkDevice device, uint32_t familyIndex) {

  VkCommandPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  createInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  createInfo.queueFamilyIndex = familyIndex;

  VkCommandPool commandPool = 0;
  VK_CHECK(vkCreateCommandPool(device, &createInfo, 0, &commandPool));

  return commandPool;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {

  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    if((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type!");
}

VkRenderPass createRenderPass(VkDevice device, VkFormat format) {

  VkAttachmentDescription attachments[1] = {};
  attachments[0].format = format;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colorAttachments = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachments;

  VkRenderPassCreateInfo createInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
  createInfo.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
  createInfo.pAttachments = attachments;
  createInfo.subpassCount = 1;
  createInfo.pSubpasses = &subpass;

  VkRenderPass renderPass = 0;
  VK_CHECK(vkCreateRenderPass(device, &createInfo, 0, &renderPass));

  return renderPass;
}

VkFramebuffer createFramebuffer(VkDevice device, VkRenderPass renderPass, VkImageView imageView, uint32_t width, uint32_t height) {

  VkFramebufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };

  createInfo.renderPass = renderPass;
  createInfo.attachmentCount = 1;
  createInfo.pAttachments = &imageView;
  createInfo.width = width;
  createInfo.height = height;
  createInfo.layers = 1;

  VkFramebuffer framebuffer = 0;
  VK_CHECK(vkCreateFramebuffer(device, &createInfo, 0, &framebuffer));

  return framebuffer;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format) {

  VkImageViewCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
  createInfo.image = image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format = format;
  createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  createInfo.subresourceRange.levelCount = 1;
  createInfo.subresourceRange.layerCount = 1;

  VkImageView view = 0;
  VK_CHECK(vkCreateImageView(device, &createInfo, 0, &view));

  return view;
}

VkShaderModule loadShader(VkDevice device, const char* path) {

  FILE* file = fopen(path, "rb");
  assert(file);

  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  assert(length >= 0);
  fseek(file, 0, SEEK_SET);

  char* buffer = new char[length];
  assert(buffer);

  size_t rc = fread(buffer, 1, length, file);
  assert(rc == size_t(length));
  fclose(file);

  VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  createInfo.codeSize = length;
  createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer);

  VkShaderModule shaderModule = 0;
  VK_CHECK(vkCreateShaderModule(device, &createInfo, 0, &shaderModule));

  return shaderModule;
}

VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout) {
  
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

  VkPipelineLayout layout = 0;
  VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, 0, &layout));

  return layout;
}

struct Vertex {
  glm::vec2 position;
  glm::vec3 color;

  static VkVertexInputBindingDescription getBindingDescription() {

    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;                             // binding number (id?)
    bindingDescription.stride = sizeof(Vertex);                 // distance between two elements (bytes)
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // next data entries could be moved between vertex or instance

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescription() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
    attributeDescriptions[0].binding = 0; // where this attribute gets its data
    attributeDescriptions[0].location = 0; // where this attribute is binded
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; // attribute data (size & type)
    attributeDescriptions[0].offset = offsetof(Vertex, position); // number of bytes since start of per-vertex data

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) {

  VkDescriptorSetLayoutBinding uboLayoutBinding = {};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // Stage where the uniform will be used
  uboLayoutBinding.pImmutableSamplers = nullptr; // Optional: relevant for image sampling related descriptors

  VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &uboLayoutBinding;

  VkDescriptorSetLayout descriptorSetLayout;

  VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

  return descriptorSetLayout;
}

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, VkShaderModule vs, VkShaderModule fs, VkPipelineLayout layout) {

  VkGraphicsPipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";

  createInfo.stageCount = sizeof(stages) / sizeof(stages[0]);
  createInfo.pStages = stages;

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescription = Vertex::getAttributeDescription();

  VkPipelineVertexInputStateCreateInfo vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
  vertexInput.pVertexBindingDescriptions = &bindingDescription;
  vertexInput.pVertexAttributeDescriptions = attributeDescription.data();
  createInfo.pVertexInputState = &vertexInput;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  createInfo.pInputAssemblyState = &inputAssembly;

  VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  createInfo.pViewportState = &viewportState;

  VkPipelineRasterizationStateCreateInfo rasterizationState = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rasterizationState.lineWidth = 1;
  rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  createInfo.pRasterizationState = &rasterizationState;

  VkPipelineMultisampleStateCreateInfo multisampleState = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  createInfo.pMultisampleState = &multisampleState;

  VkPipelineDepthStencilStateCreateInfo depthstencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
  createInfo.pDepthStencilState = &depthstencilState;

  VkPipelineColorBlendAttachmentState colorAttachmentState = { };
  colorAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo colorblendState = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  colorblendState.attachmentCount = 1;
  colorblendState.pAttachments = &colorAttachmentState;
  createInfo.pColorBlendState = &colorblendState;

  VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

  VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dynamicState.dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]);
  dynamicState.pDynamicStates = dynamicStates;
  createInfo.pDynamicState = &dynamicState;

  createInfo.layout = layout;
  createInfo.renderPass = renderPass;

  VkPipeline pipeline = 0;
  VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &createInfo, 0, &pipeline));

  return pipeline;
}

VkImageMemoryBarrier imageBarrier(VkImage image, 
  VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
  VkImageLayout oldLayout, VkImageLayout newLayout) {

  VkImageMemoryBarrier result = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
  result.srcAccessMask = srcAccessMask;
  result.dstAccessMask = dstAccessMask;
  result.oldLayout = oldLayout;
  result.newLayout = newLayout;
  result.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  result.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  result.image = image;
  result.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  result.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
  result.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

  return result;
}

struct Swapchain {

  VkSwapchainKHR swapchain;

  std::vector<VkImage> images;
  std::vector<VkImageView> imageViews;
  std::vector<VkFramebuffer> framebuffers;

  uint32_t width, height;
  uint32_t imageCount;
};

void createSwapchain(Swapchain& result, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface, 
  uint32_t familyIndex, VkFormat format, VkRenderPass renderPass, VkSwapchainKHR oldSwapchain = 0) {

  VkSurfaceCapabilitiesKHR surfaceCaps;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps));

  uint32_t width = surfaceCaps.currentExtent.width;
  uint32_t height = surfaceCaps.currentExtent.height;

  VkSwapchainKHR swapchain = createSwapchain(device, surface, surfaceCaps, familyIndex, format, width, height, oldSwapchain);
  assert(swapchain);

  uint32_t imageCount = 0;
  VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, 0));

  std::vector<VkImage> images(imageCount);
  VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data()));

  std::vector<VkImageView> imageViews(imageCount);
  for (uint32_t i = 0; i < imageCount; ++i) {
    imageViews[i] = createImageView(device, images[i], format);
    assert(imageViews[i]);
  }

  std::vector<VkFramebuffer> framebuffers(imageCount);
  for (uint32_t i = 0; i < imageCount; ++i) {
    framebuffers[i] = createFramebuffer(device, renderPass, imageViews[i], width, height);
    assert(framebuffers[i]);
  }

  result.swapchain = swapchain;
  result.images = images;
  result.imageViews = imageViews;
  result.framebuffers = framebuffers;

  result.width = width;
  result.height = height;
  result.imageCount = imageCount;
}

void destroySwapchain(VkDevice device, const Swapchain& swapchain) {

  for (uint32_t i = 0; i < swapchain.imageCount; ++i)
    vkDestroyFramebuffer(device, swapchain.framebuffers[i], 0);

  for (uint32_t i = 0; i < swapchain.imageCount; ++i)
    vkDestroyImageView(device, swapchain.imageViews[i], 0);

  vkDestroySwapchainKHR(device, swapchain.swapchain, 0);
}

// TODO: Check uniform buffers recreation when modifying swapchain
// TODO: Check description pool recreation when modifying swapchain
void resizeSwapchainIfNecessary(Swapchain& result, VkPhysicalDevice physicalDevice, 
  VkDevice device, VkSurfaceKHR surface, uint32_t familyIndex, VkFormat format, VkRenderPass renderPass) {

  VkSurfaceCapabilitiesKHR surfaceCaps;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps));

  uint32_t newWidth = surfaceCaps.currentExtent.width;
  uint32_t newHeight = surfaceCaps.currentExtent.height;

  if (result.width == newWidth && result.height == newHeight)
    return;

  Swapchain old = result;
  createSwapchain(result, physicalDevice, device, surface, familyIndex, format, renderPass, old.swapchain);
  VK_CHECK(vkDeviceWaitIdle(device));
  destroySwapchain(device, old);
}

struct Buffer {
  VkBuffer buffer;
  VkDeviceMemory bufferMemory;
};

void destroyBuffer(VkDevice device, const Buffer& buffer) {

  vkDestroyBuffer(device, buffer.buffer, nullptr);
  vkFreeMemory(device, buffer.bufferMemory, nullptr);
}

void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, Buffer& buffer,
  VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {

  VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  buffer.buffer;
  VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer.buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

  buffer.bufferMemory;
  VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &buffer.bufferMemory));
  VK_CHECK(vkBindBufferMemory(device, buffer.buffer, buffer.bufferMemory, 0));
}

void copyBuffer(VkDevice device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, 
  VkCommandPool commandPool, VkQueue graphicsQueue) {

  VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

  VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = 0;
  copyRegion.dstOffset = 0;
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

Buffer createVertexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, 
  const std::vector<Vertex>& vertices, VkCommandPool commandPool, VkQueue graphicsQueue) {

  VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  Buffer stagingBuffer;
  createBuffer(device, physicalDevice, stagingBuffer, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  void* data;
  VK_CHECK(vkMapMemory(device, stagingBuffer.bufferMemory, 0, bufferSize, 0, &data));
  memcpy(data, vertices.data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBuffer.bufferMemory);

  Buffer buffer;
  createBuffer(device, physicalDevice, buffer, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  copyBuffer(device, stagingBuffer.buffer, buffer.buffer,
    bufferSize, commandPool, graphicsQueue);

  vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
  vkFreeMemory(device, stagingBuffer.bufferMemory, nullptr);

  return buffer;
}

Buffer createIndexBuffer(VkDevice device, VkPhysicalDevice physicalDevice, 
  const std::vector<uint16_t>& indices, VkCommandPool commandPool, VkQueue graphicsQueue) {

  VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

  Buffer stagingBuffer;
  createBuffer(device, physicalDevice, stagingBuffer, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  void* data;
  VK_CHECK(vkMapMemory(device, stagingBuffer.bufferMemory, 0, bufferSize, 0, &data));
  memcpy(data, indices.data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBuffer.bufferMemory);

  Buffer buffer;
  createBuffer(device, physicalDevice, buffer, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  copyBuffer(device, stagingBuffer.buffer, buffer.buffer,
    bufferSize, commandPool, graphicsQueue);

  vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
  vkFreeMemory(device, stagingBuffer.bufferMemory, nullptr);

  return buffer;
}

void createUniformBuffers(VkDevice device, VkPhysicalDevice physicalDevice, 
  Swapchain& swapChain, std::vector<Buffer>& uniformBuffers) {

  VkDeviceSize bufferSize = sizeof(UniformBufferObject);
  uniformBuffers.resize(swapChain.images.size());

  for (size_t i = 0; i < swapChain.images.size(); ++i) {
    createBuffer(device, physicalDevice, uniformBuffers[i], bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  }
}

VkDescriptorPool createDescriptorPool(VkDevice device, const Swapchain& swapChain) {

  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(swapChain.images.size());

  VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = static_cast<uint32_t>(swapChain.images.size());

  VkDescriptorPool descriptorPool;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

  return descriptorPool;
}

std::vector<VkDescriptorSet> createDescriptorSets(VkDevice device, const Swapchain& swapchain, VkDescriptorPool descriptorPool,
  VkDescriptorSetLayout descriptorSetLayout, std::vector<Buffer>& uniformBuffers) {

  std::vector<VkDescriptorSetLayout> layouts(swapchain.images.size(), descriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapchain.images.size());
  allocInfo.pSetLayouts = layouts.data();

  std::vector<VkDescriptorSet> descriptorSets;
  descriptorSets.resize(swapchain.images.size());
  VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));

  for (size_t i = 0; i < swapchain.images.size(); ++i) {
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = uniformBuffers[i].buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);
    
    VkWriteDescriptorSet descriptorWrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    descriptorWrite.dstSet = descriptorSets[i];
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
  }

  return descriptorSets;
}

const std::vector<Vertex> vertices = {
  {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
  {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
  {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
  {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices{
  0, 1, 2, 2, 3, 0
};

void updateUniformBuffer(VkDevice device, std::vector<Buffer>& uniformBuffers, uint32_t currentImage) {

  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
  
  UniformBufferObject ubo = {};
  ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.proj = glm::perspective(glm::radians(45.0f), 1024.0f / 768.0f , 0.1f, 10.0f);
  //ubo.proj[1][1] *= -1; // TODO: I already compensate de Y axis in the viewport

  void* data;
  vkMapMemory(device, uniformBuffers[currentImage].bufferMemory, 0, sizeof(ubo), 0, &data);
  memcpy(data, &ubo, sizeof(ubo));
  vkUnmapMemory(device, uniformBuffers[currentImage].bufferMemory);
}


int main() {

  int rc = glfwInit();
  assert(rc);

  VK_CHECK(volkInitialize());

  VkInstance instance = createInstance();
  assert(instance);

  volkLoadInstance(instance);

  VkDebugReportCallbackEXT debugCallback = registerDebugCallback(instance);

  VkPhysicalDevice physicalDevices[16];
  uint32_t physicalDeviceCount = sizeof(physicalDevices) / sizeof(physicalDevices[0]);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices));

  VkPhysicalDevice physicalDevice = pickPhysicalDevice(physicalDevices, physicalDeviceCount);
  assert(physicalDevice);

  uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevice);
  assert(familyIndex != VK_QUEUE_FAMILY_IGNORED);
  
  VkDevice device = createDevice(instance, physicalDevice, familyIndex);
  assert(device);

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(1024, 768, "vulkan_render", 0, 0);
  assert(window);

  VkSurfaceKHR surface = createSurface(instance, window);
  assert(surface);

  VkBool32 presentSupported = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &presentSupported));
  assert(presentSupported);

  VkFormat swapchainFormat = getSwapchainFormat(physicalDevice, surface);

  VkSemaphore acquireSemaphore = createSemaphore(device);
  assert(acquireSemaphore);

  VkSemaphore releaseSemaphore = createSemaphore(device);
  assert(releaseSemaphore);
  
  VkQueue queue = 0;
  vkGetDeviceQueue(device, familyIndex, 0, &queue);

  VkRenderPass renderPass = createRenderPass(device, swapchainFormat);
  assert(renderPass);

  VkShaderModule triangleVS = loadShader(device, "../shaders/basic.vert.spv");
  assert(triangleVS);

  VkShaderModule triangleFS = loadShader(device, "../shaders/basic.frag.spv");
  assert(triangleFS);

  // TODO: This is critical for performance!
  VkPipelineCache pipelineCache = 0;

  VkDescriptorSetLayout descriptorSetLayout = createDescriptorSetLayout(device);
  assert(descriptorSetLayout);

  VkPipelineLayout triangleLayout = createPipelineLayout(device, descriptorSetLayout);
  assert(triangleLayout);

  VkPipeline trianglePipeline = createGraphicsPipeline(device, pipelineCache, renderPass, triangleVS, triangleFS, triangleLayout);
  assert(trianglePipeline);

  Swapchain swapchain;
  createSwapchain(swapchain, physicalDevice, device, surface, familyIndex, swapchainFormat, renderPass);

  VkCommandPool commandPool = createCommandPool(device, familyIndex);
  assert(commandPool);

  Buffer vertexBuffer = createVertexBuffer(device, physicalDevice, vertices, commandPool, queue);
  assert(vertexBuffer.buffer);
  assert(vertexBuffer.bufferMemory);

  Buffer indexBuffer = createIndexBuffer(device, physicalDevice, indices, commandPool, queue);
  assert(indexBuffer.buffer);
  assert(indexBuffer.bufferMemory);

  std::vector<Buffer> uniformBuffers;
  createUniformBuffers(device, physicalDevice, swapchain, uniformBuffers);

  VkDescriptorPool descriptorPool = createDescriptorPool(device, swapchain);
  assert(descriptorPool);

  std::vector<VkDescriptorSet> descriptorSets = createDescriptorSets(device, swapchain, descriptorPool, descriptorSetLayout, uniformBuffers);
  for (size_t i = 0; i < descriptorSets.size(); ++i)
    assert(descriptorSets[i]);

  VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  allocateInfo.commandPool = commandPool;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = 0;
  VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

  while (!glfwWindowShouldClose(window)) {
     
    glfwPollEvents();

    resizeSwapchainIfNecessary(swapchain, physicalDevice, device, surface, familyIndex, swapchainFormat, renderPass);

    uint32_t imageIndex = 0;
    VK_CHECK(vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, acquireSemaphore, VK_NULL_HANDLE, &imageIndex));

    VK_CHECK(vkResetCommandPool(device, commandPool, 0));

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkImageMemoryBarrier renderBeginBarrier = imageBarrier(swapchain.images[imageIndex], 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &renderBeginBarrier);

    VkClearColorValue color = { 48.f / 255.f, 10.f / 255.f, 36.f / 255.f, 1 };
    VkClearValue clearColor = { color };

    VkRenderPassBeginInfo passbeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    passbeginInfo.renderPass = renderPass;
    passbeginInfo.framebuffer = swapchain.framebuffers[imageIndex];
    passbeginInfo.renderArea.extent.width = swapchain.width;
    passbeginInfo.renderArea.extent.height = swapchain.height;
    passbeginInfo.clearValueCount = 1;
    passbeginInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &passbeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = { 0, float(swapchain.height), float(swapchain.width), -float(swapchain.height), 0, 1 };
    VkRect2D scissor = { { 0, 0 }, { swapchain.width, swapchain.height } };

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);

    VkBuffer vertexBuffers = { vertexBuffer.buffer };
    VkDeviceSize offset[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffers, offset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, triangleLayout, 0, 1,
      &descriptorSets[0], 0, nullptr);

    //vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    VkImageMemoryBarrier renderEndBarrier = imageBarrier(swapchain.images[imageIndex], VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &renderEndBarrier);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkPipelineStageFlags submitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    updateUniformBuffer(device, uniformBuffers, imageIndex);

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &acquireSemaphore;
    submitInfo.pWaitDstStageMask = &submitStageMask;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &releaseSemaphore;

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &releaseSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain.swapchain;
    presentInfo.pImageIndices = &imageIndex;

    VK_CHECK(vkQueuePresentKHR(queue, &presentInfo));

    VK_CHECK(vkDeviceWaitIdle(device));
  }
  
  VK_CHECK(vkDeviceWaitIdle(device)); // Wait for possible running events from the loop to finish

  vkDestroyCommandPool(device, commandPool, 0);

  destroySwapchain(device, swapchain);

  for (size_t i = 0; i < swapchain.images.size(); ++i)
    destroyBuffer(device, uniformBuffers[i]);

  vkDestroyDescriptorPool(device, descriptorPool, nullptr);

  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

  destroyBuffer(device, vertexBuffer);
  destroyBuffer(device, indexBuffer);

  vkDestroyPipeline(device, trianglePipeline, 0);
  vkDestroyPipelineLayout(device, triangleLayout, 0);

  vkDestroyShaderModule(device, triangleVS, 0);
  vkDestroyShaderModule(device, triangleFS, 0);
  vkDestroyRenderPass(device, renderPass, 0);

  vkDestroySemaphore(device, acquireSemaphore, 0);
  vkDestroySemaphore(device, releaseSemaphore, 0);

  vkDestroySurfaceKHR(instance, surface, 0);

  glfwDestroyWindow(window);

  vkDestroyDevice(device, 0);

  // Physical device memory is managed by the instance. Don't need to be destroyed manually.

#ifdef _DEBUG
  vkDestroyDebugReportCallbackEXT(instance, debugCallback, 0);
#endif

  vkDestroyInstance(instance, 0);
}

