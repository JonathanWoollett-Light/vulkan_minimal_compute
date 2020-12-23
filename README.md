# Vulkan Minimal Compute

## What?

An ultra simple demo of compute with Vulkan.

A compute shader to render the mandelbrot set, rendering it into a storage buffer.
The storage buffer is then read from the GPU, and saved as `.png`.

## Who?

For those looking to learn Vulkan.

![](image.png)

## Running

1. `cmake -B./build -S./`
2. Open solution
3. Right click `vulkan_minimal_compute` project, `Set as Startup Project`
4. Run
5. See `../vulkan_minimal_compute/mandelbrot.png` got created
