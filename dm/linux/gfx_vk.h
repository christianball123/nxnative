// The Vulkan graphics layer's entry points for the rest of the Linux host (dm/linux/GRAPHICS.md)
#pragma once
#include <stdint.h>
int vkw_wanted(void);           // DM_WINDOW is set (to anything but 0)
int vkw_active(void);           // the window is up
void vkw_present(uint64_t n);   // show present number n (stage A: a colour); makes the window on the first call
