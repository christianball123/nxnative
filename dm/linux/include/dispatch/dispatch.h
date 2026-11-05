// A small libdispatch for the host's few uses: serial and global queues (each a worker thread), async / sync / after, once, semaphores and a SIGTERM source.
// Blocks need clang's -fblocks; dm/linux/compat.c carries the blocks runtime they call.
#pragma once
#include "../dm_linux.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef void (^dispatch_block_t)(void);
typedef void (*dispatch_function_t)(void *);
typedef struct dm_dispatch_queue *dispatch_queue_t;
typedef struct dm_dispatch_sema *dispatch_semaphore_t;
typedef struct dm_dispatch_source *dispatch_source_t;
typedef const struct dm_dispatch_source_type *dispatch_source_type_t;
typedef uint64_t dispatch_time_t;
typedef long dispatch_once_t;
typedef void *dispatch_queue_attr_t;
#define DISPATCH_QUEUE_SERIAL ((dispatch_queue_attr_t)0)
#define DISPATCH_QUEUE_CONCURRENT ((dispatch_queue_attr_t)1)
#define DISPATCH_TIME_NOW 0ull
#define DISPATCH_TIME_FOREVER (~0ull)
#ifndef NSEC_PER_SEC
#define NSEC_PER_SEC 1000000000ull
#define NSEC_PER_MSEC 1000000ull
#define USEC_PER_SEC 1000000ull
#define NSEC_PER_USEC 1000ull
#endif
extern const struct dm_dispatch_source_type dm_source_type_signal;
#define DISPATCH_SOURCE_TYPE_SIGNAL (&dm_source_type_signal)
dispatch_queue_t dispatch_queue_create(const char *label, dispatch_queue_attr_t attr);
dispatch_queue_t dispatch_get_global_queue(long qos, unsigned long flags);
dispatch_queue_t dispatch_get_main_queue(void);
void dispatch_async(dispatch_queue_t q, dispatch_block_t b);
void dispatch_async_f(dispatch_queue_t q, void *ctx, dispatch_function_t f);
void dispatch_sync(dispatch_queue_t q, dispatch_block_t b);
void dispatch_sync_f(dispatch_queue_t q, void *ctx, dispatch_function_t f);
void dispatch_after(dispatch_time_t when, dispatch_queue_t q, dispatch_block_t b);
dispatch_time_t dispatch_time(dispatch_time_t base, int64_t delta);
void dispatch_once(dispatch_once_t *once, dispatch_block_t b);
void dispatch_once_f(dispatch_once_t *once, void *ctx, dispatch_function_t f);
dispatch_semaphore_t dispatch_semaphore_create(long value);
long dispatch_semaphore_signal(dispatch_semaphore_t s);
long dispatch_semaphore_wait(dispatch_semaphore_t s, dispatch_time_t timeout);
dispatch_source_t dispatch_source_create(dispatch_source_type_t type, uintptr_t handle, unsigned long mask, dispatch_queue_t q);
void dispatch_source_set_event_handler(dispatch_source_t s, dispatch_block_t b);
void dispatch_resume(void *object);
void dispatch_release(void *object);
void dispatch_retain(void *object);
#ifdef __cplusplus
}
#endif
