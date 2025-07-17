import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_sync(coro):
    """
    Run 'coro' from sync code without exploding if an event-loop
    is already running.
    """
    try:
        loop = asyncio.get_running_loop()          # ← raises if no loop
    except RuntimeError:
        # No loop yet → safe to use asyncio.run
        return asyncio.run(coro)

    # We *are* inside a running loop.
    # Option-1: run the coro in that same loop and wait for result
    #           (works because nest_asyncio has patched run_until_complete)
    from nest_asyncio import apply; apply()
    return loop.run_until_complete(coro)

    # If you prefer not to nest, alternate implementation:
    # return asyncio.run_coroutine_threadsafe(coro, loop).result() 