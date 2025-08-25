import asyncio
import threading

def run_sync(coro):
    """
    Run an async coroutine from synchronous code safely in all contexts:
    - If no event loop is running in the current thread: use asyncio.run
    - If an event loop is running (e.g., inside an async function): run the coro in a new thread with its own loop
    This avoids nested run_until_complete issues and thread loop errors on Windows.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread → safe to run directly
        return asyncio.run(coro)

    # A loop is running in this thread; execute in a dedicated background thread with its own loop
    result_holder = {"ok": False, "value": None, "error": None}

    def _runner():
        try:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
                result_holder["ok"] = True
                result_holder["value"] = result
            finally:
                try:
                    new_loop.stop()
                except Exception:
                    pass
                new_loop.close()
        except Exception as e:
            result_holder["error"] = e

    t = threading.Thread(target=_runner, name="run_sync_thread", daemon=True)
    t.start()
    t.join()

    if result_holder["ok"]:
        return result_holder["value"]
    if result_holder["error"] is not None:
        raise result_holder["error"]
    # Fallback
    raise RuntimeError("run_sync failed without a result")
