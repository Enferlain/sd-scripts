Observed CPython shield diagnostic during joined cancellation/physical-failure cases.

```text
RuntimeError exception in shielded future
future: <Future finished exception=RuntimeError('mixed physical failure after stop')>
Traceback (most recent call last):
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    value = cast(T, python_context.run(numerical_context.invoke, call))
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    return call()
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    raise RuntimeError("mixed physical failure after stop")
RuntimeError: mixed physical failure after stop
ExceptionGroup exception in shielded future
future: <Task finished name='Task-26' coro=<Lowering.producer.<locals>.producer.<locals>.activity() done, defined at /home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py:1061> exception=ExceptionGroup('unhandled errors in a TaskGroup', [PhysicalFailed('mixed physical failure after stop')])>
  + Exception Group Traceback (most recent call last):
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1100, in activity
  |     raise primary
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1074, in activity
  |     await scope.implementation(context)
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 590, in produce
  |     async with asyncio.TaskGroup() as tasks:
  |                ~~~~~~~~~~~~~~~~~^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 72, in __aexit__
  |     return await self._aexit(et, exc)
  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 174, in _aexit
  |     raise BaseExceptionGroup(
  |     ...<2 lines>...
  |     ) from None
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 166, in execute
    |     value = await asyncio.shield(pending)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    |     value = cast(T, python_context.run(numerical_context.invoke, call))
    |                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    |     return call()
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    |     raise RuntimeError("mixed physical failure after stop")
    | RuntimeError: mixed physical failure after stop
    | 
    | The above exception was the direct cause of the following exception:
    | 
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 541, in right
    |     finished = await pool.execute(invoke, threading.Event())
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 179, in execute
    |     raise failed(error) from error
    | docs_design.models-strategy-trainer.execution_candidate.workers.PhysicalFailed: mixed physical failure after stop
    +------------------------------------
RuntimeError exception in shielded future
future: <Future finished exception=RuntimeError('mixed physical failure after stop')>
Traceback (most recent call last):
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    value = cast(T, python_context.run(numerical_context.invoke, call))
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    return call()
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    raise RuntimeError("mixed physical failure after stop")
RuntimeError: mixed physical failure after stop
ExceptionGroup exception in shielded future
future: <Task finished name='Task-68' coro=<Lowering.producer.<locals>.producer.<locals>.activity() done, defined at /home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py:1061> exception=ExceptionGroup('unhandled errors in a TaskGroup', [PhysicalFailed('mixed physical failure after stop')])>
  + Exception Group Traceback (most recent call last):
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1100, in activity
  |     raise primary
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1074, in activity
  |     await scope.implementation(context)
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 590, in produce
  |     async with asyncio.TaskGroup() as tasks:
  |                ~~~~~~~~~~~~~~~~~^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 72, in __aexit__
  |     return await self._aexit(et, exc)
  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 174, in _aexit
  |     raise BaseExceptionGroup(
  |     ...<2 lines>...
  |     ) from None
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 166, in execute
    |     value = await asyncio.shield(pending)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    |     value = cast(T, python_context.run(numerical_context.invoke, call))
    |                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    |     return call()
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    |     raise RuntimeError("mixed physical failure after stop")
    | RuntimeError: mixed physical failure after stop
    | 
    | The above exception was the direct cause of the following exception:
    | 
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 541, in right
    |     finished = await pool.execute(invoke, threading.Event())
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 179, in execute
    |     raise failed(error) from error
    | docs_design.models-strategy-trainer.execution_candidate.workers.PhysicalFailed: mixed physical failure after stop
    +------------------------------------
RuntimeError exception in shielded future
future: <Future finished exception=RuntimeError('mixed physical failure after stop')>
Traceback (most recent call last):
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    value = cast(T, python_context.run(numerical_context.invoke, call))
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    return call()
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    raise RuntimeError("mixed physical failure after stop")
RuntimeError: mixed physical failure after stop
ExceptionGroup exception in shielded future
future: <Task finished name='Task-110' coro=<Lowering.producer.<locals>.producer.<locals>.activity() done, defined at /home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py:1061> exception=ExceptionGroup('unhandled errors in a TaskGroup', [PhysicalFailed('mixed physical failure after stop')])>
  + Exception Group Traceback (most recent call last):
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1100, in activity
  |     raise primary
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1074, in activity
  |     await scope.implementation(context)
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 590, in produce
  |     async with asyncio.TaskGroup() as tasks:
  |                ~~~~~~~~~~~~~~~~~^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 72, in __aexit__
  |     return await self._aexit(et, exc)
  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 174, in _aexit
  |     raise BaseExceptionGroup(
  |     ...<2 lines>...
  |     ) from None
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 166, in execute
    |     value = await asyncio.shield(pending)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    |     value = cast(T, python_context.run(numerical_context.invoke, call))
    |                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    |     return call()
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    |     raise RuntimeError("mixed physical failure after stop")
    | RuntimeError: mixed physical failure after stop
    | 
    | The above exception was the direct cause of the following exception:
    | 
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 541, in right
    |     finished = await pool.execute(invoke, threading.Event())
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 179, in execute
    |     raise failed(error) from error
    | docs_design.models-strategy-trainer.execution_candidate.workers.PhysicalFailed: mixed physical failure after stop
    +------------------------------------
RuntimeError exception in shielded future
future: <Future finished exception=RuntimeError('mixed physical failure after stop')>
Traceback (most recent call last):
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    value = cast(T, python_context.run(numerical_context.invoke, call))
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    return call()
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    raise RuntimeError("mixed physical failure after stop")
RuntimeError: mixed physical failure after stop
ExceptionGroup exception in shielded future
future: <Task finished name='Task-152' coro=<Lowering.producer.<locals>.producer.<locals>.activity() done, defined at /home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py:1061> exception=ExceptionGroup('unhandled errors in a TaskGroup', [PhysicalFailed('mixed physical failure after stop')])>
  + Exception Group Traceback (most recent call last):
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1100, in activity
  |     raise primary
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/prepare.py", line 1074, in activity
  |     await scope.implementation(context)
  |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 590, in produce
  |     async with asyncio.TaskGroup() as tasks:
  |                ~~~~~~~~~~~~~~~~~^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 72, in __aexit__
  |     return await self._aexit(et, exc)
  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/imi/.local/share/uv/python/cpython-3.14.5+freethreaded-linux-x86_64-gnu/lib/python3.14t/asyncio/taskgroups.py", line 174, in _aexit
  |     raise BaseExceptionGroup(
  |     ...<2 lines>...
  |     ) from None
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 166, in execute
    |     value = await asyncio.shield(pending)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    |     value = cast(T, python_context.run(numerical_context.invoke, call))
    |                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    |     return call()
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 535, in invoke
    |     raise RuntimeError("mixed physical failure after stop")
    | RuntimeError: mixed physical failure after stop
    | 
    | The above exception was the direct cause of the following exception:
    | 
    | Traceback (most recent call last):
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 541, in right
    |     finished = await pool.execute(invoke, threading.Event())
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 179, in execute
    |     raise failed(error) from error
    | docs_design.models-strategy-trainer.execution_candidate.workers.PhysicalFailed: mixed physical failure after stop
    +------------------------------------
```
