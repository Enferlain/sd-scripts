```text
RuntimeError exception in shielded future
future: <Future finished exception=RuntimeError('mixed physical failure after stop')>
Traceback (most recent call last):
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    value = cast(T, python_context.run(numerical_context.invoke, call))
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    return call()
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 533, in invoke
    raise RuntimeError("mixed physical failure after stop")
RuntimeError: mixed physical failure after stop
RuntimeError exception in shielded future
future: <Future finished exception=RuntimeError('mixed physical failure after stop')>
Traceback (most recent call last):
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 131, in dispatch
    value = cast(T, python_context.run(numerical_context.invoke, call))
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/workers.py", line 73, in invoke
    return call()
  File "/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/xdsl_candidate/mixed_adapter.py", line 533, in invoke
    raise RuntimeError("mixed physical failure after stop")
RuntimeError: mixed physical failure after stop
```
