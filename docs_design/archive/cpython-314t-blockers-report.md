# CPython 3.14t blockers in sd-scripts

Investigation date: 10 October 2026. Repository: `Enferlain/sd-scripts`, branch `model-strategy-trainer`, inspected commit [`33f73a4ddcfd781922cdb827172dd072046cdf8e`](https://github.com/Enferlain/sd-scripts/tree/33f73a4ddcfd781922cdb827172dd072046cdf8e). Read-only investigation; no project files or upstream sources were patched.

## 1. Executive conclusion

**Retaining CPython 3.14t as an engineering baseline is reasonable. Claiming that the present training stack is safely GIL-free is not.** Your successful eager CUDA matrix multiplication, with the GIL remaining disabled, removes PyTorch/CUDA eager execution from the current blocker list. It does not establish safety for Triton, OpenCV, gRPC, or arbitrary concurrent operations on shared training objects.

The repository currently imports Triton indirectly during ordinary startup, even when compilation and CPU offload are not selected. It also imports `cv2` through general image utilities. These are real imports, rather than merely installed dependencies. TensorBoard writer initialization is conditional on logging configuration; the inspected local event-writing path does **not** require importing gRPC.

**Triton is not a defensible declaration-only fix.** Version 3.8 contains unsynchronized native specialization caches and first-use state; its Python autotuner also has shared mutable invocation state. Its dynamically built CUDA utility extension is a second declaration and synchronization surface. Current upstream work is relevant but incomplete. OpenCV similarly exposes mutable callback registries and Python-object state whose safety cannot be established by its existing GIL attachment helpers. gRPC has documented and reproduced lifecycle races when the GIL is forcibly disabled.

A restricted eager-training recipe may realistically remain GIL-free after removing proven unnecessary dependency imports and validating the remaining stack. A full recipe using Triton compilation, Triton-dependent optimizers, and OpenCV in the same process needs upstream synchronization work and validation. Merely moving imports later postpones the interpreter-wide transition when those features are eventually used.

The smallest maintainable course is to establish the real trace, keep the existing safe OpenCV build override, remove accidental optional imports in a later reviewed change, and contribute focused synchronization fixes upstream. Accepting the GIL temporarily for feature-rich recipes is preferable to maintaining declaration-only native forks. Measure end-to-end throughput before investing heavily: GPU execution already overlaps through native code, and free-threading chiefly helps Python work that actually runs concurrently.

### Evidence boundaries

Your fresh-process package results and GPU computation are treated as supplied measurements. This investigation environment is Python 3.12.14, without the listed native packages or your GPUs. **I could not reproduce your actual trainer startup or measure its exact first GIL transition.** Section 2 establishes the import paths from source; the appendix supplies a diagnostic that identifies the native module and caller on your actual interpreter. This limitation applies to all proposed concurrency tests: they are a validation design, not claimed passing results.

Exact source tags and lockfile versions were inspected where available:

| Component | Inspected baseline |
|---|---|
| PyTorch | `v2.14.1`; your wheel is `2.14.1+cu130` |
| torchvision | `v0.29.1`; your wheel is `0.29.1+cu130` |
| Triton | `v3.8.0`, commit `c01b6774b1865984607d89d89d3a10833de92037` |
| TorchAO | `v0.18.0` |
| bitsandbytes | `0.50.2` |
| OpenCV | Exact `opencv_python-5.0.0.93.tar.gz`, SHA-256 verified against `uv.lock`; relevant binding files byte-identical to OpenCV `5.0.0` |
| grpcio | `v1.78.0` |
| Accelerate / TensorBoard | `v1.15.0` / `2.21.0`, matching `uv.lock` |
| Protobuf | `6.33.5`, source tag `v33.5`, matching `uv.lock` |

A source tag does not prove which downstream patches or Cython/build options your installed binary contains. Archive the installed distribution metadata, native file paths/hashes, compiler versions, wheel tags, and build logs with the eventual runtime trace.

CPython's declaration is a promise made by the extension author. The free-threaded ABI and import-time GIL declaration are separate requirements. GIL attachment through `PyGILState_Ensure` or Cython `with gil` does not restore mutual exclusion on a free-threaded interpreter. See the [CPython 3.14 extension guide](https://docs.python.org/3.14/howto/free-threading-extensions.html), [runtime guide](https://docs.python.org/3.14/howto/free-threading-python.html), and [Cython free-threading guide](https://docs.cython.org/en/latest/src/userguide/freethreading.html).

## 2. Actual trainer import/GIL trace

### Source-established startup

The active launcher is [`train.py`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/train.py). Its top-level order is Hydra, configuration validation/dataclasses, strategy factory, interrupts, mode factory, trainer, and logging helpers. Its `train()` then validates configuration, builds strategies/mode, constructs `Trainer`, and calls `trainer.train()`.

One unavoidable source path in the current import graph is:

```text
train.py
  library.config.config_validation
    library.optimization.optimizer_utils
      library.constants
        torchvision
          torchvision.ops.roi_align
            torch._dynamo.utils
              triton
                triton._C.libtriton
```

[`config_validation.py:32`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/config/config_validation.py#L32) imports optimizer utilities. [`optimizer_utils.py:12`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/optimization/optimizer_utils.py#L12) imports numeric regex constants. [`constants.py:5`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/constants.py#L5) imports torchvision and constructs `IMAGE_TRANSFORMS` at module load. torchvision's [`roi_align.py`](https://github.com/pytorch/vision/blob/v0.29.1/torchvision/ops/roi_align.py) imports Dynamo utilities. PyTorch's [`torch/_dynamo/utils.py`](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/_dynamo/utils.py) calls `has_triton_package()` and imports Triton at module scope; [`torch/utils/_triton.py`](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/_triton.py) imports the package to test availability.

There is an independent eager path:

```text
optimizer_utils -> optimizer_factory -> optimization.wrappers
  -> wrappers.cpu_offload -> torchao.optim / torchao.utils
    -> TorchAO package initialization -> Triton
```

[`wrappers/__init__.py:3`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/optimization/wrappers/__init__.py#L3) exports the CPU offload wrapper eagerly; [`cpu_offload.py:6–7`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/optimization/wrappers/cpu_offload.py#L6) imports TorchAO. Choosing ordinary AdamW does not avoid those top-level imports. Registry targets for many optimizers are otherwise loaded on selection. Some selected custom optimizers have their own `torch.compile` decorators; a global training compile setting alone is not a universal eager-only guarantee.

The OpenCV path is also established:

```text
strategy factory -> SD/SDXL caching -> library.data.caching_engine
  -> library.data package exports -> manifest / caption_processor / scanners
    -> library.data.image_utils -> cv2
```

[`library.data.__init__`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/data/__init__.py) eagerly exports the data subsystem. Manifest imports image-ID generation; caption processing imports `CAPTION_EXTENSIONS`; scanners import size/alpha helpers. All reach [`image_utils.py:5`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/data/image_utils.py#L5), which imports `cv2` even for helpers implemented with PIL, hashing, or filesystem operations.

### What is measured versus inferred

| Question | Answer and confidence |
|---|---|
| Does ordinary launcher import reach Triton? | **Yes, source-established**, through torchvision/Dynamo and TorchAO, before compile selection. |
| Does it reach `cv2`? | **Yes, source-established**, through the data package/image helpers. |
| Does importing the launcher itself require gRPC? | No direct project import found; the inspected local TensorBoard writer path does not require it. Other selected integrations or installed TensorFlow can change this. |
| Does trainer execution initialize TensorBoard? | Conditional. A logging directory with no explicit tracker selects TensorBoard; no logging directory and no tracker selects none. |
| Which native module first switches the actual process? | **Not measured here.** `triton._C.libtriton` is the leading known candidate. Earlier imports inside model/adapter or third-party packages may reach it sooner than the displayed path, or expose another native blocker. |
| Exact first caller/location? | Pending the appendix trace on your actual baseline. Do not present the source chain as a captured runtime timeline. |

Once the GIL has been enabled, later native imports cannot reveal their own fallback by another `False -> True` transition. Fresh-process tests therefore remain essential. A successful later import with `GIL=True` is not evidence that that module declares no-GIL safety.

### Necessity and natural ownership

| Import | Necessary for generic startup? | Natural boundary after measurement |
|---|---|---|
| torchvision from numeric constants | No evidence it is needed there. `IMAGE_TRANSFORMS` consumers found only in `library/data/_deprecated`. | Move legacy transform construction to the legacy functionality; retain simple constants without native dependency side effects. Confirm external/legacy consumers before deletion. |
| TorchAO CPU offload wrapper | Only for selected CPU offload or other TorchAO features. | Resolve the optional wrapper at its existing optimizer-selection boundary. |
| `cv2` for hashing/caption/size helpers | No; those operations are independent of OpenCV. | Import at actual OpenCV operations, without changing preprocessing semantics. |
| OpenCV `INTER_AREA` caching | Yes when selected. May run in the main process during preparation. | Keep the behavior; if strict no-GIL is required before upstream fixes, separately assess a preprocessing process. |
| Triton for `torch.compile` or selected kernels | Yes for those execution paths. | Fix upstream. Compiler isolation alone does not remove runtime CUDA extension loading and kernel launch dependencies. |
| TensorBoard local writer | Only when selected. | Existing tracker boundary is appropriate; audit optional writer APIs. |
| TensorBoard server/gRPC data services | Not part of ordinary event writing. | Run the existing server as a separate process when needed. |

No system restructuring was performed. These are candidate follow-up changes grounded in demonstrated optional boundaries, not a proposal to hide every dependency behind lazy imports.

xFormers remains optional/guarded in some model code, but no indispensable xFormers-only functionality was identified for this investigation. Native SDPA paths exist. Do not assume every current attention configuration already defaults to SDPA: the attention dataclass still has separate disabled-by-default selection flags. Confirm the intended SDPA recipe during validation; do not spend this work porting xFormers.

## 3. Triton

### Exact import cause and declaration

Triton 3.8 initializes `libtriton` in [`python/src/main.cc:51`](https://github.com/triton-lang/triton/blob/v3.8.0/python/src/main.cc#L51):

```cpp
PYBIND11_MODULE(libtriton, m) {
```

There is no `py::mod_gil_not_used()` declaration there. Its build requires pybind11 at least 2.13.1, which supports the third macro argument. **The exact metadata-only candidate patch is**:

```diff
-PYBIND11_MODULE(libtriton, m) {
+PYBIND11_MODULE(libtriton, m, py::mod_gil_not_used()) {
```

This is not an approved production fix. It changes the promise CPython sees, not Triton's synchronization. [pybind11's documentation](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#free-threading-support) explicitly leaves application thread safety to the extension author.

Current upstream main uses nanobind (`NB_MODULE`) rather than the 3.8 pybind11 initializer. A nanobind build/configuration patch for main must not be treated as the same one-line backport to 3.8. Inspected main snapshot: `ab198e4f2094e412f4a9a608a202a261387b4b9d`.

### Thread-safety classification: B, with C for the remaining audit

The evidence does not justify classification A (safe extension with missing metadata). There are identifiable GIL-dependent or otherwise unsynchronized paths:

| Source | Finding | Required work |
|---|---|---|
| [`specialize.cc`](https://github.com/triton-lang/triton/blob/v3.8.0/python/src/specialize.cc), `init_called`, `init_globals`, `native_specialize_impl` | Unsynchronized boolean first-use check and global Python class/function/string handles; multiple threads can initialize and publish state concurrently. | Error-safe initialization and publication, with Python import/callback reentrancy considered. A bare mutex held across arbitrary imports is not automatically safe. |
| Same file, `dtype_ptr2str`, `dtype2str`, `type_handler_cache` | Process-global `std::unordered_map` caches have unprotected lookup/insert paths. Concurrent mutation can produce native undefined behavior; CPython dict locks do not protect them. | Synchronize all accesses or redesign ownership/storage; audit Python-reference lifetime and cache invalidation. |
| [`linear_layout.cc`](https://github.com/triton-lang/triton/blob/v3.8.0/python/src/linear_layout.cc), `getLinearLayoutContext`; [`ir.cc`](https://github.com/triton-lang/triton/blob/v3.8.0/python/src/ir.cc), context constructor | Shared retained Python/MLIR context; the exposed context is constructed with MLIR threading disabled. Concurrent attribute/context use needs a justified ownership model. | Per-thread/per-operation contexts or properly scoped synchronization; preserve object lifetimes and verify LLVM/MLIR contracts. |
| [`llvm.cc`](https://github.com/triton-lang/triton/blob/v3.8.0/python/src/llvm.cc), `init_targets` | Target registration uses `std::call_once`, which is useful existing protection. Assignment to global `llvm::parallel::strategy` occurs outside that once block. | Place one-time global configuration inside the protected initialization or otherwise synchronize repeated writes. Audit diagnostics, callbacks, and compiler-object sharing separately. |
| [`gluon_ir.cc`](https://github.com/triton-lang/triton/blob/v3.8.0/python/src/gluon_ir.cc), `GluonLayouts` | Static retained Python handles and shared layout objects need ownership/teardown review. C++ function-static initialization protects construction, not all subsequent operations or interpreter affinity. | Audit retained objects, shutdown, and shared use; account for related upstream ownership fixes. |
| [`runtime/driver.py`](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/runtime/driver.py), `DriverConfig` | Check-then-initialize default/active driver state without an explicit once/lifetime protocol. | Concurrent first-use tests and explicit safe initialization. Do not assume an older LazyProxy fix covers the 3.8 implementation. |
| [`runtime/autotuner.py`](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/runtime/autotuner.py), `Autotuner.run` | Invocation state includes shared `self.nargs`, `best_config`, cache miss/tuning, and hooks. Independent calls on the same autotuner can overwrite each other's state. | Per-invocation state plus deduplicated tuning/compile work and exception-safe publication; define supported concurrent use. |
| [`third_party/nvidia/backend/driver.c`](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/driver.c), `PyInit_cuda_utils` | Dynamically compiled **second CPython extension**, single-phase initializer without `PyUnstable_Module_SetGIL`. Mutable static CUDA function-pointer caches and a cached Python string also have first-use paths. | Audit synchronization, type/object allocation, ABI/build flags, and callback/thread attachment, then declare only if justified. |

Native compiler bindings already release the GIL around some expensive work. This shows existing concurrency exposure and makes a focused audit useful; it does not establish safety for all surrounding bindings, global Python caches, LLVM objects, or callbacks.

Do not overstate disk-cache hazards either. [`runtime/cache.py`](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/runtime/cache.py) writes through unique temporary directories and uses `os.replace`. That is useful protection against partially published individual files. It is not evidence that every group/cache protocol, duplicate compile, or same-key operation is atomic. Test both shared and separate cache keys.

### Relevant upstream status

Status below was checked through GitHub issue/PR metadata, not inferred from a search title. Open and closed-unmerged changes are not available in the installed release.

| Upstream item | Status on investigation date | Relevance |
|---|---|---|
| [Issue #8491](https://github.com/triton-lang/triton/issues/8491) | Open | Direct free-threading support request; declaration absence is not the entire engineering problem. |
| [Issue #6721](https://github.com/triton-lang/triton/issues/6721) | Open | Historical 3.13t failure/segfault involving compiled CUDA modules. It establishes a relevant failure surface, not a reproduced 3.8 crash here. |
| [PR #8437](https://github.com/triton-lang/triton/pull/8437) | Open, unmerged | Autotuner concurrency work. Discussion acknowledges a real shared-state problem and postpones merging pending a complete no-GIL plan. |
| [PR #12221](https://github.com/triton-lang/triton/pull/12221) | Open, unmerged; opened 10 October | Propagates free-threaded build configuration and nanobind `NB_FREE_THREADED` on current main; adds an import/GIL test. Does not synchronize Triton's application caches. |
| [PR #9438](https://github.com/triton-lang/triton/pull/9438) | Closed, unmerged | Heap-type work for generated CUDA utility/free-threading problems. Not a released fix to cherry-pick as established support. |
| [PR #12049](https://github.com/triton-lang/triton/pull/12049) | Related ownership work | Gluon retained handles/builder ownership; useful audit context, not a comprehensive free-threading certification. |

The proposed #12221 test skips when the parent already has its GIL enabled. Since test collection can import Triton first, that can hide exactly the failure under test. A proper test selects the free-threaded build using `sysconfig.get_config_var("Py_GIL_DISABLED")`, starts a clean child, and fails if the default import enables the GIL. Current main still contains important specialization and CUDA-state audit surfaces; upgrading to an arbitrary main snapshot is not a safety conclusion.

### Smallest maintainable fix

The smallest **declaration** is one line; the smallest **safe complete fix** is a scoped upstream patch series. First repair native specialization initialization/caches, then LLVM/context/retained-object ownership, driver first-use state and generated CUDA utilities, then Python autotuner/JIT concurrency. Avoid a large local fork or a coarse lock held while calling Python callbacks. A coarse lock around selected native critical regions can be an initial correctness solution if ownership, reentrancy, and lock ordering are explicit; it need not serialize all compilation forever.

For the generated single-phase CUDA module, the eventual declaration would use `PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED)` under `#ifdef Py_GIL_DISABLED`, **after checking that module creation succeeded**. This is a separate conditional candidate after safety work, not an instruction to add the declaration now. Inventory every runtime-generated extension and check status after its first load, not only after `import triton`.

`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` remains the supported NixOS runtime workaround for driver library discovery. Keep it in the NixOS/runtime environment. It solves the `/sbin/ldconfig` assumption, not free-threading safety.

### Validation required before declaring no-GIL support

Run the following against the exact cp314t build, both GPUs, and the candidate patches. Use normal default-GIL behavior for declaration checks. Use `-X gil=0` only in explicitly experimental, disposable test processes before declaration, with crash reporting and no valuable training/checkpoint work in those processes.

| Test group | Concrete design and acceptance criteria |
|---|---|
| Default import/declaration | Clean subprocess asserts free-threaded build, `GIL=False` before import and after import. Capture warnings. Repeat after first compile, CUDA driver utility import, first kernel execution, teardown, and each secondary package. Fail, never skip, on an unexpected transition. |
| Concurrent imports | Barrier-release 1/2/8/32 Python threads importing Triton and different submodules, in at least 100 fresh children. Ordinary import locks serialize a given module, so combine this with distinct native first-use operations; no deadlocks, partial initialization, or lost errors. |
| Cold compiler initialization | Empty private cache, no warm-up; barrier-release specialization, LLVM initialization and driver creation from multiple threads. Include fresh dtype/type-handler cache entries. Instrument the native state repairs. |
| Concurrent compilation | Same kernel/same specialization, same kernel/different specializations, and distinct kernels. Test direct Triton compilation and JIT warm-up separately from Dynamo. Same-key work must publish valid results and propagate one coherent error/result to waiters. |
| Disk/cache concurrency | Cold/warm caches; shared cache directory and per-thread directories; identical and differing keys; threads plus multiple processes; interrupt a writer and retry. Validate metadata groups and artifacts, no truncated files, hangs, or permanent poisoned entries. |
| Compile/destroy cycles | Thousands of compile/load/launch/destroy cycles; drop references, collect garbage, create/destroy contexts. Track RSS/native allocations, refcount/lifetime failures, and interpreter shutdown. Use separate processes for fresh initialization. |
| Error handling | Simultaneous invalid kernels, static assertions, type errors and out-of-resource compilation. Verify original exception/message per invocation, all waiters released, hooks/state cleared, successful compilation afterward. |
| CUDA devices/streams | Set device per worker. Exercise 3090 alone, 3080 alone, both concurrently, separate streams and repeated device changes. Compare actual outputs after synchronization to eager reference; check no incorrect pointer/device/stream reuse. |
| Generated kernels | Elementwise/reduction/matmul kernels, FP32/BF16 where appropriate, distinct shapes and strides. Validate numerical tolerances and guard buffers, not just returned tensor shape. |
| Autotuning/shared state | Same autotuner across threads with different arguments/shapes/devices, plus separate autotuners; pre/post hooks, cache hit/miss, resets, failures. Ensure arguments and best configurations never cross invocations. |
| Repeated `torch.compile` | Distinct functions/modules plus repeated wrappers of the same callable; shape changes, graph breaks, forward/backward, repeated optimizer steps. Check outputs/gradients and GIL status. Respect documented Dynamo supported-concurrency contracts; a Dynamo failure is not automatically a libtriton defect. |
| Long stress | One-hour mixed-load run, followed by an overnight run once focused tests pass. Randomized workloads with recorded seed, bounded GPU memory, monitored subprocess watchdogs, faulthandler/core dumps. Check GIL status and outputs periodically, not only at process start. |
| Native race/lifetime tools | Host-side ThreadSanitizer build of relevant CPython/bindings/LLVM components where practical; separate AddressSanitizer/UndefinedBehaviorSanitizer configurations for lifetime and undefined behavior. Document unsanitized library blind spots and narrowly justified suppressions. Use CUDA Compute Sanitizer separately for device memory/race issues; it does not establish host binding safety. |

Upstream already has some genuine multiple-thread tests: [`test_cache.py`](https://github.com/triton-lang/triton/blob/v3.8.0/python/test/unit/runtime/test_cache.py) exercises `AsyncCompileMode` with a two-thread executor, including a compilation error; [`test_cache_determinism.py`](https://github.com/triton-lang/triton/blob/v3.8.0/python/test/unit/runtime/test_cache_determinism.py) hashes dependencies concurrently; [`test_driver.py`](https://github.com/triton-lang/triton/blob/v3.8.0/python/test/unit/runtime/test_driver.py) launches from another thread. These are useful foundations. They do not assert the GIL remains disabled and do not constitute full free-threaded cold-initialization/ownership coverage.

Passing stress tests alone cannot prove freedom from races. Require a reviewed state/ownership audit, meaningful race tests, and CI on default-no-GIL builds before making a supported declaration.

## 4. OpenCV

### Build compatibility is already addressed locally

The locked `opencv-python==5.0.0.93` sdist was downloaded and verified against SHA-256 `66aac3e5b5faa48d4025816592f3af19e4bfc2c68dec067bae2dbb4ca10aa9e2`. Its root `setup.py:184` adds `-DPYTHON3_LIMITED_API=ON`. Its `cv2.cpp`, `cv2_highgui.cpp`, and `cv2_util.cpp` match core OpenCV 5.0.0 byte for byte.

The repository's existing setting is appropriate and should remain:

```toml
[tool.uv.extra-build-variables]
opencv-python = { CMAKE_ARGS = "-DPYTHON3_LIMITED_API=OFF" }
```

This enables a compatible source build. It does not certify runtime thread safety. The Limited API/stable ABI is not a substitute for the free-threaded ABI.

Upstream [opencv-python PR #1051](https://github.com/opencv/opencv-python/pull/1051) is open and unmerged. It makes Limited API selection conditional on `sysconfig.get_config_var("Py_GIL_DISABLED")`. This is a small, reasonable upstream build fix; the existing override is a smaller maintenance burden than a local package fork until an appropriate release includes it. Related packaging discussions: [#1029](https://github.com/opencv/opencv-python/issues/1029), [#1146](https://github.com/opencv/opencv-python/issues/1146).

### Runtime declaration and binding hazards

[`cv2.cpp`](https://github.com/opencv/opencv/blob/5.0.0/modules/python/src2/cv2.cpp) defines a single-phase `PyInit_cv2`, calls `PyModule_Create`, and initializes the module body without a no-GIL declaration. The module definition uses process-global state rather than an audited per-interpreter ownership model.

Identified hazards include:

* [`cv2_highgui.cpp`](https://github.com/opencv/opencv/blob/5.0.0/modules/python/src2/cv2_highgui.cpp): mouse, trackbar, and button registration uses static `std::map` registries of Python callback objects. Replacing entries decrefs previous objects while native callbacks can retain raw userdata. `PyGILState_Ensure` in the callback attaches the thread; it does not serialize the registry or keep a replaced callback alive in free-threaded execution. A map mutex alone is insufficient without callback lifetime and reentrancy handling.
* [`cv2_util.cpp`](https://github.com/opencv/opencv/blob/5.0.0/modules/python/src2/cv2_util.cpp): `pycvRedirectError` maintains a static Python callback and swaps references while the global native error redirect can invoke callbacks. Safe ownership must span registration, invocation, replacement, and cleanup.
* [`cv2.cpp`](https://github.com/opencv/opencv/blob/5.0.0/modules/python/src2/cv2.cpp) / [`cv2_convert.cpp`](https://github.com/opencv/opencv/blob/5.0.0/modules/python/src2/cv2_convert.cpp): `pycvRegisterMatType` publishes global `pyopencv_Mat_TypePtr`; conversion reads it. Determine whether registration is constrained to initialization, or synchronize and preserve reference lifetime.
* [`cv2_numpy.cpp`](https://github.com/opencv/opencv/blob/5.0.0/modules/python/src2/cv2_numpy.cpp): array allocation/deallocation uses Python thread attachment helpers. Audit their reference/lifetime behavior together with generated conversion code and shared arrays. Existing GIL release around independent native image operations is promising for those operations, but does not certify the whole module.

Headless OpenCV removes HighGUI surfaces but does not automatically remove global error redirect, Mat type state, conversion, or allocator concerns. Independent-array resizing and color conversion may be a safe subset after review; the extension declaration covers the module, including exposed callback paths.

Upstream [issue #27933](https://github.com/opencv/opencv/issues/27933) tracks free-threading. [PR #30016](https://github.com/opencv/opencv/pull/30016), open/unmerged, is explicitly a build-detection phase: detect free-threaded Python, choose the correct Windows library, disable Limited API. [PR #30134](https://github.com/opencv/opencv/pull/30134), also open/unmerged, proposes a module declaration and reports image-operation thread stress. Those tests do not cover callback registration/lifetime or all binding state. Its proposed declaration was placed before the module-null check, which also needs correction. Do not equate this pending declaration PR with reviewed full safety.

**Assessment: B for identified state, C for the rest of the bindings.** A local Limited API override is reasonable. A local declaration-only patch is not. A small synchronized subset of bindings could be upstreamable, but complete validation is broader than one macro and should live upstream rather than in sd-scripts.

### Actual project use

Active SD and SDXL caching uses PIL for its default resizing: HAMMING on downscale, LANCZOS on upscale. Explicit `resize_interpolation == "area"` imports OpenCV and calls `cv2.resize(..., INTER_AREA)` in [`sd/caching.py:321–326`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/strategies/sd/caching.py#L321) and [`sdxl/caching.py:401–406`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/strategies/sdxl/caching.py#L401). These are image preparation/cache operations, not GPU model arithmetic. Preparation can still run in the trainer process.

The general image helper also uses OpenCV color conversion and resizing, including around some PIL paths; most found consumers of its older resize pipeline are deprecated. Confirm call sites before restricting imports. Do not silently substitute PIL for `INTER_AREA`, which changes preprocessing semantics. Removing the unconditional import can benefit default PIL recipes; selecting AREA later in the same interpreter still enables the GIL until OpenCV is fixed.

## 5. gRPC / TensorBoard

### gRPC has real synchronization/lifecycle work

[`cygrpc.pyx`](https://github.com/grpc/grpc/blob/v1.78.0/src/python/grpcio/grpc/_cython/cygrpc.pyx) in grpcio 1.78.0 does not declare `freethreading_compatible=True`. Cython's generated module defaults to requiring the GIL. Turning on the directive would assert safety, not supply it.

There is concrete contrary evidence. [`channel.pyx.pxi`](https://github.com/grpc/grpc/blob/v1.78.0/src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi), `get_registered_call_handle`, explicitly notes it is not thread-safe. Its cache lookup/handle construction is not synchronized against channel shutdown. `close()` destroys the native channel under its condition; a registration path without the same lifecycle protection can use the destroyed pointer. This is a native lifetime problem, not merely a dictionary insertion race.

In [PR #43278](https://github.com/grpc/grpc/pull/43278), reviewers reproduced channel register/close crashes with forced no-GIL execution and no crashes in their corresponding GIL-enabled control runs. A later scoped fix addressed that reproducer, but maintainers rejected a global compatibility declaration pending an extension-wide audit and CI. The PR is **closed, unmerged**. The extracted lifecycle fix [PR #43455](https://github.com/grpc/grpc/pull/43455) is **open, unmerged**, and does not itself certify the extension.

The async completion queue has additional native cross-thread state; see [`aio/completion_queue.pyx.pxi`](https://github.com/grpc/grpc/blob/v1.78.0/src/python/grpcio/grpc/_cython/_cygrpc/aio/completion_queue.pyx.pxi) and its `_shutdown` flag. Its polling thread already runs without the GIL, so some native races are not exclusive to free-threaded Python. Existing queue mutexes and reference-count locks are useful; they do not eliminate the need to audit adjacent lifecycle flags and callbacks. The atomic shutdown proposal [PR #43429](https://github.com/grpc/grpc/pull/43429) was **closed, unmerged**.

Other relevant work: support [issue #38762](https://github.com/grpc/grpc/issues/38762) remains open; declaration-only [PR #40820](https://github.com/grpc/grpc/pull/40820) remains open/unmerged; investigative [PR #43172](https://github.com/grpc/grpc/pull/43172) remains open/unmerged. Maintainer activity exists, so older backlog comments should not be read as a permanent refusal. None establishes support in grpcio 1.78.0.

**Assessment: B.** Scope is an extension-wide project across dozens of Cython include files: sync/aio channel and call lifecycle, server shutdown, completion queues, callback reference lifetime, observability/security/fork paths, tests, generated code/builds, wheels, and free-threaded/race CI. This is substantial engineering, plausibly multiple person-weeks or more depending on audit findings; there is no defensible exact ETA from this pass. sd-scripts should not take ownership of a gRPC rewrite or declare it safe for the sake of event logging.

### TensorBoard installation does not imply gRPC import

The project does not directly import gRPC. [`trainer_utils.py:265–274`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/training/trainer_utils.py#L265) selects tracker configuration. [`Trainer.train():827`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/training/runners/trainer.py#L827) calls [`logging.metrics.init_trackers`](https://github.com/Enferlain/sd-scripts/blob/33f73a4ddcfd781922cdb827172dd072046cdf8e/library/logging/metrics.py#L576), which delegates to Accelerate.

Accelerate's [`TensorBoardTracker.start`](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/tracking.py) imports `torch.utils.tensorboard` and constructs `SummaryWriter` only at tracker startup. PyTorch's [`writer.py`](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/utils/tensorboard/writer.py) writes protobuf events through TensorBoard's [`event_file_writer.py`](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/summary/writer/event_file_writer.py). That path uses an asynchronous queue, record writer, CRC, and TensorFlow compatibility shim. It does not import gRPC for ordinary event-file writing.

gRPC is used by server/data-provider paths such as [`grpc_provider.py`](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/data/grpc_provider.py), [`server_ingester.py`](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/data/server_ingester.py), and [`grpc_util.py`](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/util/grpc_util.py). Those naturally belong in the TensorBoard server process. If actual TensorFlow is installed, the lazy compatibility shim can select it instead of the stub, widening the import graph. Selected tracker integrations and APIs such as images/hparams can also expand imports. Trace initialization, scalar/image/hparam writing, flush, and close separately.

**Near-term recommendation:** preserve optional event logging, verify its actual imports, and keep the server external. Do not disable useful logging solely because grpcio is installed. If a particular selected writer/integration genuinely loads gRPC, isolate that integration or temporarily accept the GIL; do not patch gRPC's declaration.

### Additional dependency to check: Protobuf

`uv.lock` pins `protobuf==6.33.5`; its recorded wheels include stable-ABI wheels and a pure-Python wheel, but no cp314t wheel. A normal free-threaded resolver may therefore select the Python wheel. **Do not assume your process has loaded the native upb backend.** Inspect `google.protobuf.internal.api_implementation.Type()` and loaded native module paths in a fresh process.

The matching [`v33.5/python/protobuf.c`](https://github.com/protocolbuffers/protobuf/blob/v33.5/python/protobuf.c) creates `google._upb._message` without a no-GIL declaration and contains unprotected weak/object caches and a mutable allocation-trimming counter. A custom native build can add another blocker after Triton/OpenCV. [Issue #23158](https://github.com/protocolbuffers/protobuf/issues/23158) tracks backend-specific free-threading work. Current upstream work must not be backdated to the locked release. This is a diagnostic addition, not a measured extra failure on your machine, nor evidence that the pure-Python backend is unsafe for TensorBoard's ordinary usage.

## 6. torchvision / torchao / bitsandbytes

Their supplied `GIL=True` import results must not be attributed to their own native modules without isolating Triton.

| Package | Exact-version import path | Assessment |
|---|---|---|
| torchvision 0.29.1 | [`__init__.py`](https://github.com/pytorch/vision/blob/v0.29.1/torchvision/__init__.py) eagerly imports ops; [`roi_align.py`](https://github.com/pytorch/vision/blob/v0.29.1/torchvision/ops/roi_align.py) imports Dynamo utilities, which load Triton in Torch 2.14.1. | Triton explains the known transition. Native `_C_stable`/image libraries are registered through `torch.ops.load_library`, rather than the same CPython module-import declaration mechanism. No separate missing declaration identified here. |
| TorchAO 0.18.0 | [`__init__.py`](https://github.com/pytorch/ao/blob/v0.18.0/torchao/__init__.py) loads custom-op shared objects and quantization; kernel imports reach [`intmm_triton.py`](https://github.com/pytorch/ao/blob/v0.18.0/torchao/kernel/intmm_triton.py), which imports Triton. | Triton explains the known transition; package import alone does not establish safety for every AO operation. Skipping native SO loading does not remove its Python Triton imports. |
| bitsandbytes 0.50.2 | Root imports default backend operations; [`backends/utils.py`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/utils.py) imports Triton to determine availability. [`cextension.py`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/cextension.py) loads native libraries through ctypes. | Triton explains the known transition; ctypes loading is not a no-GIL declaration for a CPython extension. Its own shared native/Python state still needs operation-specific review. |

This is source-supported attribution, **not proof that Triton is the only possible transition in your particular installed artifacts**. Importing a submodule such as `torchvision.transforms` normally runs the parent initializer, so it is not an independent isolation test.

For isolated native checks, locate installed libraries using distribution metadata without importing the parent package. In separate fresh children, import the known-good Torch baseline, then load one torchvision/TorchAO custom-op library by its actual path; check GIL status and execute a relevant operation. Trace ctypes library loading and operation execution for bitsandbytes separately. Do not load duplicate copies of a registration library into the same process.

After a reviewed Triton safety patch/build, import each whole package normally and record every additional extension/warning. Forced `gil=0` cannot answer whether another extension lacks the declaration because it suppresses automatic GIL fallback; use it only for concurrency experiments.

bitsandbytes has a separate audit target in [`functional.py`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py): `CUBLAS_Context` singleton publication and per-device handle initialization are check-then-create state. Its optimizer manager also owns shared state. Determine handle/stream concurrency contracts before calling the whole library thread-safe. For practical recipe validation, test separate optimizer instances with quantize/dequantize and real 8-bit/paged optimizer steps on both GPUs, gradients, state save/load, and cleanup. Do not share mutable optimizer/model state arbitrarily across threads. Similarly validate AO offload/quantization and vision operations actually selected by the recipe.

## 7. Recommended fixes, ordered by value and effort

| Priority | Blocker / action | Best near-term classification | Effort / value |
|---|---|---|---|
| 1 | Capture default-runtime full launcher/training trace and native inventory | Investigation/validation prerequisite | Small; prevents fixing the wrong first import or missing a later blocker. |
| 2 | Keep OpenCV Limited API override; follow #1051 and #30016 | Small safe local build fix; upgrade after upstream inclusion | Very small maintenance burden; already working. |
| 3 | Remove legacy torchvision side effect from generic constants; resolve CPU offload at selection | Avoid/import optional features lazily for now | Small to medium; removes proven unnecessary Triton imports without changing selected feature behavior. Re-test HF/Dynamo paths. |
| 4 | Restrict `cv2` loading to actual OpenCV helpers | Avoid/import lazily for PIL recipes | Small; preserves semantics. AREA recipes still need an upstream fix, isolation assessment, or temporary GIL. |
| 5 | Validate TensorBoard local event writer and Protobuf backend | Existing optional boundary; external server | Small; likely avoids unnecessary gRPC work entirely. |
| 6 | Triton specialization/first-use caches and generated CUDA utilities | Upstreamable synchronization patches; substantial overall support work | Medium/large, high value for compilation and dependent packages. Follow #8437/#12221 rather than declaring only. |
| 7 | OpenCV callback and binding-state safety | Upstreamable patch series, broader binding audit | Medium/large; avoid a declaration-only local fork. |
| 8 | grpcio no-GIL support | Substantial synchronization/validation work upstream | Large, low sd-scripts-specific value if writer trace excludes it. Keep server separate; accept GIL if a required in-process integration needs it. |
| 9 | Full-feature training during upstream work | Accept GIL temporarily | Safe operational choice; keep 3.14t as a tested baseline, distinguish it from a no-GIL guarantee. |

**None of the three supplied native blockers is established as already safely fixed in its inspected release.** Pending upstream PRs are not an upgrade recommendation. The small safe local change identified is the OpenCV build setting, which is already present. Root native fixes are preferable to restructuring the trainer when their scope is manageable; accidental optional imports should still be corrected at their natural boundaries.

## 8. Concrete implementation tasks

These are issue-ready specifications, not changes applied during this pass.

### T1 — Capture the actual first GIL transition and a recipe matrix

Run the appendix harness under your real CPython 3.14.5t interpreter with no GIL override. First import `train`, then run the real Hydra command/config. Record JSON trace and warnings, interpreter executable/configuration, installed distribution/native hashes, selected recipe, CUDA versions/devices, and environment needed for Nix driver discovery. Run every case in a fresh child: logging off/on, default PIL/AREA preparation, AdamW/CPU offload/BNB, compile off/on, and each active SD/SDXL/SD3 family. Acceptance: name/file/caller stack for the first transition, checkpoint results for later phases, and a separate fresh-process blocker inventory. A 3.14t build with `GIL=True` must fail the strict no-GIL test, not be skipped.

### T2 — Remove legacy transform dependency from generic constants

Audit all `IMAGE_TRANSFORMS` consumers, including legacy entrypoints/external imports. Move construction to the owning deprecated implementation or an explicit legacy helper. Keep regex/suffix constants free of torchvision initialization. Acceptance: equivalent legacy transform outputs and active configuration imports no longer loading torchvision solely for constants. Re-run T1; do not claim this eliminates all Triton paths.

### T3 — Resolve the optional TorchAO wrapper only when selected

Change the existing wrapper/optimizer selection boundary so plain optimizers do not initialize `wrappers.cpu_offload`. Preserve wrapper public contracts and selected CPU offload behavior. Acceptance: an AdamW recipe does not load TorchAO through wrapper exports; CPU offload still constructs and runs correctly, errors remain clear, and startup trace identifies other Triton paths if present. Respect the repository's ordinary import policy outside this demonstrated optional feature.

### T4 — Separate image metadata helpers from OpenCV operations

Remove unconditional `cv2` loading from hashing/caption/PIL header operations. Import at functions that actually need OpenCV, preserving BGR/RGB and alpha conversion behavior. Preserve SD/SDXL AREA output exactly. Acceptance: PIL-only startup/preparation remains free of cv2; AREA tests show unchanged arrays and explicitly document the current GIL transition. No silent interpolation substitution.

### T5 — Validate event logging without gRPC

Fresh-process test of `SummaryWriter`, then Accelerate tracker startup, scalar/image/hparam writes, flush and close. Record native imports, Protobuf backend and GIL status. Read events back to verify valid output; keep server launch separate. Acceptance: identify whether any selected API actually loads `grpc._cython.cygrpc`, including optional TensorFlow. If it does, name the exact feature and decide only that feature's process boundary.

### T6 — Upstream Triton native-state repairs

Use a source branch pinned to the installed-compatible release or coordinate with nanobind main. Add synchronization/ownership fixes for specialization initialization/maps, LLVM global strategy initialization, shared MLIR/layout objects, and generated CUDA utility first-use state. Inventory Python callbacks and retained references. Define lock order and avoid holding non-reentrant locks across Python import/callback/decref paths. Acceptance: targeted reproducer/race tests, sanitizer results with coverage notes, two-device numerical runs, and reviewed lifetime/exception handling. Only afterward add declarations for every safe module.

### T7 — Complete Triton Python runtime concurrency and CI

Coordinate with #8437 and #12221. Make autotuner invocation state local; define same-key in-flight cache publication and failures; test JIT/driver first-use and destruction. Fix import test selection so pre-enabled parent GIL does not cause a false skip. Acceptance: the Section 3 matrix on normal default-no-GIL children, including generated extensions and prolonged stress; fail on unexpected GIL enablement. Keep experimental override tests in separate CI jobs with no production environment setting.

### T8 — Track OpenCV build versus runtime support independently

Retain the current override until an upstream release resolves Limited API detection. For runtime contributions, first target callback registry/error redirect lifetime and Mat type publication; include HighGUI/error callbacks, replacement, concurrent execution, GC and shutdown. Correct declaration placement/null handling if revisiting #30134. Acceptance: reviewed synchronization and ownership plus cp314t tests beyond image stress. Do not carry a blanket local module flag while known paths remain unaudited.

### T9 — Limit gRPC work to upstream monitoring and selected integration needs

Track #38762/#43172 and registration lifecycle fixes such as #43455. Confirm whether T5 exposes any actual trainer dependency. If none, leave gRPC outside the trainer and avoid a fork. Acceptance: a documented runtime import finding, rather than a dependency-list assumption. Adopt upstream support only with release/build verification and default-no-GIL sync/aio tests appropriate to the used integration.

### T10 — Establish package and performance acceptance criteria

Independently verify vision/AO native loading and selected BNB operations, then whole imports after Triton work. Include remaining native dependencies reached by actual recipes, rather than assuming the supplied list is exhaustive. Benchmark a representative training workload under the same 3.14t build with default behavior versus explicitly GIL-enabled control, measuring examples/second, CPU utilization, startup/cache time, RAM/VRAM and correctness. Acceptance: a reproducible benefit and a named supported recipe; no general claim of arbitrary model/optimizer thread safety.

## 9. Things we must not do

* Do not put global `PYTHON_GIL=0` or `-X gil=0` in project/Nix production settings to suppress extension warnings. Experimental isolated tests are the only justified use before safety is established.
* Do not add `Py_MOD_GIL_NOT_USED`, `py::mod_gil_not_used()`, nanobind free-threaded flags, or Cython `freethreading_compatible=True` as a warning-only fix.
* Do not infer compatibility from a cp314t wheel, successful source build, ordinary-thread tests, a returned tensor shape, or one successful `torch.compile` call.
* Do not infer incompatibility of torchvision/TorchAO/bitsandbytes from Triton's import contamination; do not infer complete thread safety from removing that contamination either.
* Do not treat `PyGILState_Ensure`, Cython `with gil`, atomic reference counts, or thread-safe individual Python dict operations as replacement synchronization for compound native/Python state and lifetime.
* Do not test declaration fallback only after another module enabled the GIL; do not let a strict free-threading test skip because the GIL has already become enabled.
* Do not relabel regular CPython/abi3 artifacts as cp314t or reuse stale generated-extension caches compiled against different interpreter headers/ABI.
* Do not make `/sbin/ldconfig` or Nix driver-path workarounds project configuration; retain the existing runtime `TRITON_LIBCUDA_PATH` solution.
* Do not rewrite gRPC for local TensorBoard event files, silently change AREA interpolation, remove logging without evidence, or revive xFormers without a demonstrated requirement.
* Do not claim a lazy import makes a feature GIL-free when that feature is later used in the same interpreter.

---

## Appendix — Default-runtime startup trace harness

This diagnostic records native extension creation/execution and captures the caller stack at any `False -> True` transition. It observes ordinary import behavior and refuses forced GIL settings. It is for startup/import diagnosis, not a race stress test; its logging lock and stack collection perturb timing.

Save the following as `gil_trace.py` outside the repository, then run it from the repository root using the **same Python executable as training**:

```bash
python /path/to/gil_trace.py --out gil-import.jsonl --import train
python /path/to/gil_trace.py --out gil-training.jsonl --run train.py -- <your usual Hydra arguments>
```

The second command performs the training invocation you supply. It does not invent a new project configuration. Keep warnings on stderr, optionally capture them separately. JSON events include extension names/origins, before/after GIL status, transitions and stacks. A final `loaded` event identifies `cv2`, Triton, gRPC, TensorBoard and the Protobuf native backend if loaded. Missing later events after a failed import must be treated as an incomplete trace, not compatibility.

```python
import sys

initial_gil = getattr(sys, "_is_gil_enabled", lambda: None)()
import argparse
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sysconfig
import threading
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--import", dest="module")
mode.add_argument("--run", dest="script")
parser.add_argument("args", nargs=argparse.REMAINDER)
options = parser.parse_args()

if sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
    parser.error("Requires the actual free-threaded training interpreter")
if "PYTHON_GIL" in os.environ or "gil" in sys._xoptions:
    parser.error("Remove PYTHON_GIL / -X gil overrides for declaration diagnosis")
if initial_gil is not False or sys._is_gil_enabled():
    parser.error("GIL already enabled before tracing; inspect site/startup imports")

lock = threading.Lock()
stream = open(options.out, "w", encoding="utf-8", buffering=1)

def emit(event, **fields):
    with lock:
        stream.write(json.dumps({"event": event, **fields}) + "\n")

emit("baseline", executable=sys.executable, version=sys.version,
     gil=sys._is_gil_enabled(),
     soabi=sysconfig.get_config_var("SOABI"),
     preloaded_native=[name for name, mod in list(sys.modules.items())
                       if any(str(getattr(mod, "__file__", "")).endswith(s)
                              for s in importlib.machinery.EXTENSION_SUFFIXES)])

loader = importlib.machinery.ExtensionFileLoader
original_create = loader.create_module
original_exec = loader.exec_module

def observe(phase, name, origin, operation):
    before = sys._is_gil_enabled()
    outcome = "ok"
    try:
        return operation()
    except BaseException:
        outcome = "error"
        raise
    finally:
        after = sys._is_gil_enabled()
        emit("native", phase=phase, name=name, origin=origin,
             before=before, after=after, outcome=outcome)
        if before is False and after is True:
            emit("transition", phase=phase, name=name, origin=origin,
                 stack=[{"file": f.filename, "line": f.lineno,
                         "function": f.name}
                        for f in traceback.extract_stack()[:-1]])

def create(self, spec):
    return observe("create", spec.name, spec.origin,
                   lambda: original_create(self, spec))

def execute(self, module):
    return observe("exec", module.__name__, self.path,
                   lambda: original_exec(self, module))

loader.create_module = create
loader.exec_module = execute
try:
    if options.module:
        sys.path.insert(0, str(Path.cwd()))
        importlib.import_module(options.module)
    else:
        script = Path(options.script).resolve()
        tail = options.args
        if tail[:1] == ["--"]:
            tail = tail[1:]
        sys.argv = [str(script), *tail]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
finally:
    loader.create_module = original_create
    loader.exec_module = original_exec
    interesting = ("triton", "cv2", "grpc", "tensorboard",
                   "google._upb", "google.protobuf.pyext")
    emit("loaded", gil=sys._is_gil_enabled(), modules={
        name: str(getattr(mod, "__file__", ""))
        for name, mod in list(sys.modules.items())
        if any(name == p or name.startswith(p + ".") for p in interesting)
    })
    stream.close()
```

Because nested extension loads can make an outer loader span also observe the same transition, use the earliest emitted `transition` as the native cause; outer entries provide the import context. Built-in/frozen or custom nonstandard loaders are outside this hook, so retain CPython warnings and investigate any final GIL change without a corresponding extension transition. If startup/site hooks enable the GIL before this diagnostic can attach, the baseline check intentionally fails; inspect that phase separately rather than disabling the warning.

The harness was syntax-checked and its non-free-threaded interpreter rejection checked here. Actual cp314t native tracing and training execution remain pending in your environment.
