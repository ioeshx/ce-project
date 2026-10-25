# Shared SD1.4 sampling protocol

All SD1.4 image generation, reference generation, benchmark runs, and new method
scripts must use `benchmark/configs/sd14_sampling.json` unless an explicit
experiment override is declared and recorded in its manifest.

- Model: CompVis/stable-diffusion-v1-4.
- Scheduler: DPMSolverMultistepScheduler, algorithm_type=dpmsolver++,
  solver_order=2, solver_type=midpoint, lower_order_final=true,
  final_sigmas_type=zero, timestep_spacing=linspace, steps_offset=1,
  no Karras/exponential/beta/LU/flow sigmas. Full scheduler config is in JSON.
- Steps=20, CFG=7.5, 512x512, CUDA model dtype=float16, negative prompt="".
- One CPU torch.Generator per record seed. Draw initial noise in float32 on CPU,
  then cast to model dtype and device. Do NOT draw float16 noise directly.
- Instance/style: seeds 0..9 per prompt. MS-COCO: first 1000 captions, seed=row
  index. Batch partition must not change initial noise.
- Text encoding: native CLIP 77-token context. Legacy max_sequence_length=256
  is a metadata field unused by SD1 CLIP, retained for cache compatibility.
- Safety checker disabled as in existing experiments.
- Enable SD1 attention slicing (`pipe.enable_attention_slicing()`, auto slice
  size), matching legacy SlicedAttnProcessor. Default SDPA is not pixel-identical
  in float16 and must not silently share the legacy reference cache.
- Use benchmark to calculate metrics and store results. Match sampling for
  original, erased, repaired, and comparison models. Never repoint a manifest
  at images generated with different settings. Preserve historical manifests.
- Explicit CPU smoke tests use float32. Closed-form solves/Jacobians may use
  float32/float64; this does not set deployment sampling precision.
- output_response's differentiable DDIM response is an explicitly declared
  float32 optimization proxy (20 steps), because its short-prefix derivative
  implementation relies on DDIM. Its benchmark/deployment uses the shared DPM
  profile. Record this exception in method metadata, not just prose.
- Non-SD1 architectures retain their explicitly declared native defaults.

Use `benchmark.sampling_profile` for defaults, scheduler construction, noise,
and canonical cache protocol. Test cache compatibility before launching large
reference sampling jobs. Existing COCO cache a971b6e6... is the compatibility
reference; do not delete previous caches or checkpoints during migration.
