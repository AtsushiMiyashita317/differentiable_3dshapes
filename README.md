# differentiable-3dshapes

Install from Git repository.

CPU:

```bash
pip install "git+https://github.com/AtsushiMiyashita317/differentiable_3dshapes.git#egg=differentiable-3dshapes[cpu]"
```

CUDA 12.4:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  "git+https://github.com/AtsushiMiyashita317/differentiable_3dshapes.git#egg=differentiable-3dshapes[cu124]"
```

Public API (from `differentiable_3dshapes`) re-exports the non-underscore symbols from `differentiable_3dshapes`.
