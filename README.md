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

## Segmentation output

`Differentiable3Dshapes.forward(..., return_segmentation=True)` returns
`(image, segmentation)`. With `return_grad=True`, it returns
`(jacobians, image, segmentation)`.

`segmentation.labels` is a `LongTensor` of class ids, and
`segmentation.masks` is a `FloatTensor` of soft alpha-contribution masks.
Classes are ordered as:

```python
(
    "background",
    "floor",
    "wall_back",
    "wall_front",
    "wall_left",
    "wall_right",
    "object",
)
```

Generate image, differential, and segmentation samples:

```bash
python generate_segmentation_samples.py --device auto
```
