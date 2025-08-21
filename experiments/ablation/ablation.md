## Ablation Studies

To quantify the contribution of each architectural component, **SeiPlant** provides three main ablation switches:  

1. **Dilated Convolutions**  
   - Switch: `remove_dilated`  
   - Purpose: capture long-range cis-dependencies.  
   - If set to `True`, the dilated convolution stack (`dconv1`–`dconv5`) is removed, leaving only the standard convolution/pooling backbone.  

2. **Residual Connections**  
   - Switch: `remove_residual`  
   - Purpose: stabilize training and alleviate vanishing gradients.  
   - If set to `True`, all skip-connections (`+ lout*`) are disabled, converting the network into a purely sequential structure.  

3. **B-spline Transformation**  
   - Switch: `remove_spline` (or use the `Sei_NoSpline` class)  
   - Purpose: project convolutional outputs into interpretable local-global bases.  
   - If set to `True`, the `BSplineTransformation` layer is bypassed and raw convolutional outputs are flattened directly into the classifier.  

**Training setup remains consistent across all ablations**:  
- **Loss**: `nn.BCELoss()` (multi-label binary cross-entropy)  
- **Optimizer**: SGD (`lr=...`, `momentum=0.9`, `weight_decay=1e-7`)  

---

### Running Examples

```python
from seiplant.models.sei import Sei, Sei_NoSpline

# Baseline
model = Sei(n_genomic_features=21907,
            remove_dilated=False, remove_residual=False, remove_spline=False)

# Without dilated convs
model = Sei(n_genomic_features=21907,
            remove_dilated=True, remove_residual=False, remove_spline=False)

# Without residuals
model = Sei(n_genomic_features=21907,
            remove_dilated=False, remove_residual=True, remove_spline=False)

# Without B-spline
model = Sei(n_genomic_features=21907,
            remove_dilated=False, remove_residual=False, remove_spline=True)

# NoSpline variant (dynamic classifier head)
model = Sei_NoSpline(n_genomic_features=21907,
                     remove_dilated=False, remove_residual=False)

