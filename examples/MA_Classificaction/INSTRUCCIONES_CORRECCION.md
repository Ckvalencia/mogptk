# Instrucciones para Corrección de SoftmaxLikelihood

## Archivos Modificados

### 1. mogptk/gpr/likelihood.py
✅ **YA CORREGIDO** - La clase `SoftmaxLikelihood` ya está implementada correctamente.

### 2. mogptk/gpr/model.py
⚠️ **REQUIERE MODIFICACIÓN MANUAL**

## Modificación Requerida en model.py

Busca la clase `SparseHensman` (aproximadamente línea 767) y su método `elbo()`.

**ANTES** (el método original):
```python
def elbo(self):
    if self.mean is not None:
        y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
    else:
        y = self.y  # Nx1

    if self.is_sparse:
        qf_mu, qf_var_diag = self._predict_f(self.X, full=False)
    else:
        Kff = self.kernel(self.X)
        Lff = self._cholesky(Kff, add_jitter=True)  # NxN

        qf_mu = Lff.mm(self.q_mu())
        if self.mean is not None:
            qf_mu -= self.mean(self.X).reshape(-1,1)  # Sx1

        qf_sqrt = Lff.mm(self.q_sqrt().tril())
        qf_var_diag = qf_sqrt.mm(qf_sqrt.T).diagonal().reshape(-1,1)

    var_exp = self.likelihood.variational_expectation(self.X, y, qf_mu, qf_var_diag)
    kl = self.kl_gaussian(self.q_mu(), self.q_sqrt())
    return var_exp - kl
```

**DESPUÉS** (agregar manejo especial para SoftmaxLikelihood):
```python
def elbo(self):
    if self.mean is not None:
        y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
    else:
        y = self.y  # Nx1

    if self.is_sparse:
        qf_mu, qf_var_diag = self._predict_f(self.X, full=False)
    else:
        Kff = self.kernel(self.X)
        Lff = self._cholesky(Kff, add_jitter=True)  # NxN

        qf_mu = Lff.mm(self.q_mu())
        if self.mean is not None:
            qf_mu -= self.mean(self.X).reshape(-1,1)  # Sx1

        qf_sqrt = Lff.mm(self.q_sqrt().tril())
        qf_var_diag = qf_sqrt.mm(qf_sqrt.T).diagonal().reshape(-1,1)

    # ========== AGREGAR ESTAS LÍNEAS ==========
    # Special handling for SoftmaxLikelihood which expects (N, K) instead of (N*K, 1)
    from . import SoftmaxLikelihood
    if isinstance(self.likelihood, SoftmaxLikelihood):
        K = self.likelihood.num_classes
        N = qf_mu.shape[0] // K
        if qf_mu.shape[0] != N * K:
            raise RuntimeError(f"Data shape mismatch: expected {N*K} points for {N} samples and {K} classes")
        # Reshape: (N*K, 1) -> (N, K)
        # Data is organized as [class0_all_points, class1_all_points, ..., classK_all_points]
        qf_mu = qf_mu.reshape(K, N).T
        qf_var_diag = qf_var_diag.reshape(K, N).T
        y = y[:N]
    # ========== FIN DE LÍNEAS A AGREGAR ==========

    var_exp = self.likelihood.variational_expectation(self.X, y, qf_mu, qf_var_diag)
    kl = self.kl_gaussian(self.q_mu(), self.q_sqrt())
    return var_exp - kl
```

## Explicación de la Modificación

1. **Detección**: Verifica si el likelihood es de tipo `SoftmaxLikelihood`
2. **Reshape**: Reorganiza `qf_mu` y `qf_var_diag` de (N*K, 1) a (N, K)
   - Los datos están organizados como: [todos_puntos_clase0, todos_puntos_clase1, todos_puntos_clase2]
   - Por eso hacemos `.reshape(K, N).T` para obtener (N, K)
3. **Etiquetas**: Toma solo las primeras N etiquetas (ya que están repetidas K veces)

## Cómo Aplicar la Modificación

1. Abre el archivo: `mogptk/gpr/model.py`
2. Busca la clase `SparseHensman` (usa Ctrl+F para buscar "class SparseHensman")
3. Encuentra el método `elbo(self):` dentro de esa clase
4. Agrega las líneas marcadas con "AGREGAR ESTAS LÍNEAS" justo **ANTES** de la línea:
   ```python
   var_exp = self.likelihood.variational_expectation(...)
   ```

## Verificación

Después de hacer la modificación, ejecuta:
```bash
python Multiclass_demo_CORRECTED.py
```

El entrenamiento debería funcionar correctamente sin errores.

## Archivos de Referencia

- `EXPLICACION_SOFTMAX.md` - Explicación detallada de cómo funciona SoftmaxLikelihood
- `Multiclass_demo_CORRECTED.py` - Demo corregido para clasificación multiclase

