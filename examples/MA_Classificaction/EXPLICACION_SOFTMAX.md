# Explicación: SoftmaxLikelihood para Clasificación Multiclase

## ¿Cómo funciona el SoftmaxLikelihood?

### 1. **Concepto básico**

Para clasificación multiclase con K clases, usamos K funciones latentes Gaussianas:
- f = [f_1, f_2, ..., f_K]
- Cada f_k ~ GP(0, K) donde K es el kernel

La probabilidad de que un punto pertenezca a la clase k está dada por:

```
p(y=k|f) = exp(f_k) / Σ_j exp(f_j) = softmax(f)_k
```

### 2. **Componentes del SoftmaxLikelihood**

#### a) `validate_y(X, y)`
- Verifica que y contenga índices de clase válidos: enteros en [0, K-1]
- Ejemplo: para 3 clases, y debe tener valores 0, 1, o 2

#### b) `log_prob(X, y, f)`
- Calcula log p(y|f) = log(softmax(f)_y)
- **Entrada**: 
  - y: (N,1) índices de clase
  - f: (N,K) valores de funciones latentes
- **Salida**: (N,1) log probabilidades

#### c) `variational_expectation(X, y, mu, var)`
- Calcula E_q[log p(y|f)] donde q(f) ~ N(mu, diag(var))
- Usa **Monte Carlo** con `mc_samples` muestras
- **Entrada**:
  - y: (N,1) índices de clase observados
  - mu: (N,K) media de q(f)
  - var: (N,K) varianza diagonal de q(f)
- **Salida**: escalar - ELBO term

**Paso a paso del Monte Carlo:**
1. Generar S muestras: f_s ~ N(mu, diag(var)) para s=1,...,S
2. Para cada muestra, calcular: log p(y|f_s) = log(softmax(f_s)_y)
3. Promediar: E[log p(y|f)] ≈ (1/S) Σ_s log p(y|f_s)
4. Sumar sobre todos los puntos N

#### d) `conditional_mean(X, f)`
- Retorna las probabilidades de clase: softmax(f)
- **Entrada**: f: (N,K)
- **Salida**: (N,K) probabilidades p(y=k|f)

#### e) `predict(X, mu, var, ci, n)`
- Integra sobre la distribución posterior para obtener probabilidades predictivas
- **Si var es pequeño**: usa softmax(mu) directamente
- **Si var > 0**: integración Monte Carlo con n muestras
- **Salida**: (N,K) probabilidades de clase

### 3. **Configuración Multi-Output**

Para clasificación multiclase, necesitas:

```python
K = 3  # número de clases

# 1. Likelihood con output_dims = K
likelihood = mogptk.gpr.SoftmaxLikelihood(num_classes=K, mc_samples=100)

# 2. Kernel multi-output (K kernels independientes)
kernels = [
    mogptk.gpr.SquaredExponentialKernel(),
    mogptk.gpr.SquaredExponentialKernel(),
    mogptk.gpr.SquaredExponentialKernel()
]
kernel = mogptk.gpr.IndependentMultiOutputKernel(*kernels)

# 3. IMPORTANTE: Preparar los datos correctamente
# X: (N, D) - coordenadas de entrada
# y: (N, 1) - índices de clase (0, 1, 2)
# PERO para multi-output, necesitamos agregar canal ID

# Formato correcto para multi-output:
N = 120
D = 1  # dimensión de entrada original
K = 3  # clases

# Crear X_multi: (N*K, D+1) donde primera columna es canal ID
X_multi = np.zeros((N*K, D+1))
for k in range(K):
    X_multi[k*N:(k+1)*N, 0] = k  # canal ID
    X_multi[k*N:(k+1)*N, 1:] = X  # coordenadas originales

# y_multi: (N*K, 1) - repetir y para cada canal
y_multi = np.tile(y, (K, 1))
```

### 4. **Problema común y solución**

**Problema**: El sistema mogptk actual espera datos en forma (N,1), pero SoftmaxLikelihood 
necesita trabajar con K funciones latentes simultáneamente.

**Solución**: Usar el formato multi-output con canales. Cada canal representa una función 
latente f_k.

### 5. **Flujo de entrenamiento**

1. **Forward pass**: 
   - Model predice mu, var para cada canal k: (N, 1) por canal
   - Internamente se concatena a (N*K, 1)

2. **ELBO calculation**:
   - Reorganiza mu, var a forma (N, K)
   - Llama variational_expectation(X, y, mu, var)
   - Aplica Monte Carlo para aproximar integral

3. **Backward pass**:
   - Gradientes fluyen hacia parámetros del kernel
   - Optimizador actualiza parámetros

### 6. **Consideraciones importantes**

- **mc_samples**: Más muestras = mejor aproximación pero más lento (defecto: 100)
- **Estabilidad numérica**: log_softmax es más estable que log(softmax())
- **Varianza**: Se clampea a mínimo 1e-10 para evitar problemas numéricos
- **Kernels independientes**: Cada clase tiene su propio GP independiente

### 7. **Predicción**

```python
# Después del entrenamiento
mu, var = model.gpr.predict_f(X_test)  # (N_test*K, 1)

# Reshape para SoftmaxLikelihood
mu_reshaped = mu.reshape(N_test, K)
var_reshaped = var.reshape(N_test, K)

# Obtener probabilidades
probs = likelihood.predict(X_test, mu_reshaped, var_reshaped)  # (N_test, K)

# Clases predichas
y_pred = torch.argmax(probs, dim=1)  # (N_test,)
```

### 8. **Próximos pasos**

Para tu demo, necesitarás:
1. Preparar datos en formato multi-output correcto
2. Usar IndependentMultiOutputKernel con K kernels
3. Configurar correctamente el DataSet para multi-output
4. Adaptar funciones de predicción y visualización

