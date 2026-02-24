#%%
import numpy as np
import torch
import torch.nn.functional as F
import mogptk

#%%
torch.manual_seed(1)
np.random.seed(1)

#%%
# ============================================================================
# DEMO: Clasificación Multiclase con Softmax Likelihood
# ============================================================================

# 1. GENERAR DATOS SINTÉTICOS
# ----------------------------------------------------------------------------
print("=" * 70)
print("1. GENERANDO DATOS SINTÉTICOS")
print("=" * 70)

N = 120  # Número de puntos
K = 3    # Número de clases

# Generar puntos X
X = np.random.rand(N, 1)

# Generar funciones latentes para cada clase
t1 = np.sin(2 * np.pi * X)      # Clase 0
t2 = np.cos(2 * np.pi * X)      # Clase 1
t3 = -np.sin(2 * np.pi * X)     # Clase 2

# La clase es la que tiene el valor más alto
T = np.hstack([t1, t2, t3])  # (N, 3)
y = np.argmax(T, axis=1).reshape(-1, 1)  # (N, 1), valores 0, 1, 2

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Clases presentes: {np.unique(y)}")
print(f"Distribución de clases: {[np.sum(y == k) for k in range(K)]}")

#%%
# 2. PREPARAR DATOS EN FORMATO MULTI-OUTPUT
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. PREPARANDO DATOS EN FORMATO MULTI-OUTPUT")
print("=" * 70)

# Para mogptk multi-output, necesitamos:
# - X con canal ID en la primera columna
# - Un Data por cada canal

# Crear lista de Data objects (uno por clase/canal)
data_list = []
for k in range(K):
    # Para cada canal, usar todos los puntos X con etiqueta de canal
    X_channel = np.hstack([np.full((N, 1), k), X])  # (N, 2): [canal_id, x]
    y_channel = y.copy()  # Todos los canales usan las mismas etiquetas

    data_list.append(mogptk.Data(X_channel, y_channel, name=f"Class_{k}"))
    print(f"Canal {k}: X shape = {X_channel.shape}, y shape = {y_channel.shape}")

# Crear DataSet
dataset = mogptk.DataSet(*data_list)
print(f"\nDataSet creado con {len(dataset)} canales")

#%%
# 3. CONFIGURAR MODELO
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. CONFIGURANDO MODELO GP CON SOFTMAX LIKELIHOOD")
print("=" * 70)

# Likelihood: SoftmaxLikelihood
likelihood = mogptk.gpr.SoftmaxLikelihood(num_classes=K, mc_samples=50)
print(f"Likelihood: {likelihood.name()}")
print(f"  - num_classes: {likelihood.num_classes}")
print(f"  - mc_samples: {likelihood.mc_samples}")
print(f"  - output_dims: {likelihood.output_dims}")

# Kernels: Un kernel independiente por clase
kernels = [
    mogptk.gpr.SquaredExponentialKernel(input_dims=1),  # Para clase 0
    mogptk.gpr.SquaredExponentialKernel(input_dims=1),  # Para clase 1
    mogptk.gpr.SquaredExponentialKernel(input_dims=1),  # Para clase 2
]

# Multi-output kernel
kernel = mogptk.gpr.IndependentMultiOutputKernel(*kernels)
print(f"\nKernel: {kernel.name()}")
print(f"  - output_dims: {kernel.output_dims}")

#%%
# 4. CREAR MODELO CON INFERENCIA HENSMAN
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. CREANDO MODELO")
print("=" * 70)

# Inferencia: Hensman (variational inference para non-Gaussian likelihoods)
inference = mogptk.Hensman(
    inducing_points=None,  # Usa todos los puntos (no sparse)
    likelihood=likelihood,
    jitter=1e-6
)

# Modelo
model = mogptk.Model(dataset, kernel, inference=inference)
print(f"\n{model}")

#%%
# 5. VERIFICAR FORMAS DE PARÁMETROS VARIACIONALES
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. PARÁMETROS VARIACIONALES")
print("=" * 70)

print(f"q_mu shape: {model.gpr.q_mu().shape}")
print(f"q_sqrt shape: {model.gpr.q_sqrt().shape}")
print(f"X shape en GPR: {model.gpr.X.shape}")
print(f"y shape en GPR: {model.gpr.y.shape}")

#%%
# 6. ENTRENAMIENTO
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. ENTRENANDO MODELO")
print("=" * 70)

try:
    model.train(iters=100, verbose=True, lr=0.01)
    print("\n✓ Entrenamiento completado exitosamente")

    # Graficar pérdidas
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(model.losses)
    plt.xlabel('Iteración')
    plt.ylabel('Pérdida (Negative ELBO)')
    plt.title('Curva de Entrenamiento')
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"\n✗ Error durante entrenamiento: {e}")
    import traceback
    traceback.print_exc()

#%%
# 7. PREDICCIÓN
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. PREDICCIÓN")
print("=" * 70)

try:
    # Puntos de prueba
    X_test = np.linspace(0, 1, 50).reshape(-1, 1)
    N_test = X_test.shape[0]

    # Formato multi-output para predicción
    X_test_multi = []
    for k in range(K):
        X_test_k = np.hstack([np.full((N_test, 1), k), X_test])
        X_test_multi.append(X_test_k)
    X_test_multi = np.vstack(X_test_multi)  # (N_test*K, 2)

    print(f"X_test_multi shape: {X_test_multi.shape}")

    # Convertir a tensor
    X_test_tensor = torch.tensor(X_test_multi, dtype=torch.float32)

    # Predicción de funciones latentes
    with torch.no_grad():
        mu, var = model.gpr.predict_f(X_test_tensor)

    print(f"mu shape: {mu.shape}")
    print(f"var shape: {var.shape}")

    # Reshape para likelihood (N_test, K)
    mu_reshaped = mu.reshape(N_test, K)
    var_reshaped = var.reshape(N_test, K)

    print(f"mu_reshaped shape: {mu_reshaped.shape}")
    print(f"var_reshaped shape: {var_reshaped.shape}")

    # Obtener probabilidades de clase
    probs = likelihood.predict(X_test_tensor[:N_test], mu_reshaped, var_reshaped)
    print(f"probs shape: {probs.shape}")

    # Clases predichas
    y_pred = torch.argmax(probs, dim=1)
    print(f"y_pred shape: {y_pred.shape}")

    # Visualizar resultados
    plt.figure(figsize=(15, 4))

    # Subplot 1: Datos originales
    plt.subplot(1, 3, 1)
    for k in range(K):
        mask = (y.flatten() == k)
        plt.scatter(X[mask], y[mask], label=f'Clase {k}', alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Clase')
    plt.title('Datos de Entrenamiento')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Probabilidades predichas
    plt.subplot(1, 3, 2)
    for k in range(K):
        plt.plot(X_test, probs[:, k].detach().numpy(), label=f'P(y={k}|x)')
    plt.xlabel('X')
    plt.ylabel('Probabilidad')
    plt.title('Probabilidades Predichas')
    plt.legend()
    plt.grid(True)

    # Subplot 3: Clases predichas
    plt.subplot(1, 3, 3)
    plt.scatter(X_test, y_pred.numpy(), c=y_pred.numpy(), cmap='viridis', alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Clase Predicha')
    plt.title('Clasificación Predicha')
    plt.colorbar(label='Clase')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n✓ Predicción completada exitosamente")

except Exception as e:
    print(f"\n✗ Error durante predicción: {e}")
    import traceback
    traceback.print_exc()

#%%
# 8. MÉTRICAS DE EVALUACIÓN
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("8. MÉTRICAS DE EVALUACIÓN")
print("=" * 70)

try:
    # Predicción en datos de entrenamiento
    X_train_multi = []
    for k in range(K):
        X_train_k = np.hstack([np.full((N, 1), k), X])
        X_train_multi.append(X_train_k)
    X_train_multi = np.vstack(X_train_multi)
    X_train_tensor = torch.tensor(X_train_multi, dtype=torch.float32)

    with torch.no_grad():
        mu_train, var_train = model.gpr.predict_f(X_train_tensor)

    mu_train = mu_train.reshape(N, K)
    var_train = var_train.reshape(N, K)

    probs_train = likelihood.predict(X_train_tensor[:N], mu_train, var_train)
    y_pred_train = torch.argmax(probs_train, dim=1)

    # Accuracy
    accuracy = (y_pred_train.numpy() == y.flatten()).mean()
    print(f"Accuracy en entrenamiento: {accuracy:.2%}")

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y.flatten(), y_pred_train.numpy())
    print(f"\nMatriz de Confusión:")
    print(cm)

    print("\n✓ Evaluación completada")

except Exception as e:
    print(f"\n✗ Error durante evaluación: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DEMO COMPLETADO")
print("=" * 70)

