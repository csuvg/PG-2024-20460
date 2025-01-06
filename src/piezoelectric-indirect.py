import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuración de precisión de TensorFlow
tf.keras.backend.set_floatx('float64')

# Parámetros del material
c_E = 70e9          # Módulo elástico (Pa)
e_31 = 12.4         # Constante pizoelectrica (C/m^2)
epsilon_S = 8.854e-12  # Permisividad dieléctrica (F/m)

L = 1.0             # Longitud del dominio (m)
U0 = 1e-6           # Escala de desplazamiento (m)
phi0 = 1000.0       # Escala de potencial eléctrico (V)

# Normalización de constantes
c_E_norm = c_E / c_E  # = 1
e_31_norm = e_31 / c_E
epsilon_S_norm = epsilon_S / c_E
# z
# Construcción de la red neuronal
def DNN_builder(in_shape=1, out_shape=2, n_hidden_layers=8, neuron_per_layer=50, actfn='swish'):
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    hidden = input_layer
    for _ in range(n_hidden_layers):
        hidden = tf.keras.layers.Dense(neuron_per_layer, activation=actfn)(hidden)
    output_layer = tf.keras.layers.Dense(out_shape)(hidden)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Construir el modelo
model = DNN_builder()

# Funciones de salida de la red neuronal
def net_u_phi(x):
    x = tf.cast(x, tf.float64)
    uv_phi = model(x)
    u = uv_phi[:, 0:1]
    phi = uv_phi[:, 1:2]
    return u, phi

# Función para calcular los residuos de las ecuaciones diferenciales
def compute_residuals(x):
    x = tf.cast(x, tf.float64)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            u, phi = net_u_phi(x)
        du_dx = tape1.gradient(u, x)
        dphi_dx = tape1.gradient(phi, x)
    d2u_dx2 = tape2.gradient(du_dx, x)
    d2phi_dx2 = tape2.gradient(dphi_dx, x)
    del tape1
    del tape2

    # Cálculo de las ecuaciones constitutivas
    sigma = c_E_norm * du_dx + e_31_norm * dphi_dx
    D = e_31_norm * du_dx - epsilon_S_norm * dphi_dx

    # Residuos de las ecuaciones de equilibrio
    res_mechanical = tf.gradients(sigma, x)[0]
    res_electrical = tf.gradients(D, x)[0]

    return res_mechanical, res_electrical

# Función para calcular la pérdida total
def compute_loss(x_colloc, x_bc_u, u_bc, x_bc_phi, phi_bc):
    # Residuos en puntos de colación
    res_mech, res_elec = compute_residuals(x_colloc)
    mse_residual = tf.reduce_mean(tf.square(res_mech)) + tf.reduce_mean(tf.square(res_elec))

    # Condiciones de frontera para u
    u_pred, _ = net_u_phi(x_bc_u)
    mse_bc_u = tf.reduce_mean(tf.square(u_pred - u_bc))

    # Condiciones de frontera para phi
    _, phi_pred = net_u_phi(x_bc_phi)
    mse_bc_phi = tf.reduce_mean(tf.square(phi_pred - phi_bc))

    # Pérdida total
    total_loss = mse_residual + 0.1 * (mse_bc_u + mse_bc_phi)  
    return total_loss

# Generación de puntos de colación y condiciones de frontera
Nc = 2000  
x_colloc = np.linspace(0, L, Nc).reshape(-1, 1)

# Condiciones de frontera
x_bc_u = np.array([[0.0]])
u_bc = np.array([[0.0]])  

x_bc_phi = np.array([[L]])
phi_bc = np.array([[1.0]])  

# Conversión a tensores de TensorFlow
x_colloc_tf = tf.constant(x_colloc, dtype=tf.float64)
x_bc_u_tf = tf.constant(x_bc_u, dtype=tf.float64)
u_bc_tf = tf.constant(u_bc, dtype=tf.float64)
x_bc_phi_tf = tf.constant(x_bc_phi, dtype=tf.float64)
phi_bc_tf = tf.constant(phi_bc, dtype=tf.float64)

# Configuración del optimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Función de entrenamiento
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = compute_loss(x_colloc_tf, x_bc_u_tf, u_bc_tf, x_bc_phi_tf, phi_bc_tf)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Ciclo de entrenamiento
loss_history = []
epochs = 7000

for epoch in range(epochs):
    loss = train_step()
    loss_history.append(loss.numpy())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Pérdida: {loss.numpy():.6e}")

# Graficar la evolución de la pérdida
plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.show()

# Evaluación del modelo
x_test = np.linspace(0, L, 100).reshape(-1, 1)
x_test_tf = tf.constant(x_test, dtype=tf.float64)
u_pred, phi_pred = net_u_phi(x_test_tf)
u_pred = u_pred.numpy()
phi_pred = phi_pred.numpy()

# Desnormalización (en este caso, u y phi están normalizados)
u_pred_actual = u_pred * U0
phi_pred_actual = phi_pred * phi0

# Graficar los resultados
plt.figure(figsize=(12, 5))

# Desplazamiento
plt.subplot(1, 2, 1)
plt.plot(x_test, u_pred_actual, label='Desplazamiento predicho')
plt.xlabel('x (m)')
plt.ylabel('u(x) (m)')
plt.title('Desplazamiento')
plt.legend()

# Potencial eléctrico
plt.subplot(1, 2, 2)
plt.plot(x_test, phi_pred_actual, label='Potencial eléctrico predicho', color='orange')
plt.xlabel('x (m)')
plt.ylabel('φ(x) (V)')
plt.title('Potencial eléctrico')
plt.legend()

plt.tight_layout()
plt.show()
