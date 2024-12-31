import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def call(self, xy):

        x, y = [ xy[..., i, tf.newaxis] for i in range(xy.shape[-1]) ]
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(x)
            ggg.watch(y)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                gg.watch(y)
                with tf.GradientTape(persistent=True) as g:
                    g.watch(x)
                    g.watch(y)
                    psi_p = self.model(tf.concat([x, y], axis=-1))
                    psi = psi_p[..., 0, tf.newaxis]
                    p   = psi_p[..., 1, tf.newaxis]
                u   =  g.batch_jacobian(psi, y)[..., 0]
                v   = -g.batch_jacobian(psi, x)[..., 0]
                p_x =  g.batch_jacobian(p,   x)[..., 0]
                p_y =  g.batch_jacobian(p,   y)[..., 0]
                del g
            u_x = gg.batch_jacobian(u, x)[..., 0]
            u_y = gg.batch_jacobian(u, y)[..., 0]
            v_x = gg.batch_jacobian(v, x)[..., 0]
            v_y = gg.batch_jacobian(v, y)[..., 0]
            del gg
        u_xx = ggg.batch_jacobian(u_x, x)[..., 0]
        u_yy = ggg.batch_jacobian(u_y, y)[..., 0]
        v_xx = ggg.batch_jacobian(v_x, x)[..., 0]
        v_yy = ggg.batch_jacobian(v_y, y)[..., 0]
        del ggg

        p_grads = p, p_x, p_y
        u_grads = u, u_x, u_y, u_xx, u_yy
        v_grads = v, v_x, v_y, v_xx, v_yy

        return psi, p_grads, u_grads, v_grads



class Network:
  
    def __init__(self):
        self.activations = {
            'tanh': 'tanh',
            'swish': self.swish,
            'mish': self.mish,
        }

    def swish(self, x):
        """ Swish activation function. """
        return x * tf.math.sigmoid(x)

    def mish(self, x):
        return x * tf.math.tanh(tf.softplus(x))

    def build(self, num_inputs=2, layers=[32, 16, 16, 32], activation='swish', num_outputs=2):
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=self.activations[activation], kernel_initializer='he_normal')(x)
        outputs = tf.keras.layers.Dense(num_outputs, kernel_initializer='he_normal')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    
class L_BFGS_B:

    def __init__(self, model, x_train, y_train, factr=10, pgtol=1e-10, m=50, maxls=50, maxiter=200):
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.progbar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params({'verbose': 1, 'epochs': 1, 'steps': self.maxiter, 'metrics': self.metrics})
        self.loss_values = []

    def set_weights(self, flat_weights):
        shapes = [w.shape for w in self.model.get_weights()]
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])
        weights = [flat_weights[from_id:to_id].reshape(shape)
                   for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.logcosh(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        self.set_weights(weights)
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        self.loss_values.append(loss)
        return loss, grads

    def callback(self, weights):
        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
        initial_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
                                     factr=self.factr, pgtol=self.pgtol, m=self.m,
                                     maxls=self.maxls, maxiter=self.maxiter, callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
        
        
        
class PINN:
    

    def __init__(self, network, rho=1, nu=0.01):
        

        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        xy_eqn = tf.keras.layers.Input(shape=(2,))
        xy_bnd = tf.keras.layers.Input(shape=(2,))

        _, p_grads, u_grads, v_grads = self.grads(xy_eqn)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads

        u_eqn = u*u_x + v*u_y + p_x/self.rho - self.nu*(u_xx + u_yy)
        v_eqn = u*v_x + v*v_y + p_y/self.rho - self.nu*(v_xx + v_yy)
        uv_eqn = tf.concat([u_eqn, v_eqn], axis=-1)

        psi_bnd, _, u_grads_bnd, v_grads_bnd = self.grads(xy_bnd)
        psi_bnd = tf.concat([psi_bnd, psi_bnd], axis=-1)
        uv_bnd = tf.concat([u_grads_bnd[0], v_grads_bnd[0]], axis=-1)

        return tf.keras.models.Model(
            inputs=[xy_eqn, xy_bnd], outputs=[uv_eqn, psi_bnd, uv_bnd])
    

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec



def uv(network, xy):

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u =  psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()

def contour(grid, x, y, z, title, levels=50):

    vmin = np.min(z)
    vmax = np.max(z)

    plt.subplot(grid)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)

if __name__ == '__main__':

    num_train_samples = 1000
    num_test_samples = 10

    u0 = 1
    rho = 1
    nu = 0.01

    network = Network().build()
    network.summary()
    pinn = PINN(network, rho=rho, nu=nu).build()




xy_eqn = np.random.rand(num_train_samples, 2)
xy_ub = np.random.rand(num_train_samples//2, 2) 
xy_ub[..., 1] = np.round(xy_ub[..., 1])          
xy_lr = np.random.rand(num_train_samples//2, 2)  
xy_lr[..., 0] = np.round(xy_lr[..., 0])          
xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))
x_train = [xy_eqn, xy_bnd]


zeros = np.zeros((num_train_samples, 2))
uv_bnd = np.zeros((num_train_samples, 2))
uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])
y_train = [zeros, zeros, uv_bnd]

lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
lbfgs.fit()


x = np.linspace(0, 1, num_test_samples)
y = np.linspace(0, 1, num_test_samples)
x, y = np.meshgrid(x, y)

xy = np.stack([x.flatten(), y.flatten()], axis=-1)
psi_p = network.predict(xy, batch_size=len(xy))
psi, p = [ psi_p[..., i].reshape(x.shape) for i in range(psi_p.shape[-1]) ]


u, v = uv(network, xy)
u = u.reshape(x.shape)
v = v.reshape(x.shape)

fig = plt.figure(figsize=(6, 5))
gs = GridSpec(2, 2)
contour(gs[0, 0], x, y, psi, 'psi')
contour(gs[0, 1], x, y, p, 'p')
contour(gs[1, 0], x, y, u, 'u')
contour(gs[1, 1], x, y, v, 'v')
plt.tight_layout()
plt.show()


plt.title("Boundary Data Points and Collocation Points")
plt.scatter(xy_eqn[:, 0], xy_eqn[:, 1], s=0.2, marker=".", c="r", label="Collocation Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.title("Loss Function Evolution")
plt.semilogy(lbfgs.loss_values, label="PINN")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

