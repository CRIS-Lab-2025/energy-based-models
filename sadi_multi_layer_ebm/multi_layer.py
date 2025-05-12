import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import time, os
from IPython.display import clear_output, display

# --- Device Setup ---
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# --- Activations ---
def hard_sigmoid(x): return torch.clamp(x, 0, 1)
def hard_sigmoid_deriv(x): return ((x >= 0) & (x <= 1)).float()

# --- Dataset ---
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, data_home='./mnist_cache', parser='auto')
    X = mnist.data.to_numpy().astype('float32') / 255.0
    le = LabelEncoder()
    y = le.fit_transform(mnist.target)
    y_1hot = np.eye(10)[y].astype('float32')
    return (X[:60000], y_1hot[:60000], y[:60000]), (X[60000:], y_1hot[60000:], y[60000:])

# --- Core EBM Class ---
class EnergyBasedModel(nn.Module):
    def __init__(self, model=[784,32,32,10], dt=0.2, beta=0.5, device=device,
                 epochs=10, batch_size=1024, subset_size=10000, lr=0.01,
                 n_free=50, n_nudge=10):
        #self = torch.compile(self, mode="reduce-overhead")
        self.input_dim = input_dim = model[0]
        self.hidden_dim = hidden_dim = model[1]
        self.output_dim = output_dim = model[2]
        self.model = model
        self.n_layers = len(model)
        self.total_dim = sum(model)
        self.num_state_units = self.total_dim - input_dim
        self.device = device
        self.dt = dt
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.lr_ih = lr
        self.lr_ho = lr
        self.lr_b = lr
        self.n_free = n_free
        self.n_nudge = n_nudge


        self.idx = []
        dims = np.cumsum(model)
        for i in range(self.n_layers):
            if i == 0:
                self.idx.append(torch.arange(0, dims[i]))
            else:
                self.idx.append(torch.arange(dims[i-1], dims[i]))
        self.idx_s = torch.arange(self.input_dim, self.total_dim, device=device)
        self.idx_s_out = torch.tensor(np.where(np.isin(self.idx_s.cpu(), self.idx[-1].cpu()))[0], device=device)

        self.W = torch.zeros(self.total_dim, self.total_dim, device=device)
        self.b = torch.zeros(self.total_dim, device=device)
        self.init_weights()

        # Clamp masks
        self.clamp_mask = torch.zeros(self.total_dim, device=device)
        self.clamp_values = torch.zeros(self.total_dim, device=device)

    def clamp(self, idx, values):
        self.clamp_mask[idx] = 1.0
        self.clamp_values[idx] = values

    def unclamp(self, idx):
        self.clamp_mask[idx] = 0.0

    def init_weights(self):
        def glorot(in_dim, out_dim):
            limit = np.sqrt(6 / (in_dim + out_dim))
            return torch.empty((in_dim, out_dim), device=self.device).uniform_(-limit, limit)
        
        self.W = []
        for i in range(self.n_layers-1):
            self.W.append(glorot(self.model[i], self.model[i+1] if i < self.n_layers - 1 else 0))


    def compute_energy(self, s, u):
        rho_u = hard_sigmoid(u)
        s_full = torch.zeros_like(rho_u)
        s_full[:, s.shape[1] * -1:] = s
        diff = s_full - rho_u
        term1 = 0.5 * torch.sum(diff**2, dim=1)
        term2 = -torch.sum(rho_u * self.b, dim=1)
        # Initialize bilinear term of the energy
        term3 = 0
        # Compute energy contributions from all layer connections
        for i in range(self.n_layers-1):
            term3 += torch.sum((rho_u[:, self.idx[i]] @ self.W[i]) * rho_u[:, self.idx[i+1]], dim=1)
        term3 = -0.5 * term3
        return term1 + term2 + term3

    def energy_grad(self, s, rho_u, rho_prime, y_target=None, h_target=None):
        out = rho_u.new_zeros(rho_u.shape)        # [B, N]
        # in → hidden  (W[idx_in, idx_h] = W_ih)
        for i in range(1,self.n_layers-1):
            out[:, self.idx[i]]    = rho_u[:, self.idx[i-1]]   @ self.W[i-1] + rho_u[:, self.idx[i+1]]  @ self.W[i].T

        # hidden → in   (W[idx_h, idx_in] = W_ih.T)
        out[:, self.idx[0]]  = rho_u[:, self.idx[1]]     @ self.W[0].T

        # hidden → out  (W[idx_h, idx_out] = W_ho)
        out[:, self.idx[-1]] = rho_u[:, self.idx[-2]]     @ self.W[-1]

        full_term = out
        grad_E = s - rho_prime * full_term[:, self.idx_s]

        grad_C = torch.zeros_like(s)
        if self.beta > 0:
            if y_target is not None:
                grad_C[:, self.idx_s_out] = s[:, self.idx_s_out] - y_target
            if h_target is not None:
                idx_s_hid = torch.tensor(np.where(np.isin(self.idx_s.cpu(), self.idx[1].cpu()))[0], device=self.device)
                grad_C[:, idx_s_hid] = s[:, idx_s_hid] - h_target

        return grad_E + self.beta * grad_C


    def simulate(self, s_init, x, beta, y_target, n_iter, learn_input=False, stochastic=False):
        """
        Run inference dynamics on s (state = [h, y]) for an EBM using energy gradients.

        Args:
            s_init: Initial state (B, num_state_units)
            x: Input (B, input_dim), always clamped
            beta: Clamping factor (0.0 = free phase, >0.0 = weakly clamped)
            y_target: Target outputs (B, output_dim), used only if beta > 0
            n_iter: Number of inference steps
            learn_input: If True, uses optimizer to learn u[:, idx_in]
            stochastic: If True, adds Langevin noise for MCMC-style inference

        Returns:
            (final_s, final_u): state trajectory and full u vector (detached)
        """
        B = s_init.shape[0]
        s = s_init.clone() if learn_input else s_init.clone().detach()

        u = torch.zeros(B, self.total_dim, device=self.device, requires_grad=learn_input)

        if x is not None:
            with torch.no_grad():
                u[:, self.idx[0]] = x

        optimizer_u = None
        if learn_input:
            optimizer_u = torch.optim.SGD([u], lr=0.1)

        for _ in range(n_iter):
            # Clamp input x every step unless learning it
            if not learn_input and x is not None:
                with torch.no_grad():
                    u[:, self.idx[0]] = x

            # Sync u's s-part to s
            with torch.no_grad():
                u[:, self.idx_s] = s

            rho_u = hard_sigmoid(u)
            rho_prime = hard_sigmoid_deriv(s.detach() if learn_input else s)

            grad_s = self.energy_grad(s, rho_u.detach(), rho_prime, y_target)

            # --- Stochastic MCMC-style Langevin step ---
            noise = torch.randn_like(s) * (np.sqrt(2 * self.dt) if stochastic else 0.0)
            s_new = s - self.dt * grad_s.detach() + noise

            # Apply clamping to relevant units
            if self.clamp_mask[self.idx_s].sum() > 0:
                s_new = s_new * (1 - self.clamp_mask[self.idx_s]) + self.clamp_values[self.idx_s] * self.clamp_mask[self.idx_s]
            # Do batch normalization
            idx_s_hid = torch.tensor(np.where(np.isin(self.idx_s.cpu(), self.idx[1].cpu()))[0], device=self.device)
            s_mean = s_new[:, idx_s_hid].mean(0)
            s_std = s_new[:, idx_s_hid].std(0)
            s_new[:, idx_s_hid] = (s_new[:, idx_s_hid] - s_mean) / (s_std + 1e-8) #* 0.1 + s_mean
            s = torch.clamp(s_new, 0, 1).detach()

            # Optional input learning via energy minimization
            if learn_input:
                optimizer_u.zero_grad()
                rho_u_for_energy = hard_sigmoid(u)
                energy = self.compute_energy(s.detach(), u).sum()
                energy.backward()
                optimizer_u.step()

        return s, u.detach()



    def run_free_phase(self, s0, x):
        return self.simulate(s0, x, beta=0, y_target=None, n_iter=self.n_free)

    def run_nudged_phase(self, s_free, x, y_target):
        return self.simulate(s_free, x, beta=self.beta, y_target=y_target, n_iter=self.n_nudge)

    def evaluate(self, X, y_true, n_iter):
        with torch.no_grad():
            preds = []
            for i in range(0, X.shape[0], 512):
                xb = X[i:i+512]
                s0 = torch.rand(xb.shape[0], self.num_state_units, device=self.device) * 0.1
                s_out, _ = self.simulate(s0, xb, beta=0, y_target=None, n_iter=n_iter)
                preds.append(torch.argmax(s_out[:, self.idx_s_out], dim=1))

            preds = torch.cat(preds)
            y_true_t = torch.tensor(y_true[:len(preds)], device=self.device)
            return (preds == y_true_t).float().mean().item()  # ✅ all torch



    @classmethod
    def train_model(cls, **kwargs):
        model = cls(**kwargs)
        model.train()
        return model

    def train(self, cutoff_accuracy = 1.0):
        (X_train, y_train, _), (X_test, y_test_1hot, y_test) = load_mnist()
        X_train = torch.tensor(X_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)
        X_test = torch.tensor(X_test).to(self.device)
        y_test_1hot = torch.tensor(y_test_1hot).to(self.device)

        W,b = self.W, self.b
        idx_in, idx_h, idx_out = self.idx[0], self.idx[1], self.idx[2]
        idx_s, idx_s_out = self.idx_s, self.idx_s_out

        train_losses, test_accs, energy_free_hist, energy_nudge_hist = [], [], [], []

        live_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        display(plt.gcf())

        for epoch in range(self.epochs):
            dW = []
            for i in range(self.n_layers-1):
                dW.append(torch.zeros_like(W[i], device=self.device))
            db = torch.zeros(self.total_dim-self.input_dim, device=self.device)
            total_loss, total_free_E, total_nudge_E = 0, 0, 0

            idx = torch.randperm(X_train.shape[0], device=self.device)[:self.subset_size]
            for i in range(0, self.subset_size, self.batch_size):
                xb = X_train[idx[i:i+self.batch_size]]
                yb = y_train[idx[i:i+self.batch_size]]
                s0 = torch.rand(xb.shape[0], self.num_state_units, device=self.device) * 0.1

                s_free, u_free = self.run_free_phase(s0, xb)
                s_nudge, u_nudge = self.run_nudged_phase(s_free, xb, yb)

                rho_free = hard_sigmoid(torch.cat([xb, s_free], dim=1))
                rho_nudge = hard_sigmoid(torch.cat([xb, s_nudge], dim=1))

                delta_W = (torch.einsum('bi,bj->ij', rho_nudge, rho_nudge) - torch.einsum('bi,bj->ij', rho_free, rho_free)) / self.batch_size / self.beta
                for j in range(self.n_layers-1):
                    dW[j] += delta_W[self.idx[j][:, None], self.idx[j+1]]

                delta_b = (rho_nudge - rho_free).mean(0) / self.beta
                db += delta_b[self.idx_s]

                loss = 0.5 * ((s_free[:, idx_s_out] - yb)**2).sum(1).mean()
                total_loss += loss.item() * xb.shape[0]

                total_free_E += self.compute_energy(s_free, u_free).mean().item()
                total_nudge_E += self.compute_energy(s_nudge, u_nudge).mean().item()

            # Update
            # Print mean of dW  for each layer
           # print(s_nudge.mean(dim=0))
            for j in range(self.n_layers-1):
                W[j] += self.lr_ih * dW[j] / (self.subset_size // self.batch_size)
            b[idx_s] += self.lr_b * db / (self.subset_size // self.batch_size)

            acc = self.evaluate(X_test, y_test, n_iter=self.n_free)
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/self.subset_size:.6f} | Acc: {acc:.4f}")

            train_losses.append(total_loss/self.subset_size)
            test_accs.append(acc)
            energy_free_hist.append(total_free_E / (self.subset_size // self.batch_size))
            energy_nudge_hist.append(total_nudge_E / (self.subset_size // self.batch_size))

            ax1.clear(); ax2.clear()
            ax1.plot(train_losses, label='Train Loss')
            ax1.plot(energy_free_hist, label='Free Energy')
            ax1.plot(energy_nudge_hist, label='Nudged Energy')
            ax1.set_title('Loss / Energy'); ax1.set_xlabel('Epoch'); ax1.grid(); ax1.legend()
            ax2.plot(test_accs)
            ax2.set_title('Test Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylim(0, 1); ax2.grid()
            clear_output(wait=True)
            display(plt.gcf())

            if acc >= cutoff_accuracy:
                print(f"✅ Stopped early at epoch {epoch+1} after reaching {acc:.2%}")
                break


    def make_state(self, x, h, y):
        """
        Build full state tensors (s and u) given input x and optionally h, y.

        Args:
            x (B, input_dim)
            h (B, hidden_dim) or None → defaults to zeros
            y (B, output_dim) or None → defaults to zeros

        Returns:
            s (B, hidden_dim + output_dim), u (B, total_dim)
        """
        B = x.shape[0]
        device = self.device

        h = h if h is not None else torch.zeros(B, self.hidden_dim, device=device)
        y = y if y is not None else torch.zeros(B, self.output_dim, device=device)

        s = torch.cat([h, y], dim=1)
        u = torch.zeros(B, self.total_dim, device=device)
        u[:, self.idx[0]] = x
        u[:, self.idx[1]] = h
        u[:, self.idx[2]] = y
        return s, u

    def sample_state(self, x):
        B = x.shape[0]
        h = torch.rand(B, self.hidden_dim, device=self.device)
        y = torch.rand(B, self.output_dim, device=self.device)
        return self.make_state(x, h, y)


    def energy_from_parts(self, x, h=None, y=None):
        s, u = self.make_state(x, h, y)
        return self.compute_energy(s, u)


    def compute_energy_from_u(self, u):
        """
        Computes energy given full state vector u = [x, h, y].
        Automatically extracts s = [h, y] from u.
        """
        s = u[:, self.idx_s]  # just pull out [h, y]
        return self.compute_energy(s, u)


    def get_x(self, u):
        return u[:, self.idx[0]]

    def get_h(self, u):
        return u[:, self.idx[1]]

    def get_y(self, u):
        return u[:, self.idx[2]]


    def set_x(self, u, x):
        u[:, self.idx[0]] = x
        return u

    def set_h(self, u, h):
        u[:, self.idx[1]] = h
        return u

    def set_y(self, u, y):
        u[:, self.idx[2]] = y
        return u




    def save_to_file(self, path):
        state = {
            'config': self.get_config(),
            'W': self.W.cpu(),
            'b': self.b.cpu(),
            'clamp_mask': self.clamp_mask.cpu(),
            'clamp_values': self.clamp_values.cpu(),
        }
        torch.save(state, path)
        print(f"EBM saved to {path}")


    @classmethod
    def load_from_file(cls, path, device=None):
        ckpt = torch.load(path, map_location=device or torch.device("cpu"))
        config = ckpt['config']
        model = cls(**config)
        model.W.data.copy_(ckpt['W'].to(model.device))
        model.b.data.copy_(ckpt['b'].to(model.device))
        model.clamp_mask.data.copy_(ckpt['clamp_mask'].to(model.device))
        model.clamp_values.data.copy_(ckpt['clamp_values'].to(model.device))
        print(f"EBM loaded from {path}")
        return model

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dt': self.dt,
            'beta': self.beta,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'subset_size': self.subset_size,
            'lr_ih': self.lr_ih,
            'lr_ho': self.lr_ho,
            'lr_b': self.lr_b,
            'n_free': self.n_free,
            'n_nudge': self.n_nudge,
        }




