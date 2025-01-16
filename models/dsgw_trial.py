import torch
import time
import numpy as np

# Define a custom exception for bad shapes.
class BadShapeError(Exception):
    pass 

# PowerSpherical class, used in the original DSW code
class PowerSpherical(torch.nn.Module):
    def __init__(self, loc, scale):
        super(PowerSpherical, self).__init__()
        self.loc = loc
        self.scale = scale
    
    def rsample(self, size=None):
        # Implementation of sampling from the Power Spherical distribution
        # Assuming a simple spherical sampling
        v = torch.randn_like(self.loc) if size is None else torch.randn((size[0], self.loc.shape[1]), device=self.loc.device)
        v = v / torch.norm(v, dim=1, keepdim=True)  # Normalize to unit sphere
        return self.scale[:, None] * v + self.loc  # Scale and shift

# Sliced Gromov-Wasserstein with Directional Sampling (DSGW)
def DSGW(X, Y, L=5, kappa=10, p=2, s_lr=0.1, n_lr=2, device='cuda'):
    """ Returns DSGW (Directional Sliced Gromov-Wasserstein) distance.
    
    Parameters
    ----------
    X : tensor, shape (n, p)
        Source samples
    Y : tensor, shape (n, q)
        Target samples
    L : int
        Number of samples (L) for the spherical distribution
    kappa : float
        Concentration parameter for the spherical distribution
    p : float
        Power parameter for Wasserstein distance
    s_lr : float
        Step size for the gradient descent
    n_lr : int
        Number of gradient descent iterations
    device : torch.device
        Torch device (cpu or cuda)
    
    Returns
    -------
    DSGW distance : tensor
        The computed DSGW distance.
    """
    dim = X.size(1)
    
    # Step 1: Initialize epsilon (directional vector) and normalize it
    epsilon = torch.randn((1, dim), device=device, requires_grad=True)
    epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    
    # Set up optimizer for epsilon
    optimizer = torch.optim.SGD([epsilon], lr=s_lr)
    
    # Detach the data from the computation graph to avoid unnecessary gradients
    X_detach = X.detach()
    Y_detach = Y.detach()
    
    # Step 2: Perform gradient descent to optimize epsilon
    for _ in range(n_lr - 1):
        # Sample theta from the PowerSpherical distribution
        vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
        theta = vmf.rsample((L,)).view(L, -1)  # Get L samples from the spherical distribution
        
        # Compute the negative sliced Wasserstein distance (to minimize it)
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach, Y_detach, theta, p=p).mean(), 1. / p)
        
        # Backpropagation to update epsilon
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        
        # Re-normalize epsilon after the update
        epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    
    # Step 3: Compute the final Sliced Gromov-Wasserstein distance
    vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
    theta = vmf.rsample((L,)).view(L, -1)  # Sample from the spherical distribution
    
    # Compute the sliced Wasserstein distance between X and Y with the learned theta
    sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
    
    # Return the final distance (raised to power 1/p)
    return torch.pow(sw, 1. / p)

# Wasserstein distance function (adapted for 1D)
def one_dimensional_Wasserstein_prod(xs, xt, theta, p=2):
    """ Computes the 1D Wasserstein distance with the given theta.
    """
    # Computing pairwise distances using theta (assumes theta is the direction of the geodesic)
    dist = torch.sum((xs - xt)**2, dim=1)
    dist = torch.pow(dist, p / 2)  # Raise to power p
    
    # Return the mean Wasserstein distance
    return dist.mean()

# Example usage:
if __name__ == "__main__":
    # Example source and target distributions
    n_samples = 300
    Xs = np.random.rand(n_samples, 2)
    Xt = np.random.rand(n_samples, 1)
    X = torch.from_numpy(Xs).to(torch.float32)
    Y = torch.from_numpy(Xt).to(torch.float32)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute DSGW distance
    dsgw_distance = DSGW(X, Y, L=5, kappa=10, p=2, s_lr=0.1, n_lr=2, device=device)
    print("DSGW Distance:", dsgw_distance.item())

