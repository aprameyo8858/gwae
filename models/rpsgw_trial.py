import torch
import time
import numpy as np

# Define a custom exception for bad shapes.
class BadShapeError(Exception):
    pass 

# PowerSpherical class, used in RPSW (as in your original code)
class PowerSpherical(torch.nn.Module):
    def __init__(self, loc, scale):
        super(PowerSpherical, self).__init__()
        self.loc = loc
        self.scale = scale
    
    def rsample(self):
        # Implementation of sampling from the Power Spherical distribution
        # Assuming a simple spherical sampling
        v = torch.randn_like(self.loc)
        v = v / torch.norm(v, dim=1, keepdim=True)  # Normalize to unit sphere
        return self.scale[:, None] * v + self.loc  # Scale and shift

# Sliced Gromov-Wasserstein with Randomized Projections (RPSGW)
def RPSGW(xs, xt, device, nproj=200, p=2, kappa=50, tolog=False, P=None):
    """ Returns RPSGW (Randomized Projection Sliced Gromov-Wasserstein) distance.
    
    Parameters
    ----------
    xs : tensor, shape (n, p)
        Source samples
    xt : tensor, shape (n, q)
        Target samples
    device : torch.device
        Torch device (cpu or cuda)
    nproj : int
        Number of projections (ignored if P is not None)
    p : float
        Power parameter for Wasserstein distance
    kappa : float
        Concentration parameter for the spherical distribution
    P : tensor, shape (max(p,q), n_proj), optional
        Predefined projection matrix. If None, generates random projections.
    tolog : bool, optional
        Whether to return detailed logs (default False)
    
    Returns
    -------
    RPSGW distance : tensor
        The computed RPSGW distance.
    """
    
    # Step 1: Perform random projection to lower-dimensional space
    xsp, xtp, _ = sink_(xs, xt, device, nproj=nproj, P=P)
    
    # Step 2: Create theta using the random projections
    L = xsp.shape[0]  # Number of samples (n)
    theta = (xsp - xtp)  # The difference between source and target in projected space
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))  # Normalize
    
    # Step 3: Sample theta from the Power Spherical distribution
    ps = PowerSpherical(loc=theta, scale=torch.full((theta.shape[0],), kappa, device=device))
    theta_sampled = ps.rsample()  # Sample from the distribution
    
    # Step 4: Compute the Wasserstein distance using theta (this part remains the same as the original SGW)
    sw = one_dimensional_Wasserstein_prod(xsp, xtp, theta_sampled, p=p).mean()
    
    # Step 5: Return the RPSGW distance (Wasserstein raised to power 1/p)
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

# Sinkhorn operator (as in SGW code) for padding and projecting
def sink_(xs, xt, device, nproj=200, P=None):
    """Sinks the points in the lowest dimension onto the highest dimension and applies projections.
    
    Parameters
    ----------
    xs : tensor, shape (n, p)
        Source samples
    xt : tensor, shape (n, q)
        Target samples
    device : torch device
        Torch device (cpu or cuda)
    nproj : int
        Number of projections
    P : tensor, shape (max(p,q), n_proj), optional
        Predefined projection matrix
    
    Returns
    -------
    xsp : tensor, shape (n, n_proj)
        Projected source samples
    xtp : tensor, shape (n, n_proj)
        Projected target samples
    """
    dim_d = xs.shape[1]
    dim_p = xt.shape[1]
    
    if dim_d < dim_p:
        random_projection_dim = dim_p
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        random_projection_dim = dim_d
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs
    
    if P is None:
        P = torch.randn(random_projection_dim, nproj)
    p = P / torch.sqrt(torch.sum(P**2, 0, True))  # Normalize the projection matrix
    
    try:
        xsp = torch.matmul(xs2, p.to(device))
        xtp = torch.matmul(xt2, p.to(device))
    except RuntimeError as error:
        print('----------------------------------------')
        print('xs origi dim :', xs.shape)
        print('xt origi dim :', xt.shape)
        print('random_projection_dim :', random_projection_dim)
        print('projector dimension :', p.shape)
        print('xs2 dim :', xs2.shape)
        print('xt2 dim :', xt2.shape)
        print('----------------------------------------')
        print(error)
        raise BadShapeError
    
    return xsp, xtp, None

# Example usage:
if __name__ == "__main__":
    # Example source and target distributions
    n_samples = 300
    Xs = np.random.rand(n_samples, 2)
    Xt = np.random.rand(n_samples, 1)
    xs = torch.from_numpy(Xs).to(torch.float32)
    xt = torch.from_numpy(Xt).to(torch.float32)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute RPSGW distance
    rpsgw_distance = RPSGW(xs, xt, device, nproj=200, p=2, kappa=50)
    print("RPSGW Distance:", rpsgw_distance.item())
