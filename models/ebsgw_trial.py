import torch

# Define the Sliced Gromov-Wasserstein (SGW) function for 1D projections
def one_dimensional_Wasserstein_prod(xs, xt, theta, p=2):
    """Computes the 1D Wasserstein distance between two distributions xs and xt using theta (projections)."""
    # Project the source and target distributions using the random projections (theta)
    x_proj = torch.matmul(xs, theta.T)  # Project the source data
    t_proj = torch.matmul(xt, theta.T)  # Project the target data
    
    # Compute pairwise distances in the projected space (L2 norm)
    dist = torch.norm(x_proj[:, None] - t_proj[None, :], dim=2, p=p)
    return dist

# Define a function to generate random projections (L-dimensional)
def rand_projections(dim, L, device='cuda'):
    """Generate random projections with L directions for the data of dimensionality `dim`."""
    theta = torch.randn(L, dim, device=device)  # Random normal projections
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))  # Normalize the projections
    return theta

# Define the Sink function that prepares the source and target data
def sink_(xs, xt, device, nproj=200, P=None):
    """Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator."""
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
    p = P / torch.sqrt(torch.sum(P ** 2, 0, True))
    
    xsp = torch.matmul(xs2, p.to(device))
    xtp = torch.matmul(xt2, p.to(device))
    
    return xsp, xtp

# Define the Empirical Barycenter Sliced Gromov-Wasserstein (EBSGW) function
def EBSGW(X, Y, L=10, p=2, device='cuda'):
    """Computes the Empirical Barycenter Sliced Gromov-Wasserstein (EBSGW) distance."""
    dim = X.size(1)  # Dimension of the data
    
    # Generate random projections
    theta = rand_projections(dim, L, device)
    
    # Sink the source and target distributions to the same dimension
    xsp, xtp = sink_(X, Y, device, nproj=L)
    
    # Compute the Sliced Gromov-Wasserstein distances for each projection
    wasserstein_distances = one_dimensional_Wasserstein_prod(xsp, xtp, theta, p=p)
    wasserstein_distances = wasserstein_distances.view(1, L)  # Reshape to (1, L)
    
    # Compute the weights using softmax over the distances
    weights = torch.softmax(-wasserstein_distances, dim=1)  # Negative sign for minimization
    
    # Compute the weighted sum of the distances
    sw = torch.sum(weights * wasserstein_distances, dim=1).mean()
    
    # Return the final EBSGW distance raised to the power of 1/p
    return torch.pow(sw, 1. / p)

# Example usage
if __name__ == "__main__":
    # Example source and target distributions
    n_samples = 300
    Xs = torch.rand(n_samples, 2)  # Source distribution (e.g., 2D)
    Xt = torch.rand(n_samples, 1)  # Target distribution (e.g., 1D)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute EBSGW distance
    ebsgw_distance = EBSGW(Xs, Xt, L=10, p=2, device=device)
    print("EBSGW Distance:", ebsgw_distance.item())
