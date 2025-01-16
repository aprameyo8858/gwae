import torch

# Assuming we have a method for computing 1D Wasserstein distance between projections
def one_dimensional_Wasserstein_prod(xs, xt, theta, p=2):
    """Computes the 1D Wasserstein distance between two distributions xs and xt using theta (projections)."""
    # Project the source and target distributions using the random projections (theta)
    x_proj = torch.matmul(xs, theta.T)  # Project the source data
    t_proj = torch.matmul(xt, theta.T)  # Project the target data
    
    # Compute pairwise distances in the projected space (L2 norm)
    dist = torch.norm(x_proj[:, None] - t_proj[None, :], dim=2, p=p)
    return dist

def sink_(xs, xt, device, nproj=200, P=None):
    """Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections."""
    dim_d = xs.shape[1]
    dim_p = xt.shape[1]
    
    # Ensure we are working with the correct dimensions, performing zero-padding if necessary
    if dim_d < dim_p:
        random_projection_dim = dim_p
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        random_projection_dim = dim_d
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs
    
    # If no predefined projection matrix is provided, generate a random one
    if P is None:
        P = torch.randn(random_projection_dim, nproj)
    
    # Normalize the projection matrix
    p = P / torch.sqrt(torch.sum(P ** 2, 0, True))
    
    try:
        # Perform the projection of both source and target samples
        xsp = torch.matmul(xs2, p.to(device))  # Project the source samples
        xtp = torch.matmul(xt2, p.to(device))  # Project the target samples
    except RuntimeError as error:
        print('Error in projection:', error)
        raise
    
    return xsp, xtp

def DSGW(X, Y, L=5, kappa=10, p=2, s_lr=0.1, n_lr=10, device='cuda', adam=False, nproj=200):
    """Returns the Differentiable Sliced Gromov-Wasserstein (DSGW) distance."""
    
    dim_x = X.size(1)  # Dimension of source distribution X
    dim_y = Y.size(1)  # Dimension of target distribution Y
    
    # Initialize epsilon as a random vector and normalize it
    epsilon = torch.randn((1, dim_x), device=device, requires_grad=True)
    epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    
    # Optimizer for epsilon (Adam or SGD)
    if adam:
        optimizer = torch.optim.Adam([epsilon], lr=s_lr)
    else:
        optimizer = torch.optim.SGD([epsilon], lr=s_lr)
    
    X_detach = X.detach()  # Detach to avoid unnecessary gradients during optimization
    Y_detach = Y.detach()  # Detach to avoid unnecessary gradients during optimization
    
    # Optimization loop to update epsilon
    for _ in range(n_lr - 1):
        # Create a von Mises-Fisher distribution for random projections
        vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
        theta = vmf.rsample((L,)).view(L, -1)
        
        # Compute the negative SGW distance (maximize the distance)
        negative_dsgw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach, Y_detach, theta, p=p).mean(), 1. / p)
        
        # Backpropagate and update epsilon
        optimizer.zero_grad()
        negative_dsgw.backward()
        optimizer.step()
        
        # Re-normalize epsilon after each update
        epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    
    # Create a von Mises-Fisher distribution with optimized epsilon
    vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
    theta = vmf.rsample((L,)).view(L, -1)
    
    # Sink the source and target samples using the sink_ function
    xsp, xtp = sink_(X, Y, device, nproj=nproj)
    
    # Compute the final DSGW distance using the optimized theta
    dsgw = one_dimensional_Wasserstein_prod(xsp, xtp, theta, p=p).mean()
    
    # Return the DSGW distance (raised to power 1/p)
    return torch.pow(dsgw, 1. / p)

# Example usage
if __name__ == "__main__":
    # Example source and target distributions
    n_samples = 300
    Xs = torch.rand(n_samples, 2)  # Source distribution (e.g., 2D)
    Xt = torch.rand(n_samples, 1)  # Target distribution (e.g., 1D)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute DSGW distance
    dsgw_distance = DSGW(Xs, Xt, L=5, kappa=10, p=2, s_lr=0.1, n_lr=10, device=device)
    print("DSGW Distance:", dsgw_distance.item())
