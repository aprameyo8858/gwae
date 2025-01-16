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

def MaxSGW(X, Y, p=2, s_lr=0.1, n_lr=10, device='cuda', adam=False, nproj=200):
    """Returns the Max-Sliced Gromov-Wasserstein (Max-SGW) distance."""
    
    dim_x = X.size(1)  # Dimension of source distribution X
    dim_y = Y.size(1)  # Dimension of target distribution Y
    
    # Initialize theta (projection matrix) as a random tensor and normalize it
    theta = torch.randn((dim_x, dim_y), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
    
    # Optimizer for theta (Adam or SGD)
    if adam:
        optimizer = torch.optim.Adam([theta], lr=s_lr)
    else:
        optimizer = torch.optim.SGD([theta], lr=s_lr)
    
    X_detach = X.detach()  # Detach to avoid unnecessary gradients during optimization
    Y_detach = Y.detach()  # Detach to avoid unnecessary gradients during optimization
    
    # Optimization loop to update theta
    for _ in range(n_lr - 1):
        # Sink the source and target samples using the sink_ function
        xsp, xtp = sink_(X_detach, Y_detach, device, nproj=nproj)
        
        # Compute the negative SGW distance (maximize the distance)
        negative_sgw = -torch.pow(one_dimensional_Wasserstein_prod(xsp, xtp, theta, p=p).mean(), 1. / p)
        
        # Backpropagate and update theta
        optimizer.zero_grad()
        negative_sgw.backward()
        optimizer.step()
        
        # Re-normalize theta after each update
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1, keepdim=True))
    
    # Sink the source and target samples one last time after optimization
    xsp, xtp = sink_(X, Y, device, nproj=nproj)
    
    # Compute the final Max-SGW distance using the optimized theta
    max_sgw = one_dimensional_Wasserstein_prod(xsp, xtp, theta, p=p).mean()
    
    # Return the Max-SGW distance (raised to power 1/p)
    return torch.pow(max_sgw, 1. / p)

# Example usage
if __name__ == "__main__":
    # Example source and target distributions
    n_samples = 300
    Xs = torch.rand(n_samples, 2)  # Source distribution (e.g., 2D)
    Xt = torch.rand(n_samples, 1)  # Target distribution (e.g., 1D)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute Max-SGW distance
    max_sgw_distance = MaxSGW(Xs, Xt, p=2, s_lr=0.1, n_lr=10, device=device)
    print("Max-SGW Distance:", max_sgw_distance.item())
