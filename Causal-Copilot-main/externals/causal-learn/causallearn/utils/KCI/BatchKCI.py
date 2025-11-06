import torch
import gpytorch
import numpy as np
from numpy import sqrt
from numpy.linalg import eigh, eigvalsh
from scipy import stats
from typing import Tuple, List, Optional


class BatchKCI_CInd:
    """
    Batch implementation of Kernel-based Conditional Independence test using GPyTorch.
    Processes multiple independence tests at the same depth simultaneously.
    """
    
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', 
                 nullss=5000, est_width='empirical', use_gp=True, approx=True, 
                 polyd=2, kwidthx=None, kwidthy=None, kwidthz=None, device=None):
        """
        Initialize the BatchKCI_CInd model.
        
        Parameters
        ----------
        kernelX, kernelY, kernelZ: kernel functions
        nullss: sample size for null distribution
        est_width: method to estimate kernel width
        use_gp: whether to use Gaussian processes
        approx: whether to use gamma approximation
        polyd: polynomial kernel degree
        kwidthx, kwidthy, kwidthz: manual kernel widths
        device: torch device (None for auto-selection)
        """
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.kwidthz = kwidthz
        self.nullss = nullss
        self.epsilon_x = 1e-3
        self.epsilon_y = 1e-3
        self.use_gp = use_gp
        self.thresh = 1e-5
        self.approx = approx
        
        # Set device (CPU or CUDA if available)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_pvalues_batch(self, data_pairs: List[Tuple], data_z: np.ndarray) -> List[Tuple[float, float]]:
        """
        Compute p-values for multiple pairs of variables conditioned on Z in batch.
        
        Parameters
        ----------
        data_pairs: List of (data_x, data_y) tuples to test
        data_z: conditioning data, same for all tests
        
        Returns
        -------
        List of (p-value, test_statistic) tuples
        """
        try:
            import gpytorch
        except ImportError:
            print("GPyTorch not installed. Installing required packages...")
            import subprocess
            subprocess.check_call(["pip", "install", "gpytorch"])
            import gpytorch
            
        # Convert all data to PyTorch tensors
        batch_size = len(data_pairs)
        
        # Z data is the same for all tests in current depth
        tensor_z = torch.tensor(stats.zscore(data_z, ddof=1, axis=0), dtype=torch.float32, device=self.device)
        tensor_z[torch.isnan(tensor_z)] = 0.0
        
        # Prepare batch data for X and Y
        tensor_x_list = []
        tensor_y_list = []
        
        for data_x, data_y in data_pairs:
            # Normalize and convert to tensor
            tx = torch.tensor(stats.zscore(data_x, ddof=1, axis=0), dtype=torch.float32, device=self.device)
            tx[torch.isnan(tx)] = 0.0
            
            ty = torch.tensor(stats.zscore(data_y, ddof=1, axis=0), dtype=torch.float32, device=self.device)
            ty[torch.isnan(ty)] = 0.0
            
            # For X, concatenate with Z as in the original implementation
            tx_concat = torch.cat([tx, 0.5 * tensor_z], dim=1)
            
            tensor_x_list.append(tx_concat)
            tensor_y_list.append(ty)
        
        # Use batch GP processing if GP is enabled
        if self.use_gp and self.kernelZ == 'Gaussian':
            batch_results = self._process_batch_with_gp(tensor_x_list, tensor_y_list, tensor_z)
            if batch_results is not None:
                return batch_results
            # Fall back to non-GP method if batch_results is None
        
        # Process each test individually if not using batch GP
        batch_results = []
        for i in range(batch_size):
            data_x = tensor_x_list[i]
            data_y = tensor_y_list[i]
            
            # Compute kernel matrices
            Kx, Ky, Kz = self._compute_kernel_matrices(data_x, data_y, tensor_z)
            
            # Compute test statistic
            test_stat, KxR, KyR = self._compute_test_statistic(Kx, Ky, Kz, tensor_z if self.use_gp else None)
            
            # Compute p-value
            if self.approx:
                # Gamma approximation
                k_appr, theta_appr = self._get_kappa(KxR, KyR)
                pvalue = 1 - stats.gamma.cdf(test_stat.cpu().numpy(), k_appr, 0, theta_appr)
            else:
                # Spectral approximation - more computationally intensive
                uu_prod, size_u = self._get_uuprod(KxR.cpu().numpy(), KyR.cpu().numpy())
                null_samples = self._null_sample_spectral(uu_prod, size_u, KxR.shape[0])
                pvalue = sum(null_samples > test_stat.cpu().numpy()) / float(self.nullss)
            
            batch_results.append((float(pvalue), float(test_stat.cpu().numpy())))
            
        return batch_results
    
    def _compute_kernel_matrices(self, data_x, data_y, data_z):
        """
        Compute kernel matrices for X, Y, and Z
        
        Parameters
        ----------
        data_x: tensor for x data
        data_y: tensor for y data  
        data_z: tensor for z data
        
        Returns
        -------
        Kx, Ky, Kz: kernel matrices
        """
        # Compute kernel matrices based on kernel type
        if self.kernelX == 'Gaussian':
            width_x = self._estimate_width(data_x)
            Kx = self._gaussian_kernel(data_x, width_x)
        else:
            raise NotImplementedError(f"Kernel type {self.kernelX} not implemented")
            
        if self.kernelY == 'Gaussian':  
            width_y = self._estimate_width(data_y)
            Ky = self._gaussian_kernel(data_y, width_y)
        else:
            raise NotImplementedError(f"Kernel type {self.kernelY} not implemented")
            
        if self.kernelZ == 'Gaussian':
            if self.use_gp:
                # Will be handled in test statistic computation
                Kz = None
            else:
                width_z = self._estimate_width(data_z)
                Kz = self._gaussian_kernel(data_z, width_z)
        else:
            raise NotImplementedError(f"Kernel type {self.kernelZ} not implemented")
        
        # Center kernel matrices
        Kx = self._center_kernel_matrix(Kx)
        Ky = self._center_kernel_matrix(Ky)
        
        if Kz is not None:
            Kz = self._center_kernel_matrix(Kz)
            
        return Kx, Ky, Kz
    
    def _estimate_width(self, data):
        """
        Estimate kernel width based on the specified method
        
        Parameters
        ----------
        data: input data tensor
        
        Returns
        -------
        width: estimated kernel width
        """
        if self.est_width == 'empirical':
            # Empirical width based on dimensionality
            return 1.0 if data.shape[1] == 0 else 1.0 / data.shape[1]
        elif self.est_width == 'median':
            # Median heuristic
            n = data.shape[0]
            # Compute pairwise distances
            X2 = (data**2).sum(1).view(-1, 1)
            distance_squared = X2 + X2.transpose(0, 1) - 2.0 * torch.mm(data, data.transpose(0, 1))
            distance = torch.sqrt(torch.clamp(distance_squared, min=0))
            
            # Get the median
            median_dist = torch.median(distance.view(-1))
            return 1.0 / (2 * median_dist**2)
        elif self.est_width == 'manual':
            if self.kwidthx is not None:
                return self.kwidthx
            else:
                # Default fallback
                return 1.0 / data.shape[1]
        else:
            # Default 
            return 1.0 / data.shape[1]
    
    def _gaussian_kernel(self, data, width):
        """
        Compute Gaussian kernel matrix
        
        Parameters
        ----------
        data: input data tensor
        width: kernel width
        
        Returns
        -------
        K: kernel matrix
        """
        n = data.shape[0]
        
        # Compute squared pairwise distances
        X2 = (data**2).sum(1).view(-1, 1)
        distance_squared = X2 + X2.transpose(0, 1) - 2.0 * torch.mm(data, data.transpose(0, 1))
        
        # Apply kernel function
        K = torch.exp(-0.5 * distance_squared * width)
        return K
    
    def _center_kernel_matrix(self, K):
        """
        Center kernel matrix
        
        Parameters
        ----------
        K: kernel matrix
        
        Returns
        -------
        Kc: centered kernel matrix
        """
        n = K.shape[0]
        # Create identity and ones matrices
        I = torch.eye(n, device=self.device, dtype=torch.float32)
        ones = torch.ones(n, 1, device=self.device, dtype=torch.float32)
        
        # Compute centering matrix H = I - 1/n 11^T
        H = I - ones.mm(ones.transpose(0, 1)) / n
        
        # Center kernel matrix
        Kc = H.mm(K.mm(H))
        return Kc
    
    def _compute_test_statistic(self, Kx, Ky, Kz=None, tensor_z=None):
        """
        Compute KCI test statistic
        
        Parameters
        ----------
        Kx: kernel matrix for X
        Ky: kernel matrix for Y  
        Kz: kernel matrix for Z
        tensor_z: tensor for z data
        
        Returns
        -------
        test_stat: test statistic
        KxR, KyR: residual kernel matrices
        """
        n = Kx.shape[0]
        I = torch.eye(n, device=self.device)
        
        if self.use_gp and self.kernelZ == 'Gaussian':
            try:
                import gpytorch
            except ImportError:
                print("GPyTorch not installed. Using non-GP method.")
                self.use_gp = False
                
            if self.use_gp and tensor_z is not None:
            # try:
                # Eigendecomposition for feature extraction for GP model
                # Symmetrize matrices to ensure real eigenvalues
                Kx_sym = 0.5 * (Kx + Kx.t())
                Ky_sym = 0.5 * (Ky + Ky.t())
                
                # Eigendecomposition
                wx, vx = torch.linalg.eigh(Kx_sym)
                wy, vy = torch.linalg.eigh(Ky_sym)
                
                # Sort eigenvalues in descending order
                idx = torch.argsort(wx, descending=True)
                idy = torch.argsort(wy, descending=True)
                
                wx = wx[idx]
                vx = vx[:, idx]
                wy = wy[idy]
                vy = vy[:, idy]
                
                # Keep significant eigenvectors
                vx = vx[:, wx > wx.max() * self.thresh]
                wx = wx[wx > wx.max() * self.thresh]
                vy = vy[:, wy > wy.max() * self.thresh]
                wy = wy[wy > wy.max() * self.thresh]
                
                # Scale eigenvectors by eigenvalues
                vx = vx @ torch.diag(torch.sqrt(wx))
                vy = vy @ torch.diag(torch.sqrt(wy))
                
                # Define GP models and likelihood
                class GPModel(gpytorch.models.ExactGP):
                    def __init__(self, train_x, train_y, likelihood):
                        super(GPModel, self).__init__(train_x, train_y, likelihood)
                        self.mean_module = gpytorch.means.ZeroMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.RBFKernel()
                        )
                    
                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                
                # Get training data
                # We'll use the first eigenvector for each matrix
                train_x = vx[:, 0:1].cpu() if vx.shape[1] > 0 else torch.randn(n, 1)
                train_y = vy[:, 0:1].cpu() if vy.shape[1] > 0 else torch.randn(n, 1)
                
                # Create likelihoods and models
                likelihood_x = gpytorch.likelihoods.GaussianLikelihood()
                likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
                
                # Use the provided tensor_z directly for GP
                tensor_z_cpu = tensor_z.cpu()
                
                model_x = GPModel(tensor_z_cpu, train_x, likelihood_x)
                model_y = GPModel(tensor_z_cpu, train_y, likelihood_y)
                
                # Training mode
                model_x.train()
                likelihood_x.train()
                model_y.train()
                likelihood_y.train()
                
                # Use the Adam optimizer
                optimizer_x = torch.optim.Adam(model_x.parameters(), lr=0.1)
                optimizer_y = torch.optim.Adam(model_y.parameters(), lr=0.1)
                
                # "Loss" for GPs - the marginal log likelihood
                mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_x, model_x)
                mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_y, model_y)
                
                # Training the model
                for i in range(50):
                    # Zero gradients from previous iteration
                    optimizer_x.zero_grad()
                    optimizer_y.zero_grad()
                    
                    # Output from model
                    output_x = model_x(tensor_z_cpu)
                    output_y = model_y(tensor_z_cpu)
                    
                    # Calculate loss and backpropagation
                    loss_x = -mll_x(output_x, train_x.squeeze())
                    loss_y = -mll_y(output_y, train_y.squeeze())
                    
                    loss_x.backward()
                    loss_y.backward()
                    
                    # Update parameters
                    optimizer_x.step()
                    optimizer_y.step()
                
                # Get final covariance matrices 
                model_x.eval()
                model_y.eval()
                
                with torch.no_grad():
                    # Get Kzx and Kzy
                    Kzx = model_x.covar_module(tensor_z_cpu).evaluate().to(self.device, dtype=torch.float32)
                    Kzy = model_y.covar_module(tensor_z_cpu).evaluate().to(self.device, dtype=torch.float32)
                    
                    # Get residual matrices
                    self.epsilon_x = likelihood_x.noise.item()
                    self.epsilon_y = likelihood_y.noise.item()
                    
                    Rx = I - Kzx @ torch.inverse(Kzx + self.epsilon_x * I).to(torch.float32)
                    Ry = I - Kzy @ torch.inverse(Kzy + self.epsilon_y * I).to(torch.float32)

                    Rx = Rx.to(torch.float32)
                    Ry = Ry.to(torch.float32)
                    
                    KxR = Rx @ Kx @ Rx
                    KyR = Ry @ Ky @ Ry
                
                # Compute test statistic
                test_stat = torch.sum(KxR * KyR)
                
                return test_stat, KxR, KyR
                # except Exception as e:
                #     print(f"Error in GP computation: {e}")
                #     self.use_gp = False
        
        if not self.use_gp or self.kernelZ != 'Gaussian':
            # Use non-GP method with pre-computed Kz
            if Kz is not None:
                # Compute residuals using matrix inversion
                Rz = I - Kz.mm(torch.inverse(Kz + self.epsilon_x * I))
                KxR = Rz.mm(Kx.mm(Rz))
                KyR = Rz.mm(Ky.mm(Rz))
            else:
                # No conditioning (just use centered matrices)
                KxR = Kx
                KyR = Ky
        
        # Compute test statistic
        test_stat = torch.sum(KxR * KyR)
        
        return test_stat, KxR, KyR
    
    def _get_kappa(self, KxR, KyR):
        """
        Get parameters for the gamma approximation
        
        Parameters
        ----------
        KxR, KyR: residual kernel matrices
        
        Returns
        -------
        k_appr, theta_appr: gamma distribution parameters
        """
        n = KxR.shape[0]
        
        # Move to CPU for numpy compatibility
        KxR_np = KxR.cpu().numpy()
        KyR_np = KyR.cpu().numpy()
        
        # Mean approximation
        mean_appr = np.trace(KxR_np) * np.trace(KyR_np) / n
        
        # Variance approximation
        var_appr = 2 * np.sum(KxR_np**2) * np.sum(KyR_np**2) / (n*n)
        
        # Gamma distribution parameters
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        
        return k_appr, theta_appr
    
    def _get_uuprod(self, KxR, KyR):
        """
        Compute the product of eigenvectors for spectral sampling
        
        Parameters
        ----------
        KxR, KyR: residual kernel matrices (numpy arrays)
        
        Returns
        -------
        uu_prod: product of eigenvectors
        size_u: number of eigenvectors
        """
        from numpy.linalg import eigh
        from numpy import sqrt
        
        # Symmetrize matrices
        KxR = 0.5 * (KxR + KxR.T)
        KyR = 0.5 * (KyR + KyR.T)
        
        # Eigen-decomposition of KxR
        wx, vx = eigh(KxR)
        idx = np.argsort(-wx)
        wx = wx[idx]
        vx = vx[:, idx]
        
        # Keep only significant eigenvectors/values
        vx = vx[:, wx > wx.max() * self.thresh]
        wx = wx[wx > wx.max() * self.thresh]
        
        # Eigen-decomposition of KyR
        wy, vy = eigh(KyR)
        idy = np.argsort(-wy)
        wy = wy[idy]
        vy = vy[:, idy]
        
        # Keep only significant eigenvectors/values
        vy = vy[:, wy > wy.max() * self.thresh]
        wy = wy[wy > wy.max() * self.thresh]
        
        # Get number of eigenvectors
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        
        # Scale eigenvectors by eigenvalues
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))
        
        # Calculate products
        T = KxR.shape[0]
        size_u = num_eigx * num_eigy
        uu = np.zeros((T, size_u))
        
        # Compute element-wise products of eigenvectors
        for i in range(num_eigx):
            for j in range(num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]
        
        # Compute final product
        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)
        
        return uu_prod, size_u
    
    def _null_sample_spectral(self, uu_prod, size_u, T):
        """
        Sample from null distribution using spectral method
        
        Parameters
        ----------
        uu_prod: product of eigenvectors
        size_u: number of eigenvectors
        T: number of samples
        
        Returns
        -------
        null_dstr: samples from null distribution
        """
        from numpy.linalg import eigvalsh
        
        # Get eigenvalues
        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        
        # Keep only significant eigenvalues
        eig_uu = eig_uu[0:min(T, size_u)]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]
        
        # Generate chi-square samples
        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        
        # Compute null distribution samples
        null_dstr = eig_uu.T.dot(f_rand)
        
        return null_dstr

    def _process_batch_with_gp(self, tensor_x_list, tensor_y_list, tensor_z):
        """
        Process a batch of tests using GPyTorch with shared conditioning variable tensor_z
        
        Parameters
        ----------
        tensor_x_list: List of X tensors
        tensor_y_list: List of Y tensors
        tensor_z: Z tensor (shared across all tests)
        
        Returns
        -------
        batch_results: List of (p-value, test_statistic) tuples
        """
        try:
            import gpytorch
        except ImportError:
            print("GPyTorch not installed. Using non-GP method.")
            return None
            
        # try:
        batch_size = len(tensor_x_list)
        n = tensor_z.shape[0]
        
        # Process in smaller sub-batches to avoid memory issues
        max_sub_batch = 5  # Adjust based on memory constraints
        batch_results = []
        
        for sub_batch_start in range(0, batch_size, max_sub_batch):
            sub_batch_end = min(sub_batch_start + max_sub_batch, batch_size)
            sub_batch_size = sub_batch_end - sub_batch_start
            
            # Setup batch GP models
            class BatchGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
                    # Batch mean and covariance modules
                    self.mean_module = gpytorch.means.ZeroMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(batch_shape=torch.Size([sub_batch_size])),
                        batch_shape=torch.Size([sub_batch_size])
                    )
                
                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
            # Compute kernel matrices for this sub-batch
            sub_batch_Kx = []
            sub_batch_Ky = []
            sub_batch_train_x = []
            sub_batch_train_y = []
            
            for i in range(sub_batch_start, sub_batch_end):
                # Get the tensors
                x_tensor = tensor_x_list[i]
                y_tensor = tensor_y_list[i]
                
                # Compute kernel matrices
                width_x = self._estimate_width(x_tensor)
                width_y = self._estimate_width(y_tensor)
                
                Kx = self._gaussian_kernel(x_tensor, width_x)
                Ky = self._gaussian_kernel(y_tensor, width_y)
                
                # Center kernel matrices
                Kx = self._center_kernel_matrix(Kx)
                Ky = self._center_kernel_matrix(Ky)
                
                sub_batch_Kx.append(Kx)
                sub_batch_Ky.append(Ky)
                
                # Get eigenvectors for GP training
                Kx_sym = 0.5 * (Kx + Kx.t())
                Ky_sym = 0.5 * (Ky + Ky.t())
                
                wx, vx = torch.linalg.eigh(Kx_sym)
                wy, vy = torch.linalg.eigh(Ky_sym)
                
                idx = torch.argsort(wx, descending=True)
                idy = torch.argsort(wy, descending=True)
                
                wx = wx[idx]
                vx = vx[:, idx]
                wy = wy[idy]
                vy = vy[:, idy]
                
                # Use first eigenvector as target
                sub_batch_train_x.append(vx[:, 0].cpu())
                sub_batch_train_y.append(vy[:, 0].cpu())
            
            # Stack tensors for batch processing
            batch_train_x = torch.stack(sub_batch_train_x)
            batch_train_y = torch.stack(sub_batch_train_y)
            tensor_z_cpu = tensor_z.cpu()
            
            # Create batch likelihood and models
            likelihood_x = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([sub_batch_size]))
            likelihood_y = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([sub_batch_size]))
            
            model_x = BatchGPModel(tensor_z_cpu, batch_train_x, likelihood_x)
            model_y = BatchGPModel(tensor_z_cpu, batch_train_y, likelihood_y)
            
            # Training mode
            model_x.train()
            likelihood_x.train()
            model_y.train()
            likelihood_y.train()
            
            # Use the Adam optimizer
            optimizer_x = torch.optim.Adam(model_x.parameters(), lr=0.1)
            optimizer_y = torch.optim.Adam(model_y.parameters(), lr=0.1)
            
            # "Loss" for GPs - the marginal log likelihood
            mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_x, model_x)
            mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_y, model_y)
            
            # Training the models
            for i in range(50):  # Adjust number of iterations as needed
                # Zero gradients from previous iteration
                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                
                # Output from model
                output_x = model_x(tensor_z_cpu)
                output_y = model_y(tensor_z_cpu)
                
                # Calculate loss and backpropagation
                loss_x = -mll_x(output_x, batch_train_x)
                loss_y = -mll_y(output_y, batch_train_y)
                
                loss_x.sum().backward()
                loss_y.sum().backward()
                
                # Update parameters
                optimizer_x.step()
                optimizer_y.step()
            
            # Get final covariance matrices
            model_x.eval()
            model_y.eval()
            
            # Identity matrix for residual computation
            I = torch.eye(n, device=self.device)
            
            # Compute test statistics and p-values for each item in batch
            with torch.no_grad():
                # Get batch kernel matrices
                batch_Kzx = model_x.covar_module(tensor_z_cpu).evaluate().to(self.device)
                batch_Kzy = model_y.covar_module(tensor_z_cpu).evaluate().to(self.device)
                
                # Get noise parameters
                batch_epsilon_x = likelihood_x.noise.to(self.device)
                batch_epsilon_y = likelihood_y.noise.to(self.device)
                
                for i in range(sub_batch_size):
                    # Get kernel matrices for this test
                    Kx = sub_batch_Kx[i]
                    Ky = sub_batch_Ky[i]
                    Kzx = batch_Kzx[i]
                    Kzy = batch_Kzy[i]
                    epsilon_x = batch_epsilon_x[i].item()
                    epsilon_y = batch_epsilon_y[i].item()
                    
                    # Compute residual matrices
                    Rx = I - Kzx @ torch.inverse(Kzx + epsilon_x * I)
                    Ry = I - Kzy @ torch.inverse(Kzy + epsilon_y * I)

                    Rx = Rx.to(torch.float32)
                    Ry = Ry.to(torch.float32)
                    
                    KxR = Rx @ Kx @ Rx
                    KyR = Ry @ Ky @ Ry
                    
                    # Compute test statistic
                    test_stat = torch.sum(KxR * KyR)
                    
                    # Compute p-value using gamma approximation
                    if self.approx:
                        k_appr, theta_appr = self._get_kappa(KxR, KyR)
                        pvalue = 1 - stats.gamma.cdf(test_stat.cpu().numpy(), k_appr, 0, theta_appr)
                    else:
                        # Spectral approximation - more computationally intensive
                        uu_prod, size_u = self._get_uuprod(KxR.cpu().numpy(), KyR.cpu().numpy())
                        null_samples = self._null_sample_spectral(uu_prod, size_u, KxR.shape[0])
                        pvalue = sum(null_samples > test_stat.cpu().numpy()) / float(self.nullss)
                    
                    batch_results.append((float(pvalue), float(test_stat.cpu().numpy())))
        
        return batch_results
            
        # except Exception as e:
        #     print(f"Error in batch GP computation: {e}")
        #     return None