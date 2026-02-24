import torch
import numpy as np
import torch.nn.functional as F
from . import config, Parameter

def identity(x):
    """
    The identity link function given by

    $$ y = x $$
    """
    return x

def square(x):
    """
    The square link function given by

    $$ y = x^2 $$
    """
    return torch.square(x)

def exp(x):
    """
    The exponential link function given by

    $$ y = e^{x} $$
    """
    return torch.exp(x)

def probit(x):
    """
    The probit link function given by

    $$ y = \\sqrt{2} \\operatorname{erf}^{-1}(2x-1)\\right) $$
    """
    return np.sqrt(2)*torch.erfinv(2.0*x - 1.0)

def inv_probit(x):
    """
    The inverse probit link function given by

    $$ y = \\frac{1}{2} \\left(1 + \\operatorname{erf}(x/\\sqrt{2})\\right) $$
    """
    jitter = 1e-3
    return 0.5*(1.0+torch.erf(x/np.sqrt(2.0))) * (1.0-2.0*jitter) + jitter

# also inv_logit or logistic
def sigmoid(x):
    """
    The logistic, inverse logit, or sigmoid link function given by

    $$ y = \\frac{1}{1 + e^{-x}} $$
    """
    return 1.0/(1.0+torch.exp(-x))

def log_logistic_distribution(loc, scale):
    return torch.distributions.transformed_distribution.TransformedDistribution(
        base_distribution=torch.distributions.uniform.Uniform(0.0,1.0),
        transforms=[
            torch.distributions.transforms.SigmoidTransform().inv,
            torch.distributions.transforms.AffineTransform(loc=loc, scale=scale),
            torch.distributions.transforms.ExpTransform(),
        ],
    )

class GaussHermiteQuadrature:
    def __init__(self, deg=20, t_scale=None, w_scale=None):
        t, w = np.polynomial.hermite.hermgauss(deg)
        t = t.reshape(-1,1)
        w = w.reshape(-1,1)
        if t_scale is not None:
            t *= t_scale
        if w_scale is not None:
            w *= w_scale
        self.t = torch.tensor(t, device=config.device, dtype=config.dtype)  # degx1
        self.w = torch.tensor(w, device=config.device, dtype=config.dtype)  # degx1
        self.deg = deg

    def __call__(self, mu, var, F):
        return F(mu + var.sqrt().mm(self.t.T)).mm(self.w)  # Nx1

class Likelihood(torch.nn.Module):
    """
    Base likelihood.

    Args:
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, quadratures=20):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature(deg=quadratures, t_scale=np.sqrt(2), w_scale=1.0/np.sqrt(np.pi))
        self.output_dims = None

    def name(self):
        return self.__class__.__name__

    def __setattr__(self, name, val):
        if name == 'train':
            for p in self.parameters():
                p.train = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val._name is None:
            val._name = '%s.%s' % (self.__class__.__name__, name)
        elif isinstance(val, torch.nn.ModuleList):
            for i, item in enumerate(val):
                for p in item.parameters():
                    p._name = '%s[%d].%s' % (self.__class__.__name__, i, p._name)
        super().__setattr__(name, val)

    def _channel_indices(self, X):
        c = X[:,0].long()
        m = [c==j for j in range(self.output_dims)]
        r = [torch.nonzero(m[j], as_tuple=False).reshape(-1) for j in range(self.output_dims)]
        return r

    def validate_y(self, X, y):
        """
        Validate whether the y input is within the likelihood's support.
        """
        pass

    def log_prob(self, X, y, f):
        """
        Calculate the log probability density

        $$ \\log(p(y|f)) $$

        with \\(p(y|f)\\) our likelihood.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            y (torch.tensor): Values for y of shape (data_points,).
            f (torch.tensor): values for f of shape (data_points,quadratures).

        Returns:
            torch.tensor: Log probability density of shape (data_points,quadratures).
        """
        raise NotImplementedError()

    def variational_expectation(self, X, y, mu, var):
        """
        Calculate the variational expectation

        $$ \\int \\log(p(y|X,f)) \\; q(f) \\; df $$

        where \\(q(f) \\sim \\mathcal{N}(\\mu,\\Sigma)\\) and \\(p(y|f)\\) our likelihood. By default this uses Gauss-Hermite quadratures to approximate the integral.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            y (torch.tensor): Values for y of shape (data_points,).
            mu (torch.tensor): Mean of the posterior \\(q(f)\\) of shape (data_points,1).
            var (torch.tensor): Variance of the posterior \\(q(f)\\) of shape (data_points,1).

        Returns:
            torch.tensor: Expected log density of shape ().
        """
        q = self.quadrature(mu, var, lambda f: self.log_prob(X,y,f))
        return q.sum()

    def conditional_mean(self, X, f):
        """
        Calculate the mean of the likelihood conditional on \\(f\\).

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            f (torch.tensor): Posterior values for f of shape (data_points,quadratures).

        Returns:
            torch.tensor: Mean of the predictive posterior \\(p(y|f)\\) of shape (data_points,quadratures).
        """
        raise NotImplementedError()

    def conditional_sample(self, X, f):
        """
        Sample from likelihood distribution.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            f (torch.tensor): Samples of f values of shape (n,data_points).

        Returns:
            torch.tensor: Samples of shape (n,data_points).
        """
        # f: nxN
        raise NotImplementedError()

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        """
        Calculate the mean and variance of the predictive distribution

        $$ \\mu = \\iint y \\; p(y|f) \\; q(f) \\; df dy $$
        $$ \\Sigma = \\iint y^2 \\; p(y|f) \\; q(f) \\; df dy - \\mu^2 $$

        where \\(q(f) \\sim \\mathcal{N}(\\mu,\\Sigma)\\) and \\(p(y|f)\\) our likelihood. By default this uses Gauss-Hermite quadratures to approximate both integrals.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            mu (torch.tensor): Mean of the posterior \\(q(f)\\) of shape (data_points,1).
            var (torch.tensor): Variance of the posterior \\(q(f)\\) of shape (data_points,1).
            ci (list of float): Two percentages [lower, upper] in the range of [0,1] that represent the confidence interval.
            sigma (float): Number of standard deviations of the confidence interval. Only used for short-path for Gaussian likelihood.
            n (int): Number of samples used from distribution to estimate quantile.

        Returns:
            torch.tensor: Mean of the predictive posterior \\(p(y|f)\\) of shape (data_points,1).
            torch.tensor: Lower confidence boundary of shape (data_points,1).
            torch.tensor: Upper confidence boundary of shape (data_points,1).
        """
        # mu,var: Nx1
        mu = self.quadrature(mu, var, lambda f: self.conditional_mean(X,f))
        if ci is None:
            return mu

        samples_f = torch.distributions.normal.Normal(mu, var).sample([n]) # nxNx1
        samples_y = self.conditional_sample(X, samples_f) # nxNx1
        if samples_y is None:
            return mu, mu, mu
        samples_y, _ = samples_y.sort(dim=0)
        lower = int(ci[0]*n + 0.5)
        upper = int(ci[1]*n + 0.5)
        return mu, samples_y[lower,:], samples_y[upper,:]

class MultiOutputLikelihood(Likelihood):
    """
    Multi-output likelihood to assign a different likelihood per channel.

    Args:
        likelihoods (mogptk.gpr.likelihood.Likelihood): List of likelihoods equal to the number of output dimensions.
    """
    def __init__(self, *likelihoods):
        super().__init__()

        if isinstance(likelihoods, tuple):
            if len(likelihoods) == 1 and isinstance(likelihoods[0], list):
                likelihoods = likelihoods[0]
            else:
                likelihoods = list(likelihoods)
        elif not isinstance(likelihoods, list):
            likelihoods = [likelihoods]
        if len(likelihoods) == 0:
            raise ValueError("must pass at least one likelihood")
        for i, likelihood in enumerate(likelihoods):
            if not issubclass(type(likelihood), Likelihood):
                raise ValueError("must pass likelihoods")
            elif isinstance(likelihood, MultiOutputLikelihood):
                raise ValueError("can not nest MultiOutputLikelihoods")

        self.output_dims = len(likelihoods)
        self.likelihoods = torch.nn.ModuleList(likelihoods)

    def name(self):
        names = [likelihood.name() for likelihood in self.likelihoods]
        return '[%s]' % (','.join(names),)

    def validate_y(self, X, y):
        if self.output_dims == 1:
            self.likelihoods[0].validate_y(X, y)
            return

        r = self._channel_indices(X)
        for i in range(self.output_dims):
            self.likelihoods[i].validate_y(X, y[r[i],:])

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].log_prob(X, y[r[i],:], f[r[i],:])
        return res  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        q = torch.tensor(0.0, dtype=config.dtype, device=config.device)
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            q += self.likelihoods[i].variational_expectation(X, y[r[i],:], mu[r[i],:], var[r[i],:]).sum()  # sum over N
        return q

    def conditional_mean(self, X, f):
        # f: NxQ
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].conditional_mean(X, f[r[i],:])
        return res  # NxQ

    def conditional_sample(self, X, f):
        # f: nxN
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            f[:,r[i]] = self.likelihoods[i].conditional_sample(X, f[:,r[i]])
        return f  # nxN

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        # mu,var: Nx1
        r = self._channel_indices(X)
        res = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        if ci is None:
            for i in range(self.output_dims):
                res[r[i],:] = self.likelihoods[i].predict(X, mu[r[i],:], var[r[i],:], ci=ci, sigma=sigma, n=n)
            return res

        lower = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        upper = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:], lower[r[i],:], upper[r[i],:] = self.likelihoods[i].predict(X, mu[r[i],:], var[r[i],:], ci=ci, sigma=sigma, n=n)
        return res, lower, upper  # Nx1

class MultiLatentLikelihood(Likelihood):
    """
    Multi-latent likelihood wrapper.

    A single likelihood whose parameters are modeled by multiple latent functions.

    Example:
        Heteroscedastic Gaussian:
            f[:,0] -> mean
            f[:,1] -> log variance
    """
    def __init__(self, likelihood, mc_samples=10):
        super().__init__()

        if not issubclass(type(likelihood), Likelihood):
            raise ValueError("likelihood must be an instance of Likelihood")

        if isinstance(likelihood, MultiOutputLikelihood):
            raise ValueError("MultiLatentLikelihood wraps a single likelihood only")

        self.likelihood = likelihood
        # Obtener latent_dims del likelihood específico
        if not hasattr(likelihood, 'latent_dims'):
            raise ValueError("El likelihood debe definir el atributo 'latent_dims'")

        self.latent_dims = likelihood.latent_dims
        self.output_dims = self.latent_dims
        if self.latent_dims <= 0:
            raise ValueError("latent_dims debe ser positivo")

        self.has_closed_form = getattr(likelihood, 'has_closed_form', False)
        self.mc_samples = mc_samples
    def name(self):
        return f"MultiLatent[{self.likelihood.name()}, Q={self.latent_dims}]"

    def validate_y(self, X, y):
        """
        Validate observations.
        """
        # Calcular N (número de puntos reales)
        total_points = X.shape[0]
        N = total_points // self.latent_dims

        # Solo validar los primeros N puntos
        y_real = y[:N, :]
        self.likelihood.validate_y(X, y_real)

    def _reshape_mogptk_to_latent(self, X, tensor):
        """
        Convierte de formato mogptk (N*Q, 1) a formato latent (N, Q)
        """
        total_points = tensor.shape[0]  # N*Q (ejemplo: 400)
        N = total_points // self.latent_dims  # N (ejemplo: 200)
        Q = self.latent_dims  # Q (ejemplo: 2)


        # Reshape: (N*Q, 1) -> (N*Q,) -> (Q, N) -> (N, Q)
        tensor_flat = tensor.squeeze(-1)  # (400,)

        tensor_reshaped = tensor_flat.reshape(Q, N).T  # (2, 200) -> (200, 2)

        return tensor_reshaped

    def _reshape_latent_to_mogptk(self, X, tensor):
        """
        Convierte de formato latent (N, Q) o (N,) a formato mogptk (N*Q, 1)

        Ejemplo: [[f0,g0], [f1,g1], ..., [f199,g199]] -> [f0, f1, ..., f199, g0, g1, ..., g199]
        """
        if tensor.ndim == 1:
            # Si tensor es (N,), replicar para todos los canales
            N = tensor.shape[0]
            Q = self.latent_dims
            tensor_expanded = tensor.unsqueeze(-1).expand(N, Q)  # (N, Q)
            tensor_flat = tensor_expanded.T.reshape(-1, 1)  # (N*Q, 1)
        else:
            # Si tensor es (N, Q)
            N, Q = tensor.shape
            tensor_flat = tensor.T.reshape(-1, 1)  # (Q, N) -> (N*Q, 1)

        return tensor_flat

    def log_prob(self, X, y, f):
        """
        Args:
            X : inputs
            y : observations, shape (N, 1)
            f : latent functions, shape (N*Q, 1)

        Returns:
            log p(y | f1, ..., fQ)
        """
       # if f.ndim != 2:
       #     raise ValueError("f must be a 2D tensor (N, Q)")

        #if f.shape[1] != self.latent_dims:
        #   raise ValueError(
        #        f"Expected {self.latent_dims} latent functions, got {f.shape[1]}"
        #    )
        # Convertir f de (N*Q, 1) a (N, Q)
        f_reshaped = self._reshape_mogptk_to_latent(X, f)

        # Obtener y real (solo primer canal)
        r = self._channel_indices(X)
        y_real = y[r[0], :]

        # Calcular log_prob - resultado es (N,) o (N, 1)
        logp = self.likelihood.log_prob(X, y_real, f_reshaped)

        if logp.ndim == 1:
            logp = logp.unsqueeze(-1)

        # Convertir de (N, 1) a (N*Q, 1) - replicar para todos los canales
        return self._reshape_latent_to_mogptk(X, logp.squeeze(-1))

    def variational_expectation(self, X, y, mu, var):
        """
        Args:
            X : inputs, shape (N*Q, input_dims)
            y : observations, shape (N*Q, 1)
            mu : mean, shape (N*Q, 1)
            var : variance, shape (N*Q, 1)

        Returns:
            scalar: E_q[log p(y | f)]
        """


        # Convertir de (N*Q, 1) a (N, Q)
        mu_reshaped = self._reshape_mogptk_to_latent(X, mu)
        var_reshaped = self._reshape_mogptk_to_latent(X, var)

        # Obtener y real (primeros N puntos)
        N = mu_reshaped.shape[0]
        Q = self.latent_dims
        y_real = y[:N, :]


        if self.has_closed_form:
            return self.likelihood.variational_expectation(X, y_real, mu_reshaped, var_reshaped )

        S = self.mc_samples

        # Reparametrización
        eps = torch.randn(S, N, Q, device=mu_reshaped.device, dtype=mu_reshaped.dtype)

        f_samples = mu_reshaped.unsqueeze(0) + eps * torch.sqrt(var_reshaped).unsqueeze(0)

        # Calcular log p(y | f) para cada muestra
        logp_sum = 0.0
        for s in range(S):
            logp_s = self.likelihood.log_prob(X, y_real, f_samples[s])
            logp_sum += logp_s.sum()

        # Promedio Monte Carlo
        return logp_sum / S

    def conditional_mean(self, X, f):
        """
        Media condicional E[y | f1, ..., fQ]

        Args:
            X: inputs, shape (N*Q, input_dims)
            f: funciones latentes, shape (N*Q, n_cols)
                n_cols puede ser 1 (normal) o deg (cuadratura)

        Returns:
            mean: E[y | f], shape (N*Q, n_cols)
        """
        # Guardar shape original
        n_cols = f.shape[1] if f.ndim == 2 else 1
        if n_cols == 1 and f.ndim == 1:
            f = f.unsqueeze(-1)

        # Calcular dimensiones
        N = f.shape[0] // self.latent_dims
        Q = self.latent_dims

        # Procesar cada columna (punto de cuadratura) por separado
        means_list = []
        for col_idx in range(n_cols):
            # Extraer columna
            f_col = f[:, col_idx:col_idx + 1]  # (N*Q, 1)

            # Reshape: (N*Q, 1) -> (N, Q)
            f_col_flat = f_col.squeeze(-1)  # (N*Q,)
            f_reshaped = f_col_flat.reshape(Q, N).T  # (N, Q)

            # Llamar likelihood interno
            mean_col = self.likelihood.conditional_mean(X, f_reshaped)  # (N, 1)

            # Reshape de vuelta: (N, 1) -> (N*Q, 1)
            mean_col_flat = mean_col.squeeze(-1)  # (N,)
            mean_col_expanded = mean_col_flat.unsqueeze(-1).expand(N, Q)  # (N, Q)
            mean_col_mogptk = mean_col_expanded.T.reshape(-1, 1)  # (N*Q, 1)

            means_list.append(mean_col_mogptk)

        # Concatenar columnas
        result = torch.cat(means_list, dim=1)  # (N*Q, n_cols)

        return result

    def conditional_sample(self, X, f):
        """
        Muestrea y ~ p(y | f1, ..., fQ)

        Args:
            X: inputs, shape (N*Q, input_dims)
            f: funciones latentes, shape (N*Q, n_cols)

        Returns:
            sample: muestra de y, shape (N*Q, n_cols)
        """
        # Guardar shape original
        n_cols = f.shape[1] if f.ndim == 2 else 1
        if n_cols == 1 and f.ndim == 1:
            f = f.unsqueeze(-1)

        # Calcular dimensiones
        N = f.shape[0] // self.latent_dims
        Q = self.latent_dims

        # Procesar cada columna por separado
        samples_list = []
        for col_idx in range(n_cols):
            # Extraer columna
            f_col = f[:, col_idx:col_idx + 1]  # (N*Q, 1)

            # Reshape: (N*Q, 1) -> (N, Q)
            f_col_flat = f_col.squeeze(-1)  # (N*Q,)
            f_reshaped = f_col_flat.reshape(Q, N).T  # (N, Q)

            # Llamar likelihood interno
            sample_col = self.likelihood.conditional_sample(X, f_reshaped)  # (N, 1)

            # Reshape de vuelta: (N, 1) -> (N*Q, 1)
            sample_col_flat = sample_col.squeeze(-1)  # (N,)
            sample_col_expanded = sample_col_flat.unsqueeze(-1).expand(N, Q)  # (N, Q)
            sample_col_mogptk = sample_col_expanded.T.reshape(-1, 1)  # (N*Q, 1)

            samples_list.append(sample_col_mogptk)

        # Concatenar columnas
        result = torch.cat(samples_list, dim=1)  # (N*Q, n_cols)

        return result

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        """
        Wrapper genérico para predict.

        Delega al likelihood interno, manejando solo la conversión de shapes.

        Args:
            X: inputs, shape (N*Q, input_dims)
            mu: media posterior, shape (N*Q, 1)
            var: varianza posterior, shape (N*Q, 1)
            ci: intervalo de confianza
            sigma: número de desviaciones estándar
            n: número de muestras

        Returns:
            mean: shape (N*Q, 1)
            lower: shape (N*Q, 1) (opcional)
            upper: shape (N*Q, 1) (opcional)
        """
        # Convertir de (N*Q, 1) a (N, Q)
        mu_reshaped = self._reshape_mogptk_to_latent(X, mu)
        var_reshaped = self._reshape_mogptk_to_latent(X, var)

        # Llamar al método predict del likelihood interno
        result = self.likelihood.predict(X, mu_reshaped, var_reshaped, ci=ci, sigma=sigma, n=n)

        # Manejar diferentes formatos de retorno
        if isinstance(result, tuple):
            # Retorna (mean, lower, upper)
            pred_mean, lower, upper = result

            # Convertir cada uno de (N, 1) a (N*Q, 1)
            mean_mogptk = self._reshape_latent_to_mogptk(X, pred_mean.squeeze(-1))
            lower_mogptk = self._reshape_latent_to_mogptk(X, lower.squeeze(-1))
            upper_mogptk = self._reshape_latent_to_mogptk(X, upper.squeeze(-1))

            return mean_mogptk, lower_mogptk, upper_mogptk
        else:
            # Solo retorna mean
            pred_mean = result

            # Convertir de (N, 1) a (N*Q, 1)
            mean_mogptk = self._reshape_latent_to_mogptk(X, pred_mean.squeeze(-1))

            return mean_mogptk

class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood given by

    $$ p(y|f) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(y-f)^2}{2\\sigma^2}} $$

    with \\(\\sigma\\) the scale.

    Args:
        scale (float,torch.tensor): Scale as float or as tensor of shape (output_dims,) when considering a multi-output Gaussian likelihood.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = Parameter(scale, lower=config.positive_minimum)
        if self.scale.ndim == 1:
            self.output_dims = self.scale.shape[0]

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -0.5 * (np.log(2.0*np.pi) + 2.0*self.scale().log() + ((y-f)/self.scale()).square())
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.scale().square()
        p -= np.log(2.0 * np.pi)
        p -= 2.0*self.scale().log()
        return 0.5*p.sum()  # sum over N

    def conditional_mean(self, X, f):
        return f

    def conditional_sample(self, X, f):
        return torch.distributions.normal.Normal(f, scale=self.scale()).sample()

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        if ci is None and sigma is None:
            return mu

        if self.output_dims is not None:
            var = self.scale()**2
            r = self._channel_indices(X)
            lower = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
            upper = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
            for i in range(self.output_dims):
                if sigma is None:
                    ci = torch.tensor(ci, device=config.device, dtype=config.dtype)
                    lower[r[i],:] = mu[r[i],:] + torch.sqrt(2.0*var[i])*self.scale()[i]*torch.erfinv(2.0*ci[0] - 1.0)
                    upper[r[i],:] = mu[r[i],:] + torch.sqrt(2.0)*self.scale()[i]*torch.erfinv(2.0*ci[1] - 1.0)
                else:
                    lower[r[i],:] = mu[r[i],:] - sigma*self.scale()[i]
                    upper[r[i],:] = mu[r[i],:] + sigma*self.scale()[i]
            return mu, lower, upper  # Nx1

        var += self.scale()**2
        if sigma is None:
            ci = torch.tensor(ci, device=config.device, dtype=config.dtype)
            lower = mu + torch.sqrt(2.0*var)*torch.erfinv(2.0*ci[0] - 1.0)
            upper = mu + torch.sqrt(2.0*var)*torch.erfinv(2.0*ci[1] - 1.0)
        else:
            lower = mu - sigma*var.sqrt()
            upper = mu + sigma*var.sqrt()
        return mu, lower, upper



class GaussianHeteroLikelihood(Likelihood):
    """
    Gaussian heteroscedastic likelihood:
        y | f, g ~ N(f, exp(g))

    Latent dimensions:
        f[:,0] -> mean
        f[:,1] -> log-variance
    """

    def __init__(self):
        super().__init__()
        self.latent_dims = 2
        self.has_closed_form_ve = True

    def safe_exp(self, x):
        """
        Exponencial segura para evitar overflow.
        """
        UPPER_LIM = torch.log(torch.tensor(705.0, device=x.device, dtype=x.dtype))
        x_clipped = torch.clamp(x, -UPPER_LIM, UPPER_LIM)
        return torch.exp(x_clipped)

    def log_prob(self, X, y, f):
        """
        Calcula log p(y | f, g)

        Args:
            X: inputs (no usado pero requerido por interfaz)
            y: observaciones, shape (N, 1)
            f: funciones latentes, shape (..., N, 2)
                Puede ser (N, 2) o (S, N, 2) para muestras Monte Carlo

        Returns:
            log p(y|f,g), shape (..., N) o escalar
        """
        # Extraer f y g
        F = f[..., 0]  # (..., N) - media
        G = f[..., 1]  # (..., N) - log-varianza

        # y puede ser (N, 1), convertir a (N,)
        y_flat = y.squeeze(-1)  # (N,)

        # Calcular log p(y | F, G)
        # log p = -(1/2) * [log(2π) + G + (y - F)²/exp(G)]
        var = self.safe_exp(G)
        log_prob = -0.5 * (
                torch.log(2 * torch.tensor(np.pi, device=y.device, dtype=y.dtype)) +
                G +
                (y_flat - F) ** 2 / var
        )

        return log_prob  # (..., N)

    def variational_expectation(self, X, y, mu, var):
        """
        Calcula E_q[log p(y | f, g)] usando forma cerrada.

        Args:
            X: inputs (no usado)
            y: observaciones, shape (N, 1)
            mu: media posterior, shape (N, 2)
                mu[:,0] = E[f]
                mu[:,1] = E[g]
            var: varianza posterior, shape (N, 2)
                var[:,0] = Var[f]
                var[:,1] = Var[g]

        Returns:
            E_q[log p(y|f,g)], escalar
        """
        # Extraer componentes
        Fmu = mu[:, 0:1]  # (N, 1) - media de f
        Gmu = mu[:, 1:2]  # (N, 1) - media de g
        Fvar = var[:, 0:1]  # (N, 1) - varianza de f
        Gvar = var[:, 1:2]  # (N, 1) - varianza de g

        # Constantes
        N = torch.tensor(y.shape[0], dtype=y.dtype, device=y.device)
        π = torch.tensor(np.pi, dtype=y.dtype, device=y.device)

        # Término de varianza usando la fórmula cerrada
        # Gmu + exp(-Gmu + Gvar/2) * (Y² - 2Y*Fmu + Fvar + Fmu²)
        exp_term = torch.exp(-Gmu + Gvar / 2)
        var_term = Gmu + exp_term * (y ** 2 - 2 * y * Fmu + Fvar + Fmu ** 2)

        # E_q[log p(y|f,g)] = -(1/2) * [N*log(2π) + sum(var_term)]
        ve = -0.5 * (N * torch.log(2 * π) + var_term.sum())

        return ve

    def conditional_mean(self, X, f):
        """
        Media condicional: E[y | f, g] = f

        Args:
            X: inputs, shape (N, input_dims) - no usado pero requerido
            f: funciones latentes, shape (N, 2)
                f[:,0] = media
                f[:,1] = log-varianza

        Returns:
            mean: E[y | f, g], shape (N, 1)
        """
        # La media condicional de y dado f,g es simplemente f (la componente de media)
        # Ignoramos g (log-varianza) porque no afecta la media
        return f[:, 0:1]  # (N, 1)

    def conditional_sample(self, X, f):
        """
        Genera muestras de y | f, g

        Args:
            X: inputs
            f: funciones latentes, shape (N, 2)

        Returns:
            samples, shape (N,)
        """
        F = f[:, 0]  # Media
        G = f[:, 1]  # Log-varianza

        var = self.safe_exp(G)
        std = torch.sqrt(var)

        # Muestrear de N(F, exp(G))
        samples = torch.distributions.normal.Normal(F, std).sample()

        return samples

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        """
        Predicción usando forma cerrada para Gaussian heteroscedastic.

        Args:
            X: inputs, shape (N, input_dims)
            mu: media posterior, shape (N, 2)
                mu[:,0] = E[f]
                mu[:,1] = E[g]
            var: varianza posterior, shape (N, 2)
                var[:,0] = Var[f]
                var[:,1] = Var[g]
            ci: intervalo de confianza [lower, upper]
            sigma: número de desviaciones estándar
            n: número de muestras (no usado, forma cerrada)

        Returns:
            mean: media predictiva, shape (N, 1)
            lower: límite inferior (opcional), shape (N, 1)
            upper: límite superior (opcional), shape (N, 1)
        """
        # Extraer componentes
        Fmu = mu[:, 0:1]  # (N, 1) - E[f]
        Gmu = mu[:, 1:2]  # (N, 1) - E[g]
        Fvar = var[:, 0:1]  # (N, 1) - Var[f]
        Gvar = var[:, 1:2]  # (N, 1) - Var[g]

        # Media predictiva: E[y] = E[E[y|f,g]] = E[f]
        pred_mean = Fmu  # (N, 1)

        # Varianza predictiva: Var[y] = E[Var[y|f,g]] + Var[E[y|f,g]]
        #                             = E[exp(g)] + Var[f]
        # donde E[exp(g)] = exp(E[g] + Var[g]/2) por propiedad lognormal
        expected_noise_var = torch.exp(Gmu + Gvar / 2)
        pred_var = Fvar + expected_noise_var  # (N, 1)

        if ci is None and sigma is None:
            return pred_mean

        # Calcular intervalos usando aproximación Gaussiana
        pred_std = torch.sqrt(pred_var)

        if sigma is not None:
            # Usar número de desviaciones estándar
            lower = pred_mean - sigma * pred_std
            upper = pred_mean + sigma * pred_std
        else:
            # Usar cuantiles de la distribución normal
            # Crear distribución normal
            dist = torch.distributions.Normal(pred_mean, pred_std)

            # Calcular cuantiles
            lower_q = torch.tensor(ci[0], device=mu.device, dtype=mu.dtype)
            upper_q = torch.tensor(ci[1], device=mu.device, dtype=mu.dtype)

            lower = dist.icdf(lower_q)
            upper = dist.icdf(upper_q)

        return pred_mean, lower, upper

class StudentTLikelihood(Likelihood):
    """
    Student's t likelihood given by

    $$ p(y|f) = \\frac{\\Gamma\\left(\\frac{\\nu+1}{2}\\right)}{\\Gamma\\left(\\frac{\\nu}{2}\\right)\\sqrt{\\pi\\nu}\\sigma} \\left( 1 + \\frac{(y-f)^2}{\\nu\\sigma^2} \\right)^{-(\\nu+1)/2} $$

    with \\(\\nu\\) the degrees of freedom and \\(\\sigma\\) the scale.

    Args:
        dof (float): Degrees of freedom.
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, dof=3, scale=1.0, quadratures=20):
        super().__init__(quadratures)
        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -0.5 * (self.dof+1.0)*torch.log1p(((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * (torch.log(self.dof) + np.log(np.pi) + self.scale().square().log())
        return p  # NxQ

    def conditional_mean(self, X, f):
        if self.dof <= 1.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f

    # TODO: implement predict for certain values of dof

    def conditional_sample(self, X, f):
        return torch.distributions.studentT.StudentT(self.dof, f, self.scale()).sample()

class ExponentialLikelihood(Likelihood):
    """
    Exponential likelihood given by

    $$ p(y|f) = 1/h(f) e^{-y/h(f)} $$

    with \\(h\\) the link function and \\(y \\in [0.0,\\infty)\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link

    def validate_y(self, X, y):
        if torch.any(y < 0.0):
            raise ValueError("y must be positive")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = -y/self.link(f) - f
        else:
            p = -y/self.link(f) - self.link(f).log()
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        if self.link != exp:
            super().variational_expectation(X, y, mu, var)

        p = -mu - y * torch.exp(var/2.0 - mu)
        return p.sum()

    def conditional_mean(self, X, f):
        return self.link(f)

    #TODO: implement predict?

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        rate = 1.0/self.link(f)
        return torch.distributions.exponential.Exponential(rate).sample().log()

class LaplaceLikelihood(Likelihood):
    """
    Laplace likelihood given by

    $$ p(y|f) = \\frac{1}{2\\sigma}e^{-\\frac{|y-f|}{\\sigma}} $$

    with \\(\\sigma\\) the scale.

    Args:
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, quadratures=20):
        super().__init__(quadratures)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxQ

    def conditional_mean(self, X, f):
        return f

    #def variational_expectation(self, X, y, mu, var):
    #    # TODO: doesn't seem right
    #    # y,mu,var: Nx1
    #    p = -y/self.scale() + mu/self.scale() - torch.log(2.0*self.scale())
    #    p += torch.sqrt(2.0*var/np.pi)/self.scale()
    #    return p.sum()

    #TODO: implement predict

    def conditional_sample(self, X, f):
        return torch.distributions.laplace.Laplace(f, self.scale()).sample()

class BernoulliLikelihood(Likelihood):
    """
    Bernoulli likelihood given by

    $$ p(y|f) = h(f)^k (1-h(f))^{n-k} $$

    with \\(h\\) the link function, \\(k\\) the number of \\(y\\) values equal to 1, \\(n\\) the number of data points, and \\(y \\in \\{0.0,1.0\\}\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=inv_probit):
        super().__init__()
        self.link = link

    def validate_y(self, X, y):
        if torch.any((y != 0.0) & (y != 1.0)):
            raise ValueError("y must have only 0.0 and 1.0 values")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = self.link(f)
        return torch.log(torch.where(0.5 <= y, p, 1.0-p))  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        return None

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        if self.link != inv_probit:
            return super().predict(X, mu, var, ci=ci, sigma=sigma, n=n)

        p = self.link(mu / torch.sqrt(1.0 + var))
        if ci is None and sigma is None:
            return p
        return p, p, p

class SoftmaxLikelihood(Likelihood):
    """
    Softmax likelihood para clasificación multiclase.

    Para clasificación multiclase con K clases, usamos K funciones latentes f = [f_1, ..., f_K],
    y la probabilidad de la clase k está dada por:

    $$ p(y=k|f) = \\frac{\\exp(f_k)}{\\sum_{j=1}^K \\exp(f_j)} $$

    Args:
        num_classes (int): Número de clases K.
        mc_samples (int): Número de muestras Monte Carlo para aproximar la expectativa variacional.

    Attributes:
        num_classes (int): Número de clases.
        mc_samples (int): Número de muestras Monte Carlo.
    """
    def __init__(self, num_classes, mc_samples=100):
        super().__init__(quadratures=20)
        if num_classes is None or num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = int(num_classes)
        self.mc_samples = int(mc_samples)
        self.latent_dims = self.num_classes
        self.output_dims = num_classes
        self.has_closed_form_ve = False

    def validate_y(self, X, y):
        if y is None:
            return
        y_t = torch.as_tensor(y, dtype=torch.long)
        y_flat = y_t.view(-1)
        if not torch.all((y_flat >= 0) & (y_flat < self.num_classes)):
            raise ValueError(f"y must contain integers in [0, {self.num_classes-1}], got min={y_flat.min()}, max={y_flat.max()}")

    def log_prob(self, X, y, f):
        """
        Calcula log p(y|f) para cada punto de datos.

        Args:
            X: (N,D) - no usado directamente pero mantiene compatibilidad
            y: (N,1) - índices de clase (valores entre 0 y K-1)
            f: (N,Q) donde Q puede ser num_classes (para evaluación directa)
                o cualquier número de muestras de quadratura

        Returns:
            torch.tensor: (N,Q) log probabilidades
        """
        # f puede venir en diferentes formas dependiendo del contexto
        # Para softmax, esperamos que las últimas dimensiones sean las clases
        y_idx = y.view(-1).long()
        N = y_idx.shape[0]

        if f.dim() == 2:
            # f es (N, K) donde K = num_classes
            if f.shape[1] != self.num_classes:
                raise RuntimeError(f"Expected f.shape[1]={self.num_classes}, got {f.shape[1]}")
            logp = F.log_softmax(f, dim=1)  # (N, K)
            # Seleccionar log-prob de la clase observada
            vals = logp[torch.arange(N, device=logp.device), y_idx]  # (N,)
            return vals.view(N, 1)
        else:
            raise RuntimeError(f"SoftmaxLikelihood.log_prob: unexpected f.dim()={f.dim()}")

    def conditional_mean(self, X, f):
        """
        Retorna las probabilidades de clase: p(y=k|f).

        Args:
            X: (N,D) - entrada
            f: (N,K) - valores de funciones latentes

        Returns:
            torch.tensor: (N,K) probabilidades de clase
        """
        return F.softmax(f, dim=-1)

    def conditional_sample(self, X, f):
        """
        Muestrea clases de la distribución categórica.

        Args:
            X: (N,D) - entrada
            f: (n_samples, N, K) o (N,K) - valores de funciones latentes

        Returns:
            torch.tensor: índices de clase muestreados
        """
        if f.dim() == 3:
            # f: (n_samples, N, K)
            n_samples, N, K = f.shape
            probs = F.softmax(f, dim=2)  # (n_samples, N, K)
            samples = torch.zeros(n_samples, N, device=f.device, dtype=torch.long)
            for i in range(n_samples):
                samples[i] = torch.multinomial(probs[i], num_samples=1).view(-1)
            return samples.float()
        else:
            # f: (N,K)
            probs = F.softmax(f, dim=1)
            return torch.multinomial(probs, num_samples=1).view(-1, 1).float()

    def variational_expectation(self, X, y, mu, var):
        """
        Aproximación Monte Carlo de E_q[log p(y|f)] donde q(f) ~ N(mu, diag(var)).

        Args:
            X: (N,D) - entrada
            y: (N,1) - índices de clase observados
            mu: (N,K) - media de la distribución variacional
            var: (N,K) - varianza diagonal de la distribución variacional

        Returns:
            torch.tensor: escalar - suma de expectativas variacional sobre todos los puntos
        """
        mu = torch.as_tensor(mu)
        var = torch.as_tensor(var)
        y_idx = y.view(-1).long()

        N = mu.shape[0]
        K = mu.shape[1] if mu.dim() == 2 else 1

        if K != self.num_classes:
            raise RuntimeError(f"Expected mu with K={self.num_classes} latent functions, got {K}")

        # Muestreo Monte Carlo
        S = self.mc_samples
        device = mu.device
        dtype = mu.dtype

        # Clamping para estabilidad numérica
        var = torch.clamp(var, min=1e-10)
        sqrt_var = torch.sqrt(var)

        # Muestras: (S, N, K)
        eps = torch.randn(S, N, K, device=device, dtype=dtype)
        f_samples = mu.unsqueeze(0) + eps * sqrt_var.unsqueeze(0)

        # Log softmax: (S, N, K)
        logp = F.log_softmax(f_samples, dim=2)

        # Seleccionar log-prob de clases observadas: (S, N)
        idx = torch.arange(N, device=device)
        logp_obs = logp[:, idx, y_idx]

        # Promedio sobre muestras y suma sobre datos
        return logp_obs.mean(dim=0).sum()

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        """
        Predice probabilidades de clase integrando sobre la distribución posterior.

        Args:
            X: (N,D) - puntos de entrada
            mu: (N,K) - media posterior
            var: (N,K) - varianza posterior diagonal
            ci: [lower, upper] - percentiles para intervalo de confianza
            sigma: no usado para softmax
            n: número de muestras Monte Carlo

        Returns:
            Si ci es None: (N,K) - probabilidades de clase
            Si ci especificado: ((N,K), (N,K), (N,K)) - media, lower, upper
        """
        mu = torch.as_tensor(mu)
        var = torch.as_tensor(var)

        if var is None or torch.all(var < 1e-8):
            # Predicción determinística
            p = F.softmax(mu, dim=-1)
            if ci is None:
                return p
            return p, p, p

        # Integración Monte Carlo
        N, K = mu.shape
        device = mu.device
        dtype = mu.dtype

        var = torch.clamp(var, min=1e-10)
        sqrt_var = torch.sqrt(var)

        # Muestras: (n, N, K)
        eps = torch.randn(n, N, K, device=device, dtype=dtype)
        f_samples = mu.unsqueeze(0) + eps * sqrt_var.unsqueeze(0)

        # Probabilidades: (n, N, K)
        probs = F.softmax(f_samples, dim=2)

        # Media
        p_mean = probs.mean(dim=0)

        if ci is None:
            return p_mean

        # Cuantiles para intervalo de confianza
        probs_sorted, _ = torch.sort(probs, dim=0)
        lower_idx = int(ci[0] * n)
        upper_idx = int(ci[1] * n)
        lower = probs_sorted[lower_idx]
        upper = probs_sorted[upper_idx]

        return p_mean, lower, upper

class BetaLikelihood(Likelihood):
    """
    Beta likelihood given by

    $$ p(y|f) = \\frac{\\Gamma(\\sigma)}{\\Gamma\\left(h(f)\\sigma\\right)\\Gamma\\left((1-h(f)\\sigma\\right)} y^{h(f)\\sigma} (1-y)^{(1-h(f))\\sigma} $$

    with \\(h\\) the link function, \\(\\sigma\\) the scale, and \\(y \\in (0.0,1.0)\\).

    Args:
        scale (float): Scale.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, link=inv_probit, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any((y <= 0.0) | (1.0 <= y)):
            raise ValueError("y must be in the range (0.0,1.0)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = self.scale() - alpha

        p = (alpha-1.0)*y.log()
        p += (beta-1.0)*torch.log1p(-y)
        p += torch.lgamma(alpha+beta)
        p -= torch.lgamma(alpha)
        p -= torch.lgamma(beta)
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        if self.link != inv_probit:
            raise ValueError("only inverse probit link function is supported")
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = self.scale() - alpha
        return probit(torch.distributions.beta.Beta(alpha, beta).sample())

class GammaLikelihood(Likelihood):
    """
    Gamma likelihood given by

    $$ p(y|f) = \\frac{1}{\\Gamma(k)h(f)^k} y^{k-1} e^{-y/h(f)} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -y/self.link(f)
        p += (self.shape()-1.0)*y.log()
        p -= torch.lgamma(self.shape())
        if self.link == exp:
            p -= self.shape()*f
        else:
            p -= self.shape()*self.link(f).log()
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        if self.link != exp:
            super().variational_expectation(X, y, mu, var)

        p = -self.shape()*mu
        p -= torch.lgamma(self.shape())
        p += (self.shape() - 1.0) * y.log()
        p -= y * torch.exp(var/2.0 - mu)
        return p.sum()

    def conditional_mean(self, X, f):
        return self.shape()*self.link(f)

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        rate = 1.0/self.link(f)
        return torch.distributions.gamma.Gamma(self.shape(), rate).sample().log()

class PoissonLikelihood(Likelihood):
    """
    Poisson likelihood given by

    $$ p(y|f) = \\frac{1}{y!} h(f)^y e^{-h(f)} $$

    with \\(h\\) the link function and \\(y \\in \\mathbb{N}_0\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link

    def validate_y(self, X, y):
        if torch.any(y < 0.0):
            raise ValueError("y must be in the range [0.0,inf)")
        if not torch.all(y == y.long()):
            raise ValueError("y must have integer count values")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = y*f
        else:
            p = y*self.link(f).log()
        p -= torch.lgamma(y+1.0)
        p -= self.link(f)
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        if self.link != exp:
            super().variational_expectation(X, y, mu, var)

        p = y*mu - torch.exp(var/2.0 + mu) - torch.lgamma(y+1.0)
        return p.sum()

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        rate = self.link(f)
        return torch.distributions.poisson.Poisson(rate).sample().log()

class WeibullLikelihood(Likelihood):
    """
    Weibull likelihood given by

    $$ p(y|f) = \\frac{k}{h(f)} \\left( \\frac{y}{h(f)} \\right)^{k-1} e^{-(y/h(f))^k} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = -self.shape()*f
        else:
            p = -self.shape()*self.link(f).log()
        p += self.shape().log() + (self.shape()-1.0)*y.log()
        p -= (y/self.link(f))**self.shape()
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f) * torch.lgamma(1.0 + 1.0/self.shape()).exp()

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        scale = self.link(f)
        return torch.distributions.weibull.Weibull(scale, self.shape()).sample().log()

class LogLogisticLikelihood(Likelihood):
    """
    Log-logistic likelihood given by

    $$ p(y|f) = \\frac{(k/h(f)) (y/h(f))^{k-1}}{\\left(1 + (y/h(f))^k\\right)^2} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y < 0.0):
            raise ValueError("y must be in the range [0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = -self.shape()*f
        else:
            p = -self.shape()*self.link(f).log()
        p -= 2.0*torch.log1p((y/self.link(f))**self.shape())
        p += self.shape().log()
        p += (self.shape()-1.0)*y.log()
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f) / torch.sinc(1.0/self.shape())

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        return log_logistic_distribution(f, 1.0/self.shape()).sample().log()

class LogGaussianLikelihood(Likelihood):
    """
    Log-Gaussian likelihood given by

    $$ p(y|f) = \\frac{1}{y\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(\\log(y) - f)^2}{2\\sigma^2}} $$

    with \\(\\sigma\\) the scale and \\(y \\in (0.0,\\infty)\\).

    Args:
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, quadratures=20):
        super().__init__(quadratures)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        logy = y.log()
        p = -0.5 * (np.log(2.0*np.pi) + 2.0*self.scale().log() + ((logy-f)/self.scale()).square())
        p -= logy
        return p  # NxQ

    def conditional_mean(self, X, f):
        return torch.exp(f + 0.5*self.scale().square())

    #TODO: implement variational_expectation?
    #TODO: implement predict

    def conditional_sample(self, X, f):
        return torch.distributions.log_normal.LogNormal(f, self.scale()).sample().log()

class ChiSquaredLikelihood(Likelihood):
    """
    Chi-squared likelihood given by

    $$ p(y|f) = \\frac{1}{2^{f/2}\\Gamma(f/2)} y^{f/2-1} e^{-y/2} $$

    with \\(y \\in (0.0,\\infty)\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        f = self.link(f)
        p = -0.5*f*np.log(2.0) - torch.lgamma(f/2.0) + (f/2.0-1.0)*y.log() - 0.5*y
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        return torch.distributions.chi2.Chi2(self.link(f)).sample().log()


class MultiClassMA(Likelihood):
    """
    Multi-Class Multi-Annotator Likelihood

    Modelo de clasificación multiclase con múltiples anotadores de confiabilidad variable.

    Args:
        K (int): Número de clases
        R (int): Número de anotadores
        iAnn (torch.Tensor): Matriz de anotaciones (N, R) donde iAnn[n,m]=1 si
                             el anotador m etiquetó el dato n
        Y (torch.Tensor): Etiquetas observadas (N, R) con valores en {1, ..., K} o 0 si no anotado

    Funciones latentes:
        - f[:,0:K]: funciones de clasificación (softmax)
        - f[:,K:K+R]: funciones de confiabilidad de anotadores (sigmoid)

    Total: K + R funciones latentes
    """

    def __init__(self, K, R, iAnn=None, Y=None):
        super().__init__()

        # Validaciones
        if K < 2:
            raise ValueError(f"K debe ser >= 2, got {K}")
        if R < 1:
            raise ValueError(f"R debe ser >= 1, got {R}")

        self.K = K  # Número de clases
        self.R = R  # Número de anotadores

        # Total de funciones latentes: K para clasificación + R para anotadores
        self.latent_dims = K + R

        # Metadata de anotaciones
        if iAnn is not None:
            if not isinstance(iAnn, torch.Tensor):
                iAnn = torch.tensor(iAnn, dtype=torch.float32)
            self.iAnn = iAnn  # (N, R)
        else:
            self.iAnn = None

        # Etiquetas observadas
        if Y is not None:
            if not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y, dtype=torch.float32)
            self.Y = Y  # (N, R)
        else:
            self.Y = None

        # No tiene forma cerrada - requiere cuadratura
        self.has_closed_form = False

        # Parámetros para cuadratura Gauss-Hermite
        self.gh_degree = 12  # Por defecto, ajustable
        self._gh_points = None  # Se calculan lazy

    def name(self):
        return f"MultiClassMA(K={self.K}, R={self.R})"

    def __repr__(self):
        return f"MultiClassMA(K={self.K}, R={self.R}, latent_dims={self.latent_dims})"

    def validate_y(self, X, y):
        """
        Valida las observaciones.

        En MultiClassMA, las observaciones verdaderas (Y, iAnn) se pasan en el constructor.
        El 'y' que viene de mogptk es solo para validar dimensiones.

        Args:
            X: inputs, shape (N, input_dims) - ya procesado por MultiLatentLikelihood
            y: placeholder, shape (N, 1)
        """
        # 1. Validar shape básico
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(f"y debe tener shape (N, 1), got {y.shape}")

        N = y.shape[0]

        # 2. Verificar que tenemos metadata
        if self.Y is None or self.iAnn is None:
            raise ValueError(
                "MultiClassMA requiere metadata. Usa set_metadata(iAnn, Y) "
                "o pasa iAnn y Y en el constructor."
            )

        # 3. Validar dimensiones de metadata
        if self.Y.shape[0] != N:
            raise ValueError(
                f"Y debe tener {N} filas (igual que y), got {self.Y.shape[0]}"
            )

        if self.iAnn.shape[0] != N:
            raise ValueError(
                f"iAnn debe tener {N} filas (igual que y), got {self.iAnn.shape[0]}"
            )

        if self.Y.shape[1] != self.R or self.iAnn.shape[1] != self.R:
            raise ValueError(
                f"Y e iAnn deben tener {self.R} columnas (R anotadores), "
                f"got Y: {self.Y.shape[1]}, iAnn: {self.iAnn.shape[1]}"
            )

        # 4. Validar rango de etiquetas (solo donde hay anotaciones)
        mask = self.iAnn > 0  # Máscara de anotaciones presentes
        Y_annotated = self.Y[mask]  # Solo etiquetas anotadas

        if Y_annotated.numel() > 0:
            y_min = Y_annotated.min().item()
            y_max = Y_annotated.max().item()

            if y_min < 1 or y_max > self.K:
                raise ValueError(
                    f"Las etiquetas deben estar en {{1, ..., {self.K}}}, "
                    f"got min={y_min}, max={y_max}"
                )

        # 5. Validar consistencia entre Y e iAnn
        # Donde iAnn=0, Y debería ser 0 (o ignorarse)
        inconsistent = (self.iAnn == 0) & (self.Y != 0)
        if inconsistent.any():
            n_inconsistent = inconsistent.sum().item()
            raise ValueError(
                f"Encontradas {n_inconsistent} etiquetas donde iAnn=0 pero Y!=0. "
                "Y debe ser 0 donde no hay anotaciones."
            )

        # 6. Verificar que hay al menos una anotación
        if self.iAnn.sum() == 0:
            raise ValueError("iAnn está vacío - no hay anotaciones")

        # 7. Opcional: advertir si algún dato no tiene anotaciones
        annotations_per_point = self.iAnn.sum(dim=1)
        points_without_annotations = (annotations_per_point == 0).sum().item()

        if points_without_annotations > 0:
            import warnings
            warnings.warn(
                f"{points_without_annotations}/{N} puntos no tienen anotaciones. "
                "El modelo no aprenderá de estos puntos."
            )