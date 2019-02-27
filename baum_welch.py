import numpy as np
from scipy.stats import multivariate_normal


def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N,T,M])  # [N,T,M]
    alpha_sum = np.zeros([N,T])  # [N,T], normalizer for alpha
    beta = np.zeros([N,T,M])  # [N,T,M]
    gamma = np.zeros([N,T,M])  # [N,T,M]
    xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]

    # Forward messages
    for n in range(N):
        for m in range(M):
            alpha[n,0,m] = pi[m] * multivariate_normal.pdf(X[n,0,:], mu[m], sigma2[m] * np.identity(K))
        alpha_sum[n,0] = np.sum(alpha[n,0,:], axis=-1)
        alpha[n,0,:] /= alpha_sum[n,0,np.newaxis]

        for t in range(T-1):
            for m in range(M):
                alpha[n,t+1,m] = multivariate_normal.pdf(X[n,t+1,:], mu[m], sigma2[m] * np.identity(K)) * alpha[n,t,:].dot(A[:,m])
            alpha_sum[n,t+1] = np.sum(alpha[n,t+1,:], axis=-1)
            alpha[n,t+1,:] /= alpha_sum[n,t+1,np.newaxis]

    # Backward messages
    for n in range(N):
        beta[n,T-1,:] = 1
        
        for t in range(T-1)[::-1]:
            prob = np.zeros(M)
            for m in range(M):
                prob[m] = multivariate_normal.pdf(X[n,t+1,:], mu[m], sigma2[m] * np.identity(K))
            for m in range(M):
                beta[n,t,m] = np.sum(A[m,:] * beta[n,t+1,:] * prob[:])
            beta[n,t,:] /= alpha_sum[n,t+1,np.newaxis]

    # Sufficient statistics
    gamma = (alpha * beta) / np.sum(alpha * beta, axis=-1, keepdims=True)
    for n in range(N):
        for t in range(T-1):
            prob = np.zeros(M)
            for m in range(M):
                prob[m] = multivariate_normal.pdf(X[n,t+1,:], mu[m], sigma2[m] * np.identity(K))
            for m in range(M):
                xi[n,t,m,:] = alpha[n,t,m] * beta[n,t+1,:] * prob * A[m,:]
            xi[n,t,:,:] /= np.sum(xi[n,t,:,:])

    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    """M-step: MLE"""
    pi = np.mean(gamma[:,0,:], axis=0)

    p_ij = np.einsum('ntij->ij', xi)
    p_i = np.einsum('nti->i', gamma[:,:-1,:])
    A = np.einsum('ij,i->ij', p_ij,1.0/p_i)

    mu = np.einsum('ntk, ntm->mk', X, gamma) / np.einsum('ntm->m', gamma)[:,np.newaxis]

    M = gamma.shape[-1]
    K = X.shape[-1]
    sigma2 = np.zeros(M)
    for m in range(M):
        t = np.einsum('ntk,ntk->nt',X-mu[m],X-mu[m]) / K
        sigma2[m] = np.einsum('nt, nt->', t, gamma[:,:,m]) / np.einsum('nt->', gamma[:,:,m])

    return pi, A, mu, sigma2


def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        _, alpha_sum, _, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T] samples."""
    M, K = mu.shape
    Y = np.zeros([N,T], dtype=int) 
    X = np.zeros([N,T,K], dtype=float)
    for n in range(N):
        Y[n,0] = np.random.choice(M, p=pi)  # [1,]
        X[n,0,:] = multivariate_normal.rvs(mu[Y[n,0],:], sigma2[Y[n,0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n,t+1] = np.random.choice(M, p=A[Y[n,t],:])  # [1,]
            X[n,t+1,:] = multivariate_normal.rvs(mu[Y[n,t+1],:], sigma2[Y[n,t+1]] * np.eye(K))  # [K,]
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    N, T, M, K = 10, 100, 4, 2
    pi = np.array([.0, .0, .0, 1.])  # [M,]
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    A_init = np.random.rand(M, M)
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    sigma2_init = np.ones(M)
    pi, A, mu, sigma2 = hmm_train(X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()

