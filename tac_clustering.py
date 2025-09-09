import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation, SpectralClustering
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
max_iter = 1000
tol = 1e-10
num_repl = 5

# X is always a p x n matrix
# k is always the number of components
# output labels or probabilities should be k x p
# output means should be k x n
# output covariances should be k x n

def kmeans_plusplus(X, k, dist='euclidean'):
    p, n = X.shape
    # choose first centroid at random from X
    cluster_idx = np.random.choice(p,p=None)
    all_cluster_idx = np.zeros(k,dtype=int)
    all_cluster_idx[0] = cluster_idx

    C = np.zeros((k,n))
    C[0] = X[cluster_idx]

    for i in range(k):
        if dist=='euclidean':
            distance = np.linalg.norm(X[:,:,None] - C.T[None,:,:], axis=1)**2
        elif dist=='crosscorr':
            X_f = np.fft.rfft(X, axis=-1)
            X_f[:,0] = 0 #remove the DC component
            C_f = np.fft.rfft(C, axis=-1)
            C_f[:,0] = 0
            X_sqnorm = np.linalg.norm(X-np.mean(X,-1,keepdims=True), axis=-1)**2
            cross_corr = np.fft.irfft(np.conj(X_f[:,None,:]) * C_f[None,:,:], axis=-1)
            normalized_cross_corr = cross_corr / np.sqrt(X_sqnorm[:,None,None] * np.linalg.norm(C-np.mean(C,-1,keepdims=True), axis=1)[None,:,None]**2)
            max_normalized_cross_corr = np.max(normalized_cross_corr, axis=-1)
            distance = 1 - max_normalized_cross_corr
        else:
            raise ValueError('Invalid distance metric')
        distance = np.clip(distance,0,None) # (p,k)

        mindis = np.nanmin(distance,axis=1) #choose the distance to the closest centroid for each point

        if i==k-1:
            X_part = np.nanargmin(distance,axis=1)
            obj = np.mean(mindis)
            break
        
        prob_dist = mindis/np.sum(mindis) # construct the prob. distribution
        cluster_idx = np.random.choice(p,p=prob_dist) #assume existing idx can never be chosen because p=0

        C[i+1] = X[cluster_idx]

    return np.eye(k)[X_part], C
        
def ashburner1996(X, k, max_iter=max_iter, tol=tol, num_repl=num_repl):
    p, n = X.shape
    B = np.sum(X, axis=1)  # Compute the integral image B
    
    best_log_likelihood = -np.inf
    best_P = np.zeros((p, k))
    best_mu = np.zeros((n, k))
    best_variance = np.zeros((n, k))
    best_obj = np.inf
    
    for repl in range(num_repl):
        # P,_,_ = kmeans_plusplus(X, k)
        # P = P.T
        # P = np.random.rand(p, k)  # Initialize belonging probabilities randomly
        # P /= np.sum(P, axis=1, keepdims=True)  # Normalize P to sum to 1 per row
        P,_ = kmeans_plusplus(X, k)
        # P = U.T
        prev_log_likelihood = -np.inf
        
        for it in range(max_iter):
            print('Ashburner 1996, replication: ', str(repl),' Iteration: ', str(it),' loglik: ',str(prev_log_likelihood), end='\r')
            G = np.sum(P, axis=0)  # Compute the number of pixel vectors per partition
            
            # Compute the mean pixel vector for each partition
            numerator = np.sum(X[:, :, None] * P[:,None,:] * B[:, None,None], axis=0) # (n, k)
            denominator = np.sum(P * B[:, None]**2, axis=0) # (k,)
            mu = numerator / denominator[None] # (n, k)
            
            # Compute variance for each frame
            variance = 1/p * np.sum((X[:,:,None]-mu[None,:,:]*B[:,None,None])**2 * P[:,None,:], axis=0) # (n,k)

            # compute probability densities
            log_Q = np.log(G)-0.5*np.sum(np.log(variance), axis=0) - 0.5 * np.sum((X[:,:,None]-mu[None,:,:]*B[:,None,None])**2 /variance[None,:,:], axis=1) # (p, k)

            # Compute belonging probabilities
            # P = np.exp(log_Q) / np.sum(np.exp(log_Q), axis=1, keepdims=True)
            logsum_density = np.logaddexp.reduce(log_Q,axis=1) #sum over the K components
            P = np.exp(log_Q-logsum_density[:,None])
            
            if np.any(np.isnan(P)) or np.any(np.isclose(np.sum(P,0),0)):
                print('Nan or zero sum in P, breaking. Try increasing num_repl')
                log_likelihood = -np.inf
                # ashburner1996(X, k, max_iter, tol, num_repl)
                break
            # Compute log likelihood
            log_likelihood = np.sum(logsum_density) #sum over the N samples
            # log_likelihood = np.sum(np.log(np.sum(np.exp(log_Q), axis=1)))
            
            # Check for convergence
            if log_likelihood - prev_log_likelihood < tol:
                break
            prev_log_likelihood = log_likelihood
        
        # Keep the best result
        if log_likelihood > best_log_likelihood:
            best_obj = 1/2*np.linalg.norm(X - P@mu.T)**2
            best_log_likelihood = log_likelihood
            best_P = P.copy()
            best_mu = mu.copy()
            best_variance = variance.copy()
    
    return best_P, best_mu.T, best_variance.T, best_obj

def gmm_spherical(X, k, max_iter=max_iter, tol=tol, num_repl=num_repl):
    gmm = GaussianMixture(n_components=k, covariance_type='spherical', max_iter=max_iter, tol=tol, n_init=num_repl)
    gmm.fit(X)
    ls_loss = 1/2*np.sum((X - gmm.means_[gmm.predict(X)])**2)
    return gmm.predict_proba(X), gmm.means_, gmm.covariances_, ls_loss

def gmm_diagonal(X, k, max_iter=max_iter, tol=tol, num_repl=num_repl):
    gmm = GaussianMixture(n_components=k, covariance_type='diag', max_iter=max_iter, tol=tol, n_init=num_repl)
    gmm.fit(X)
    ls_loss = 1/2*np.sum((X - gmm.means_[gmm.predict(X)])**2)
    return gmm.predict_proba(X), gmm.means_, gmm.covariances_, ls_loss

def kmeans(X, k, max_iter=max_iter, tol=tol, num_repl=num_repl):
    p, n = X.shape
    best_labels = None
    best_centers = None
    best_obj = np.inf
    
    for repl in range(num_repl):
        _,centers = kmeans_plusplus(X, k)
        
        obj = np.inf

        for it in range(max_iter):
            print('K-means, replication: ', str(repl),' Iteration: ', str(it),' obj: ',str(obj), end='\r')
            # Compute Mahalanobis distances and assign clusters

            diff = X[:, None, :] - centers[None, :, :]  # Shape (p, k, n)
            distances = np.sum(diff**2, axis=2)  # Shape (p, k)
            
            # Compute final inertia
            new_obj = 1/2*sum(np.min(distances, axis=1))

            # Check for convergence
            if obj-new_obj < tol:
                break
            labels = np.argmin(distances, axis=1)
            
            # Compute new cluster centers
            centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])

            obj = new_obj
        
        # Keep the best clustering solution
        if obj < best_obj:
            best_obj = obj
            best_labels = labels
            best_centers = centers
    
    return np.eye(k)[best_labels], best_centers, best_obj

def fuzzy_cmeans(X, k, m=2.0, max_iter=max_iter, tol=tol, num_repl=num_repl):
    p, n = X.shape  # Consider X as p different n-dimensional vectors
    best_U, best_centers, best_obj = None, None, np.inf
    
    for repl in range(num_repl):
        # Initialize membership matrix randomly and normalize
        # U = np.random.rand(k, p)
        # U /= np.sum(U, axis=0, keepdims=True)
        U,_ = kmeans_plusplus(X, k)

        # Initialize cluster centers randomly
        centers = np.random.rand(n,k)

        obj = np.inf
        
        for it in range(max_iter):
            print('Fuzzy C-means, replication: ', str(repl),' Iteration: ', str(it),' obj: ',str(obj), end='\r')

            distance = np.linalg.norm(X[:, :, None] - centers[None, :, :], axis=1)**2
            distance = np.fmax(distance, np.finfo(np.float64).eps)

            U = (1/distance)**(1/(m-1))/np.sum((1/distance)**(1/(m-1)),axis=1)[:,None]

            centers = ((U.T**m @ X) / np.sum(U**m, axis=0)[:,None]).T

            obj_new = 1/2*np.sum(U**m*np.linalg.norm(X[:, :, None] - centers[None, :, :], axis=1)**2)
            # new_obj = 1/2*sum(np.min(distance, axis=1))
            # Check for convergence
            if obj-obj_new < tol:
                # obj = new_obj
                break
            obj = obj_new
        
        if obj < best_obj:
            best_U, best_centers, best_obj = U, centers, obj
    
    return best_U, best_centers.T, best_obj  # Return the best membership matrix and cluster centers

def kmeans_mahalanobis(X, k, W, max_iter=max_iter, tol=tol, num_repl=num_repl):
    p, n = X.shape
    best_labels = None
    best_centers = None
    best_obj = np.inf
    
    for repl in range(num_repl):
        # Initialize cluster centers randomly
        # indices = np.random.choice(p, k, replace=False)
        # centers = X[indices]
        _,centers = kmeans_plusplus(X, k)
        
        # Compute the inverse covariance matrix using provided weights
        inv_W = np.diag(1/W)
        
        obj = np.inf

        for it in range(max_iter):
            print('K-means (mahalanobis), replication: ', str(repl),' Iteration: ', str(it),' obj: ',str(obj), end='\r')
            # Compute Mahalanobis distances and assign clusters

            diff = X[:, None, :] - centers[None, :, :]  # Shape (p, k, n)
            distances = np.sum(diff @ inv_W * diff, axis=2)  # Shape (p, k)
            
            # Compute final inertia (sum of squared Mahalanobis distances)
            new_obj = 1/2*sum(np.min(distances**2, axis=1))

            # Check for convergence
            if obj-new_obj < tol:
                break
            labels = np.argmin(distances, axis=1)
            
            # Compute new cluster centers
            centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])

            obj = new_obj
        
        # Keep the best clustering solution
        if obj < best_obj:
            best_obj = obj
            best_labels = labels
            best_centers = centers
    
    return np.eye(k)[best_labels], best_centers, best_obj

def affinity_propagation(X, max_iter=max_iter, tol=tol, num_repl=num_repl):
    best_labels = None
    best_centers = None
    best_obj = np.inf
    
    for _ in range(num_repl):
        af = AffinityPropagation(max_iter=max_iter)
        af.fit(X)
        labels = af.labels_
        centers = af.cluster_centers_
        obj = af.inertia_
        
        if obj < best_obj:
            best_obj = obj
            best_labels = labels
            best_centers = centers
    
    return best_labels, best_centers, best_obj

def spectral_clustering(X, k, max_iter=max_iter, tol=tol, num_repl=num_repl):
    print('Spectral clustering')
    sc = SpectralClustering(n_clusters=k, n_init=num_repl, affinity='rbf', n_neighbors=10, verbose=False)
    fit = sc.fit(X)
    labels = fit.labels_
    centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else X[j] for j in range(k)])
    obj = 1/2*np.linalg.norm((X - np.eye(k)[labels]@centers))**2
    
    return np.eye(k)[labels], centers, obj

def roll(a, shifts, axis):
    assert a.shape[axis] == len(shifts)
    return np.stack([
        np.roll(np.take(a, i, axis), shifts[i]) for i in range(len(shifts))
    ], axis)

def kshape(X, k, max_iter=max_iter, tol=tol, num_repl=num_repl):
    p, n = X.shape
    best_labels = None
    best_centers = None
    best_obj = np.inf
    X_f = np.fft.rfft(X, axis=-1)
    # X_f[:,[0,-1]] /= np.sqrt(2)
    # X_f2 = X_f.copy()
    # X_f2[:,0] = 0
    X_sqnorm = np.linalg.norm(X-np.mean(X,-1,keepdims=True), axis=-1)**2

    f = (-1j*2*np.pi*np.arange(n)/n)
    f = f[:n//2+1]
    
    for repl in range(num_repl):
        # Initialize cluster centers randomly
        _, centers = kmeans_plusplus(X, k, dist='crosscorr') #size (k,n)
        centers_f = np.fft.rfft(centers, axis=-1) # (k,n)
        
        # Compute the inverse covariance matrix using provided weights
        obj = np.inf

        for it in range(max_iter):
            print('K-means (crosscorr), replication: ', str(repl),' Iteration: ', str(it),' obj: ',str(obj), end='\r')

            # conj on X means that -N+1 is a shift to the right. 
            # conj on centers means that -N+1 is a shift to the left.
            # centers_f = np.fft.rfft(centers-np.mean(centers,-1, keepdims=True), axis=-1) # (k,n)
            modified_centers_f = centers_f.copy()
            modified_centers_f[:,0] = 0#modified_centers_f[:,0]/np.sqrt(2)
            modified_centers_f[:,-1] = modified_centers_f[:,-1]/np.sqrt(2)
            centers_f_sqnorm = 2/n*np.sum(np.abs(modified_centers_f)**2, axis=-1)

            # cross_spec = np.conj(X_f[:,None,:]) * modified_centers_f[None,:,:]
            cross_spec = np.conj(X_f[:,None,:]) * centers_f[None,:,:]
            cross_spec[:,:,0] = 0
            cross_cov = np.fft.irfft(cross_spec, axis=-1, n=n) # (p,k,n)
            # normalized_cross_corr = cross_corr / np.sqrt(X_sqnorm[:,None,None] * np.linalg.norm(centers-np.mean(centers,-1, keepdims=True), axis=1)[None,:,None]**2)
            normalized_cross_cov = cross_cov / np.sqrt(X_sqnorm[:,None,None] * centers_f_sqnorm[None,:,None])
            max_normalized_cross_cov = np.max(normalized_cross_cov, axis=-1) # (p,k)
            distances = np.clip(1 - max_normalized_cross_cov, 0, None)
            delay = np.argmax(normalized_cross_cov, axis=-1) # (p,k)
            delay = delay - X.shape[-1]

            # so-called "shape-based distance", which is non-sense since time-series still need to be normalized...
            new_obj = np.sum(np.min(distances, axis=1))
            labels = np.argmin(distances, axis=1)

            # Check for convergence
            if obj-new_obj < tol:
                print('K-means (crosscorr), replication: ', str(repl),' Iteration: ', str(it),' obj: ',str(obj))
                break
            
            # Compute new cluster centers
            for j in range(k):
                #shift the time series assigned to cluster j by the delay
                # X_shifted = roll(X[labels==j], delay[labels==j,j], axis=0)
                # centers[j] = X_shifted.mean(axis=0)
                
                # centers[j] = np.fft.irfft(np.mean(X_f[labels==j]*np.exp(f[None] * delay[labels==j,j][:,None]), axis=0), axis=-1)
                centers_f[j] = np.mean(X_f[labels==j]*np.exp(f[None] * delay[labels==j,j][:,None]), axis=0)

            obj = new_obj
        
        
        # Keep the best clustering solution
        if obj < best_obj:
            centers = np.fft.irfft(centers_f, axis=-1, n=n)
            best_obj = obj
            best_labels = labels
            #shift centers according to the delay to the time-series with the highest cross-correlation value
            for j in range(k):
                cross_spec = np.conj(X_f[labels==j,:]) * centers_f[j][None,:]
                cross_spec[:,0] = 0
                cross_cov = np.fft.irfft(cross_spec, axis=-1, n=n) # (p,n)
                normalized_cross_cov = cross_cov / np.sqrt(X_sqnorm[labels==j] * np.linalg.norm(centers[j])**2)[:,None] # (p,n)
                max_normalized_cross_cov = np.max(normalized_cross_cov, axis=-1) # (p,)
                voxel_with_max_cross_cov = np.argmax(max_normalized_cross_cov)
                delay_final = np.argmax(normalized_cross_cov[voxel_with_max_cross_cov], axis=-1) - X.shape[-1]
                centers[j] = np.roll(centers[j], -delay_final) #minus because the delay is applied to the centers
            best_centers = centers

            # compute least squares loss like the other models
            A = np.eye(k)[labels]
            A_f = A[:,None,:]*np.exp(f[None,:,None] * -delay[:,None,:])
            diff = X_f - np.sum(A_f*centers_f.T[None],-1) # (P, N)
            diff[:,[0,-1]] /= np.sqrt(2)
            ls_loss = 1/n*np.linalg.norm(diff)**2
    
    return np.eye(k)[best_labels], best_centers, ls_loss