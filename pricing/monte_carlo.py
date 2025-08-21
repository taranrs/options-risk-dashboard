import numpy as np

def mc_price(S, K, T, r, sigma, q=0.0, option="call", n_paths=20000, steps=252):
    dt = T/steps
    Z = np.random.normal(size=(n_paths, steps))
    drift = (r - q - 0.5*sigma**2)*dt
    vol   = sigma*np.sqrt(dt)
    S_paths = S * np.exp(np.cumsum(drift + vol*Z, axis=1))
    ST = S_paths[:, -1]
    payoff = np.maximum(ST - K, 0.0) if option=="call" else np.maximum(K - ST, 0.0)
    return np.exp(-r*T) * payoff.mean()
