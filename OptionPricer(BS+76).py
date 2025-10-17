import numpy as np
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Model 1: Black-Scholes Model (1973) for European Options
# -----------------------------------------------------------------------------
def black_scholes_call_put(S, K, T, r, sigma, q=0):
    """
    Calculates the price of European call and put options using the Black-Scholes model.

    Args:
        S (float): Current stock price.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying stock (annual).
        q (float, optional): Continuous dividend yield. Defaults to 0.

    Returns:
        tuple: A tuple containing the call price and the put price.
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate call and put prices
    # N(d1) and N(d2) are the cumulative distribution functions (CDF) for a standard normal distribution.
    call_price = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))

    return call_price, put_price

# -----------------------------------------------------------------------------
# Model 2: Black '76 Model for European Options on Futures
# -----------------------------------------------------------------------------
def black_76_call_put(F, K, T, r, sigma):
    """
    Calculates the price of European call and put options on futures using the Black '76 model.

    Args:
        F (float): Current futures price.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying future (annual).

    Returns:
        tuple: A tuple containing the call price and the put price.
    """
    # Calculate d1 and d2 for the Black '76 model
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate call and put prices
    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    return call_price, put_price

# -----------------------------------------------------------------------------
# --- Main Execution Block ---
# This block runs when the script is executed directly.
# It demonstrates how to use the functions above with sample data.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. Black-Scholes (European Options) ---
    print("1. Black-Scholes (European Options)")
    S_eur = 100.0  # Current stock price
    K_eur = 100.0  # Strike price
    T_eur = 1.0    # Time to maturity in years
    r_eur = 0.05   # Annual risk-free rate (e.g., 5%)
    sigma_eur = 0.20 # Annual volatility (e.g., 20%)
    q_eur = 0.03 # <<< This is the dividend yield that adjusts the price

    # Calculate prices with dividend by passing q=q_eur into the function
    call_eur, put_eur = black_scholes_call_put(S_eur, K_eur, T_eur, r_eur, sigma_eur, q=q_eur)
    print(f"   Parameters (with dividend): S={S_eur}, K={K_eur}, T={T_eur}, r={r_eur}, sigma={sigma_eur}, q={q_eur}")
    print(f"   European Call Price (with dividend): {call_eur:.4f}")
    print(f"   European Put Price (with dividend):  {put_eur:.4f}\n")

    # --- 2. Black '76 (European Options on Futures) ---
    print("2. Black '76 (European Options on Futures)")
    F_76 = 100.0   # Current futures price
    K_76 = 100.0   # Strike price
    T_76 = 1.0     # Time to maturity in years
    r_76 = 0.05    # Annual risk-free rate
    sigma_76 = 0.20 # Annual volatility
    
    call_76, put_76 = black_76_call_put(F_76, K_76, T_76, r_76, sigma_76)
    print(f"   Parameters: F={F_76}, K={K_76}, T={T_76}, r={r_76}, sigma={sigma_76}")
    print(f"   European Call on Future Price: {call_76:.4f}")
    print(f"   European Put on Future Price:  {put_76:.4f}\n")
