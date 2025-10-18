# -*- coding: utf-8 -*-
"""
Option Pricing Models
This script provides implementations for pricing financial options using various models:
1. Black-Scholes Model: For European call and put options.
2. Black '76 Model: For European call and put options on futures contracts.
3. Barone-Adesi and Whaley (1987) Model: A quadratic approximation method
   for pricing American call and put options.
"""

import numpy as np
from scipy.stats import norm

# N(x) is the cumulative distribution function for a standard normal distribution
N = norm.cdf

def black_scholes_call_put(S, K, T, r, sigma, q=0):
    """
    Calculates the price of European call and put options using the Black-Scholes model.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the stock price (annual)
        q (float): Continuous dividend yield (optional, default is 0)


    Returns:
        tuple: A tuple containing the call price and the put price.
    """
    if T <= 0:
        call_price = max(0, S - K)
        put_price = max(0, K - S)
        return call_price, put_price

    b = r - q # Cost of carry
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = (S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2))
    put_price = (K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1))

    return call_price, put_price

def black_76_call_put(F, K, T, r, sigma):
    """
    Calculates the price of European options on futures contracts using the Black '76 model.

    Args:
        F (float): Current futures price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the futures price (annual)

    Returns:
        tuple: A tuple containing the call price and the put price.
    """
    if T <= 0:
        call_price = max(0, F - K)
        put_price = max(0, K - F)
        return call_price, put_price
        
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = np.exp(-r * T) * (F * N(d1) - K * N(d2))
    put_price = np.exp(-r * T) * (K * N(-d2) - F * N(-d1))

    return call_price, put_price

def _critical_price(K, T, r, sigma, q, is_call):
    """ Helper function for Barone-Adesi and Whaley to find the critical stock price. """
    b = r - q
    
    # Define parameters for the quadratic approximation
    if is_call:
        m = 2 * r / sigma**2
        n = 2 * b / sigma**2
        q2 = (-(n - 1) + np.sqrt((n - 1)**2 + 4 * m)) / 2
        
        # Newton-Raphson method to find the critical stock price S*
        s_star = K  # Initial guess
        for _ in range(100): # Iterate to find the root
            bs_price, _ = black_scholes_call_put(s_star, K, T, r, sigma, q)
            lhs = s_star - K
            d1 = (np.log(s_star / K) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
            rhs = bs_price + (1 - np.exp(-q * T) * N(d1)) * s_star / q2
            
            # Derivative for Newton-Raphson
            derivative = (1 - np.exp(-q * T) * N(d1)) * (1 - 1/q2) + np.exp(-q * T) * norm.pdf(d1) / (sigma * np.sqrt(T) * q2)

            if abs(lhs - rhs) < 1e-6 or abs(derivative) < 1e-6:
                break
            
            s_star -= (lhs - rhs) / derivative
        return s_star

    else: # is_put
        m = 2 * r / sigma**2
        n = 2 * b / sigma**2
        q1 = (-(n - 1) - np.sqrt((n - 1)**2 + 4 * m)) / 2

        # Newton-Raphson method to find the critical stock price S*
        s_star = K  # Initial guess
        for _ in range(100): # Iterate to find the root
            _, bs_price = black_scholes_call_put(s_star, K, T, r, sigma, q)
            lhs = K - s_star
            d1 = (np.log(s_star / K) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
            # The early exercise premium A1 is -(s_star/q1) * (1 - exp(-qT)N(-d1))
            rhs = bs_price - (s_star / q1) * (1 - np.exp(-q * T) * N(-d1))
            
            # Corrected derivative for Newton-Raphson
            derivative = -1 - ( (1/q1) * (1 - np.exp(-q*T)*N(-d1)) + np.exp(-q*T)*norm.pdf(-d1)/(sigma*np.sqrt(T)) )

            if abs(lhs - rhs) < 1e-6 or abs(derivative) < 1e-6:
                break
                
            s_star -= (lhs - rhs) / derivative
        return s_star

def barone_adesi_whaley_call(S, K, T, r, sigma, q):
    """
    Calculates the price of an American call option using the Barone-Adesi and Whaley (1987) approximation.
    """
    if T <= 0:
        return max(0, S - K)
        
    b = r - q
    
    if b >= r: # Never optimal to exercise early
        call_price, _ = black_scholes_call_put(S, K, T, r, sigma, q)
        return call_price

    s_star = _critical_price(K, T, r, sigma, q, is_call=True)
    bs_price, _ = black_scholes_call_put(S, K, T, r, sigma, q)
    
    if S >= s_star:
        return S - K
    else:
        m = 2 * r / sigma**2
        n = 2 * b / sigma**2
        q2 = (-(n - 1) + np.sqrt((n - 1)**2 + 4 * m)) / 2
        
        d1 = (np.log(s_star / K) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
        A2 = (s_star / q2) * (1 - np.exp(-q * T) * N(d1))
        
        return bs_price + A2 * (S / s_star)**q2

def barone_adesi_whaley_put(S, K, T, r, sigma, q):
    """
    Calculates the price of an American put option using the Barone-Adesi and Whaley (1987) approximation.
    """
    if T <= 0:
        return max(0, K - S)

    b = r - q
    s_star = _critical_price(K, T, r, sigma, q, is_call=False)
    _, bs_price = black_scholes_call_put(S, K, T, r, sigma, q)
    
    if S <= s_star:
        return K - S
    else:
        m = 2 * r / sigma**2
        n = 2 * b / sigma**2
        q1 = (-(n - 1) - np.sqrt((n - 1)**2 + 4 * m)) / 2
        
        d1 = (np.log(s_star / K) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
        A1 = -(s_star / q1) * (1 - np.exp(-q * T) * N(-d1))
        
        return bs_price + A1 * (S / s_star)**q1


# --- Main execution block for demonstration ---
if __name__ == '__main__':
    print("--- Option Pricing Model Demonstration ---")

    # --- 1. Black-Scholes for European Options ---
    print("\n1. Black-Scholes (European Options)")
    S_bs = 100  # Stock Price
    K_bs = 100  # Strike Price
    T_bs = 1.0  # Time to Maturity (1 year)
    r_bs = 0.05 # Risk-free rate (5%)
    sigma_bs = 0.2 # Volatility (20%)
    
    call_bs, put_bs = black_scholes_call_put(S_bs, K_bs, T_bs, r_bs, sigma_bs)
    print(f"   Parameters: S={S_bs}, K={K_bs}, T={T_bs}, r={r_bs}, sigma={sigma_bs}")
    print(f"   European Call Price: {call_bs:.4f}")
    print(f"   European Put Price:  {put_bs:.4f}")

    # --- 2. Black '76 for European Options on Futures ---
    print("\n2. Black '76 (European Options on Futures)")
    F_b76 = 100  # Futures Price
    K_b76 = 100  # Strike Price
    T_b76 = 1.0  # Time to Maturity (1 year)
    r_b76 = 0.05 # Risk-free rate (5%)
    sigma_b76 = 0.2 # Volatility (20%)

    call_b76, put_b76 = black_76_call_put(F_b76, K_b76, T_b76, r_b76, sigma_b76)
    print(f"   Parameters: F={F_b76}, K={K_b76}, T={T_b76}, r={r_b76}, sigma={sigma_b76}")
    print(f"   Futures Call Price: {call_b76:.4f}")
    print(f"   Futures Put Price:  {put_b76:.4f}")

    # --- 3. Barone-Adesi and Whaley for American Options ---
    print("\n3. Barone-Adesi and Whaley (American Options Approximation)")
    print("   (Note: There is no simple closed-form solution like Black-Scholes for American options.)")
    S_am = 100   # Stock Price
    K_am = 100   # Strike Price
    T_am = 1.0   # Time to Maturity (1 year)
    r_am = 0.05  # Risk-free rate (5%)
    sigma_am = 0.2 # Volatility (20%)
    q_am = 0.03  # Dividend yield (3%)

    call_am = barone_adesi_whaley_call(S_am, K_am, T_am, r_am, sigma_am, q_am)
    put_am = barone_adesi_whaley_put(S_am, K_am, T_am, r_am, sigma_am, q_am)
    print(f"   Parameters: S={S_am}, K={K_am}, T={T_am}, r={r_am}, sigma={sigma_am}, q={q_am}")
    print(f"   American Call Price: {call_am:.4f}")
    print(f"   American Put Price:  {put_am:.4f}")
    
    # Comparison: American vs European
    # Merton's model adjusts Black-Scholes for dividends by using (r-q) as the effective risk-free rate.
    call_eur_comp, put_eur_comp = black_scholes_call_put(S_am, K_am, T_am, r_am, sigma_am, q_am)
    print("\n   Comparison with equivalent European option (using Merton's dividend-adjusted model):")
    print(f"   European Call Price: {call_eur_comp:.4f} (American call value is higher due to early exercise premium on dividends)")
    print(f"   European Put Price:  {put_eur_comp:.4f} (American put value is higher due to early exercise premium)")



