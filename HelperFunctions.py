def GeometricSeries(p0, r, t):
    # Inputs:
    # p0    - Initial Amount
    # r     - growth or decay rate +/-(%)
    # t     - time for growth

    # Outputs: 
    # p     - Amount at time t
    # st    - Sum over time frame t  

    # Example: GeometricSeries(10,-5,2)  
    # Principal amount of 10 with 5 % decay for 2 time periods

    theta   = 1 + (r/100)                       # Common Ratio
    p       = p0*theta**t                       # Geometric series value at time
    st      = p0*(1 - theta**(t+1))/(1 - theta) # Sum of Geometric Series

    return p, st

