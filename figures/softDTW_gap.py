# SoftDTW with gap penalty
import numpy as np
from tslearn.backend import instantiate_backend
from tslearn.metrics import SquaredEuclidean
import matplotlib.pyplot as plt
from itertools import product

def soft_dtw_with_gaps(ts1, ts2, gamma=1.0, gap_penalty=1.0, be=None):
    """
    Modified SoftDTW with insertion and deletion operations.
    
    Parameters
    ----------
    ts1 : array-like
        First time series
    ts2 : array-like
        Second time series
    gamma : float
        Softmin parameter
    gap_penalty : float
        Penalty for insertion/deletion operations
    be : Backend object
        Computation backend
    """
    be = instantiate_backend(be, ts1, ts2)
    ts1 = be.array(ts1)
    ts2 = be.array(ts2)
    
    # Compute distance matrix
    D = SquaredEuclidean(ts1, ts2, be=be).compute()
    m, n = be.shape(D)
    
    # Initialize matrices
    R = be.zeros((m + 1, n + 1))  # Cost matrix
    P = be.zeros((m + 1, n + 1))  # Path matrix (1: match, 2: deletion, 3: insertion)
    
    # Initialize first row and column with gap penalties
    for i in range(1, m + 1):
        R[i, 0] = i * gap_penalty
        P[i, 0] = 2  # Deletion
    for j in range(1, n + 1):
        R[0, j] = j * gap_penalty
        P[0, j] = 3  # Insertion
    
    # Dynamic programming recursion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Compute costs for each operation
            match_cost = R[i-1, j-1] + D[i-1, j-1]  # Changed: removed negative sign
            delete_cost = R[i-1, j] + gap_penalty
            insert_cost = R[i, j-1] + gap_penalty
            
            # Soft minimum of the three operations
            costs = be.array([match_cost, delete_cost, insert_cost])
            if hasattr(be, 'logsumexp'):
                R[i, j] = gamma * be.logsumexp(-costs/gamma)  # Changed: removed negative sign
            else:
                R[i, j] = gamma * np.log(np.sum(np.exp(-costs/gamma)))
            
            # Store the operation that led to the minimum
            if hasattr(be, 'argmin'):
                P[i, j] = be.argmin(costs) + 1
            else:
                P[i, j] = np.argmin(costs) + 1
    
    # Backtrack to find alignment
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if P[i, j] == 1:  # Match
            alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif P[i, j] == 2:  # Deletion
            alignment.append((i-1, None))
            i -= 1
        else:  # Insertion
            alignment.append((None, j-1))
            j -= 1
    
    return R[m, n], alignment[::-1]  # Return cost and alignment path

# Print the alignment
def format_alignment(ts1, ts2, alignment):
    """
    Format the alignment as a string showing matches, insertions, and deletions.
    
    Parameters
    ----------
    ts1 : array-like
        First time series
    ts2 : array-like
        Second time series
    alignment : list of tuples
        List of (i, j) pairs from soft_dtw_with_gaps
        
    Returns
    -------
    str
        Formatted alignment string
    """
    # Convert time series to lists if they aren't already
    ts1 = list(ts1)
    ts2 = list(ts2)
    
    # Initialize strings for each line
    top_line = []      # First sequence
    middle_line = []   # Match/Insertion/Deletion indicators
    bottom_line = []   # Second sequence
    
    # Process each alignment step
    for i, j in alignment:
        if i is not None and j is not None:  # Match
            top_line.append(str(ts1[i]))
            middle_line.append('|')  # Vertical bar for match
            bottom_line.append(str(ts2[j]))
        elif i is not None:  # Deletion
            top_line.append(str(ts1[i]))
            middle_line.append(' ')  # Space for deletion
            bottom_line.append('-')  # Gap in second sequence
        else:  # Insertion
            top_line.append('-')     # Gap in first sequence
            middle_line.append(' ')  # Space for insertion
            bottom_line.append(str(ts2[j]))
    
    # Join the lines and add some spacing for readability
    return '\n'.join([
        ' '.join(top_line),
        ' '.join(middle_line),
        ' '.join(bottom_line)
    ])

def analyze_parameters(ts1, ts2, gap_penalties, gammas):
    """
    Analyze the effect of different gap penalties and gamma values on alignment.
    
    Parameters
    ----------
    ts1 : array-like
        First time series
    ts2 : array-like
        Second time series
    gap_penalties : array-like
        List of gap penalty values to test
    gammas : array-like
        List of gamma values to test
        
    Returns
    -------
    dict
        Dictionary containing results for each parameter combination
    """
    results = {}
    
    for gap_penalty, gamma in product(gap_penalties, gammas):
        cost, alignment = soft_dtw_with_gaps(ts1, ts2, gamma=gamma, gap_penalty=gap_penalty)
        
        # Count matches, insertions, and deletions
        matches = sum(1 for i, j in alignment if i is not None and j is not None)
        insertions = sum(1 for i, j in alignment if i is None)
        deletions = sum(1 for i, j in alignment if j is None)
        
        results[(gap_penalty, gamma)] = {
            'cost': cost,
            'matches': matches,
            'insertions': insertions,
            'deletions': deletions,
            'alignment': alignment
        }
    
    return results

def plot_parameter_effects(results, gap_penalties, gammas):
    """
    Create plots showing the effect of parameters on alignment.
    
    Parameters
    ----------
    results : dict
        Results from analyze_parameters
    gap_penalties : array-like
        List of gap penalty values tested
    gammas : array-like
        List of gamma values tested
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total cost vs parameters
    costs = np.array([[results[(gp, g)]['cost'] for g in gammas] for gp in gap_penalties])
    im1 = axes[0, 0].imshow(costs, aspect='auto', origin='lower')
    axes[0, 0].set_title('Total Cost')
    axes[0, 0].set_xlabel('Gamma')
    axes[0, 0].set_ylabel('Gap Penalty')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Number of matches vs parameters
    matches = np.array([[results[(gp, g)]['matches'] for g in gammas] for gp in gap_penalties])
    im2 = axes[0, 1].imshow(matches, aspect='auto', origin='lower')
    axes[0, 1].set_title('Number of Matches')
    axes[0, 1].set_xlabel('Gamma')
    axes[0, 1].set_ylabel('Gap Penalty')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Number of insertions vs parameters
    insertions = np.array([[results[(gp, g)]['insertions'] for g in gammas] for gp in gap_penalties])
    im3 = axes[1, 0].imshow(insertions, aspect='auto', origin='lower')
    axes[1, 0].set_title('Number of Insertions')
    axes[1, 0].set_xlabel('Gamma')
    axes[1, 0].set_ylabel('Gap Penalty')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot 4: Number of deletions vs parameters
    deletions = np.array([[results[(gp, g)]['deletions'] for g in gammas] for gp in gap_penalties])
    im4 = axes[1, 1].imshow(deletions, aspect='auto', origin='lower')
    axes[1, 1].set_title('Number of Deletions')
    axes[1, 1].set_xlabel('Gamma')
    axes[1, 1].set_ylabel('Gap Penalty')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Set tick labels
    for ax in axes.flat:
        ax.set_xticks(range(len(gammas)))
        ax.set_yticks(range(len(gap_penalties)))
        ax.set_xticklabels([f'{g:.2f}' for g in gammas])
        ax.set_yticklabels([f'{gp:.2f}' for gp in gap_penalties])
    
    plt.tight_layout()
    return fig

def print_parameter_summary(results, gap_penalties, gammas):
    """
    Print a summary of the parameter effects.
    
    Parameters
    ----------
    results : dict
        Results from analyze_parameters
    gap_penalties : array-like
        List of gap penalty values tested
    gammas : array-like
        List of gamma values tested
    """
    print("Parameter Effect Summary:")
    print("-" * 50)
    
    # Find best parameters for different criteria
    best_cost = min(results.items(), key=lambda x: x[1]['cost'])
    best_matches = max(results.items(), key=lambda x: x[1]['matches'])
    
    print(f"Best cost ({best_cost[1]['cost']:.2f}) achieved with:")
    print(f"  Gap penalty: {best_cost[0][0]:.2f}")
    print(f"  Gamma: {best_cost[0][1]:.2f}")
    
    print(f"\nMost matches ({best_matches[1]['matches']}) achieved with:")
    print(f"  Gap penalty: {best_matches[0][0]:.2f}")
    print(f"  Gamma: {best_matches[0][1]:.2f}")
    
    print("\nDetailed results for selected parameter combinations:")
    print("-" * 50)
    for gap_penalty in [min(gap_penalties), max(gap_penalties)]:
        for gamma in [min(gammas), max(gammas)]:
            result = results[(gap_penalty, gamma)]
            print(f"\nGap penalty: {gap_penalty:.2f}, Gamma: {gamma:.2f}")
            print(f"Cost: {result['cost']:.2f}")
            print(f"Matches: {result['matches']}")
            print(f"Insertions: {result['insertions']}")
            print(f"Deletions: {result['deletions']}")
            print("Alignment:")
            print(format_alignment(ts1, ts2, result['alignment']))

# Test the analysis
if __name__ == "__main__":
    # Example sequences
    print("Test case 1")
    ts1 = [1, 2, 3, 4]
    ts2 = [1, 2, 4, 5, 6]

    # Get the alignment with a higher gap penalty
    cost, alignment = soft_dtw_with_gaps(ts1, ts2, gamma=1.0, gap_penalty=5.0)  # Increased gap penalty
    
    # Print the formatted alignment
    print("gamma:", 1.0)
    print("gap_penalty:", 5.0)
    print("Cost:", cost)
    print("Alignment:")
    print(format_alignment(ts1, ts2, alignment))

    # Get the alignment (issue with too low a gap penalty)
    cost, alignment = soft_dtw_with_gaps(ts1, ts2, gamma=1.0, gap_penalty=2.0)
    
    # Print the formatted alignment
    print("\ngamma:", 1.0)
    print("gap_penalty:", 2.0)
    print("Cost:", cost)
    print("Alignment:")
    print(format_alignment(ts1, ts2, alignment))
    

    # Define parameter ranges
    gap_penalties = np.linspace(0.5, 10.0, 10)
    gammas = np.linspace(0.1, 10, 10)
    
    # Run analysis
    results = analyze_parameters(ts1, ts2, gap_penalties, gammas)
    
    # Create plots
    fig = plot_parameter_effects(results, gap_penalties, gammas)
    plt.savefig('parameter_effects.png')
    
    # Print summary
    print_parameter_summary(results, gap_penalties, gammas)

# Example usage with the previous soft_dtw_with_gaps function
# Test the function
