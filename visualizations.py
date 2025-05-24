# Displays graphs/charts comparing results.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def display_performance_comparison(sequential_times, parallel_times, parallel_method):
    """
    Display a bar chart comparing sequential vs parallel performance.
    
    Args:
        sequential_times: List of sequential processing times
        parallel_times: List of parallel processing times
        parallel_method: Name of the parallel method used
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    image_labels = [f"Image {i+1}" for i in range(len(sequential_times))]
    x = np.arange(len(image_labels))
    width = 0.35
    
    # Plot bars
    rects1 = ax.bar(x - width/2, sequential_times, width, label='Sequential')
    rects2 = ax.bar(x + width/2, parallel_times, width, label=parallel_method)
    
    # Add labels and title
    ax.set_xlabel('Images')
    ax.set_ylabel('Processing Time (seconds)')
    ax.set_title('Processing Time Comparison: Sequential vs. Parallel')
    ax.set_xticks(x)
    ax.set_xticklabels(image_labels)
    ax.legend()
    
    # Add speedup text on top of bars
    for i, (seq, par) in enumerate(zip(sequential_times, parallel_times)):
        speedup = seq / par
        ax.text(i, max(seq, par) + 0.05, 
                f"{speedup:.2f}x", 
                ha='center', va='bottom',
                color='green', fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure layout is tight
    fig.tight_layout()
    
    # Display the plot
    st.pyplot(fig)

def plot_image_comparison(original_image, processed_image, operation_name):
    """
    Display a side-by-side comparison of original and processed images.
    
    Args:
        original_image: PIL Image of the original
        processed_image: PIL Image of the processed result
        operation_name: Name of the operation performed
    """
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(processed_image, cmap='gray' if processed_image.mode == 'L' else None)
    ax2.set_title(f"After {operation_name}")
    ax2.axis('off')
    
    # Ensure tight layout
    fig.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Add a divider
    st.markdown("---")
