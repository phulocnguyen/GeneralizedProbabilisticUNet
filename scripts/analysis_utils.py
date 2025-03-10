import os
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(array, vert_value=None, fname=None):
    """
    Vẽ histogram của một mảng số và lưu hình ảnh.
    
    Args:
        array (np.ndarray): Mảng chứa dữ liệu cần vẽ histogram.
        vert_value (float, optional): Giá trị đường thẳng đứng để đánh dấu.
        fname (str, optional): Đường dẫn để lưu hình ảnh.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(array, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Histogram of Entropy')
    
    if vert_value is not None:
        plt.axvline(x=vert_value, color='red', linestyle='dashed', linewidth=2, label=f'Max Entropy ({vert_value:.2f})')
        plt.legend()
    
    if fname is not None:
        plt.savefig(fname)
    plt.close()

def plot_2d_scatterplot(matrix, fname=None):
    """
    Vẽ scatter plot 2D của ma trận mẫu và lưu hình ảnh.
    
    Args:
        matrix (np.ndarray): Ma trận có dạng (N, 2), trong đó N là số điểm dữ liệu.
        fname (str, optional): Đường dẫn để lưu hình ảnh.
    """
    if matrix.shape[1] != 2:
        raise ValueError("Ma trận đầu vào phải có dạng (N, 2) để vẽ scatter plot 2D.")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(matrix[:, 0], matrix[:, 1], alpha=0.6, c='blue', edgecolors='black')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Scatter Plot of Prior Samples')
    
    if fname is not None:
        plt.savefig(fname)
    plt.close()