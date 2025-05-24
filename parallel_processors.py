import time
import multiprocessing
import threading
import concurrent.futures
from joblib import Parallel, delayed
from PIL import Image
from functools import partial

def process_image_task(image_path, process_func, params):
    """
    Generic task for processing a single image.
    
    Args:
        image_path: Path to the image file
        process_func: The image processing function to apply
        params: Parameters for the processing function
        
    Returns:
        Processed image and processing time
    """
    result, processing_time = process_func(image_path, params)
    return result, processing_time

def process_images_sequential(image_paths, process_func, params):
    """
    Process images sequentially (no parallelism).
    
    Args:
        image_paths: List of paths to image files
        process_func: The image processing function to apply
        params: Parameters for the processing function
        
    Returns:
        List of processed images and list of processing times
    """
    results = []
    durations = []
    
    for path in image_paths:
        result, duration = process_func(path, params)
        results.append(result)
        durations.append(duration)
    
    return results, durations

def process_images_multiprocessing(image_paths, process_func, params, n_workers=None):
    """
    Process images using Python's multiprocessing module.
    
    Args:
        image_paths: List of paths to image files
        process_func: The image processing function to apply
        params: Parameters for the processing function
        n_workers: Number of processes to use (default: CPU count)
        
    Returns:
        List of processed images and list of processing times
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # Create a partial function with fixed parameters
    task = partial(process_image_task, process_func=process_func, params=params)
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=n_workers) as pool:
        # Map the task to the image paths
        results = pool.map(task, image_paths)
    
    # Unpack results
    processed_images, durations = zip(*results)
    
    return list(processed_images), list(durations)

def process_images_threading(image_paths, process_func, params, n_workers=None):
    """
    Process images using Python's threading module.
    Note: Due to Python's GIL, this may not provide speedup for CPU-bound tasks.
    
    Args:
        image_paths: List of paths to image files
        process_func: The image processing function to apply
        params: Parameters for the processing function
        n_workers: Number of threads to use (default: 2x CPU count)
        
    Returns:
        List of processed images and list of processing times
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count() * 2  # More threads than cores for I/O tasks
    
    results = [None] * len(image_paths)
    durations = [None] * len(image_paths)
    
    def thread_task(index, path):
        result, duration = process_func(path, params)
        results[index] = result
        durations[index] = duration
    
    threads = []
    for i, path in enumerate(image_paths):
        thread = threading.Thread(target=thread_task, args=(i, path))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return results, durations

def process_images_joblib(image_paths, process_func, params, n_workers=None):
    """
    Process images using joblib's Parallel and delayed functions.
    
    Args:
        image_paths: List of paths to image files
        process_func: The image processing function to apply
        params: Parameters for the processing function
        n_workers: Number of workers to use (default: CPU count)
        
    Returns:
        List of processed images and list of processing times
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # Define the task
    results = Parallel(n_jobs=n_workers)(
        delayed(process_func)(path, params) for path in image_paths
    )
    
    # Unpack results
    processed_images, durations = zip(*results)
    
    return list(processed_images), list(durations)

def process_images_concurrent_futures(image_paths, process_func, params, n_workers=None):
    """
    Process images using concurrent.futures module.
    
    Args:
        image_paths: List of paths to image files
        process_func: The image processing function to apply
        params: Parameters for the processing function
        n_workers: Number of workers to use (default: CPU count)
        
    Returns:
        List of processed images and list of processing times
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    results = []
    durations = []
    
    # Create a partial function with fixed parameters
    task = partial(process_image_task, process_func=process_func, params=params)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit tasks
        future_to_path = {executor.submit(task, path): path for path in image_paths}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            result, duration = future.result()
            results.append(result)
            durations.append(duration)
    
    # Sort results based on the original order of image_paths
    # This is necessary because as_completed() returns futures in the order they complete
    sorted_results = []
    sorted_durations = []
    path_to_index = {path: i for i, path in enumerate(image_paths)}
    
    for i in range(len(image_paths)):
        idx = path_to_index.get(future_to_path.get(list(future_to_path.keys())[i]))
        if idx is not None and idx < len(results):
            sorted_results.append(results[idx])
            sorted_durations.append(durations[idx])
    
    return sorted_results, sorted_durations
