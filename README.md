# Parallel Image Processing Comparison Tool 🖼️⚙️

This is a Streamlit-based web application that demonstrates the performance benefits of **parallel computing** (threading and multiprocessing) over sequential execution in **image processing tasks**.

---

## 📌 Features

- 📂 Upload your own images or use sample images.
- 🧠 Apply image processing operations:
  - Edge Detection
  - Gaussian Blur
  - Histogram Equalization
- ⚙️ Choose execution mode:
  - Sequential
  - Threading
  - Multiprocessing
- 📊 View performance metrics (execution time, speedup)
- 🖼️ Compare original vs. processed images
- ⬇️ Download processed images

---

## 📂 Project Structure

```
parallel-image-processing/
├── app.py                  # Main Streamlit app
├── image_processing.py     # Core image processing functions using OpenCV
├── parallel_processors.py  # Multiprocessing or multithreading logic
├── visualizations.py       # Functions for plotting and visualizing outputs
├── utils.py                # Utility functions
├── temp/                   # Temporary image storage
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```


## 📸 Image Processing Techniques

- **Edge Detection:** Using Canny algorithm
- **Gaussian Blur:** Smoothing filter using kernel
- **Histogram Equalization:** Contrast enhancement

---

## 🚀 Technologies Used

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)
- Python Standard Libraries: `multiprocessing`, `threading`, `os`, `io`, `time`, `base64`, `datetime`

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/parallel-image-processing.git
cd parallel-image-processing
pip install -r requirements.txt
```

