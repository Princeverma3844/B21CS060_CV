# Computer Vision Algorithms: Harris Corner Detection, Stereo 3D Reconstruction, and Epipolar Geometry

This repository contains implementations of key computer vision algorithms including Harris Corner Detection, Stereo 3D Reconstruction, and Epipolar Geometry. Each algorithm is implemented from scratch and compared with established libraries where applicable.

## Question 1: Harris Corner Detection

### Description
Harris Corner Detection is a popular technique used to detect corners within an image, which are points of interest used in various computer vision applications.

### Implementation Details
- **Algorithm Implementation**: Developed from scratch in a Python notebook. The process involves preprocessing the input image by applying a Gaussian filter to reduce noise.
- **Gradient Calculation**: Computed the gradients in both the X and Y directions using the separable property of Sobel filters, optimizing computational efficiency from O(n^2) to O(2n).
- **Corner Detection**: Utilized a sliding window approach to compute the response function value using the Harris corner detection formula (det(M) - k * trace(M)^2). Applied a threshold to identify corner points.
- **Performance Comparison**: Compared the scratch implementation with OpenCV's Harris Corner Detection library.

## Question 2: Stereo 3D Reconstruction

### Description
In the context of stereo vision, the task involves computing the disparity, depth, and generating a 3D point cloud from a pair of stereo images captured by cameras with known intrinsic matrices and a given baseline (translation along the x-axis).

### Implementation Details
- **Essential Matrix**: Derived using the provided intrinsic matrices and baseline information with the formula E = Tx @ R (where R is the identity matrix and Tx is the translation matrix with the x component set to the baseline value).
- **Fundamental Matrix**: Computed from the Essential matrix to describe the epipolar geometry between the two images.
- **Correspondence Map**: Established by searching for corresponding points in the left and right images, enabling the calculation of 3D coordinates using triangulation techniques.
- **3D Reconstruction**: Calculated the disparity map, depth map, and 3D point cloud of the image.

## Question 3: Epipolar Geometry

### Description
This task involves working with two images of a static scene captured from a single camera. The goal is to find and draw epipolar lines, and compute corresponding pixels on these lines.

### Implementation Details
- **Epipolar Lines**: Computed and drawn on both images using the Fundamental matrix.
- **Corresponding Pixels**: Determined by sampling points on one epipolar line and finding their correspondences on the other.
- **Visual Inspection**: Plotted the epipolar lines and corresponding points to visualize the stereo correspondence.
