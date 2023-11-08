# Homography Matrix and Image Warping

### Description:

This project implements an efficient and robust method for estimating the homography between a pair of images and applies image warping techniques to achieve perspective transformations.

### Input:

The user can select as many points as they want, however, points on both images should match up to produce the best results.<br>
Source Image (preselected points): ![source image](./images/source.jpg)<br>
Destination Image (preselected points): ![destination image](./images/dest.jpg)<br>

### Output:

Forward Warping: ![forward warping](./images/forward.jpg)<br>
Backward Warping Using Nearest Neighbor Method: ![backward warping using nn](./images/backwardnn.jpg)<br>
Backward Warping Using Bilinear Method: ![backward warping using bilinear](./images/backwardbilinear.jpg)<br>
Backward Warping Using Interp2 Function: ![backward using interp2](./images/backwardinterp2.jpg)
