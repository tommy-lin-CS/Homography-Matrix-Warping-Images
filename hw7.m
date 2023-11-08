close all;
% Read source and destination images
im_src = imread('im_src.tif');
im_dest = imread('im_dest.tif');

% change colored image to gray
im_src = im2double(im2gray(im_src)); 
im_dest = im2double(im2gray(im_dest));

% point selection on src image
imshow(im_src);
title("Select your points on source image then press enter: ")
[x_src, y_src] = ginput();

close all;

% point selection on dest image
imshow(im_dest);
title("Select your points on destination image then press enter: ")
[x_dest, y_dest] = ginput();

% Plot the points on the source image
figure(1)
imshow(im_src);
title("Selected points on the source image");
hold on; % Keep the image, and plot on top of it
plot(x_src, y_src, 'r.', 'LineWidth', 2, 'MarkerSize', 20);
hold off;

% Plot the points on the source image
figure(2);
imshow(im_dest);
title("Selected points on the destination image");
hold on; % Keep the image, and plot on top of it
plot(x_dest, y_dest, 'r.', 'LineWidth', 2, 'MarkerSize', 20);
hold off;

% Checks if the number of points selected at source == dest
if size(x_src, 1) ~= size(x_dest, 1)
    close all;
    error("Please select equal number of points in the src and dest images");
end

% Task 1
% A h = 0
A = []; % matrix to store A

N = size(x_src, 1); % number of points user has chosen

% Loops through N times and populates the A matrix 
% A matrix is a 2N X 9 matrix
for i = 1:N
    x1 = x_src(i);
    y1 = y_src(i);
    x1p = x_dest(i);
    y1p = y_dest(i);

    A = [A;
         x1, y1, 1, 0, 0, 0, -x1*x1p, -y1*x1p, -x1p;
         0, 0, 0, x1, y1, 1, -x1*y1p, -y1*y1p, -y1p];
end

% Computing the homography using eigen decomposition
[~, ~, V] = eig(A'*A);
h_eig = V(:,1); % the eigenvector corresponding to the smallest eigenvalue
disp(h_eig);
H_eig = reshape(h_eig, 3, 3)'; % reshape into a 3x3 matrix
disp('The homography matrix H using eigen decomposition is:');
disp(H_eig)

% Computing the homography using SVD
[U, D, V] = svd(A);
h_svd = V(:,end); % the last column of V corresponding to the smallest singular value
H_svd = reshape(h_svd, 3, 3)'; % reshape into a 3x3 matrix
% Display the homography matrix from SVD
disp('The homography matrix H using SVD is:');
disp(H_svd);


% Task 2
% Forward
[nrows_src, ncols_src] = size(im_src); % dimension of source image
[nrows_dest, ncols_dest] = size(im_dest); % dimension of destination image
warpIm = zeros(nrows_src, ncols_src); % initialize the warp image 

% Nested for loop to map pixels from source to destination using forward
% warping
for x=1:ncols_src
    for y=1:nrows_src
        p = [x; y; 1]; % homogeneous coordinates of source pixel
        p_prime = H_svd * p; % mapping using homography
        x_prime = p_prime(1) / p_prime(3);
        y_prime = p_prime(2) / p_prime(3);
        % bounds of image check
        if x_prime < 1 || x_prime > ncols_dest || y_prime < 1 || y_prime > nrows_dest
            continue;
        end
        % produce warp image
        warpIm(round(y_prime), round(x_prime)) = im_src(y, x);
    end
end
figure(3); % Creates a new figure window.
imshow(warpIm, []); 
title('Forward Warping of Image');

% Backward Nearest Neighbors
[nrows_src, ncols_src] = size(im_src); % dimension of source image
[nrows_dest, ncols_dest] = size(im_dest); % dimension of destination image
warpIm = zeros(nrows_dest, ncols_dest); % initialize the warp image 

% Nested for loop to map pixels from source to destination using backward
% warping (nearest neighbors)
for y_prime=1:nrows_dest
    for x_prime=1:ncols_dest
        p_prime = [x_prime; y_prime; 1];
        p = H_svd \ p_prime; % pseudoinverse!
        x = p(1) / p(3);
        y = p(2) / p(3);
        % bounds check
        if x < 1 || x > ncols_src || y < 1 || y > nrows_src
            continue;
        end
        warpIm(y_prime, x_prime) = im_src(round(y), round(x));
    end
end
figure(4); % Creates a new figure window.
imshow(warpIm, []); 
title('Backward Warping of Image Using Nearest Neighbor');

% Backward Bilinear
[nrows_src, ncols_src] = size(im_src);
[nrows_dest, ncols_dest] = size(im_dest);
warpIm = zeros(nrows_dest, ncols_dest);
for y_prime=1:nrows_dest
    for x_prime=1:ncols_dest
        p_prime = [x_prime; y_prime; 1];
        p = H_svd \ p_prime;
        x = p(1) / p(3);
        y = p(2) / p(3);
        if x < 1 || x > ncols_src || y < 1 || y > nrows_src
            continue;
        end

        % Calculate floor and ceil values for bilinear interpolation
        xfloor = floor(x);
        xceil = ceil(x);
        yfloor = floor(y);
        yceil = ceil(y);
        
        % Calculate interpolation weights
        a = x - xfloor;
        b = y - yfloor;
        warpIm(y_prime, x_prime) = (1 - a) * (1 - b) * im_src(yfloor, xfloor)...
                                 + a * (1 - b) * im_src(yfloor, xceil)...
                                 + (1 - a) * b * im_src(yceil, xfloor)...
                                 + a * b * im_src(yceil, xceil);
    end
end
figure(5); % Creates a new figure window.
imshow(warpIm, []); 
title('Backward Warping of Image Using Bilinear');


% Backward Interp2
h = inv(H_svd);

% Create the meshgrid for the destination image
[xi, yi] = meshgrid(1:ncols_dest, 1:nrows_dest);

% Apply the inverse homography to the grid to get the source coordinates
xx = (h(1,1)*xi + h(1,2)*yi + h(1,3)) ./ (h(3,1)*xi + h(3,2)*yi + h(3,3));
yy = (h(2,1)*xi + h(2,2)*yi + h(2,3)) ./ (h(3,1)*xi + h(3,2)*yi + h(3,3));

% Perform interp2
interp2edIm = interp2(im_src, xx, yy, 'linear', 0);

figure(6); 
imshow(uint8(interp2edIm * 255), []); % we need to convert it back to uint8
title("Backward Warping of Image Using Interp2");

