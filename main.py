# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
import cv2
import sys

#Author:  Dylan Hird

def main():

    print("yo")
    image_file_names = ["mural01.jpg", "mural02.jpg"]
    bgr_img0 = cv2.imread("guitar01.jpg")
    # cv2.imshow("img0", bgr_img0)

    pts1 = np.array([[75, 388], [214, 387], [70, 578], [215, 573]])

    # bgr_display = bgr_img0.copy()
    # for x, y in pts1:   cv2.drawMarker(img=bgr_display, position=(x, y), color=(255, 0, 0),
    #                                    markerType=cv2.MARKER_DIAMOND, thickness=3)
    # cv2.namedWindow("Image1", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image1", bgr_display)
    # cv2.waitKey(0)
    pts1_ortho = np.array([[350, 750], [650, 750], [350, 1150], [650, 1150]])
    H_cur_mos, _ = cv2.findHomography(srcPoints=pts1, dstPoints=pts1_ortho)
    # print(H1)
    output_width = 20000
    output_height = 6000
    # Warp the image to the orthophoto.
    bgr_ortho = cv2.warpPerspective(bgr_img0, H_cur_mos, (output_width, output_height))
    # cv2.imshow("ortho", bgr_ortho)
    imgPrev = bgr_img0
    H_prev_mos = H_cur_mos
    i_mos = bgr_ortho
    # img_height = bgr_image.shape[0]  # number of rows (y)
    # img_width = bgr_image.shape[1]  # number of columns (x)

####################################################################################################################################
    for i in range(2,13):
        imgCur = cv2.imread("guitar0" + str(i) + ".jpg")
        cv2.imshow("imgcur", imgCur)
        # cv2.waitKey(0)
        ###########################################################
        bgr_train = imgPrev
        bgr_query = imgCur
        # Extract keypoints and descriptors.
        kp_train, desc_train = detect_features(bgr_train, show_features=False)
        kp_query, desc_query = detect_features(bgr_query, show_features=False)

        matcher = cv2.BFMatcher.create(cv2.NORM_L2)

        # Match query image descriptors to the training image.
        # Use k nearest neighbor matching and apply ratio test.
        matches = matcher.knnMatch(desc_query, desc_train, k=2)
        good = []
        for m, n in matches:
            if m.distance < 1 * n.distance:
                good.append(m)
        matches = good
        # print("Number of raw matches between training and query: ", len(matches))

        bgr_matches = cv2.drawMatches(
            img1=bgr_query, keypoints1=kp_query,
            img2=bgr_train, keypoints2=kp_train,
            matches1to2=matches, matchesMask=None, outImg=None)
        # cv2.imshow("All matches", bgr_matches)

        # show_votes(bgr_query, kp_query, bgr_train, kp_train, matches)

        matches = find_cluster(bgr_query, kp_query, bgr_train, kp_train, matches,
                               show_votes=True)
        # print("Number of matches in the largest cluster:", len(matches))
        # Calculate an homography transformation from the training image to the query image.
        # srcPts = np.zeros((len(kp_train), 2))
        # for j in range(0, len(kp_train)):
        #     srcPts[j] = kp_train[j].pt
        # dstPts = np.zeros((len(kp_query), 2))
        # for k in range(0, len(kp_query)):
        #     dstPts[k] = kp_query[k].pt
        srcPts = np.float32([kp_train[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) # found this line on stack overflow "how to extract coordinates from keypoints" link: https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object/35884644
        dstPts = np.float32([kp_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) #found this line on stack overflow

        print(len(srcPts))
        # srcPts = np.float([kp_train[idx].pt for idx in range(0, len(kp_train))]).reshape(-1, 1, 2)
        # print(kp_train.shape())
        H_cur_prev, inliers = cv2.findHomography(srcPoints=dstPts, dstPoints=srcPts, method = cv2.RANSAC, ransacReprojThreshold = 4, maxIters = 2500, confidence = 0.90)
        # method = cv2.RANSAC, ransacReprojThreshold = 3, maxIters = 2000, confidence = 0.99
        H_cur_mos = H_prev_mos @ H_cur_prev
        # Draw matches between query image and training image.
        bgr_matches = cv2.drawMatches(
            img1=bgr_query, keypoints1=kp_query,
            img2=bgr_train, keypoints2=kp_train,
            matches1to2=matches, matchesMask=None, outImg=None)
        cv2.imshow("Matches in largest cluster", bgr_matches)

        # Apply the affine warp to warp the training image to the query image.
        if H_cur_prev is not None:
            # Object detected! Warp the training image to the query image and blend the images.
            print("Object detected! Found %d inlier matches" % sum(inliers))
            i_cur_warp = cv2.warpPerspective(imgCur, H_cur_mos, (output_width, output_height))

            # Blend the images.

            i_mos = fuse_color_images(i_mos, i_cur_warp)
            width = int(i_mos.shape[1] * 0.2)
            height = int(i_mos.shape[0] *0.2)
            resized = cv2.resize(i_mos,(width,height), interpolation= cv2.INTER_AREA )
            cv2.imshow("Blended", resized)
            # cv2.waitKey(0)
        else:
            print("Object not detected; can't fit an affine transform")

        imgPrev = imgCur
        H_prev_mos = H_cur_mos

        cv2.waitKey(0)

    # Detect features in the image and return the keypoints and descriptors.
def detect_features(bgr_img, show_features=False):
    detector = cv2.ORB_create(	nfeatures = 1500,	scaleFactor = 1.2, 	nlevels = 8,	edgeThreshold = 31, firstLevel = 0, WTA_K = 2 , patchSize = 31, fastThreshold = 20	)


    # Extract keypoints and descriptors from image.
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, mask=None)

    # Optionally draw detected keypoints.
    if show_features:
        # Possible flags: DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, DRAW_MATCHES_FLAGS_DEFAULT
        bgr_display = bgr_img.copy()
        cv2.drawKeypoints(image=bgr_display, keypoints=keypoints,
                          outImage=bgr_display,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Features", bgr_display)
        print("Number of keypoints: ", len(keypoints))
        # cv2.waitKey(0)

    return keypoints, descriptors

# Given the proposed matches, each match votes into a quantized "pose" space. Find the
# bin with the largest number of votes, and return the matches within that bin.
def find_cluster(query_img, keypoints_query, train_img, keypoints_train, matches,
                 show_votes=False):
    hq = query_img.shape[0]
    wq = query_img.shape[1]

    max_scale = 4.0  # Scale differences go from 0 to max_scale

    # Our accumulator array is a 4D array of empty lists. These are the number of bins
    # for each of the dimensions.
    num_bins_height = 5
    num_bins_width = 5
    num_bins_scale = 5
    num_bins_ang = 8

    # It is easier to have a 1 dimensional array instead of a 4 dimensional array.
    # Just convert subscripts (h,w,s,a) to indices idx.
    size_acc = num_bins_height * num_bins_width * num_bins_scale * num_bins_ang
    acc_array = [[] for idx in range(size_acc)]

    ht = train_img.shape[0]
    wt = train_img.shape[1]

    # Vote into accumulator array.
    for match in matches:
        qi = match.queryIdx  # Index of query keypoint
        ti = match.trainIdx  # Index of training keypoint that matched

        # Get data for training image.
        kp_train = keypoints_train[ti]
        at = kp_train.angle
        st = kp_train.size
        pt = np.array(kp_train.pt)  # training keypoint location
        mt = np.array([wt / 2, ht / 2])  # Center of training image
        vt = mt - pt  # Vector from keypoint to center

        # Get data for query image.
        kp_query = keypoints_query[qi]
        aq = kp_query.angle
        sq = kp_query.size
        pq = np.array(kp_query.pt)

        # Rotate and scale the vector to the marker point.
        scale_factor = sq / st
        angle_diff = aq - at
        angle_diff = (angle_diff + 360) % 360  # Force angle to between 0..360 degrees
        vq = rotate_and_scale(vt, scale_factor, angle_diff)
        mq = pq + vq

        if show_votes:
            # print("Scale diff %f, angle diff %f" % (scale_factor, angle_diff))

            # Display training image.
            train_img_display = train_img.copy()
            cv2.drawKeypoints(image=train_img_display, keypoints=[kp_train],
                              outImage=train_img_display,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawMarker(img=train_img_display, position=(int(mt[0]), int(mt[1])),
                           color=(255, 0, 0),
                           markerType=cv2.MARKER_DIAMOND)
            cv2.line(img=train_img_display,
                     pt1=(int(pt[0]), int(pt[1])), pt2=(int(mt[0]), int(mt[1])),
                     color=(255, 0, 0), thickness=2)
            # cv2.imshow("Training keypoint", train_img_display)

            # Display query image.
            query_img_display = query_img.copy()
            cv2.drawKeypoints(image=query_img_display, keypoints=[kp_query],
                              outImage=query_img_display,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.line(img=query_img_display,
                     pt1=(int(pq[0]), int(pq[1])), pt2=(int(mq[0]), int(mq[1])),
                     color=(255, 0, 0), thickness=2)
            # cv2.imshow("Query keypoint", query_img_display)
            # cv2.waitKey(100)

        # Compute the cell of the accumulator array, that this match should be stored in.
        row_subscript = int(round(num_bins_height * (mq[1] / hq)))
        col_subscript = int(round(num_bins_width * (mq[0] / wq)))
        if row_subscript >= 0 and row_subscript < num_bins_height and col_subscript >= 0 and col_subscript < num_bins_width:
            scale_subscript = int(num_bins_scale * (scale_factor / max_scale))
            if scale_subscript > num_bins_scale:
                scale_subscript = num_bins_scale - 1

            ang_subscript = int(num_bins_ang * (angle_diff / 360))
            # print(row_subscript,col_subscript, scale_subscript, ang_subscript)

            # Note: the numpy functions ravel_multi_index(), and unravel_index() convert
            # subscripts to indices, and vice versa.
            idx = np.ravel_multi_index(
                (row_subscript, col_subscript, scale_subscript, ang_subscript),
                (num_bins_height, num_bins_width, num_bins_scale, num_bins_ang))

            acc_array[idx].append(match)

    # Count matches in each bin.
    counts = [len(acc_array[idx]) for idx in range(size_acc)]

    # Find the bin with maximum number of counts.
    idx_max = np.argmax(np.array(counts))

    # Return the matches in the largest bin.
    return acc_array[idx_max]

    # Calculate an affine transformation from the training image to the query image.
def calc_affine_transformation(matches_in_cluster, kp_train, kp_query):
    if len(matches_in_cluster) < 3:
        # Not enough matches to calculate affine transformation.
        return None, None

    # Estimate affine transformation from training to query image points.
    # Use the "least median of squares" method for robustness. It also detects outliers.
    # Outliers are those points that have a large error relative to the median of errors.
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    dst_pts = np.float32([kp_query[m.queryIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    A_train_query, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.LMEDS)

    return A_train_query, inliers

def rotate_and_scale(vt, scale_factor, angle_diff):
    theta = np.radians(angle_diff)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    vq = R @ vt
    vq = vq * scale_factor
    return vq

def fuse_color_images(A, B):
    assert (A.ndim == 3 and B.ndim == 3)
    assert (A.shape == B.shape)  # Allocate result image.
    C = np.zeros(A.shape, dtype=np.uint8)# Create masks for pixels that are not zero.
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0# Compute regions of overlap.
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask
    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]
    return C


    cv2.waitKey(0)

if __name__ == "__main__":
    main()


