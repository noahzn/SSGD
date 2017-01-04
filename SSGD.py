"""
SSGD: SUPERPIXELS USING THE SHORTEST GRADIENT DISTANCE
Ning Zhang and Lin Zhang∗
School of Software Engineering, Tongji University, Shanghai, China
"""

import cv2
import numpy as np
import time
from skimage.graph import route_through_array
import scipy.io as sio
import os
import warnings

warnings.simplefilter("error")
np.set_printoptions(threshold=np.inf)
np.errstate(invalid='ignore', divide='ignore')


class SSGD:
    def __init__(self, img, the_number_of_superpixel, iteration=10, compactness=15):
        """
        :param img: input image
        :param the_number_of_superpixel: desired number of superpixels
        :param iteration: default value is 10
        :param compactness:
        :return:
        """
        self.img_name = img
        self.img_o = cv2.imread(img)
        self.img = cv2.imread(img)
        self.img_gray = cv2.imread(img, 0)
        self.height_of_img, self.width_of_img = self.img.shape[:2]
        self.iteration = iteration
        self.the_number_of_superpixel = the_number_of_superpixel
        self.step_of_superpixel = int(
            (self.img.shape[0] * self.img.shape[1] / self.the_number_of_superpixel) ** 0.5)

        self.compactness = compactness
        self.lamda = self.compactness ** 2 / self.step_of_superpixel ** 2
        self.clusters = -1 * np.ones(self.img.shape[:2])
        self.distances = float('inf') * np.ones(self.img.shape[:2])

        self.generate_superpixel()  # Let's go!

    def _bilateral_filter(self):
        """
        bilateral filtering operation in none-edged texture-rich regions
        :return:
        """
        for i in range(self.step_of_superpixel, int(self.width_of_img - self.step_of_superpixel / 2),
                       self.step_of_superpixel):
            for j in range(self.step_of_superpixel, int(self.height_of_img - self.step_of_superpixel / 2),
                           self.step_of_superpixel):

                xlow, xhigh = int(i - self.step_of_superpixel), int(i + self.step_of_superpixel)
                ylow, yhigh = int(j - self.step_of_superpixel), int(j + self.step_of_superpixel)

                xlow = max(0, xlow)
                xhigh = min(self.width_of_img, xhigh)
                ylow = max(0, ylow)
                yhigh = min(self.height_of_img, yhigh)

                crop = self.img_gray[ylow:yhigh, xlow:xhigh]
                hist = cv2.calcHist([crop], [0], None, [256], (0, 255))

                if np.count_nonzero(hist) >= 100:
                    self.img[ylow:yhigh, xlow:xhigh] = cv2.bilateralFilter(self.img[ylow:yhigh, xlow:xhigh], 4, 40, 10)

        self.lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB).astype(np.float64)

    def _edge_detection(self):
        """
        calculate the gradient map
        :return:
        """
        img = np.float32(self.img)
        img /= 255.
        detect = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
        edge1 = detect.detectEdges(img)
        edge1 *= 255
        edge1 = edge1.astype('uint8')
        self.edge = np.zeros_like(edge1)

        gradx = cv2.Sobel(self.img_gray, cv2.CV_16S, 1, 0, ksize=1)
        grady = cv2.Sobel(self.img_gray, cv2.CV_16S, 0, 1, ksize=1)
        absX = cv2.convertScaleAbs(gradx)
        absY = cv2.convertScaleAbs(grady)
        edge2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.edge2 = edge2

        idx1 = (edge1 > 10)
        idx2 = (edge2 > 10)
        for i in range(self.img.shape[1]):
            for j in range(self.img.shape[0]):
                if (idx1[j, i] and idx2[j, i]) or (idx1[j, i] > 30):
                    self.edge[j, i] = (int(edge1[j, i]) + int(edge2[j, i])) / 2

    def _initialize_cluster_centers(self):
        """
        initialize cluster centers and move the centers to the lowest gradient position in a 3x3 neighborhood
        """
        new_centers = []
        for i in range(self.step_of_superpixel, int(self.width_of_img - self.step_of_superpixel / 2),
                       self.step_of_superpixel):
            for j in range(self.step_of_superpixel, int(self.height_of_img - self.step_of_superpixel / 2),
                           self.step_of_superpixel):
                new_center = self.find_lowest_gradient2((i, j))
                color = self.lab_img[new_center[1], new_center[0]]
                center = [color[0], color[1], color[2], new_center[0], new_center[1]]
                new_centers.append(center)

        self.center_counts = np.zeros(len(new_centers))
        self.new_centers = np.array(new_centers)  # (n, 5)

    def find_lowest_gradient2(self, old_center):
        """
        :param: old_center: the original centers of clusters
        :return:
        """
        minimum_gradient = float('inf')
        new_center = old_center

        # eight pixels around the old center
        for i in range(old_center[0] - 1, old_center[0] + 2):
            for j in range(old_center[1] - 1, old_center[1] + 2):
                gradx = cv2.Sobel(self.lab_img[j, i], cv2.CV_16S, 1, 0, ksize=3)
                gradx = cv2.convertScaleAbs(gradx)
                grady = cv2.Sobel(self.lab_img[j, i], cv2.CV_16S, 0, 1, ksize=3)
                grady = cv2.convertScaleAbs(grady)
                grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

                if grad[1] < minimum_gradient:
                    minimum_gradient = grad[1]
                    new_center = [i, j]

        return new_center

    def _compute_distance(self):
        index_of_distance = np.mgrid[0:self.height_of_img, 0:self.width_of_img].swapaxes(0, 2).swapaxes(0,
                                                                                                        1)  # (512, 512, 2)
        # 2S x 2S neighborhood around the cluster center
        for i in range(self.iteration):
            print('iteration: %d' % i)
            self.distances = float('inf') * np.ones(self.img.shape[:2])

            for j in range(self.new_centers.shape[0]):
                xlow, xhigh = int(self.new_centers[j][3] - self.step_of_superpixel), int(
                    self.new_centers[j][3] + self.step_of_superpixel)
                ylow, yhigh = int(self.new_centers[j][4] - self.step_of_superpixel), int(
                    self.new_centers[j][4] + self.step_of_superpixel)

                xlow = max(0, xlow)
                xhigh = min(self.width_of_img, xhigh)
                ylow = max(0, ylow)
                yhigh = min(self.height_of_img, yhigh)

                neighborhood_of_color = self.lab_img[ylow:yhigh, xlow:xhigh]

                # distance of color in CIELAB space
                distance_of_color = (np.sum(np.square(neighborhood_of_color - self.lab_img[int(self.new_centers[j][4]),
                                                                                           int(self.new_centers[j][
                                                                                                   3])]),
                                            axis=2)) ** 0.5

                yy, xx = np.ogrid[ylow:yhigh, xlow:xhigh]
                distance_of_space = ((xx - self.new_centers[j][3]) ** 2 + (yy - self.new_centers[j][4]) ** 2) ** 0.5

                neighborhood_of_gradient = self.edge[ylow:yhigh, xlow:xhigh]

                # calculate the shortest gradient distance, we get a cost-matrix
                # notice: route_through_array function has been modified to return the whole cost-matrix
                distance_of_space_g_temp = route_through_array(neighborhood_of_gradient,
                                                               [int(self.new_centers[j][4] - ylow),
                                                                int(self.new_centers[j][3] - xlow)], [0, 0],
                                                               fully_connected=False)

                if np.max(distance_of_space_g_temp) == 0:
                    distance_of_space_g = distance_of_space_g_temp
                else:
                    distance_of_space_g = (distance_of_space_g_temp - np.min(distance_of_space_g_temp)) / \
                                          (np.max(distance_of_space_g_temp) - np.min(distance_of_space_g_temp))

                distance_of_space_g += (abs(distance_of_space_g - neighborhood_of_gradient[
                    int(self.new_centers[j][4] - ylow), int(self.new_centers[j][3] - xlow)]) / 255)

                distance = distance_of_color + self.lamda * (
                               0.3 * distance_of_space + 0.7 * (distance_of_space * np.exp(distance_of_space_g)))

                distance_of_neighborhood = self.distances[ylow:yhigh, xlow:xhigh]
                index = distance < distance_of_neighborhood
                distance_of_neighborhood[index] = distance[index]

                self.distances[ylow:yhigh, xlow:xhigh] = distance_of_neighborhood
                self.clusters[ylow:yhigh, xlow:xhigh][index] = j

            # adjust the barycenters of centers
            for k in range(len(self.new_centers)):
                index = (self.clusters == k)
                color_of_same_kind_pixels = self.lab_img[index]
                location_of_same_kind_pixels = index_of_distance[index]
                center_color_of_same_kind_pixels = np.empty([0, 3], dtype=np.float64)
                center_location_of_same_kind_pixels = np.empty([0, 2], dtype=np.float64)

                if len(color_of_same_kind_pixels) != 0:
                    color_diff = color_of_same_kind_pixels - self.new_centers[k][:3]
                    std_color = np.std(color_of_same_kind_pixels, axis=0)
                    for s in range(color_diff.shape[0]):
                        if (abs(color_diff[s]) <= 3 * std_color).all():
                            center_color_of_same_kind_pixels = np.vstack(
                                (center_color_of_same_kind_pixels, color_of_same_kind_pixels[s]))
                            center_location_of_same_kind_pixels = np.vstack(
                                (center_location_of_same_kind_pixels, location_of_same_kind_pixels[s]))

                    if len(center_color_of_same_kind_pixels) != 0:
                        self.new_centers[k][:3] = np.sum(center_color_of_same_kind_pixels, axis=0)
                        sum_y, sum_x = np.sum(center_location_of_same_kind_pixels, axis=0)
                        self.new_centers[k][3:] = sum_x, sum_y
                        self.new_centers[k][:3] /= center_color_of_same_kind_pixels.shape[0]
                        self.new_centers[k][3:] /= center_location_of_same_kind_pixels.shape[0]

    def _enforce_connectivity(self):
        """
        :return:
        """
        dx4 = [-1, 0, 1, 0]
        dy4 = [0, -1, 0, 1]
        label = 0
        adjacent_label = 0
        lims = self.width_of_img * self.height_of_img / self.new_centers.shape[0]
        self.new_clusters = -1 * np.ones(self.img.shape[:2]).astype(np.int64)
        elements = []

        for i in range(self.width_of_img):
            for j in range(self.height_of_img):
                if self.new_clusters[j, i] == -1:
                    elements = [(j, i)]
                    for dx, dy in zip(dx4, dy4):
                        x = elements[0][1] + dx
                        y = elements[0][0] + dy
                        if 0 <= x < self.width_of_img and 0 <= y < self.height_of_img and self.new_clusters[y, x] >= 0:
                            adjacent_label = self.new_clusters[y, x]

                count = 1
                c = 0
                while c < count:
                    for dx, dy in zip(dx4, dy4):
                        x = elements[c][1] + dx
                        y = elements[c][0] + dy
                        if 0 <= x < self.width_of_img and 0 <= y < self.height_of_img:
                            if self.new_clusters[y, x] == -1 and self.clusters[j, i] == self.clusters[y, x]:
                                elements.append((y, x))
                                self.new_clusters[y, x] = label
                                count += 1
                    c += 1

                if count <= int(lims) >> 4:
                    for c in range(count):
                        self.new_clusters[elements[c]] = adjacent_label
                    label -= 1

                label += 1

        for i in range(self.width_of_img):
            for j in range(self.height_of_img):
                self.clusters[j, i] = self.new_clusters[j, i]

    def _draw_contours(self, color):
        """
        :param color: RGB color
        :return:
        """
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]
        is_taken = np.zeros(self.img.shape[:2], np.bool)
        contours = []

        for i in range(self.width_of_img):
            for j in range(self.height_of_img):
                nr_p = 0
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if 0 <= x < self.width_of_img and 0 <= y < self.height_of_img:
                        if is_taken[y, x] == False and self.new_clusters[j, i] != self.new_clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    is_taken[j, i] = True
                    contours.append([j, i])

        for i in range(len(contours)):
            self.img_o[contours[i][0], contours[i][1]] = color

        colors = np.zeros([np.max(self.new_clusters) + 1, 3], dtype=np.float64)
        self.result = np.zeros((self.height_of_img, self.width_of_img, 3), dtype=np.uint8)
        img_o2 = cv2.imread(self.img_name)
        for x in range(self.width_of_img):
            for y in range(self.height_of_img):
                try:
                    index = self.new_clusters[y, x]
                    color2 = img_o2[y, x]
                    colors[index][0] += color2[0]
                    colors[index][1] += color2[1]
                    colors[index][2] += color2[2]

                except Exception as e:
                    input(e)

        for i in range(np.max(self.new_clusters) + 1):
            idx = self.new_clusters == i
            colors[i][0] /= idx.sum() + 1
            colors[i][1] /= idx.sum() + 1
            colors[i][2] /= idx.sum() + 1

        for x in range(self.width_of_img):
            for y in range(self.height_of_img):
                n_color = colors[self.new_clusters[y][x]]
                self.result[y, x][0] = n_color[0]
                self.result[y, x][1] = n_color[1]
                self.result[y, x][2] = n_color[2]

    def generate_superpixel(self):
        self._bilateral_filter()
        self._edge_detection()
        self._initialize_cluster_centers()
        self._compute_distance()
        self._enforce_connectivity()
        self._enforce_connectivity()
        self._draw_contours((0, 255, 0))


seg = SSGD('183066.jpg', 500)
cv2.imshow('segmentation', seg.img_o)
cv2.imshow('artistic reconstruction', seg.result)
cv2.waitKey()
cv2.destroyAllWindows()
