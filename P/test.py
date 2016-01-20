img = read_img('001', 'L', '1')
# segment pupil
nb_clusters = 22
img_k = kmeans(img, nb_clusters)
darkest_cluster = segment_lowest_cluster(img_k)
x, y = cluster_mean(darkest_cluster)
# Clean pupil
clean_pupil, rad = find_pupil(darkest_cluster)
# Find center and radio
center_x, center_y = center(clean_pupil, rad)
radio_pupil = radius(clean_pupil, center_x, center_y)
# Segment iris
ii_ad = bin_image(img)
ii_ad_canny = canny(ii_ad)
# find radio for iris
radio_iris = radius_iris(ii_ad_canny, center_x, center_y, rad)
