from pose_format import Pose
import numpy as np

data_buffer = open("/home/saiteja/Desktop/guideddiff/HumanPoser/videos/output_segmented_cropped.pose", "rb").read()
pose = Pose.read(data_buffer)

numpy_data = pose.body.data
confidence_measure = pose.body.confidence

frame_no = 0
points = numpy_data[frame_no][0]

# drop the negative
x = points[:, 0]
y = points[:, 1]

# get all the negative values locations of x
x_neg_points = np.where(x < 0)
y_neg_points = np.where(y < 0)

# set of all the negative points
negative_points = set(x_neg_points[0]).union(set(y_neg_points[0]))

# drop the negative points
x = np.delete(x, list(negative_points))
y = np.delete(y, list(negative_points))

y_diff = np.max(y) - np.min(y)
x_diff = np.max(x) - np.min(x)
diff = max(y_diff, x_diff)
print("x_min: ", np.min(x))
print("diff: ", diff)
