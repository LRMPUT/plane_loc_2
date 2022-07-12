#!/usr/bin/env python

import sys
import argparse
import numpy as np
import os
import cv2
import rospy
import roslib
roslib.load_manifest('sensor_msgs')
import plane_loc_py as plpy
import fnmatch
import struct
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from rosbag import Bag
from scipy.spatial.transform import Rotation as rot
from plane_loc.msg import Serialized
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


def vector_to_matrix(v):
    T = np.identity(4, dtype=np.float)
    T[0:3, 3] = v[0:3]
    R = rot.from_quat(v[3:7])
    T[0:3, 0:3] = R.as_matrix()

    return T


def matrix_to_vector(T):
    v = np.zeros([7], dtype=np.float)
    v[0:3] = T[0:3, 3]
    R = rot.from_matrix(T[0:3, 0:3])
    v[3:7] = R.as_quat()

    return v


def create_trans_stamped_msg(seq, frame_id, child_frame_id, stamp, v):
    trans_msg = TransformStamped()
    trans_msg.header.seq = seq
    trans_msg.header.stamp = stamp
    trans_msg.header.frame_id = frame_id
    trans_msg.child_frame_id = child_frame_id
    trans_msg.transform.translation.x = v[0]
    trans_msg.transform.translation.y = v[1]
    trans_msg.transform.translation.z = v[2]
    trans_msg.transform.rotation.x = v[3]
    trans_msg.transform.rotation.y = v[4]
    trans_msg.transform.rotation.z = v[5]
    trans_msg.transform.rotation.w = v[6]

    return trans_msg


def main():

    parser = argparse.ArgumentParser(description='Pack plane detection results to rosbag.')
    parser.add_argument('dataset_dir',
                        help='dataset directory')
    parser.add_argument('scene_id',
                        help='scene name')
    parser.add_argument('output_bag',
                        help='output bag file with topics merged')
    parser.add_argument('-v', '--verbose', action="store_true", default=False,
                        help='verbose output')
    parser.add_argument('-f', '--freq', type=float, default=30.0,
                        help='frequency used to calculate timestamps')
    parser.add_argument('-d', '--depth', action="store_true", default=False,
                        help='use external depth instead of plane parameters')

    args = parser.parse_args()

    if args.verbose:
        print("Writing bag file: " + args.output_bag)

    rospy.init_node('pack_rosbag')
    stamp_start = rospy.Time.now()

    # pub = rospy.Publisher('point_cloud_debug', PointCloud2, queue_size=10)
    pub_tf = rospy.Publisher('/tf', TFMessage, queue_size=10)
    pub_color = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
    pub_det_image = rospy.Publisher('/camera/det_image/image_raw', Image, queue_size=10)
    pub_depth = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
    pub_objs = rospy.Publisher('/detector/objs', Serialized, queue_size=10)
    pub_pc = rospy.Publisher('/detector/point_cloud_lab', PointCloud2, queue_size=10)

    bridge = CvBridge()

    scene_dir = os.path.join(args.dataset_dir, 'scenes', args.scene_id)
    color_dir = os.path.join(scene_dir, 'frames', 'color_left')
    annotation_dir = os.path.join(scene_dir, 'annotation_plane_params_det')
    # annotation_dir = os.path.join(scene_dir, 'annotation_baseline_det')
    det_image_dir = os.path.join(annotation_dir, 'det_left')
    depth_dir = os.path.join(annotation_dir, 'depth_left')
    depth_gt_dir = os.path.join(scene_dir, 'frames', 'depth_left')
    depth_stddev_dir = os.path.join(annotation_dir, 'depth_stddev_left')
    pose_dir = os.path.join(scene_dir, 'frames', 'pose_left')
    segmentation_dir = os.path.join(annotation_dir, 'segmentation_left')
    planes_filename = os.path.join(annotation_dir, 'planes.npy')
    descs_filename = os.path.join(annotation_dir, 'descs.npy')

    frame_nums = sorted(os.listdir(segmentation_dir))
    frame_nums = [frame_num.split('.')[0] for frame_num in frame_nums]

    plane_eqs = np.load(planes_filename)
    descs = np.load(descs_filename)
    # descs = np.zeros((plane_eqs.shape[0], 64), dtype=np.float32)


    if args.verbose:
        print("Found %d frames" % len(frame_nums))
        print("Found %d planes" % plane_eqs.shape[0])

    K = np.eye(3, dtype=np.float32)
    baseline = 0
    im_w = 0
    im_h = 0
    scene_filename = os.path.join(scene_dir, args.scene_id + '.txt')
    with open(scene_filename) as f:
        for line in f:
            line = line.strip()
            tokens = [token for token in line.split(' ') if token.strip() != '']
            if tokens[0] == "fx_depth":
                K[0, 0] = float(tokens[2])
            if tokens[0] == "fy_depth":
                K[1, 1] = float(tokens[2])
            if tokens[0] == "mx_depth":
                K[0, 2] = float(tokens[2])
            if tokens[0] == "my_depth":
                K[1, 2] = float(tokens[2])
            if tokens[0] == "depthWidth":
                im_w = int(tokens[2])
            if tokens[0] == "depthHeight":
                im_h = int(tokens[2])
            if tokens[0] == "baseline":
                baseline = float(tokens[2])

    # point_cloud_fields = [PointField('x', 0, PointField.FLOAT32, 1),
    #                       PointField('y', 4, PointField.FLOAT32, 1),
    #                       PointField('z', 8, PointField.FLOAT32, 1),
    #                       PointField('rgba', 12, PointField.UINT32, 1),
    #                       PointField('label', 16, PointField.UINT32, 1)]
    point_cloud_fields = [PointField('x', 0, PointField.FLOAT32, 1),
                          PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1),
                          PointField('rgb', 12, PointField.UINT32, 1),
                          PointField('label', 16, PointField.UINT32, 1)]

    next_id = 0
    with Bag(args.output_bag, 'w') as o:
        T_vo_prev = np.identity(4, dtype=np.float)
        # for i in range(len(rgb_list)):
        for i, frame_num in enumerate(frame_nums[:]):
            # if frame_num != '001482':
            #     continue
            timestamp = stamp_start + rospy.Duration.from_sec(i / float(args.freq))

            if args.verbose and i % 10 == 0:
                print('i = %d' % i)

            pose_file = os.path.join(pose_dir, frame_num + '.txt')
            T_w_c = np.loadtxt(pose_file, delimiter=' ')

            color_file = os.path.join(color_dir, frame_num + '.jpg')
            color = cv2.imread(color_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            det_image_file = os.path.join(det_image_dir, frame_num + '.jpg')
            det_image = cv2.imread(det_image_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            depth_file = os.path.join(depth_dir, frame_num + '.png')
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 1000.0

            depth_stddev_file = os.path.join(depth_stddev_dir, frame_num + '.png')
            depth_stddev = cv2.imread(depth_stddev_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_stddev = depth_stddev.astype(np.float32) / 1000.0
            # depth_stddev = 0.02 * np.ones_like(depth)
            depth_covar = np.square(depth_stddev)

            segmentation_file = os.path.join(segmentation_dir, frame_num + '.png')
            segmentation = cv2.imread(segmentation_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.int32)

            segmentation = (segmentation[:, :, 2] * 256 * 256 +
                            segmentation[:, :, 1] * 256 +
                            segmentation[:, :, 0]) // 100 - 1

            depth_gt_file = os.path.join(depth_gt_dir, frame_num + '.png')
            depth_gt = cv2.imread(depth_gt_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_gt = depth_gt.astype(np.float32) / 1000.0
            # depth = depth_gt

            plane_ids = np.unique(segmentation)
            if plane_ids[0] == -1:
                plane_ids = plane_ids[1:]
            cur_plane_eqs = plane_eqs[plane_ids, :]
            cur_descs = descs[plane_ids, :]
            masks = np.zeros((cur_plane_eqs.shape[0], segmentation.shape[0], segmentation.shape[1]), dtype=np.float32)
            for idx, plane_id in enumerate(plane_ids):
                masks[idx, :, :] = np.where(segmentation == plane_id, 1.0, 0.0)

            if np.isnan(cur_plane_eqs).any() or (np.linalg.norm(cur_plane_eqs, axis=1) < 1e-5).any():
                print(cur_plane_eqs)

            T_c_w_inv_t = T_w_c.transpose()
            cur_plane_eqs_c = np.matmul(T_c_w_inv_t, cur_plane_eqs.transpose()).transpose()
            cur_planes = cur_plane_eqs_c[:, 0:3] * (-cur_plane_eqs_c[:, 3:4])

            header_cam = Header()
            header_cam.seq = i
            header_cam.stamp = timestamp
            header_cam.frame_id = 'camera'

            color_msg = bridge.cv2_to_imgmsg(color, encoding="passthrough")
            color_msg.header = header_cam
            color_msg.encoding = 'bgr8'

            det_image_msg = bridge.cv2_to_imgmsg(det_image, encoding="passthrough")
            det_image_msg.header = header_cam
            det_image_msg.encoding = 'bgr8'

            depth_msg = bridge.cv2_to_imgmsg(depth, encoding="passthrough")
            depth_msg.header = header_cam
            depth_msg.encoding = 'mono16'

            obj_instances_msg = Serialized()
            obj_instances_msg.header = header_cam
            obj_instances_msg.type = 'vectorObjInstance'

            tss = np.array([timestamp.to_nsec()], dtype=np.uint64)
            tss = np.broadcast_to(tss, plane_ids.shape)

            obj_instances_msg.data = plpy.createObjInstanceViews(masks,
                                                                 cur_planes,
                                                                 plane_ids,
                                                                 tss,
                                                                 cur_descs,
                                                                 color,
                                                                 K,
                                                                 depth,
                                                                 depth_covar,
                                                                 not args.depth)

            point_cloud_pts_float = plpy.createPointCloud(masks,
                                                          cur_planes,
                                                          plane_ids,
                                                          tss,
                                                          cur_descs,
                                                          color,
                                                          K,
                                                          depth,
                                                          depth_covar,
                                                          not args.depth)
            # point_cloud_pts_float = np.squeeze(np.vsplit(point_cloud_pts_float, point_cloud_pts_float.shape[0]))
            point_cloud_pts = []
            for pt_idx in range(point_cloud_pts_float.shape[0]):
                pt_float = point_cloud_pts_float[pt_idx, :]
                r = int(round(pt_float[3]))
                g = int(round(pt_float[4]))
                b = int(round(pt_float[5]))
                rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                pt = (pt_float[0], pt_float[1], pt_float[2], rgba, int(round(pt_float[6])))
                # pt = (pt_float[0], pt_float[1], pt_float[2], rgba)
                point_cloud_pts.append(pt)
            point_cloud_msg = pc2.create_cloud(header_cam,
                                               point_cloud_fields,
                                               point_cloud_pts)

            T_gt = T_w_c
            T_vo = T_gt
            vo_corr = True

            tf_msg = TFMessage()

            # if odometry is correct
            if vo_corr >= 0:
                T_d = np.matmul(T_gt, np.linalg.inv(T_vo))
                v_d = matrix_to_vector(T_d)
                # print(v_d)

                trans_d_msg = create_trans_stamped_msg(i, 'map', 'odom',
                                                       header_cam.stamp,
                                                       v_d)

                trans_vo_msg = create_trans_stamped_msg(i, 'odom', 'camera',
                                                       header_cam.stamp,
                                                       matrix_to_vector(T_vo))

                # v_gt = matrix_to_vector(T_gt)
                # trans_gt_msg = create_trans_stamped_msg(i, 'map', 'camera',
                #                                        header_cam.stamp,
                #                                        v_gt)

                tf_msg.transforms = [trans_d_msg, trans_vo_msg]
                # tf_msg.transforms = [trans_gt_msg]

                T_vo_prev = T_vo
            else:
                # assuming odometry didn't change
                T_d = np.matmul(T_gt, np.linalg.inv(T_vo_prev))
                v_d = matrix_to_vector(T_d)

                trans_d_msg = create_trans_stamped_msg(i, 'map', 'odom',
                                                       header_cam.stamp,
                                                       v_d)

                tf_msg.transforms = [trans_d_msg]

            # transformation published before image, so it can be used for image transformation
            # o.write("/tf", tf_msg, tf_msg.transforms[0].header.stamp - rospy.Duration.from_sec(0.01 / float(args.freq)))
            o.write("/tf", tf_msg, tf_msg.transforms[0].header.stamp)
            o.write("/camera/color/image_raw", color_msg, header_cam.stamp)
            # o.write("/camera/det_image/image_raw", det_image_msg, header_cam.stamp)
            o.write("/camera/depth/image_raw", depth_msg, header_cam.stamp)
            o.write("/detector/objs", obj_instances_msg, header_cam.stamp)
            o.write("/detector/point_cloud_lab", point_cloud_msg, header_cam.stamp)

            pub_tf.publish(tf_msg)
            pub_color.publish(color_msg)
            pub_det_image.publish(det_image_msg)
            pub_depth.publish(depth_msg)
            pub_objs.publish(obj_instances_msg)
            pub_pc.publish(point_cloud_msg)

            next_id += 1000


if __name__ == "__main__":
    main()
