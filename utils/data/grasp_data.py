import numpy as np
from PIL import Image
import tqdm
import torch
import torch.utils.data

import random


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=480, include_depth=True, include_rgb=False, random_rotate=False,  # changed from 300 to 480
                 random_zoom=False, input_only=False, use_saved_grasp_map=False, min_angle=False, max_width=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = False
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.use_saved_grasp_map = use_saved_grasp_map
        self.min_angle = min_angle
        self.max_width = max_width

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()
    
    def get_grasp_map(self, idx):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        if self.use_saved_grasp_map:
            pos_img, ang_img, width_img = self.get_grasp_map(idx, rot, zoom_factor)
            # if self.min_angle:
            #     ang_img = self.get_min_angle(idx, rot, zoom_factor)
            # if self.max_width:
            #     width_img = self.get_max_width(idx, rot, zoom_factor)
        
        else:
            bbs = self.get_gtbb(idx, rot, zoom_factor)
            pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
            if self.min_angle:
                ang_img = self.get_min_angle(idx, rot, zoom_factor)
            if self.max_width:
                width_img = self.get_max_width(idx, rot, zoom_factor)

        width_img = np.clip(width_img, 0.0, 150.0)/150.0

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
            
            # rgbd_img = np.concatenate(
            #         (np.expand_dims(depth_img, 0),
            #          rgb_img),
            #         0
            #     )
            # print("{},{},{},{}".format(rgbd_img.max(), rgbd_img.min(), np.mean(rgbd_img), np.std(rgbd_img)))

        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)

    def save_grasp_map_images(self):

        for i in tqdm.tqdm(range(len(self))):
            pos_path = self.grasp_files[i].replace("grasps.txt", "grasp_quality.tiff")
            ang_path = self.grasp_files[i].replace("grasps.txt", "grasp_angle.tiff")
            width_path = self.grasp_files[i].replace("grasps.txt", "grasp_width.tiff")

            bbs = self.get_gtbb(i)

            pos_out = np.zeros((self.output_size, self.output_size))
            ang_out = np.zeros((self.output_size, self.output_size))
            width_out = np.zeros((self.output_size, self.output_size))
            
            all_grs_pos_out = np.zeros((self.output_size, self.output_size, len(bbs.grs)))
            all_grs_ang_out = np.zeros((self.output_size, self.output_size, len(bbs.grs)))
            all_grs_width_out = np.zeros((self.output_size, self.output_size, len(bbs.grs)))

            for i, gr in enumerate(bbs.grs):
                rr, cc = gr.compact_polygon_coords((self.output_size, self.output_size))
                pos_out = gr.normalize_rect(pos_out, rr, cc)
                all_grs_pos_out[:, :, i] = pos_out
                ang_out[rr, cc] = gr.angle
                all_grs_ang_out[:, :, i] = ang_out
                width_out[rr, cc] = gr.length
                all_grs_width_out[:, :, i] = width_out
                pos_out = np.zeros((self.output_size, self.output_size))
                ang_out = np.zeros((self.output_size, self.output_size))
                width_out = np.zeros((self.output_size, self.output_size))

            pos_out = all_grs_pos_out.max(2)
            max_pos_ind = np.argmax(all_grs_pos_out, axis=2)
            ang_out = np.take_along_axis(all_grs_ang_out, max_pos_ind.reshape(self.output_size, self.output_size, 1), axis=2)
            width_out = np.take_along_axis(all_grs_width_out, max_pos_ind.reshape(self.output_size, self.output_size, 1), axis=2)
                
            pos_img = Image.fromarray(pos_out)
            ang_out = ang_out.astype(np.float32).reshape((self.output_size, self.output_size))
            ang_img = Image.fromarray(ang_out)
            width_out = width_out.astype(np.float32).reshape((self.output_size, self.output_size))
            width_img = Image.fromarray(width_out)

            pos_img.save(pos_path)
            ang_img.save(ang_path)
            width_img.save(width_path)

    def save_min_angle_map(self):
        import os
        for i in tqdm.tqdm(range(len(self))):
            ang_path = self.grasp_files[i].replace("grasps.txt", "min_angle.tiff")
            if os.path.exists(ang_path):
                os.remove(ang_path)

            bbs = self.get_gtbb(i)

            ang_out = np.full((self.output_size, self.output_size), 10.0)
            
            all_grs_ang_out = np.full((self.output_size, self.output_size, len(bbs.grs)), 10.0)

            for i, gr in enumerate(bbs.grs):
                rr, cc = gr.compact_polygon_coords((self.output_size, self.output_size))
                ang_out[rr, cc] = gr.angle
                all_grs_ang_out[:, :, i] = ang_out
                ang_out = np.full((self.output_size, self.output_size), 10.0)

            ang_out = all_grs_ang_out.min(2)
            ang_out[ang_out == 10.0] = 0.0
                
            ang_out = ang_out.astype(np.float32).reshape((self.output_size, self.output_size))
            ang_img = Image.fromarray(ang_out)

            ang_img.save(ang_path)
    
    def save_max_width_map(self):
        import os
        for i in tqdm.tqdm(range(len(self))):
            width_path = self.grasp_files[i].replace("grasps.txt", "max_width.tiff")
            if os.path.exists(width_path):
                os.remove(width_path)

            bbs = self.get_gtbb(i)

            width_out = np.full((self.output_size, self.output_size), -10.0)
            
            all_grs_width_out = np.full((self.output_size, self.output_size, len(bbs.grs)), -10.0)

            for i, gr in enumerate(bbs.grs):
                rr, cc = gr.compact_polygon_coords((self.output_size, self.output_size))
                width_out[rr, cc] = gr.length
                all_grs_width_out[:, :, i] = width_out
                ang_out = np.full((self.output_size, self.output_size), -10.0)

            width_out = all_grs_width_out.max(2)
            width_out[width_out == -10.0] = 0.0
                
            width_out = width_out.astype(np.float32).reshape((self.output_size, self.output_size))
            width_out = Image.fromarray(width_out)

            width_out.save(width_path)
