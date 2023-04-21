import os
import glob
import random
import numpy as np

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


class JacquardDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, image_wise=False, random_seed=10, stratified=False, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(JacquardDataset, self).__init__(**kwargs)
        random.seed(random_seed)
        self.grasp_files_jacquard = []
        self.grasp_files_poly = []
        if stratified:
            self.grasp_files = []
            self.depth_files = []
            self.rgb_files = []
            self.mask_files = []
            graspf = glob.glob(os.path.join(file_path, '*', '*', '*_grasps.txt'))
            graspf.sort()
            l = len(graspf)
            if l == 0:
                raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
            graspf_jacquard = list(filter(lambda x: "Polygonal_Dataset" not in x, graspf))
            graspf_poly = list(filter(lambda x: "Polygonal_Dataset" in x, graspf))
            random.shuffle(graspf_poly)
            l_jacquard = len(graspf_jacquard)
            l_poly = len(graspf_poly)
            if ds_rotate:
                graspf_jacquard = graspf_jacquard[int(l*ds_rotate):] + graspf_jacquard[:int(l*ds_rotate)]
                graspf_poly = graspf_poly[int(l*ds_rotate):] + graspf_poly[:int(l*ds_rotate)]

            depthf_jacquard = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf_jacquard]
            rgbf_jacquard = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf_jacquard]
            maskf_jacquard = [f.replace('perfect_depth.tiff', 'mask.png') for f in depthf_jacquard]

            depthf_poly = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf_poly]
            rgbf_poly = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf_poly]
            maskf_poly = [f.replace('perfect_depth.tiff', 'mask.png') for f in depthf_poly]

            self.grasp_files_jacquard = graspf_jacquard[int(l_jacquard*start):int(l_jacquard*end)]
            self.depth_files_jacquard = depthf_jacquard[int(l_jacquard*start):int(l_jacquard*end)]
            self.rgb_files_jacquard = rgbf_jacquard[int(l_jacquard*start):int(l_jacquard*end)]
            self.mask_files_jacquard = maskf_jacquard[int(l_jacquard*start):int(l_jacquard*end)]

            self.grasp_files_poly = graspf_poly[int(l_poly*start):int(l_poly*end)]
            self.depth_files_poly = depthf_poly[int(l_poly*start):int(l_poly*end)]
            self.rgb_files_poly = rgbf_poly[int(l_poly*start):int(l_poly*end)]
            self.mask_files_poly = maskf_poly[int(l_poly*start):int(l_poly*end)]

            self.grasp_files.extend(self.grasp_files_jacquard + self.grasp_files_poly)
            self.depth_files.extend(self.depth_files_jacquard + self.depth_files_poly)
            self.rgb_files.extend(self.rgb_files_jacquard + self.rgb_files_poly)
            self.mask_files.extend(self.mask_files_jacquard + self.mask_files_poly)
            if self.min_angle:
                self.angle_files = []
                anglef = [f.replace('perfect_depth.tiff', 'min_angle.tiff') for f in depthf]
                self.angle_files_jacquard = anglef[int(l_jacquard*start):int(l_jacquard*end)]
                self.angle_files_poly = anglef[int(l_poly*start):int(l_poly*end)]
                self.angle_files.extend(self.angle_files_jacquard + self.angle_files_poly)
            
            if self.max_width:
                self.width_files = []
                widthf = [f.replace('perfect_depth.tiff', 'max_width.tiff') for f in depthf]
                self.width_files_jacquard = widthf[int(l_jacquard*start):int(l_jacquard*end)]
                self.width_files_poly = widthf[int(l_poly*start):int(l_poly*end)]
                self.width_files.extend(self.width_files_jacquard + self.width_files_poly)

            if self.use_saved_grasp_map:
                self.quality_files = []
                self.angle_files = []
                self.width_files = []
                
                qualityf_jacquard = [f.replace('perfect_depth.tiff', 'grasp_quality.tiff') for f in depthf_jacquard]
                qualityf_poly = [f.replace('perfect_depth.tiff', 'grasp_quality.tiff') for f in depthf_poly]
                if self.min_angle:
                    anglef_poly = [f.replace('perfect_depth.tiff', 'min_angle.tiff') for f in depthf_poly]
                    anglef_jacquard = [f.replace('perfect_depth.tiff', 'min_angle.tiff') for f in depthf_jacquard]
                else:
                    anglef_poly = [f.replace('perfect_depth.tiff', 'grasp_angle.tiff') for f in depthf_poly]
                    anglef_jacquard = [f.replace('perfect_depth.tiff', 'grasp_angle.tiff') for f in depthf_jacquard]
                
                if self.max_width:
                    widthf_poly = [f.replace('perfect_depth.tiff', 'max_width.tiff') for f in depthf_poly]
                    widthf_jacquard = [f.replace('perfect_depth.tiff', 'max_width.tiff') for f in depthf_jacquard]
                else:
                    widthf_poly = [f.replace('perfect_depth.tiff', 'grasp_width.tiff') for f in depthf_poly]
                    widthf_jacquard = [f.replace('perfect_depth.tiff', 'grasp_width.tiff') for f in depthf_jacquard]

                self.quality_files_jacquard = qualityf_jacquard[int(l_jacquard*start):int(l_jacquard*end)]
                self.angle_files_jacquard = anglef_jacquard[int(l_jacquard*start):int(l_jacquard*end)]
                self.width_files_jacquard = widthf_jacquard[int(l_jacquard*start):int(l_jacquard*end)]

                self.quality_files_poly = qualityf_poly[int(l_poly*start):int(l_poly*end)]
                self.angle_files_poly = anglef_poly[int(l_poly*start):int(l_poly*end)]
                self.width_files_poly = widthf_poly[int(l_poly*start):int(l_poly*end)]

                self.quality_files.extend(self.quality_files_jacquard + self.quality_files_poly)
                self.angle_files.extend(self.angle_files_jacquard + self.angle_files_poly)
                self.width_files.extend(self.width_files_jacquard + self.width_files_poly)

            
            print("Jacquard images number: " + str(len(self.grasp_files_jacquard)))
            print("Polygonal images number: " + str(len(self.grasp_files_poly)))
            #print("Jacquard images quality files number: " + str(len(self.quality_files_jacquard)))
            #print("Polygonal images quality files number: " + str(len(self.quality_files_poly)))
        else:
            graspf = glob.glob(os.path.join(file_path, '*', '*', '*_grasps.txt'))
            graspf.sort()
            l = len(graspf)

            if l == 0:
                raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

            if ds_rotate:
                graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

            depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
            rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
            maskf = [f.replace('perfect_depth.tiff', 'mask.png') for f in depthf]

            self.grasp_files = graspf[int(l*start):int(l*end)]
            self.depth_files = depthf[int(l*start):int(l*end)]
            self.rgb_files = rgbf[int(l*start):int(l*end)]
            self.mask_files = maskf[int(l*start):int(l*end)]

            if self.min_angle:
                anglef = [f.replace('perfect_depth.tiff', 'min_angle.tiff') for f in depthf]
                self.angle_files = anglef[int(l*start):int(l*end)]
            
            if self.max_width:
                widthf = [f.replace('perfect_depth.tiff', 'max_width.tiff') for f in depthf]
                self.width_files = widthf[int(l*start):int(l*end)]

            if self.use_saved_grasp_map:
                qualityf = [f.replace('perfect_depth.tiff', 'grasp_quality.tiff') for f in depthf]
                if self.min_angle:
                    anglef = [f.replace('perfect_depth.tiff', 'min_angle.tiff') for f in depthf]
                else:
                    anglef = [f.replace('perfect_depth.tiff', 'grasp_angle.tiff') for f in depthf]
                if self.max_width:
                    widthf = [f.replace('perfect_depth.tiff', 'max_width.tiff') for f in depthf]
                else:
                    widthf = [f.replace('perfect_depth.tiff', 'grasp_width.tiff') for f in depthf]
            
                self.quality_files = qualityf[int(l*start):int(l*end)]
                self.angle_files = anglef[int(l*start):int(l*end)]
                self.width_files = widthf[int(l*start):int(l*end)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # print("{},{},{},{}".format(depth_img.max(), depth_img.min(), np.mean(depth_img), np.std(depth_img)))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # print("{},{},{},{}".format(rgb_img.max(), rgb_img.min(), np.mean(rgb_img), np.std(rgb_img)))
        return rgb_img.img
    
    def get_grasp_map(self, idx, rot=0, zoom=1.0):
        quality_img = image.DepthImage.from_tiff(self.quality_files[idx])
        quality_img.img = quality_img.img.astype(np.float64)
        quality_img.rotate(rot)
        quality_img.zoom(zoom)
        quality_img.resize((self.output_size, self.output_size))

        angle_img = image.DepthImage.from_tiff(self.angle_files[idx])
        angle_img.img = angle_img.img.astype(np.float64)
        angle_img.rotate(rot)
        angle_img.zoom(zoom)
        angle_img.resize((self.output_size, self.output_size))

        width_img = image.DepthImage.from_tiff(self.width_files[idx])
        width_img.img = width_img.img.astype(np.float64)
        width_img.rotate(rot)
        width_img.zoom(zoom)
        width_img.resize((self.output_size, self.output_size))

        return quality_img.img, angle_img.img, width_img.img

    def get_min_angle(self, idx, rot=0, zoom=1.0):

        angle_img = image.DepthImage.from_tiff(self.angle_files[idx])
        angle_img.img = angle_img.img.astype(np.float64)
        angle_img.rotate(rot)
        angle_img.zoom(zoom)
        angle_img.resize((self.output_size, self.output_size))
        return angle_img.img
    
    def get_max_width(self, idx, rot=0, zoom=1.0):
        width_img = image.DepthImage.from_tiff(self.width_files[idx])
        width_img.img = width_img.img.astype(np.float64)
        width_img.rotate(rot)
        width_img.zoom(zoom)
        width_img.resize((self.output_size, self.output_size))
        return width_img.img

    def get_mask(self, idx, rot=0, zoom=1.0):
        mask_img = image.Image.from_file(self.mask_files[idx])
        mask_img.rotate(rot)
        mask_img.zoom(zoom)
        mask_img.resize((self.output_size, self.output_size))
        mask_img.img = mask_img.img.astype(np.float32)/255.0
        return mask_img.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])
