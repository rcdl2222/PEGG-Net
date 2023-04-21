import numpy as np
import matplotlib.pyplot as plt

from .grasp import GraspRectangles, detect_grasps


def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img.T)
    # for g in gs:
    #     g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    plot = ax.imshow(grasp_width_img, cmap='gray')
    # for g in gs:
    #     g.plot(ax)
    ax.set_title('Grasp Width')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Quality')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Grasp Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.savefig("sample_gqn_output.png")
    # plt.show()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False

def coverage_linear_func(coverage):
    return coverage / 0.75 if coverage <= 0.75 else  -(4 * coverage) + 4

def calculate_score(grasp_q, grasp_angle, mask_img, input_size, ground_truth_bbs, no_grasps=1, grasp_width=None, debug_path=None):
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    curr_score_sum = 0.0
    scores = (0.0, 0.0, 0.0)
    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    for g in gs:
        gr = g.as_gr
        coverage_score = calculate_coverage(gr, mask_img, input_size, debug_path=debug_path)
        angle_score, center_score = calculate_angle_and_center_score(gr, gt_bbs)
        score_sum = coverage_score + angle_score + center_score
        if score_sum > curr_score_sum:
            scores = (coverage_score, angle_score, center_score)
    return scores

def calculate_coverage(gr, mask_img, input_size, debug_path=None):
    grasp_mask = np.zeros(input_size)
    rr, cc = gr.polygon_coords(input_size)


    grasp_mask[rr, cc] = 1.0
    intersection = np.minimum(grasp_mask, mask_img)
    if np.count_nonzero(grasp_mask) == 0:
        return 0.0
    coverage = np.count_nonzero(intersection) / np.count_nonzero(grasp_mask)
    print("Coverage: "+ str(coverage))
    return coverage_linear_func(coverage)

def calculate_angle_and_center_score(gr, gt_bbs):
    highest_combined_score = 0.0
    highest_separate_score = (0.0, 0.0)
    for gt_gr in gt_bbs:
        min_angle_diff = np.pi / 6
        dist_proportion = (np.linalg.norm(abs(gr.center - gt_gr.center)) / np.sqrt(gt_gr.length ** 2 + gt_gr.width ** 2))
        if dist_proportion > 1.0:
            # Do not consider score when center distances are larger than diagonal of grasp rectangle
            continue
        curr_angle_diff = abs((gr.angle - gt_gr.angle + np.pi/2) % np.pi - np.pi/2)
        curr_center_diff = 1 - dist_proportion
        if curr_angle_diff < (np.pi / 6):
            min_angle_diff = curr_angle_diff
        combined_score = 1 - (min_angle_diff / (np.pi / 6)) + curr_center_diff
        if combined_score > highest_combined_score:
            highest_separate_score = (1 - (min_angle_diff / (np.pi / 6)), curr_center_diff)
            highest_combined_score = combined_score
    return highest_separate_score
