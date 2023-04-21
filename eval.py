import argparse
import logging

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from models import get_network

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')
    parser.add_argument('--input-size', type=int, default=480, help='Size of input image (must be a multiple of 8)')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--image-wise', action='store_true', help='Split the Cornell dataset image-wise')
    parser.add_argument('--random-seed', type=int, default=10, help='Random seed for dataset shuffling.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--stratified', action='store_true', help='Stratified')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--score-eval', action='store_true', help='Compute success based on score metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    ggcnn = get_network("peggnet")
    net = ggcnn(input_channels=1)
    net.load_state_dict(torch.load(args.network, map_location=torch.device("cuda:0")))
    net.eval()
    device = torch.device("cuda:0")
    net = net.to(device)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    
    test_dataset = Dataset(file_path=args.dataset_path, 
                           output_size=args.input_size, 
                           start=args.split, 
                           end=1.0, 
                           ds_rotate=args.ds_rotate,
                           image_wise=args.image_wise, 
                           random_seed=args.random_seed,
                           random_rotate=args.augment, 
                           random_zoom=args.augment,
                           include_depth=args.use_depth, 
                           include_rgb=args.use_rgb,
                           stratified=args.stratified)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logging.info('Number of test images: {}'.format(len(test_data)))
    logging.info('Done')

    results = {'correct': 0, 'failed': 0, 'total_score': 0.0, 'coverage_score': 0.0,
               'angle_score': 0.0, 'center_score': 0.0, "jacquard_correct": 0, "jacquard_wrong": 0,
               "polygonal_correct": 0, "polygonal_wrong": 0, 'total_score_jacquard': 0.0, 'coverage_score_jacquard': 0.0,
               'angle_score_jacquard': 0.0, 'center_score_jacquard': 0.0, 'total_score_polygonal': 0.0, 'coverage_score_polygonal': 0.0,
               'angle_score_polygonal': 0.0, 'center_score_polygonal': 0.0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if s:
                    results['correct'] += 1
                    if "Polygonal" in test_data.dataset.grasp_files[didx]:
                        results["polygonal_correct"] += 1
                    else:
                        results["jacquard_correct"] += 1
                        
                else:
                    results['failed'] += 1
                    if "Polygonal" in test_data.dataset.grasp_files[didx]:
                        results["polygonal_wrong"] += 1
                    else:
                        results["jacquard_wrong"] += 1
            if args.score_eval:
                size = (test_dataset.output_size, test_dataset.output_size)
                coverage_score, angle_score, center_score = evaluation.calculate_score(
                    q_img, 
                    ang_img, 
                    test_dataset.get_mask(idx), 
                    size, 
                    test_data.dataset.get_gtbb(didx, rot, zoom),
                    grasp_width=width_img,
                    debug_path=test_data.dataset.grasp_files[didx]
                    )
                score = (1/3) * coverage_score + (1/3) * angle_score + (1/3) * center_score
                logging.info("Image index %i scores", idx)
                print(test_data.dataset.grasp_files[didx])
                logging.info("Coverage Score: %.3f", coverage_score)
                logging.info("Angle Score: %.3f", angle_score)
                logging.info("Center Score: %.3f", center_score)
                logging.info("Overall Score: %.3f", score)          
                results['total_score'] += score      
                results['coverage_score'] += coverage_score
                results['angle_score'] += angle_score
                results['center_score'] += center_score
                if "Polygonal" in test_data.dataset.grasp_files[didx]:
                    results['total_score_polygonal'] += score      
                    results['coverage_score_polygonal'] += coverage_score
                    results['angle_score_polygonal'] += angle_score
                    results['center_score_polygonal'] += center_score
                else:
                    results['total_score_jacquard'] += score      
                    results['coverage_score_jacquard'] += coverage_score
                    results['angle_score_jacquard'] += angle_score
                    results['center_score_jacquard'] += center_score
                

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                # if "0_cube_3_cm" in test_data.dataset.grasp_files[didx]:
                evaluation.plot_output(test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                    test_data.dataset.get_depth(didx, rot, zoom), q_img,
                                    ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)
                                

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))
        if results['jacquard_wrong'] + results["jacquard_correct"] != 0:
            logging.info('IOU Results (jacquard): %d/%d = %f' % (results['jacquard_correct'],
                                results['jacquard_correct'] + results['jacquard_wrong'],
                                results['jacquard_correct'] / (results['jacquard_correct'] + results['jacquard_wrong'])))
        if results['polygonal_wrong'] + results["polygonal_correct"] != 0:
            logging.info('IOU Results (polygonal): %d/%d = %f' % (results['polygonal_correct'],
                                results['polygonal_correct'] + results['polygonal_wrong'],
                                results['polygonal_correct'] / (results['polygonal_correct'] + results['polygonal_wrong'])))
    if args.score_eval:
        size = len(test_data)
        jac_size = len(test_data.dataset.grasp_files_jacquard)
        poly_size = len(test_data.dataset.grasp_files_poly)
        logging.info("Mean Coverage Score: %f", results['coverage_score'] / size)
        logging.info("Mean Angle Score: %f", results['angle_score'] / size)
        logging.info("Mean Center Score: %f", results['center_score'] / size)
        logging.info("CAC Score: %f", results['total_score'] / size)
        if jac_size != 0:
            logging.info("Mean Coverage Score (jacquard): %f", results['coverage_score_jacquard'] / jac_size)
            logging.info("Mean Angle Score (jacquard): %f", results['angle_score_jacquard'] / jac_size)
            logging.info("Mean Center Score (jacquard): %f", results['center_score_jacquard'] / jac_size)
            logging.info("CAC Score (jacquard): %f", results['total_score_jacquard'] / jac_size)
        if poly_size != 0:
            logging.info("Mean Coverage Score (polygonal): %f", results['coverage_score_polygonal'] / poly_size)
            logging.info("Mean Angle Score (polygonal): %f", results['angle_score_polygonal'] / poly_size)
            logging.info("Mean Center Score (polygonal): %f", results['center_score_polygonal'] / poly_size)
            logging.info("CAC Score (polygonal): %f", results['total_score_polygonal'] / poly_size)

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))
