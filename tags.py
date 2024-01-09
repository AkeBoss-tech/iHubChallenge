#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import numpy as np

import cv2 as cv
from pupil_apriltags import Detector

DISTANCE_AREA = 1
TOP_BUFFER = 0.2

def polygon_area(x, y):
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Parameters:
    - x: List of x-coordinates of polygon vertices
    - y: List of y-coordinates of polygon vertices
    
    Returns:
    - Area of the polygon
    """
    n = len(x)
    area = 0.5 * abs(sum(x[i] * (y[(i + 1) % n] - y[(i - 1) % n]) for i in range(n)))
    return area

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Detector
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    tag_size = 0.1  # replace with the true physical size of the AprilTag in meters

    elapsed_time = 0

    while True:
        start_time = time.time()

        ret, image = cap.read()
        if not ret:
            break
        # get image size
        cap_width = image.shape[1]
        cap_height = image.shape[0]

        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=tag_size,
        )

        debug_image = draw_tags(debug_image, tags, elapsed_time, cap_width, cap_height)

        elapsed_time = time.time() - start_time

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('AprilTag Detect Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
    cap_width=960,
    cap_height=540,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # Draw lines between projected 3D points
        for i in range(len(image_points)):
            p1 = tuple(map(int, image_points[i - 1].ravel()))
            p2 = tuple(map(int, image_points[i].ravel()))
            cv.line(image, p1, p2, (255, 0, 0), 2)
                
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

        # Calculate area of the tag based on corners
        area = polygon_area([corner_01[0], corner_02[0], corner_03[0], corner_04[0]],
                            [corner_01[1], corner_02[1], corner_03[1], corner_04[1]])
        print(area)

        distance = 100 / (area ** 0.5)
        distance_str = f"Distance: {distance:.2f} m"
        cv.putText(image, distance_str, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)

        # Display arrows
        arrow_left = "<"
        arrow_right = ">"
        arrow_up = "^"
        arrow_down = "v"

        choice = ""
        if distance > DISTANCE_AREA + TOP_BUFFER:
            choice = arrow_up            
        elif distance < DISTANCE_AREA - TOP_BUFFER: 
            choice = arrow_down

        turn = ""
        if center[0] > cap_width / 2 - 50:
            turn = arrow_left
        elif center[0] < cap_width / 2 + 50:
            turn = arrow_right

        cv.putText(image, turn, (int(center[0]), 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv.LINE_AA)
        cv.putText(image, choice, (int(center[0]), cap_height - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv.LINE_AA)
        # cv.putText(image, arrow_down, (int(center[0]), cap_height - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # cv.putText(image, arrow_left, (10, int(center[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        

    """ cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA) """

    return image

if __name__ == '__main__':
    main()
