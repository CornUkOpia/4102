# COMP4102A Final Project - Generating the 3D Structures of Model Cars from Videos

This is the repository for the COMP4102 final project, made by Alexander Cornish (101053176) and Gabriel Valachi (101068875).

The final project writeup is in `FinalProjectWriteUp.pdf`, in the root directory.

The presentation for the final presentation, presented by Alexander, is in `COMP4102PRES.mp4`.

To see a demo of the 3D reconstruction code and the (not-very-impressive) results of that execution, check out `test-run.mp4`.

## Running the Project Code

### project.py
`project.py` contains the 3D reconstruction code. To run it, you require Python 3 and the following dependencies:
* OpenCV for Python (PIP: `opencv-python`)
** Installed OpenCV libraries may also be needed
* Numpy (PIP: `numpy`)
* Matplotlib (PIP: `matplotlib`)
* Pillow (PIP: `pillow`)

To install a Python dependency with PIP, run `python -m pip install <package>`, where `<package>` is one of the package names following `PIP:` above.

Run `project.py` like so:

  `python project.py <source-video-or-gif> [frame-interval]`

Arguments:
* `<source-video-or-gif>` is the source video or GIF file from which to attempt to construct a 3D point cloud.
* `[frame-interval]` is the number of frames skipped between images. For instance, an interval of 2 means the first image pair uses the first and third frames, the second image pair uses the third and fifth frames, and so on.

Note the following files and directories:
* `outputImages/` contains the image sequence generated from the input video.
* `featureMatched/` contains the image pairs with features matched, for verification purposes.
* `graphs/` contains the point clouds generated for each image pair.
* `point-cloud.png` is the final point cloud. It should be generated in the same directory where you ran `project.py`.

### calibrate-camera.cpp
`calibrate-camera.cpp` contains the calibration code. To build it and run it, you need to have OpenCV installed on your system.

To build it on Linux, run the following command:

  `g++ -o calibrate-camera calibrate-camera.cpp /usr/bin/libopencv_*`

Run the generated binary like so:

  `./calibrate-camera`

Executable takes no arguments.

Values such as chessboard dimensions and number of calibration pictures are hardcoded in `calibrate-camera.cpp`.

Note the following files and directories:
* `calibPics/` contains the calibration pictures, each file being a picture of a chessboard taken at a different angle.
* `calibPics_out/` contains the calibration pictures with the detected chessboard corners.
* `calibMatrix.xml` contains the resulting camera matrix and distortion coefficients.


# Proposal: Generating the 3D Structures of Model Cars from Videos

**NOTE**: What follows is the original project proposal, and it is severely out of date. We had to take a radically different approach, that approach being Structure from Motion, because we realized our original method was not going to work. Unfortunately, the project still turned out to be unsuccessful, because we severely underestimated the difficulty of our task. Please also note that we did not adhere to our original schedule due to work in COMP4102 and other classes, and due to the above-mentioned shift in the project's direction.

## Team Members
Gabriel Valachi (101068875)
Alexander Cornish (101053176)

## Summary
The plan for this project is to create a program that takes as input a video of a toy car - with the possibility of using real-time video as input instead of a prerecorded one - and attempts to construct a 3D model of said recorded car. Currently, we plan to make the technology work on model cars, 1:18 and 1:24 scale ones, with the possible application of the technology to other non-model car related objects. The accomplishment of this project is presumed to require edge detection, corner detection, feature tracking, and depth detection/estimation; however, other technologies might be required to complete this project.

## Background
Though known to have been implemented before by various people and institutions, this project idea does not originate from any specific existing research. The field of photogrammetry has been reasonably well developed in the commercial sector, but there is no singular specific project on which this will be based. However, some components of the real-world technology will be replicated on a much lower budget. For instance, the mechanism for taking a stable video of an item would be a really expensive camera rig in a commercial environment, but for the purposes of this project, we will be using a stationary mobile phone camera and have the object resting on a lazy susan.

## The Challenge
With this project, we are hoping to learn about detecting edges, corners, tracking features, and using shared information from multiple images to estimate a 3D structure.

OpenCV provides some functions to accomplish some necessary steps, such as edge detection, and even some advanced functions such as feature detection. However, in the interest of learning about and implementing edge detection, corner detection, feature tracking, and depth estimation, we will try to avoid using these preexisting functions unless we are running out of time. OpenCV does not seem to provide any functions to reconstruct a 3D objectfrom multiple images, and even if it was present, to use such a function would be antithetical to the project goals.

## Goals and Deliverables
What we plan to achieve:
* Create a program that accepts a prerecorded video of an object and outputs a 3D model of it.

What we hope to achieve:
* Create a program that can take footage of an object and output a 3D model of it, with both the recording and the model generating in real time.

Measuring success:
* Success in this project will be measured in two ways: amount of time it takes to process and how accurate the resulting 3D model is.
* Accuracy is prioritized over processing speed in the prerecorded video; however, for the real-time project to work, this might require some sacrifice of accuracy for performance.
How realistic is it?
* This project is reasonably realistic, considering similar programs have been done before.
* However, the real-time model generation as a live video is being recorded may be infeasible, depending on the hardware requirements necessary for such a task.
* If things progress slower than expected, we may abandon our secondary objective of real-time model construction.

## Schedule
NOTE: This is primarily an outline, and this schedule may not be strictly adhered to depending on the circumstances behind project development.

February 6th to February 12th
* Laying the groundwork that will be necessary for the ongoing part of this project. This includes code that loads a video, extracts the frames from the video, and prepares them for use in the project.

February 13th to February 19th
* Starting the development of the edge detection, corner detection, and depthdetection features.

February 20th to February 26th
* Continuing development of the aforementioned features.

February 27th to March 5th
* Creating whatever code is required to combine the resulting images once they have been through the edge detection, corner detection, and depth detection features.

March 6th to March 12th
* Continuing with the creation of the code for the image to model process, including research into what we would need to use to display a 3d model onscreen.

March 13th to March 19th
* Debugging of the first stage of the project, includes testing on models of varying size, shape, detail, and colour.

March 20th to March 26th
* Working on the real-time footage code, including figuring out how to get real time footage from a phone to a computer, improving performance if needed, and handling other issues related to using real time footage.

March 27th to April 2nd
* Continuing work on the real-time footage code.

April 3rd to April 9th
* Finishing touches on the project. Improving the displaying of the resulting images maybe with a side-by-side of the real time footage and the resulting 3D model, or something else that would improve the presentation of the project.

April 10th to April 14th
* Project will be presented.
* Project should be finished by this point. If not, there should only be some finishing touches left to cover.
