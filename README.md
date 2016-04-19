# calib_video_opencv

Powerful console application for calibration (including stereo calibration) of cameras from video files based on the OpenCV library.

###News Update

The master now the upstream development branch. Use tags checkout more stable versions.

###What the heck is *sync_based_on_audio.py*?

This is for automated video syncing using sound. Sound has very high temporal resolution, much higher than video. In case the videos for calibration are obtained using cameras that were not genlocked or synchronized in any way, this utility can save you a lot of time spent finding the corresponding frames in the two videos and manually editing them to synchronize them. It finds the offset between the videos by matching groupings of frequency peaks in their audio, seeks out the calibration board (optionally) to determine what to cut off from the start and the end, and automatically recodes the videos for you to input into the provided calibration script. The offset-finding is adapted from Allison Deal's [VideoSync] (https://github.com/allisonnicoledeal/VideoSync).

###What's so powerful about *calibrate_video_opencv.py*?

It allowes you to set various ways to filter off unwanted frames. The most critical is `--frame_count_target=X` or `--ft=X`, where X is an integer, representing approximately how many frames you want to cherry-pick for the calibration. The reason this number is important is that runtime of the OpenCV calibration routine increases with the number of frames you pass it in a faster-than-linear way, i.e. conisder a I7-4790K CPU taking about 8 hours to calibrate based on 200 frames. Specifying the target frame number will cause the frame gathering algorithm to skip over even intervalls in the video(s) before sifting through frames to pick out the next one to sample.

Other filters include frame sharpness (calculated as variance of the image convolved with the Laplacian operator), minimum raw pixel difference from the previously-sampled frame, and a manual filter that uses OpenCV GUI routines to display the images to be accepted or rejected via keyboard strokes.

Another useful feature is saving/loading of cherry-picked frames and/or checkerboard inner corner positions detected in the cherry-picked frames. This allows to skip the gathering process when re-calibrating with different settings.

Finally, because there are so many command-line options, there is a simple way to save all the settings in a YAML settings file, to avoid re-entering them later. The setting file may subsequently be overridden by and/or updated with the alternative settings passed as command-line arguments.

###Requirements
*calibrate_video_opencv.py*:
* [Python 3.4](https://www.python.org/) or newer
* [OpenCV 3](http://opencv.org/) built with Python 3 support and properly installed.
* [Lxml](http://lxml.de/) *
* [PyYaml](http://pyyaml.org/) *

*sync_based_on_audio.py*:
* All of the above
* [FFmpeg](https://www.ffmpeg.org/) binaries
* [Scipy](http://www.scipy.org/) *

Using [pip](https://pypi.python.org/pypi/pip), the python packages (marked with '*') can be installed via the following commands:
* `pip install lxml`
* `pip install pyyaml`
* `pip install scipy`

**Note**: on many Linux distributions, i.e. Ubuntu-based, pip for python3 is envoked with the `pip3` command instead of `pip`. It is also recommended to install `libyaml` prior to installing `pyyaml` if it is easy to do so; on Debian-based you can try `sudo apt-get install libyaml-dev`. For `scipy` installation, a fortran compiler is required (`sudo apt-get install gfortran` for Debian-based). 

**Note for Windows Users**: Windows users are recommended to get the required python binaries and their dependencies, including Numpy, at Christoph Gohlke's page [Unofficial Python Extensions for Winodws](http://www.lfd.uci.edu/~gohlke/pythonlibs/). These can be installed via the `pip wheel <package_file_path>` command from a command prompt with administrative privileges.

###Usage

*calibrate_video_opencv.py*:
See output of `python calib_video_opencv.py --help` (again, python 3 is usually envoked via `python3` on Linux). In most Linux/Unix shells, you can also run `./calib_video_opencv.py --help` provided you grant the file necessary permissions.

*calibrate_video_opencv.py*:
See output of `python sync_based_on_audio.py --help`, all notes above for the calibration script also apply.

#####The provided calibration board

The default calibration board provided (checkerboard.pdf) is a small 9x6 checkerboard that can be easily printed on 8.5x11" US Letter paper or standard A4 paper. Print without scaling, and double-check resulting square size (against the default program settings). Checkerboard square dimensions and size can be set as command-line arguments or via settings file (see above and below).

#####Using the resulting calibration file

The resulting calibration file can be read back in by adapting the same python code (check the XML module), but the format is also fully-compatible with OpenCV's XML input-output utilities, so you can read it from your C++ OpenCV applications or libraries. Here is some ugly C++ code that does that for your convenience:

```C++
include <opencv2/core.hpp>

//.....
// then, in some function:
cv::FileStorage fs(path, cv::FileStorage::READ);

cv::FileNode stereo_calib_node = fs["Rig"];
cv::FileNode cameras_node = stereo_calib_node["Cameras"];
cv::FileNode camera_0_node = cameras_node[0];
cv::FileNode camera_1_node = cameras_node[1];
cv::FileNode intrinsics_0_node = camera_0_node["Intrinsics"]
cv::FileNode intrinsics_1_node = camera_1_node["Intrinsics"]
cv::FileNode extrinsics_node = camera_1_node["Extrinsics"]

cv::Mat K0, d0, K1, d1, R, T;
cv::Size im_size;

intrinsics_0_node["intrinsic_mat"] >> K0;
intrinsics_0_node["distortion_coeffs"] >> d0;
intrinsics_1_node["intrinsic_mat"] >> K1;
intrinsics_1_node["distortion_coeffs"] >> d1;
extrinsics_node["rotation"] >> R;
extrinsics_node["translation"] >> T;

im_size = cv::Size(static_cast<int>(intrinsics_0_node["resolution"]["width"]),
			static_cast<int>(intrinsics_0_node["resolution"]["height"]));
```


###Calibration Tips
Calibration experts: skip this section.

The provided tiny calibration board will only work well for calibrating at short distances (within half a meter or so). I recommend a larger calibration board, with larger and more squares for greater distances. Any calibration board should be snugly mounted on a completely flat, unbending surface. During calibration, aim for variety of angles and positions (including depth) of the board within the image frame. Calibration of cameras with autofocus is not supported, since the algorithm assumes camera intrinsics (inculding focal distance) are static. On such cameras, you have to find a way to fix the focus. Also, keep in mind, calibration process does not yield acutal focal length (look to your camera manufacturer for that information, as well as the actual metric size of the image sensor).

**Happy calibration!**
