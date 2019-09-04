# Spaces Dataset

We have collected 100 scenes of data for training and testing. The data was
collected with a 16 camera [rig](docs/rigs.pdf). For each scene we captured between 3 and 10
different rig positions. The rig positions are all relatively close together, so
one rig position can be used as the input to a model, with an image from a
different rig position used as the target image.

Each scene is stored in a separate directory with the following structure:

    scene_name/
      models.json
      cam_00/
        image_000.JPG
        image_001.JPG
        ...
        image_00N.JPG
      cam_01/
        image_000.JPG
        image_001.JPG
        ...
        image_00N.JPG
      ...
      cam_15
        image_000.JPG
        image_001.JPG
        ...
        image_00N.JPG

Each scene has a sub directory for each camera in the rig, and each camera
directory contains the images for that camera. The index of the image is the
index of the rig position that generated that image. Meaning
cam_00/image_000.JPG and cam_01/image_000.JPG were captured at the 0th rig
position.

The calibration is stored in the models.json file in the following format:

    for each rig position:
      for each of the 16 cameras in the rig, a dictionary containing:
        principal_point: Principal point of the camera in pixels.
        pixel_aspect_ratio: Aspect ratio of the camera.
        position: 3 element position of the camera, in meters.
        focal_length: Effective focal length of the camera.
        width: Width of the image.
        height: Height of the image.
        relative_path: Location of the image for this camera relative to the scene
          directory, e.g. 'cam_00/image_000.JPG'
        orientation: 3 element axis-angle representation of the rotation of
          the camera. The length of the axis gives the rotation angle, in
          radians.

The function ReadScene in utils.py will read the extrinsics and image paths for
all rigs in a scene. ReadViewImages(views) can be used to read the images for a
rig position.
