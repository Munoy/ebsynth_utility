import cv2
import os
import glob
import shutil
import numpy as np
import math
import mediapipe as mp

#---------------------------------
# Copied from PySceneDetect
def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)


def estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size

_kernel = None

def _detect_edges(lum: np.ndarray) -> np.ndarray:
    global _kernel
    """Detect edges using the luma channel of a frame.
    Arguments:
        lum: 2D 8-bit image representing the luma channel of a frame.
    Returns:
        2D 8-bit image of the same size as the input, where pixels with values of 255
        represent edges, and all other pixels are 0.
    """
    # Initialize kernel.
    if _kernel is None:
        kernel_size = estimated_kernel_size(lum.shape[1], lum.shape[0])
        _kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Estimate levels for thresholding.
    sigma: float = 1.0 / 3.0
    median = np.median(lum)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
    # This increases edge overlap leading to improved robustness against noise and slow
    # camera movement. Note that very large kernel sizes can negatively affect accuracy.
    edges = cv2.Canny(lum, low, high)
    return cv2.dilate(edges, _kernel)

#---------------------------------

def detect_edges(img_path, mask_path, is_invert_mask):
    im = cv2.imread(img_path)
    if mask_path:
        mask = cv2.imread(mask_path)[:,:,0]
        mask = mask[:, :, np.newaxis]
        im = im * ( (mask == 0) if is_invert_mask else (mask > 0) )
#        im = im * (mask/255)
#        im = im.astype(np.uint8)
#        cv2.imwrite( os.path.join( os.path.dirname(mask_path) , "tmp.png" ) , im)

    hue, sat, lum = cv2.split(cv2.cvtColor( im , cv2.COLOR_BGR2HSV))
    return _detect_edges(lum)

def get_mask_path_of_img(img_path, mask_dir):
    img_basename = os.path.basename(img_path)
    mask_path = os.path.join( mask_dir , img_basename )
    return mask_path if os.path.isfile( mask_path ) else None

def analyze_key_frames(png_dir, mask_dir, th, min_gap, max_gap, add_last_frame, is_invert_mask):
    keys = []
    
    frames = sorted(glob.glob( os.path.join(png_dir, "[0-9]*.png") ))
    
    key_frame = frames[0]
    keys.append( int(os.path.splitext(os.path.basename(key_frame))[0]) )
    key_edges = detect_edges( key_frame, get_mask_path_of_img( key_frame, mask_dir ), is_invert_mask )
    gap = 0
    
    for frame in frames:
        gap += 1
        if gap < min_gap:
            continue
        
        edges = detect_edges( frame, get_mask_path_of_img( frame, mask_dir ), is_invert_mask )
        
        delta = mean_pixel_distance( edges, key_edges )
        
        _th = th * (max_gap - gap)/max_gap
        
        if _th < delta:
            basename_without_ext = os.path.splitext(os.path.basename(frame))[0]
            keys.append( int(basename_without_ext) )
            key_frame = frame
            key_edges = edges
            gap = 0
    
    if add_last_frame:
        basename_without_ext = os.path.splitext(os.path.basename(frames[-1]))[0]
        last_frame = int(basename_without_ext)
        if not last_frame in keys:
            keys.append( last_frame )

    return keys

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def mp_args():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    mp_drawing_styles = mp.solutions.drawing_styles
    
    return [mp_drawing, mp_face_mesh, drawing_spec, mp_drawing_styles]

def getLandmarks(image):
    _, mp_face_mesh, _, _ = mp_args()
    
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.4, min_tracking_confidence=0.4)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    try:
        landmarks = results.multi_face_landmarks[0].landmark
    except:
        landmarks = False
    return landmarks, results

def x_y_z(landmakr, image):
    landmakrs, _ = getLandmarks(image)
    if not landmakrs:
        return 1, 1, 1
    x = int(landmakrs[landmakr].x * 100000)
    y = int(landmakrs[landmakr].y * 100000)
    z = int(landmakrs[landmakr].z * 100000 * 2)
    
    return (x, y, z)

def dist_dots(landmark1, landmark2, image):
    one = x_y_z(landmark1, image)
    two = x_y_z(landmark2, image)
    
    dx = (one[0] - two[0])**2
    dy = (one[1] - two[1])**2
    dz = (one[2] - two[2])**2

    return int(math.sqrt(dx + dy + dz))

def add_key_frame(frame_path, key_list):
    # frames = sorted(glob.glob( os.path.join(frame_path, "[0-9]*.png") ))
    
    temp_key_list = key_list.copy()
    
    i = 0
    # print(len(key_list))
    while True:
        temp_key_frame = 0
        # print('key_list: ', key_list)
        
        f_key = key_list[i]
        s_key = key_list[i+1]
        # print(f_key, s_key, i)
        
        key_frames = []
        for j in range(f_key, s_key+1):
            key_frames.append(os.path.join(frame_path, f'{str(j).zfill(5)}.png'))
            
        # print(key_frames, "\n\n----------")
        img1 = cv2.imread(key_frames[0])
        img2 = cv2.imread(key_frames[-1])
        
        top = 0
        bottom = 17
        left = 61
        right = 375
        
        top_bottom = max(dist_dots(top, bottom, img1), 1)
        left_right = max(dist_dots(left, right, img1), 1)
        
        f = int(top_bottom/left_right * 1000)
        
        top_bottom = max(dist_dots(top, bottom, img2), 1)
        left_right = max(dist_dots(left, right, img2), 1)
        
        l = int(top_bottom/left_right * 1000)
        
        
        if f < l:
            base = l
        else:
            base = f
        
        temp_key = [0, 0]
        for key in (key_frames):
            img = cv2.imread(key)
            
            top_bottom = max(dist_dots(top, bottom, img), 1)
            left_right = max(dist_dots(left, right, img), 1)
            
            n =  int(top_bottom/left_right * 1000)
            
            
            # print(now_ratio)
            if base * 1.05 < n:
                if temp_key[0] < n:
                    temp_key = [n, key]
                    # print(f'{key},f: {f}, l: {l}, n: {n}')
            # print(temp_key[1])
        if temp_key[1] == 0:
            temp_key_list.append(0)
        else:
            temp_key_list.append(int(os.path.basename(temp_key[1])[:-4]))


        i += 1
        if i == len(key_list)-1:
            break
    print(temp_key_list)
    keys = sorted(list(set(temp_key_list)))
    if keys[0] == 0:
        keys.remove(0)
    return keys


def ebsynth_utility_stage2(dbg, project_args, key_min_gap, key_max_gap, key_th, key_add_last_frame, is_invert_mask, face_analysis):
    dbg.print("stage2")
    dbg.print("")

    _, original_movie_path, frame_path, frame_mask_path, org_key_path, _, _ = project_args

    remove_pngs_in_dir(org_key_path)
    os.makedirs(org_key_path, exist_ok=True)

    fps = 30
    clip = cv2.VideoCapture(original_movie_path)
    if clip:
        fps = clip.get(cv2.CAP_PROP_FPS)
        clip.release()

    if key_min_gap == -1:
        key_min_gap = int(10 * fps/30)
    else:
        key_min_gap = max(1, key_min_gap)
        key_min_gap = int(key_min_gap * fps/30)
        
    if key_max_gap == -1:
        key_max_gap = int(300 * fps/30)
    else:
        key_max_gap = max(10, key_max_gap)
        key_max_gap = int(key_max_gap * fps/30)
    
    key_min_gap,key_max_gap = (key_min_gap,key_max_gap) if key_min_gap < key_max_gap else (key_max_gap,key_min_gap)
    
    dbg.print("fps: {}".format(fps))
    dbg.print("key_min_gap: {}".format(key_min_gap))
    dbg.print("key_max_gap: {}".format(key_max_gap))
    dbg.print("key_th: {}".format(key_th))

    keys = analyze_key_frames(frame_path, frame_mask_path, key_th, key_min_gap, key_max_gap, key_add_last_frame, is_invert_mask)

    dbg.print("keys : " + str(keys))
    print('keys1 : '+ str(keys))
    
    if face_analysis:
        keys = add_key_frame(frame_path, keys)
    
    for k in keys:
        filename = str(k).zfill(5) + ".png"
        shutil.copy( os.path.join( frame_path , filename) , os.path.join(org_key_path, filename) )
    
    print("=====================================")
    print(keys)

    dbg.print("")
    dbg.print("Keyframes are output to [" + org_key_path + "]")
    dbg.print("")
    dbg.print("[Ebsynth Utility]->[configuration]->[stage 2]->[Threshold of delta frame edge]")
    dbg.print("The smaller this value, the narrower the keyframe spacing, and if set to 0, the keyframes will be equally spaced at the value of [Minimum keyframe gap].")
    dbg.print("")
    dbg.print("If you do not like the selection, you can modify it manually.")
    dbg.print("(Delete keyframe, or Add keyframe from ["+frame_path+"])")

    dbg.print("")
    dbg.print("completed.")

