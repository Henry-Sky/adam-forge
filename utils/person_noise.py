"""Deterministic binary noise replacement that does not sample person pixels."""
import numpy as np


def replace_rectangle(rgb, box, seed, clip_id, obj_id, frame_index):
    rgb = np.asarray(rgb)
    x0,y0,x1,y1 = box
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError('Rectangle outside source frame')
    rng = np.random.default_rng(np.random.SeedSequence([seed,clip_id,obj_id,frame_index]))
    noise = rng.integers(0,2,size=(y1-y0,x1-x0),dtype=np.uint8)*255
    output = rgb.copy()
    output[y0:y1,x0:x1] = noise[:,:,None]
    return output


def _smooth_track(values, window):
    """Centered triangular smoothing with endpoint velocity extrapolation."""
    radius = window//2
    if not radius or len(values) == 1:
        return values.copy()
    span = min(radius,len(values)-1)
    first_slope = (values[span]-values[0])/span
    last_slope = (values[-1]-values[-1-span])/span
    padded = np.concatenate([
        values[0]+np.arange(-radius,0)[:,None]*first_slope,
        values,
        values[-1]+np.arange(1,radius+1)[:,None]*last_slope])
    weights = np.concatenate([np.arange(1,radius+2),np.arange(radius,0,-1)]).astype(float)
    weights /= weights.sum()
    return np.stack([np.sum(padded[i:i+window]*weights[:,None],axis=0) for i in range(len(values))])


def stabilize_rectangles(boxes, size, padding=0.1, min_padding=8, window=7):
    """Smooth centers/sizes without lag, and conservatively contain every box.

    A shared per-axis radius correction ensures smoothing never cuts into the
    expanded current frame. Even outward rounding aligns YUV420 chroma blocks.
    """
    boxes = np.asarray(boxes,dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0
            or not np.isfinite(boxes).all() or np.any(boxes[:,2:] <= boxes[:,:2])):
        raise ValueError('Invalid rectangle track')
    if window < 1 or window % 2 != 1 or padding < 0 or min_padding < 0:
        raise ValueError('Expected odd positive window and nonnegative padding')
    width,height = size
    if width % 2 or height % 2:
        raise ValueError('YUV420 requires even frame dimensions')
    dimensions = boxes[:,2:]-boxes[:,:2]
    margin = np.maximum(dimensions*padding,min_padding)
    expanded = np.concatenate([boxes[:,:2]-margin,boxes[:,2:]+margin],axis=1)
    expanded[:,[0,2]] = np.clip(expanded[:,[0,2]],0,width)
    expanded[:,[1,3]] = np.clip(expanded[:,[1,3]],0,height)
    centers = _smooth_track((expanded[:,:2]+expanded[:,2:])/2,window)
    required = np.maximum(centers-expanded[:,:2],expanded[:,2:]-centers)
    radii = _smooth_track(required,window)
    radii += np.maximum(0,(required-radii).max(axis=0))
    result = np.concatenate([np.floor((centers-radii)/2)*2,
                             np.ceil((centers+radii)/2)*2],axis=1).astype(int)
    result[:,[0,2]] = np.clip(result[:,[0,2]],0,width)
    result[:,[1,3]] = np.clip(result[:,[1,3]],0,height)
    if np.any(result[:,:2] > expanded[:,:2]) or np.any(result[:,2:] < expanded[:,2:]):
        raise ValueError('Smoothed rectangle lost current-frame coverage')
    return result,expanded


def replace_yuv_rectangle(raw, size, box, seed, clip_id, obj_id, frame_index, full_range=False, fixed_scene_noise=False):
    width,height = size
    x0,y0,x1,y1 = map(int,box)
    if width % 2 or height % 2 or any(v % 2 for v in (x0,y0,x1,y1)):
        raise ValueError('YUV420 frame and rectangle must align to two pixels')
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('Rectangle outside frame')
    result = np.frombuffer(raw,dtype=np.uint8).copy()
    if result.size != width*height*3//2:
        raise ValueError('Unexpected YUV frame length')
    y = result[:width*height].reshape(height,width)
    u = result[width*height:width*height*5//4].reshape(height//2,width//2)
    v = result[width*height*5//4:].reshape(height//2,width//2)
    if fixed_scene_noise:
        rng = np.random.default_rng(np.random.SeedSequence([seed,clip_id,obj_id]))
        bits = rng.integers(0,2,size=(height,width),dtype=np.uint8)[y0:y1,x0:x1]
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed,clip_id,obj_id,frame_index]))
        bits = rng.integers(0,2,size=(y1-y0,x1-x0),dtype=np.uint8)
    y[y0:y1,x0:x1] = bits*255 if full_range else 16+bits*219
    u[y0//2:y1//2,x0//2:x1//2] = 128
    v[y0//2:y1//2,x0//2:x1//2] = 128
    return result.tobytes()
