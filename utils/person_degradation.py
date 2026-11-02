"""Deterministic foreground ISP and 2D lighting-field degradations (BGR arrays)."""
import hashlib
import math

import cv2
import numpy as np


def linear(image):
    x = image.astype(np.float32) / 255
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def encoded(value):
    x = np.clip(value, 0, 1)
    return np.clip(np.rint(np.where(x <= .0031308, 12.92*x,
                                  1.055*x**(1/2.4)-.055)*255), 0, 255).astype(np.uint8)


ISP_RANGES = {'exposure_ev': (-5, 5), 'wb_red_ev': (-2, 2),
              'wb_green_ev': (-2, 2), 'wb_blue_ev': (-2, 2),
              'saturation': (0, 3), 'contrast': (.1, 3), 'gamma': (.1, 3),
              'black_point': (0, 254), 'white_point': (1, 255)}
LIGHT_RANGES = {'ambient_ev': (-3, 3), 'direction_deg': (0, 360),
                'directional_ev': (-3, 3), 'spot_ev': (-3, 3),
                'center_x': (0, 1), 'center_y': (0, 1), 'sigma': (.01, 2),
                'light_red_ev': (-2, 2), 'light_blue_ev': (-2, 2)}


def number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite and in [{low}, {high}]')


def validate(config):
    if not isinstance(config, dict) or set(config) != {'version', 'seed', 'variants', 'modes', 'detail', 'isp', 'relighting', 'legacy'}:
        raise ValueError('YAML requires version, seed, variants, modes, detail, isp, relighting, legacy only')
    if type(config['version']) is not int or config['version'] != 2:
        raise ValueError('unsupported config version')
    for key in ['seed', 'variants']:
        if type(config[key]) is not int or config[key] < (1 if key == 'variants' else 0):
            raise ValueError(f'{key} must be a positive integer (seed may be zero)')
    modes = config['modes']
    if not isinstance(modes, list) or not modes:
        raise ValueError('modes must be a nonempty list')
    names = set()
    for m in modes:
        if not isinstance(m, dict) or set(m) != {'name', 'weight', 'detail', 'isp', 'relighting', 'legacy'}:
            raise ValueError('invalid mode fields')
        if not isinstance(m['name'], str) or not m['name'] or m['name'] in names:
            raise ValueError('mode names must be nonempty and unique')
        names.add(m['name'])
        if m['legacy'] and any(m[k] for k in ['detail', 'isp', 'relighting']):
            raise ValueError('legacy must run as a standalone mode')
        number(m['weight'], 0, 1e6, 'weight')
        if any(type(m[k]) is not bool for k in ['detail', 'isp', 'relighting', 'legacy']) or not any(m[k] for k in ['detail', 'isp', 'relighting', 'legacy']):
            raise ValueError('mode must enable at least one boolean branch')
    if sum(m['weight'] for m in modes) <= 0:
        raise ValueError('mode weights sum must be positive')
    for section, ranges, scalars in [('isp', ISP_RANGES, {'contrast_pivot', 'channel_clip_probability'}),
                                     ('relighting', LIGHT_RANGES, {'min_gain', 'max_gain', 'bbox_smoothing'})]:
        values = config[section]
        if not isinstance(values, dict) or set(values) != set(ranges) | scalars:
            raise ValueError(f'invalid {section} fields')
        for key, (lo, hi) in ranges.items():
            bounds = values[key]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f'{section}.{key} must be [min, max]')
            for x in bounds:
                number(x, lo, hi, key)
            if bounds[0] > bounds[1]:
                raise ValueError(f'{key}: min exceeds max')
    isp, light = config['isp'], config['relighting']
    number(isp['contrast_pivot'], .001, .999, 'contrast_pivot')
    number(isp['channel_clip_probability'], 0, 1, 'channel_clip_probability')
    if isp['black_point'][1] >= isp['white_point'][0]:
        raise ValueError('black and white point ranges must not overlap')
    number(light['bbox_smoothing'], 0, .999, 'bbox_smoothing')
    number(light['min_gain'], .01, 1, 'min_gain')
    number(light['max_gain'], 1, 16, 'max_gain')
    return config


def sample(config, sample_id, variant, mode_name=None):
    digest = hashlib.sha256(f"{config['seed']}\0{sample_id}\0{variant}".encode()).digest()
    seed = int.from_bytes(digest[:8], 'big')
    rng = np.random.default_rng(seed)
    modes = config['modes']
    if mode_name:
        found = [m for m in modes if m['name'] == mode_name]
        if not found:
            raise ValueError(f'unknown mode: {mode_name}')
        mode = found[0]
    else:
        weights = np.array([m['weight'] for m in modes], dtype=float)
        mode = modes[rng.choice(len(modes), p=weights / weights.sum())]
    result = {'seed': seed, 'variant': variant, 'mode': dict(mode)}
    for section, ranges in [('isp', ISP_RANGES), ('relighting', LIGHT_RANGES)]:
        if not mode[section]:
            continue
        values = config[section]
        branch_seed = int.from_bytes(hashlib.sha256(digest+section.encode()).digest()[:8], 'big')
        branch_rng = np.random.default_rng(branch_seed)
        result[section] = {key: float(branch_rng.uniform(*values[key])) for key in ranges}
        result[section].update({k: v for k, v in values.items() if k not in ranges})
        if section == 'isp':
            result[section]['channel_clip'] = bool(branch_rng.random() < values['channel_clip_probability'])
    for branch in ['detail', 'legacy']:
        if mode[branch]:
            result[branch] = dict(config[branch])
    return result


def foreground_box(mask, previous=None, smoothing=0):
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError('empty mask')
    box = np.array([x.min(), y.min(), x.max()+1, y.max()+1], dtype=float)
    return box if previous is None else smoothing*previous+(1-smoothing)*box


def relight(image, params, box):
    """Add smooth spatial illumination; this does not remove original shadows."""
    height, width = image.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = (x+.5-box[0])/max(box[2]-box[0], 1)
    y = (y+.5-box[1])/max(box[3]-box[1], 1)
    angle = math.radians(params['direction_deg'])
    direction = np.clip((2*x-1)*math.cos(angle)+(2*y-1)*math.sin(angle), -1, 1)
    spot = np.exp(-((x-params['center_x'])**2+(y-params['center_y'])**2)/(2*params['sigma']**2))
    ev = params['ambient_ev'] + params['directional_ev']*direction + params['spot_ev']*spot
    # BGR; spatially varying colored illumination, normalized to neutral at zero EV.
    strength = .5*(direction+1)
    color = strength[..., None]*np.array([params['light_blue_ev'], 0, params['light_red_ev']])
    gain = np.clip(np.exp2(ev[..., None]+color), params['min_gain'], params['max_gain'])
    return encoded(linear(image)*gain)


def isp(image, params):
    value = linear(image)
    gains = np.exp2([params['wb_blue_ev'], params['wb_green_ev'], params['wb_red_ev']])
    gains /= np.dot(gains, [.0722, .7152, .2126])
    value = value*gains*2**params['exposure_ev']
    luminance = np.sum(value*np.array([.0722, .7152, .2126]), axis=2, keepdims=True)
    value = np.maximum(0, luminance+(value-luminance)*params['saturation'])
    # Pivoted power contrast and tone curve operate in linear RGB.
    pivot = params['contrast_pivot']
    value = pivot*np.maximum(value/pivot, 0)**params['contrast']
    value = np.maximum(value, 0)**params['gamma']
    result = encoded(value)
    if params['channel_clip']:
        result = np.clip(np.rint((result.astype(float)-params['black_point'])/
                                (params['white_point']-params['black_point'])*255), 0, 255).astype(np.uint8)
    return result
