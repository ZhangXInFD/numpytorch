import numpy as np


def BilinearInterpolation(img, pos):
    A = img.transpose(2, 0, 1)
    x, y = pos
    x0, x1 = int(x), int(x + 1)
    y0, y1 = int(y), int(y + 1)
    if x == x0 and y == y0:
        return A[:, x0:x1, y0:y1].transpose(1, 2, 0)
    return (np.array([x1 - x, x - x0]).reshape((1, 2)) @ A[:, x0:(x1 + 1), y0:(y1 + 1)] @
            np.array([y1 - y, y - y0]).reshape((2, 1))).transpose(1, 2, 0)


class Resize:

    def __init__(self, ran=(0.75, 1.0)):
        self.l, self.u = ran

    def __call__(self, image):
        input_shape = image.shape
        H, W, C = image.shape
        H_ratio = np.random.uniform(self.l, self.u)
        W_ratio = np.random.uniform(self.l, self.u)
        outH, outW = int(H*H_ratio), int(W*W_ratio)
        output = np.zeros((outH, outW, C))
        count = 0
        for h in range(int(H*H_ratio)):
            for w in range(int(W*W_ratio)):
                count += 1
                output[h:(h+1), w:(w+1), :] = BilinearInterpolation(image, (h/H_ratio, w/W_ratio))
        ph, pw = (H-outH)//2, (W-outW)//2
        output = np.pad(output, ((ph, H-outH-ph), (pw, W-outW-pw), (0, 0)),
                        "constant", constant_values=0)
        return output.reshape(input_shape)


class Translation:

    def __init__(self, distortion=0.1):
        self.distortion = distortion

    def __call__(self, image):
        input_shape = image.shape
        H, W, C = image.shape
        h, w = int(H * self.distortion), int(W * self.distortion)
        output = np.pad(image, ((h, h), (w, w), (0, 0)), "constant", constant_values=0)
        HFLAG, WFLAG = np.random.uniform(size=2) > 0.5
        output = output[2*h*HFLAG: 2*h*HFLAG+H, 2*w*WFLAG: 2*w*WFLAG+W]
        return output.reshape(input_shape)


class Rotation:

    def __init__(self, ran=(-np.pi/12, np.pi/12)):
        self.l, self.u = ran

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_shape = image.shape
        H, W, C = image.shape
        phi = np.random.uniform(self.l, self.u)
        rotate_matrix = np.array([[ np.cos(phi), np.sin(phi), 0.],
                                  [-np.sin(phi), np.cos(phi), 0.],
                                  [           0,           0, 1.]])
        rotate_matrix = np.linalg.inv(rotate_matrix)
        output = np.zeros_like(image)
        center = np.array([[H/2], [W/2], [0]])
        location = np.array([[h, w, 1.] for h in range(H) for w in range(W)]).T
        target = rotate_matrix @ (location - center) + center
        location = location.astype(np.int)
        for h, w, tx, ty in zip(location[0], location[1], target[0], target[1]):
            if tx >= H-1 or tx < 0 or ty >= W-1 or ty < 0: continue
            output[h, w, :] = BilinearInterpolation(image, (tx, ty))
        return output.reshape(input_shape)