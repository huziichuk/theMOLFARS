"""Small 3D Gaussian Splatting rasterizer in pure PyTorch (no CUDA extension needed).

It does what the browser's Spark renderer does: projects every Gaussian's 3D covariance to a 2D ellipse, sorts by
depth and alpha-composites front to back. View-independent colour only (SH degree 0), which is what the labels need.
Used by multiview_classify.py to render each 3D box from several aisle viewpoints.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene as B  # noqa: E402

C0 = 0.28209479177387814


def quat_to_rotmat(q):
    """q: (N,4) as (w,x,y,z) -> (N,3,3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


class SplatRenderer:
    def __init__(self, ply_path, device="cuda", min_opacity=0.05):
        _, props, rows = B.load_ply(ply_path)
        P = {p: i for i, p in enumerate(props)}
        rows = B.sanitize(rows, P)
        op = 1 / (1 + np.exp(-rows[:, P["opacity"]].astype(np.float64)))
        keep = op > min_opacity
        rows, op = rows[keep], op[keep]
        self.dev = device
        xyz = torch.from_numpy(B.to_world(rows[:, :3].astype(np.float64)).astype(np.float32))
        q = torch.from_numpy(rows[:, [P["rot_0"], P["rot_1"], P["rot_2"], P["rot_3"]]].astype(np.float32))
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s = torch.from_numpy(np.exp(rows[:, [P["scale_0"], P["scale_1"], P["scale_2"]]].astype(np.float32)))
        R = quat_to_rotmat(q)
        M = R * s[:, None, :]                      # R @ diag(s)
        cov = M @ M.transpose(1, 2)                # (N,3,3) in the PLY frame
        A = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # linear part of build_scene.to_world
        cov = A @ cov @ A.T
        self.xyz = xyz.to(device)
        self.cov = cov.to(device)
        self.op = torch.from_numpy(op.astype(np.float32)).to(device)
        rgb = np.clip(0.5 + C0 * rows[:, [P["f_dc_0"], P["f_dc_1"], P["f_dc_2"]]], 0, 1)
        self.rgb = torch.from_numpy(rgb.astype(np.float32)).to(device)
        self.n = len(rows)

    @torch.no_grad()
    def render(self, cam_pos, target, size=320, fov_deg=58.0, near=0.15, far=8.0, bg=0.1, max_radius=18):
        """Returns (image uint8 HxWx3 numpy, coverage = mean opacity accumulated per pixel)."""
        dev = self.dev
        cam = torch.tensor(np.asarray(cam_pos, np.float32), device=dev)
        tgt = torch.tensor(np.asarray(target, np.float32), device=dev)
        fwd = tgt - cam
        fwd = fwd / fwd.norm().clamp_min(1e-9)
        up = torch.tensor([0.0, 1.0, 0.0], device=dev)
        right = torch.cross(fwd, up, dim=0)
        if right.norm() < 1e-4:
            right = torch.tensor([1.0, 0.0, 0.0], device=dev)
        right = right / right.norm()
        up = torch.cross(right, fwd, dim=0)
        Rv = torch.stack([right, up, fwd])        # world -> camera (x right, y up, z forward)

        # frustum cull on a cheap box test first, then exact
        t = (self.xyz - cam) @ Rv.T
        z = t[:, 2]
        tan = float(np.tan(np.deg2rad(fov_deg / 2)))
        vis = (z > near) & (z < far) & (t[:, 0].abs() < z * tan * 1.4) & (t[:, 1].abs() < z * tan * 1.4)
        idx = vis.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return np.full((size, size, 3), int(bg * 255), np.uint8), 0.0
        t, z = t[idx], z[idx]
        cov = Rv @ self.cov[idx] @ Rv.T            # (M,3,3) in camera frame
        f = (size / 2) / tan
        u = f * t[:, 0] / z + size / 2
        v = -f * t[:, 1] / z + size / 2
        # Jacobian of (u,v) wrt camera xyz
        J = torch.zeros(len(idx), 2, 3, device=dev)
        J[:, 0, 0] = f / z
        J[:, 0, 2] = -f * t[:, 0] / z ** 2
        J[:, 1, 1] = -f / z
        J[:, 1, 2] = f * t[:, 1] / z ** 2
        c2 = J @ cov @ J.transpose(1, 2)           # (M,2,2)
        c2[:, 0, 0] += 0.3
        c2[:, 1, 1] += 0.3
        det = c2[:, 0, 0] * c2[:, 1, 1] - c2[:, 0, 1] ** 2
        ok = det > 1e-8
        mid = 0.5 * (c2[:, 0, 0] + c2[:, 1, 1])
        lam = mid + torch.sqrt((mid ** 2 - det).clamp_min(0))
        radius = torch.ceil(3 * torch.sqrt(lam)).clamp(1, max_radius)
        ok &= (u + radius > 0) & (u - radius < size) & (v + radius > 0) & (v - radius < size)
        sel = ok.nonzero(as_tuple=True)[0]
        idx, u, v, z, c2, det, radius = idx[sel], u[sel], v[sel], z[sel], c2[sel], det[sel], radius[sel]
        inv = torch.stack([c2[:, 1, 1], -c2[:, 0, 1], c2[:, 0, 0]], 1) / det[:, None]   # a, b, c of inverse
        op, rgb = self.op[idx], self.rgb[idx]
        order = torch.argsort(z)                   # front to back
        u, v, inv, op, rgb, radius = u[order], v[order], inv[order], op[order], rgb[order], radius[order]
        rank = torch.arange(len(u), device=dev)

        # enumerate (gaussian, pixel) pairs, bucketed by radius so small Gaussians don't pay for big grids
        pix_l, a_l, rgb_l, key_l = [], [], [], []
        r_int = radius.long()
        for lo, hi in ((1, 3), (4, 7), (8, 12), (13, max_radius)):
            m = (r_int >= lo) & (r_int <= hi)
            if not m.any():
                continue
            g = m.nonzero(as_tuple=True)[0]
            R = hi
            offs = torch.arange(-R, R + 1, device=dev)
            dy, dx = torch.meshgrid(offs, offs, indexing="ij")
            dx, dy = dx.reshape(-1).float(), dy.reshape(-1).float()
            chunk = max(1, int(6_000_000 // len(dx)))
            for s0 in range(0, len(g), chunk):
                gi = g[s0:s0 + chunk]
                cu, cv = u[gi], v[gi]
                px = torch.floor(cu)[:, None] + dx[None]
                py = torch.floor(cv)[:, None] + dy[None]
                ddx = px + 0.5 - cu[:, None]
                ddy = py + 0.5 - cv[:, None]
                a, b, c = inv[gi, 0][:, None], inv[gi, 1][:, None], inv[gi, 2][:, None]
                power = -0.5 * (a * ddx * ddx + c * ddy * ddy) - b * ddx * ddy
                alpha = (op[gi][:, None] * torch.exp(power)).clamp_max(0.99)
                valid = (alpha >= 1 / 255) & (px >= 0) & (px < size) & (py >= 0) & (py < size)
                if not valid.any():
                    continue
                gidx, pidx = valid.nonzero(as_tuple=True)
                pixel = (py[gidx, pidx].long() * size + px[gidx, pidx].long())
                pix_l.append(pixel)
                a_l.append(alpha[gidx, pidx])
                rgb_l.append(rgb[gi][gidx])
                key_l.append(pixel * len(u) + rank[gi][gidx])
        if not pix_l:
            return np.full((size, size, 3), int(bg * 255), np.uint8), 0.0
        pixel, alpha, col, key = torch.cat(pix_l), torch.cat(a_l), torch.cat(rgb_l), torch.cat(key_l)
        o = torch.argsort(key)
        pixel, alpha, col = pixel[o], alpha[o], col[o]
        # segmented exclusive cumulative transmittance per pixel
        la = torch.log1p(-alpha)
        cum = torch.cumsum(la, 0)
        excl = cum - la
        start = torch.ones_like(pixel, dtype=torch.bool)
        start[1:] = pixel[1:] != pixel[:-1]
        seg = torch.cumsum(start.long(), 0) - 1
        base = excl[start.nonzero(as_tuple=True)[0]][seg]
        T_prev = torch.exp(excl - base)
        wgt = alpha * T_prev
        img = torch.zeros(size * size, 3, device=dev)
        img.index_add_(0, pixel, wgt[:, None] * col)
        acc = torch.zeros(size * size, device=dev)
        acc.index_add_(0, pixel, wgt)
        img = img + (1 - acc)[:, None] * bg
        out = (img.clamp(0, 1).view(size, size, 3) * 255).to(torch.uint8).cpu().numpy()
        return out, float(acc.mean())
