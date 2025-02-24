import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


# Reference: https://github.com/SarroccoLuigi/CombiNeRF/blob/a445f8a2bb63c80eaa0c8fa9ab59b575dc3abc7f/nerf/info/loss.py#L125

######################################################
#          Infomation Gain Reduction Loss            #
######################################################

class SmoothingLoss:
    def __init__(self):
        super(SmoothingLoss, self).__init__()

        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    def __call__(self, sigma, rays_a):

        half_num = sigma.size(0)//2
        half_1  = rays_a[0::2]
        half_2  = rays_a[1::2]
        import pdb; pdb.set_trace()
        sigma_1 = sigma[:half_num]
        sigma_2 = sigma[half_num:]

        p = nn.functional.softmax(sigma_1, -1)
        q = nn.functional.softmax(sigma_2, -1)

        loss = self.criterion(p.log(), q)

        return loss


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3, smoothness_loss_w=1e-4):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.smoothness_loss   = SmoothingLoss()
        self.smoothness_loss_w = smoothness_loss_w

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb']-target['rgb'])**2 + 1e-5

        o = results['opacity'] + 1e-5
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])
        
        # if self.smoothness_loss_w > 0:
        #     d['smoothness'] = self.smoothness_loss_w * self.smoothness_loss(results['ws'], results['rays_a'])
        
        # import pdb; pdb.set_trace()

        return d
