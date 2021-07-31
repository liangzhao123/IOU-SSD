import numpy as np
import torch
from new_train.utils import common_utils


class ResidualCoder_v0(object):
    def __init__(self, code_size=7):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=-1)

        # need to convert boxes to z-center format
        zg = zg + hg / 2
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 1.6
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
        rt = rg - ra
        cts = [g - a for g, a in zip(cgs, cas)]
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt, *cts], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = np.split(box_encodings, box_ndim, axis=-1)

        # need to convert box_encodings to z-bottom format
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        za = za + ha / 2
        zg = zg + hg / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        za = za + ha / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra

        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def decode_with_head_direction_torch(self, box_preds, anchors, dir_cls_preds,
                                         num_dir_bins, dir_offset, dir_limit_offset, use_binary_dir_classifier=False):
        """
        :param box_preds: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param dir_cls_preds: (batch_size, H, W, num_anchors_per_locations*2)
        :return:
        """
        batch_box_preds = self.decode_torch(box_preds, anchors)

        if dir_cls_preds is not None:
            dir_cls_preds = dir_cls_preds.view(box_preds.shape[0], box_preds.shape[1], -1)
            if use_binary_dir_classifier:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
                opp_labels = (batch_box_preds[..., -1] > 0) ^ dir_labels.byte()
                batch_box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(batch_box_preds),
                    torch.tensor(0.0).type_as(batch_box_preds)
                )
            else:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

                period = (2 * np.pi / num_dir_bins)
                dir_rot = common_utils.limit_period_torch(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                )
                # 输出的角度在0-np.pi之间，且根据dir_label的增加period(np.pi)
                batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_box_preds

class ResidualCoder_v1(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        rt = rg - ra

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, rt, *cts], dim=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

if __name__ == '__main__':
    a = torch.tensor([[0,1,1],
                      [1,0,0],
                      [0,1,0]])
    b = torch.nonzero(a)
    print("done")



