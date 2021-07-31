import numpy as np
import torch
from abc import ABCMeta,abstractmethod
import torch.nn as nn
from pvdet.dataset.utils import box_utils
import torch.nn.functional as F



class Loss(object):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __call__(self,
                 prediction_tensor,
                 target_tensor,
                 ignore_nan_targets=False,
                 scope=None,
                 **params):
        """Call the loss function.
        Args:
          prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
            representing predicted quantities.
          target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
            regression or classification targets.
          ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
          scope: Op scope name. Defaults to 'Loss' if None.
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: a tensor representing the value of the loss function.
        """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor),
                                        prediction_tensor,
                                        target_tensor)
        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.
        Args:
          prediction_tensor: a tensor representing predicted quantities
          target_tensor: a tensor representing regression or classification targets
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
            anchor
        """
        pass


class SigmoidFocalClassificationLoss_V0(Loss):
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
          gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
          alpha: optional alpha weighting factor to balance positives vs negatives.
          all_zero_negative: bool. if True, will treat all zero as background.
                else, will treat first label as background. only affect alpha.
        """
        self._alpha = alpha
        self._gamma = gamma

    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      weights,
                      class_indices=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]
          class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        if weights.shape.__len__() == 2 or (weights.shape.__len__() == 1 and target_tensor.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)
        # if class_indices is not None:
        #     weights *= indices_to_dense_vector(class_indices, prediction_tensor.shape[2]
        #                                        ).view(1, 1, -1).type_as(prediction_tensor)
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits_v0(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha +
                                   (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits_v0(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


class WeightedSmoothL1LocalizationLoss_V0(Loss):
    """Smooth L1 localization loss function.
    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = torch.from_numpy(self._code_weights).cuda()
        else:
            self._code_weights = None
        self._codewise = codewise

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            code_weights = self._code_weights.type_as(prediction_tensor)
            diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
               + (abs_diff - 0.5 / (self._sigma ** 2)) * (1. - abs_diff_lt_1)
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  # * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


class WeightedSoftmaxClassificationLoss_V0(Loss):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.
        Args:
          logit_scale: When this value is high, the prediction is "diffused" and
                       when this value is low, the prediction is made peakier.
                       (default 1.0)

        """
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(
            prediction_tensor, self._logit_scale)
        per_row_cross_ent = (_softmax_cross_entropy_with_logits_v0(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)))
        return per_row_cross_ent.view(weights.shape) * weights


def _softmax_cross_entropy_with_logits_v0(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)  # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def huber_loss_v0(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return losses


def get_corner_loss_lidar_v0(pred_bbox3d, gt_bbox3d):
    """
    :param pred_bbox3d: (N, 7)
    :param gt_bbox3d: (N, 7)
    :return: corner_loss: (N)
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes3d_to_corners3d_lidar_torch(pred_bbox3d)
    gt_box_corners = box_utils.boxes3d_to_corners3d_lidar_torch(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes3d_to_corners3d_lidar_torch(gt_bbox3d_flip)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))  # (N, 8)
    corner_loss = huber_loss_v0(corner_dist, delta=1.0)  # (N, 8)

    return corner_loss.mean(dim=1)



#**********************************************************************************************************
class SigmoidFocalClassificationLoss_v1(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss_v1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits_v1(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits_v1(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss_v1(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss_v1, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss_v1(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss_v1(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss_v1(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss_v1, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar_v1(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss_v1.smooth_l1_loss_v1(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)





