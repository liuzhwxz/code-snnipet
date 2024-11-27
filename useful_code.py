import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch.nn import Module
from loguru import logger
from .temps import BoundedTemp

class HyperbolicEntailmentCones(Module):
    """
    MaxMarginOENegativeSamplingLoss
    这个嵌入模型将实体表示为 Poincaré 圆盘中的圆锥，其中圆锥的开口与原点的距离相关，
    以便在包含关系上保持传递性。
    （参见论文：https://arxiv.org/pdf/1804.01882.pdf）

    :param relative_cone_aperture_scale: 在 (0,1] 范围内的数值，表示相对于距离原点的圆锥开口大小。
        实现方式如下：
            K = relative_cone_aperture_scale * eps_bound / (1 - eps_bound^2)
        （参考上述论文的公式 (25) 了解原因）
    :param eps_bound: 将向量约束在 eps 和 1-eps 之间的环形区域内。
    """

    def __init__(
        self,
        num_entity: int,#entity的数量
        dim: int,#嵌入空间的维度
        relative_cone_aperture_scale: float = 1.0,#相对圆锥开口比例，控制圆锥的开口大小
        eps_bound: float = 0.1,#控制嵌入向量的半径范围，确保它们位于环形区域内，避免过于接近原点或边界。
    ):
        super().__init__()
        self.eps_bound = eps_bound#这是一个小的正数，用于定义向量半径的下界和上界，确保嵌入向量位于 Poincaré 圆盘内的一个安全区域，避免数值不稳定。
        assert 0 < self.eps_bound < 0.5
        self.cone_aperature_scale = (#计算圆锥开口比例，根据论文中的公式 (24)
            relative_cone_aperture_scale * self.eps_bound / (1 - self.eps_bound ** 2)
        )

        self.angles = torch.nn.Embedding(num_entity, dim)#使用 torch.nn.Embedding 为每个实体创建一个角度向量（方向），初始值为随机的。
        initial_radius_range = 0.9 * (1 - 2 * self.eps_bound)#定义BoundedTemp的初始半径的范围
        initial_radius = 0.5 + initial_radius_range * (torch.rand(num_entity) - 0.5)
        self.radii = BoundedTemp(#BoundedTemp 用于对entity/node的半径进行参数化和约束。
            num_entity, initial_radius, self.eps_bound, 1 - self.eps_bound,
        )

    def forward(self, idxs: LongTensor) -> Tensor:
        """
        Returns the score of edges between the nodes in `idxs`.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :return: score
        """
        #logger.debug(f"idxs: {idxs.shape }")
        angles = F.normalize(self.angles(idxs), p=2, dim=-1)
        '''
        训练阶段，self.angles(idxs)得到的tensor维度为(batch_size, num_pairs, 2, dim)
        F.normalize(self.angles(idxs), p=2, dim=-1)：使用L2 范数，对(batch_size, num_pairs, 2, dim)的最后一个维度进行归一化，p=2 表示使用 L2 范数
        '''
        radii = self.radii(idxs)#为每个实体/node创建一个半径值，并确保其值在 [eps_bound, 1 - eps_bound] 范围内。
        vectors = radii[..., None] * angles# 维度(batch_size, num_pairs, 2, dim)
        '''
        radii[..., None] 会将 radii 张量的形状增加一个维度，例如从 (batch_size, num_pairs) 变为 (batch_size, num_pairs, 1)。
        这个操作的意思是 在最后一个维度（num_pairs 之后）添加一个维度，使得 radii 的每个元素都可以与 angles 的每个元素进行乘法操作
        '''
        # test_vectors_radii = torch.linalg.norm(vectors, dim=-1)
        # assert (test_vectors_radii > self.eps_bound).all()
        # assert (test_vectors_radii < 1 - self.eps_bound).all()
        # assert torch.isclose(test_vectors_radii, radii).all()

        radii_squared = radii ** 2
        euclidean_dot_products = (vectors[..., 0, :] * vectors[..., 1, :]).sum(dim=-1)
        '''
        vectors[..., 0, :] 和 vectors[..., 1, :] 的形状都是(batch_size, num_pairs, dim)，再sum求和后，euclidean_dot_products 维度是(batch_size, num_pairs)
        '''
        euclidean_distances = torch.linalg.norm(
            vectors[..., 0, :] - vectors[..., 1, :], dim=-1
        )

        parent_aperature_angle_sin = (
            self.cone_aperature_scale * (1 - radii_squared[..., 0]) / radii[..., 0]
        )
        # assert (parent_aperature_angle_sin >= -1).all()
        # assert (parent_aperature_angle_sin <= 1).all()
        parent_aperature_angle = torch.arcsin(parent_aperature_angle_sin)

        min_angle_parent_rotation_cos = (
            euclidean_dot_products * (1 + radii_squared[..., 0])
            - radii_squared[..., 0] * (1 + radii_squared[..., 1])
        ) / (
            radii[..., 0]
            * euclidean_distances
            * torch.sqrt(
                1
                + radii_squared[..., 0] * radii_squared[..., 1]
                - 2 * euclidean_dot_products
            )
            + 1e-22
        )
        # assert (min_angle_parent_rotation_cos >= -1).all()
        # assert (min_angle_parent_rotation_cos <= 1).all()
        # original implementation clamps this value from -1+eps to 1-eps, however it seems as though [-1, 1] is all that
        # is required.
        min_angle_parent_rotation = torch.arccos(
            min_angle_parent_rotation_cos.clamp(-1, 1)
        )
        # The energy in the original formulation is clamped, which means gradients for negative examples may be squashed.
        return (parent_aperature_angle - min_angle_parent_rotation).clamp_max(0)