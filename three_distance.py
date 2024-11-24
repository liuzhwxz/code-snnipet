"""
双曲空间的三个距离公式spdist：
1.Poincare's ball
2.Lorentzian distance
3.Squared Lorentzian distance
"""
import torch
from manifolds.base import Manifold
from utils.math_utils import artanh, tanh

class PoincareBall(Manifold):
    """
    这里最终的公式类似于论文《Poincaré Embeddings for Learning Hierarchical Representations，NeurIPS 2017》的方法，不过使用的是双曲正切的反函数artanh计算最终的距离
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15# 防止数值误差导致除零
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}# 流形的数值精度控制

    def sqdist(self, p1, p2, c):
        '''
        sqdist 方法计算的平方距离：
        p1和p2表示双曲空间内的点
        c表示曲率
        '''
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )#计算
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def mobius_add(self, x, y, c, dim=-1):
        '''
        Möbius 加法
        
        分别计算分子 num 和分母 denom。
        num = (1+2c*⟨x,y⟩+c∥y∥2)*x+(1−c∥x∥2)*y
        denom = 1+2c*⟨x,y⟩+c2*∥x∥2*∥y∥2
        通过分子分母的比值实现加法。
        '''
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)
    
class Hyperboloid(Manifold):
    """
    这里使用的论文《Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry，ICML 2018-Maximilian Nickel, Douwe Kiela》的方法，基于Minkowski 内积计算距离
    sqdist(x,y,c)=K⋅(arcosh(−⟨x,y⟩/K​))2
    Hyperboloid manifold class.
    不同于poincareball，这里没有缩放因子_lambda_x
    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        '''
        minkowski 内积，它是 伪欧几里得内积,相当于poincare中的inner函数
        x, y: 张量，形状为 (..., d+1)
        keepdim: 是否保持维度
        返回值: 张量，形状为 (...) 或 (..., 1) 取决于 keepdim
        '''
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        #res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)#计算 x 和 y 在双曲空间中的 Minkowski内积⟨x,y⟩
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])#双曲距离度量中的夹角参数 θ,θ=−⟨x,y⟩​/K
        sqdist = K * arcosh(theta) ** 2#用 反双曲余弦函数（arcosh）来计算 双曲空间中的距离
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)#sqdist 的值会被裁剪，确保它不会超过 50.0
    
class Lorentz(Manifold):
    """
    这里使用的论文《Lorentzian Distance Learning for Hyperbolic Representation》的方法
    不同于poincareball，这里没有缩放因子_lambda_x
    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Lorentz, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def inner(self, p, c, u, v=None, keepdim = False, dim = -1):
        """
        计算洛伦兹内积，与minkowski_dot一样
        参数:
            p: 流形内的点，形状为 (..., d+1)
            c: 曲率 (暂不使用，但保留以匹配函数签名)
            u: 切向量，形状为 (..., d+1)
            v: 第二个切向量，形状为 (..., d+1)。如果为 None，则计算 u 的自内积，
            u 是切向量，不需要满足这个hyperboloid的约束要求，而是被定义在点 pp 的切空间内
            keepdim: 是否保持维度

        返回:
            洛伦兹内积，形状取决于 keepdim
        """
        if v is None:
            v=u
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(dim=dim, keepdim=True)

    def minkowski_dot(self, x, y, keepdim=True):
        '''
        minkowski 内积，它是 伪欧几里得内积,相当于poincare中的inner函数
        x, y: 张量，形状为 (..., d+1)
        keepdim: 是否保持维度
        返回值: 张量，形状为 (...) 或 (..., 1) 取决于 keepdim
        '''
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        #res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        if keepdim:
            res = res.view(res.shape + (1,))
        return res
    
    def sqdist(self, x, y, c):
        """Squared distance between pairs of points.
        洛伦兹平方距离
        """
        K = 1. / c
        #sqdist = - 2.0 * K - 2.0 * self.inner(p=x, c=c, u=x, v=y,keepdim = True)#与self.minkowski_dot方法相同
        sqdist = - 2.0 * K - 2.0 * self.minkowski_dot(x, y)#论文3
        sqdist = torch.clamp(sqdist,min=self.min_norm)#限制最小值，防止nan
        # clamp distance to avoid nans in Fermi-Dirac decoder
        u=x
        v=y
        u0 = torch.sqrt(torch.sum(torch.pow(u,2),dim=-1, keepdim=True) + K)#时间分量，
        v0 = -torch.sqrt(torch.sum(torch.pow(v,2),dim=-1, keepdim=True) + K)#时间分量
        u = torch.cat((u,u0),dim=-1)
        v = torch.cat((v,v0),dim=-1)
        #sqdist = - 2 * K - 2 *torch.sum(u * v, dim=-1, keepdim=True)#论文3的公式6的原始代码，这里不适用，因为输入x和y已经是双曲面上的点，不再需要进行约束条件的计算
        return torch.clamp(sqdist, max=50.0)#sqdist 的值会被裁剪，确保它不会超过 50.0