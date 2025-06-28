import torch
from PIL import Image
from model import EffNet, APL, prepare_image

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def mvg(x, y, ps):
    x_mean = x
    x_cov = cov(x)
    y_mean = y[0]
    y_cov = y[1]
    s = torch.linalg.solve((x_cov + y_cov) / 2, (x_mean - y_mean).t())
    s = (x_mean - y_mean).mm(s)
    s = torch.diagonal(s).abs().sqrt()
    s_map = s.reshape(ps.shape)
    q = (s_map * ps).sum() / ps.sum()
    return q


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_path = './imgs/bikes.bmp'
img = prepare_image(Image.open(img_path).convert("RGB"))

feats_total = torch.load('./MDFS_weights.pth')
feats_total = [feats_total.mean(0, keepdim=True), cov(feats_total)]

vgg = EffNet().to(device)
model = APL().to(device)

img = img.to(device)
feats = vgg(img)
out, ps = model(feats, select=0)
score = mvg(out, feats_total, ps)
print(score.item())
# score: 19.2222



