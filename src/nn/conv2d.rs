use crate::tensor::Tensor;

pub struct Conv2DConfig {
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
    bias: bool,
}

impl Default for Conv2DConfig {
    fn default() -> Self {
        Conv2DConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }
}

pub struct Conv2D {
    ws: Tensor,
    bs: Tensor,
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
}

impl Conv2D {
    pub fn new(
        vs: &mut super::var_store::Path,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: Conv2DConfig,
    ) -> Conv2D {
        let Conv2DConfig {
            stride,
            padding,
            dilation,
            groups,
            bias,
        } = config;
        let bs = if bias {
            vs.zeros(&[out_dim])
        } else {
            Tensor::zeros(&[out_dim], (crate::Kind::Float, vs.device()))
        };
        let ws = vs.kaiming_uniform(&[out_dim, in_dim, ksize, ksize]);
        Conv2D {
            ws,
            bs,
            stride: [stride, stride],
            padding: [padding, padding],
            dilation: [dilation, dilation],
            groups,
        }
    }
}

impl super::module::Module for Conv2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv2d(
            &xs,
            &self.ws,
            &self.bs,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups,
        )
    }
}
