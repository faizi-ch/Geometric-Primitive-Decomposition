import torch
import torch.nn as nn

from ShapeEvaluator import MultipleShapesEvaluation
from BooleanLayer import BooleanLayer


class Encoder(nn.Module):
    def __init__(self, num_channels=1, latent_size=256):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        # resnet = models.resnet50(pretrained = True)
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

        # modules = list(resnet.children())[:-1]      # delete the last fc layer.
        # self.resnet = nn.Sequential(*modules)

        self.conv1 = nn.Conv2d(num_channels, self.latent_size // 8, 4, 2, padding=2)
        self.conv2 = nn.Conv2d(
            self.latent_size // 8, self.latent_size // 4, 4, 2, padding=2
        )
        self.conv3 = nn.Conv2d(
            self.latent_size // 4, self.latent_size // 2, 4, 2, padding=2
        )
        self.conv4 = nn.Conv2d(self.latent_size // 2, self.latent_size, 4, 2, padding=2)
        self.conv5 = nn.Conv2d(self.latent_size, self.latent_size, 4, 2)
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        """
        Input: (batch_size, in_ch, 64, 64)
        Outputs: (batch_size, latent_size)
        """

        # y = self.resnet(x)
        # y = y.reshape(y.shape[0], -1)

        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.conv2(y))
        y = self.lrelu(self.conv3(y))
        y = self.lrelu(self.conv4(y))
        y = self.conv5(y)
        y = y.reshape(y.shape[0], -1)

        assert y.shape == torch.Size([y.shape[0], self.latent_size])

        return y


class Decoder(nn.Module):
    def __init__(self, latent_size=256):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.latent_size, self.latent_size * 2)
        self.fc2 = nn.Linear(self.latent_size * 2, self.latent_size * 4)
        self.fc3 = nn.Linear(self.latent_size * 4, self.latent_size * 8)

        self.lrelu = nn.LeakyReLU(0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Input: (batch_size, latent_size)
        Outputs: (batch_size, latent_size * 8)
        """
        y = self.lrelu(self.fc1(x))
        y = self.lrelu(self.fc2(y))
        y = self.lrelu(self.fc3(y))

        return y


class PrimitivesNet(nn.Module):
    def __init__(
        self, num_channels=1, latent_size=256, num_shapes_per_type=8, threshold=0.5
    ):
        super().__init__()
        self.encoder = Encoder(num_channels, latent_size)
        self.decoder = Decoder(latent_size)

        self.shape_evaluator = MultipleShapesEvaluation(num_shapes_per_type)

        self.scaler = Scaler()

        self.threshold = threshold  # threshold for binarization

        self.num_boolean_layers = 3

        self.num_shape_evaluators = len(self.shape_evaluator)
        self.out_shape_eval = num_shapes_per_type * self.num_shape_evaluators

        boolean_layer = []

        self.out_shape_layer = 2

        in_shape = self.out_shape_eval
        out_shape = self.out_shape_layer

        for i in range(self.num_boolean_layers):
            if i == self.num_boolean_layers - 1:
                out_shape = 1

            boolean_layer.append(
                BooleanLayer(in_shape, out_shape, self.threshold, latent_size)
            )
            in_shape = out_shape * 4 + self.out_shape_eval

        self.boolean_layer = nn.ModuleList(boolean_layer)

    def forward(
        self,
        img,
        pt,
        return_shapes_distances=False,
    ):
        latent_vec = self.encoder(img)

        shape_param = self.decoder(latent_vec)  # [batch, latent_size * 8]

        pt = pt.unsqueeze(dim=1)  # [batch, 1, num_pt, 2]
        shape_dist = self.shape_evaluator(shape_param, pt)  # [batch, num_shape, num_pt]
        shape_dist = shape_dist.permute((0, 2, 1))  # [batch, num_pt, num_shape]

        scaled_shape_dist = 1 - self.scaler(shape_dist)  # [batch, num_pt, num_shape]

        last_distances = scaled_shape_dist

        for i, layer in enumerate(self.boolean_layer):
            if i:
                last_distances = torch.cat([last_distances, scaled_shape_dist], dim=-1)

            last_distances = layer(
                last_distances, latent_vec
            )  # [batch, num_pt, num_shape_out, 4]

            assert layer.V_encode != None

        last_distances = last_distances[..., 0]
        reconstructed_shape = last_distances.clamp(0, 1)
        outputs = [reconstructed_shape]

        if return_shapes_distances:
            outputs.append(shape_dist)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def binarize(self, x):
        """
        Binarize the input image.
        """
        return (x >= self.threshold).float()


FLOAT_EPS = torch.finfo(torch.float32).eps


class Scaler(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()

        self.alpha = nn.Parameter(torch.Tensor(1, 1, 1), requires_grad=True)
        nn.init.constant_(self.alpha, 1.0)

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return (x / self.alpha.clamp(min=FLOAT_EPS)).clamp(self.min_val, self.max_val)
