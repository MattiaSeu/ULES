import math
import copy
import random
from functools import wraps, partial
from math import floor

import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# from kornia import augmentation as augs
from torchvision import transforms as augs
from kornia import filters, color

from einops import rearrange

from utils.tenprint import print_tensor


# helper functions

def identity(t):
    return t


def default(val, def_val):
    return def_val if val is None else val


def rand_true(prob):
    return random.random() < prob


def singleton_rgb(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def singleton_range(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def cutout_coordinates(image, ratio_range=(0.6, 0.8)):
    _, _, orig_h, orig_w = image.shape

    ratio_lo, ratio_hi = ratio_range
    random_ratio = ratio_lo + random.random() * (ratio_hi - ratio_lo)
    w, h = floor(random_ratio * orig_w), floor(random_ratio * orig_h)
    coor_x = floor((orig_w - w) * random.random())
    coor_y = floor((orig_h - h) * random.random())
    return ((coor_y, coor_y + h), (coor_x, coor_x + w)), random_ratio


def cutout_and_resize(image, coordinates, output_size=None, mode='nearest'):
    shape = image.shape
    output_size = default(output_size, shape[2:])
    (y0, y1), (x0, x1) = coordinates
    cutout_image = image[:, :, y0:y1, x0:x1]
    return F.interpolate(cutout_image, size=output_size, mode=mode)


# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# classes

class MLP(nn.Module):
    def __init__(self, chan, chan_out=256, inner_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, chan_out)
        )

    def forward(self, x):
        return self.net(x)


class ConvMLP(nn.Module):
    def __init__(self, chan, chan_out=256, inner_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, chan_out, 1)
        )

    def forward(self, x):
        return self.net(x)


class PPM(nn.Module):
    def __init__(
            self,
            *,
            chan,
            num_layers=1,
            gamma=2):
        super().__init__()
        self.gamma = gamma

        if num_layers == 0:
            self.transform_net = nn.Identity()
        elif num_layers == 1:
            self.transform_net = nn.Conv2d(chan, chan, 1)
        elif num_layers == 2:
            self.transform_net = nn.Sequential(
                nn.Conv2d(chan, chan, 1),
                nn.BatchNorm2d(chan),
                nn.ReLU(),
                nn.Conv2d(chan, chan, 1)
            )
        else:
            raise ValueError('num_layers must be one of 0, 1, or 2')

    def forward(self, x):
        return checkpoint(self._forward, x)

    def _forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)

        # Compute cosine similarity more efficiently
        x_norm = F.normalize(x_flat, dim=1)
        similarity = torch.bmm(x_norm.transpose(1, 2), x_norm).pow(self.gamma)

        # Apply ReLU
        similarity = F.relu(similarity)

        # Transform in a memory-efficient way
        transform_out = self.transform_net(x)
        transform_flat = transform_out.view(b, c, -1)

        # Compute the output
        out = torch.bmm(transform_flat, similarity)
        return out.view(b, c, h, w)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(
            self,
            *,
            net,
            projection_size,
            projection_hidden_size,
            layer_pixel=-2,
            layer_instance=-2
    ):
        super().__init__()
        self.net = net
        self.layer_pixel = layer_pixel
        self.layer_instance = layer_instance

        self.pixel_projector = None
        self.instance_projector = None

        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden_pixel = None
        self.hidden_instance = None
        self.hook_registered = False

    def _find_layer(self, layer_id):
        if type(layer_id) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(layer_id, None)
        elif type(layer_id) == int:
            children = [*self.net.children()]
            return children[layer_id]
        return None

    def _hook_pixel(self, _, __, output):
        setattr(self, 'hidden_pixel', output)

    def _hook_instance(self, _, __, output):
        setattr(self, 'hidden_instance', output)

    def _register_hook(self):
        pixel_layer = self._find_layer(self.layer_pixel)
        instance_layer = self._find_layer(self.layer_instance)

        assert pixel_layer is not None, f'hidden layer ({self.layer_pixel}) not found'
        assert instance_layer is not None, f'hidden layer ({self.layer_instance}) not found'

        pixel_layer.register_forward_hook(self._hook_pixel)
        instance_layer.register_forward_hook(self._hook_instance)
        self.hook_registered = True

    @singleton_rgb('pixel_projector')
    def _get_pixel_projector(self, hidden):
        _, dim, *_ = hidden.shape
        projector = ConvMLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    @singleton_rgb('instance_projector')
    def _get_instance_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    @singleton_range('pixel_projector')
    def _get_pixel_projector(self, hidden):
        _, dim, *_ = hidden.shape
        projector = ConvMLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    @singleton_range('instance_projector')
    def _get_instance_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_pixel = None
        self.hidden_instance = None
        assert hidden_pixel is not None, f'hidden pixel layer {self.layer_pixel} never emitted an output'
        assert hidden_instance is not None, f'hidden instance layer {self.layer_instance} never emitted an output'
        return hidden_pixel, hidden_instance

    def forward(self, x):
        pixel_representation, instance_representation = self.get_representation(x)
        #instance_representation = instance_representation["out"].flatten(1)  # use this when entire model
        instance_representation = instance_representation.flatten(1)

        pixel_projector = self._get_pixel_projector(pixel_representation)
        instance_projector = self._get_instance_projector(instance_representation)

        pixel_projection = pixel_projector(pixel_representation)
        instance_projection = instance_projector(instance_representation)
        return pixel_projection, instance_projection


# main class

class PixelCL_DB(nn.Module):
    def __init__(
            self,
            net,
            image_size,
            hidden_layer_pixel=-2,
            hidden_layer_instance=-2,
            projection_size=256,
            projection_hidden_size=2048,
            augment_fn=None,
            augment_fn2=None,
            prob_rand_hflip=0.25,
            moving_average_decay=0.99,
            ppm_num_layers=1,
            ppm_gamma=2,
            distance_thres=0.7,
            similarity_temperature=0.3,
            alpha=1.,
            use_pixpro=True,
            cutout_ratio_range=(0.6, 0.8),
            cutout_interpolate_mode='nearest',
            coord_cutout_interpolate_mode='bilinear',
            use_range_image=False
    ):
        super().__init__()

        DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomSolarize(threshold=0.1, p=0.5),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)
        self.prob_rand_hflip = prob_rand_hflip
        self.use_range_image = use_range_image

        self.online_encoder_rgb = NetWrapper(
            net=net,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
            layer_pixel=hidden_layer_pixel,
            layer_instance=hidden_layer_instance
        )

        net_1ch = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True,
                                                               num_classes=20, aux_loss=None)
        net_1ch = net_1ch.backbone
        if self.use_range_image:
            # net_1ch.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # use this when you are taking the entire network and not just the backbone
            net_1ch.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.online_encoder_gray = NetWrapper(
            net=net_1ch,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
            layer_pixel=hidden_layer_pixel,
            layer_instance=hidden_layer_instance
        )

        self.target_encoder_rgb = None
        self.target_encoder_gray = None
        self.target_ema_updater_rgb = EMA(moving_average_decay)
        self.target_ema_updater_range = EMA(moving_average_decay)

        self.distance_thres = distance_thres
        self.similarity_temperature = similarity_temperature
        self.alpha = alpha

        self.use_pixpro = use_pixpro

        if use_pixpro:
            self.propagate_pixels = PPM(
                chan=projection_size,
                num_layers=ppm_num_layers,
                gamma=ppm_gamma
            )

        self.cutout_ratio_range = cutout_ratio_range
        self.cutout_interpolate_mode = cutout_interpolate_mode
        self.coord_cutout_interpolate_mode = coord_cutout_interpolate_mode

        # instance level predictor
        self.online_predictor_rgb = MLP(projection_size, projection_size, projection_hidden_size)
        self.online_predictor_gray = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        if self.use_range_image:
            mock_dict = {
                "image": torch.randn(2, 3, image_size[0], image_size[1], device=device),
                "range_view": torch.randn(2, 1, image_size[0], image_size[1], device=device)
            }
            self.forward(mock_dict)
        else:
            # self.forward(torch.randn(2, 3, image_size[0], image_size[1], device=device))
            mock_dict = {
                "image": torch.randn(2, 3, image_size[0], image_size[1], device=device),
                "daolp": torch.randn(2, 3, image_size[0], image_size[1], device=device)
            }
            self.forward(mock_dict)

    @singleton_rgb('target_encoder_rgb')
    def _get_target_encoder_rgb(self):
        target_encoder_rgb = copy.deepcopy(self.online_encoder_rgb)
        set_requires_grad(target_encoder_rgb, False)
        return target_encoder_rgb

    @singleton_range('target_encoder_gray')
    def _get_target_encoder_gray(self):
        target_encoder_gray = copy.deepcopy(self.online_encoder_gray)
        set_requires_grad(target_encoder_gray, False)
        return target_encoder_gray

    def reset_moving_average_rgb(self):
        del self.target_encoder_rgb
        self.target_encoder_rgb = None

    def update_moving_average_rgb(self):
        assert self.target_encoder_rgb is not None, 'target encoder_rgb has not been created yet'
        update_moving_average(self.target_ema_updater_rgb, self.target_encoder_rgb, self.online_encoder_rgb)

    def reset_moving_average_range(self):
        del self.target_encoder_gray
        self.target_encoder_gray = None

    def update_moving_average_range(self):
        assert self.target_encoder_gray is not None, 'target encoder_range has not been created yet'
        update_moving_average(self.target_ema_updater_range, self.target_encoder_gray, self.online_encoder_gray)

    def forward(self, x, return_positive_pairs=False):
        range_view = False

        if isinstance(x, dict):
            if self.use_range_image:
                range_view = x['range_view']
                x = x['image']
            else:
                range_view = x['daolp']
                x = x['image']

        shape, device, prob_flip = x.shape, x.device, self.prob_rand_hflip

        rand_flip_fn = lambda t: torch.flip(t, dims=(-1,))

        flip_image_one, flip_image_two = rand_true(prob_flip), rand_true(prob_flip)
        flip_image_one_fn = rand_flip_fn if flip_image_one else identity
        flip_image_two_fn = rand_flip_fn if flip_image_two else identity

        cutout_coordinates_one, _ = cutout_coordinates(x, self.cutout_ratio_range)
        cutout_coordinates_two, _ = cutout_coordinates(x, self.cutout_ratio_range)

        image_one_cutout = cutout_and_resize(x, cutout_coordinates_one, mode=self.cutout_interpolate_mode)
        image_two_cutout = cutout_and_resize(x, cutout_coordinates_two, mode=self.cutout_interpolate_mode)

        range_view_one_cutout = cutout_and_resize(range_view, cutout_coordinates_one,
                                                  mode=self.cutout_interpolate_mode)
        range_view_two_cutout = cutout_and_resize(range_view, cutout_coordinates_two,
                                                  mode=self.cutout_interpolate_mode)
        range_view_one_cutout = flip_image_one_fn(range_view_one_cutout)
        range_view_two_cutout = flip_image_two_fn(range_view_two_cutout)

        image_one_cutout = flip_image_one_fn(image_one_cutout)
        image_two_cutout = flip_image_two_fn(image_two_cutout)

        image_one_cutout_tmp, image_two_cutout_tmp = self.augment1(image_one_cutout[:, :3]), self.augment2(
            image_two_cutout[:, :3])
        #  we don't apply augments to the intensity points as they pertain only to color

        image_one_cutout[:, :3] = image_one_cutout_tmp
        image_two_cutout[:, :3] = image_two_cutout_tmp

        proj_pixel_one_rgb, proj_instance_one_rgb = self.online_encoder_rgb(image_one_cutout)
        proj_pixel_two_rgb, proj_instance_two_rgb = self.online_encoder_rgb(image_two_cutout)

        proj_pixel_one_range, proj_instance_one_range = self.online_encoder_gray(range_view_one_cutout)
        proj_pixel_two_range, proj_instance_two_range = self.online_encoder_gray(range_view_two_cutout)

        image_h, image_w = shape[2:]

        proj_image_shape = proj_pixel_one_rgb.shape[2:]
        proj_image_h, proj_image_w = proj_image_shape

        coordinates = torch.meshgrid(
            torch.arange(image_h, device=device),
            torch.arange(image_w, device=device)
        )

        coordinates = torch.stack(coordinates).unsqueeze(0).float()
        coordinates /= math.sqrt(image_h ** 2 + image_w ** 2)
        coordinates[:, 0] *= proj_image_h
        coordinates[:, 1] *= proj_image_w

        proj_coors_one = cutout_and_resize(coordinates, cutout_coordinates_one, output_size=proj_image_shape,
                                           mode=self.coord_cutout_interpolate_mode)
        proj_coors_two = cutout_and_resize(coordinates, cutout_coordinates_two, output_size=proj_image_shape,
                                           mode=self.coord_cutout_interpolate_mode)

        proj_coors_one = flip_image_one_fn(proj_coors_one)
        proj_coors_two = flip_image_two_fn(proj_coors_two)

        proj_coors_one, proj_coors_two = map(lambda t: rearrange(t, 'b c h w -> (b h w) c'),
                                             (proj_coors_one, proj_coors_two))
        pdist = nn.PairwiseDistance(p=2)

        num_pixels = proj_coors_one.shape[0]

        proj_coors_one_expanded = proj_coors_one[:, None].expand(num_pixels, num_pixels, -1).reshape(
            num_pixels * num_pixels, 2)
        proj_coors_two_expanded = proj_coors_two[None, :].expand(num_pixels, num_pixels, -1).reshape(
            num_pixels * num_pixels, 2)

        distance_matrix = pdist(proj_coors_one_expanded, proj_coors_two_expanded)
        distance_matrix = distance_matrix.reshape(num_pixels, num_pixels)

        positive_mask_one_two = distance_matrix < self.distance_thres
        positive_mask_two_one = positive_mask_one_two.t()

        with torch.no_grad():
            target_encoder_rgb = self._get_target_encoder_rgb()
            target_proj_pixel_one_rgb, target_proj_instance_one_rgb = target_encoder_rgb(image_one_cutout)
            target_proj_pixel_two_rgb, target_proj_instance_two_rgb = target_encoder_rgb(image_two_cutout)
            target_encoder_gray = self._get_target_encoder_gray()
            target_proj_pixel_one_range, target_proj_instance_one_range = target_encoder_gray(range_view_one_cutout)
            target_proj_pixel_two_range, target_proj_instance_two_range = target_encoder_gray(range_view_two_cutout)

        # flatten all the pixel projections

        flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')

        target_proj_pixel_one_rgb, target_proj_pixel_two_rgb = list(
            map(flatten, (target_proj_pixel_one_rgb, target_proj_pixel_two_rgb)))

        target_proj_pixel_one_range, target_proj_pixel_two_range = list(
            map(flatten, (target_proj_pixel_one_range, target_proj_pixel_two_range)))

        # get total number of positive pixel pairs

        positive_pixel_pairs = positive_mask_one_two.sum()

        # get instance level loss

        pred_instance_one = self.online_predictor_rgb(proj_instance_one_rgb + proj_instance_one_range)
        pred_instance_two = self.online_predictor_rgb(proj_instance_two_rgb + proj_instance_two_range)

        target_proj_instance_two = target_proj_instance_two_rgb + target_proj_instance_two_range
        target_proj_instance_one = target_proj_instance_one_rgb + target_proj_instance_one_range
        loss_instance_one = loss_fn(pred_instance_one, target_proj_instance_two.detach())
        loss_instance_two = loss_fn(pred_instance_two, target_proj_instance_one.detach())

        instance_loss = (loss_instance_one + loss_instance_two).mean()
        # instance_loss_range = (loss_instance_one_range + loss_instance_two_range).mean()

        if positive_pixel_pairs == 0:
            ret = (instance_loss, 0) if return_positive_pairs else instance_loss
            return ret

        if not self.use_pixpro:
            # calculate pix contrast loss

            proj_pixel_one_rgb, proj_pixel_two_rgb = list(map(flatten, (proj_pixel_one_rgb, proj_pixel_two_rgb)))

            similarity_one_two_rgb = F.cosine_similarity(proj_pixel_one_rgb[..., :, None],
                                                         target_proj_pixel_two_rgb[..., None, :],
                                                         dim=1) / self.similarity_temperature
            similarity_two_one_rgb = F.cosine_similarity(proj_pixel_two_rgb[..., :, None],
                                                         target_proj_pixel_one_rgb[..., None, :],
                                                         dim=1) / self.similarity_temperature

            proj_pixel_one_range, proj_pixel_two_range = list(
                map(flatten, (proj_pixel_one_range, proj_pixel_two_range)))

            similarity_one_two_range = F.cosine_similarity(proj_pixel_one_range[..., :, None],
                                                           target_proj_pixel_two_range[..., None, :],
                                                           dim=1) / self.similarity_temperature
            similarity_two_one_range = F.cosine_similarity(proj_pixel_two_range[..., :, None],
                                                           target_proj_pixel_one_range[..., None, :],
                                                           dim=1) / self.similarity_temperature

            loss_pix_one_two = -torch.log(
                similarity_one_two_rgb.masked_select(positive_mask_one_two[None, ...]).exp().sum() /
                similarity_one_two_rgb.exp().sum()
            )

            loss_pix_two_one = -torch.log(
                similarity_two_one_rgb.masked_select(positive_mask_two_one[None, ...]).exp().sum() /
                similarity_two_one_rgb.exp().sum()
            )

            pix_loss = (loss_pix_one_two + loss_pix_two_one) / 2
        else:
            # calculate pix pro loss

            propagated_pixels_one = self.propagate_pixels(proj_pixel_one_rgb + proj_pixel_one_range)
            propagated_pixels_two = self.propagate_pixels(proj_pixel_two_rgb + proj_pixel_two_range)

            propagated_pixels_one, propagated_pixels_two = list(
                map(flatten, (propagated_pixels_one, propagated_pixels_two)))

            target_sum_two = (target_proj_pixel_two_rgb + target_proj_pixel_two_range)
            target_sum_one = (target_proj_pixel_one_rgb + target_proj_pixel_one_range)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            prop_pix_one = F.normalize(propagated_pixels_one, dim=1)
            targ_sum_two = F.normalize(target_sum_two, dim=1)
            propagated_similarity_one_two = torch.bmm(prop_pix_one.transpose(1, 2), targ_sum_two)

            prop_pix_two = F.normalize(propagated_pixels_two, dim=1)
            targ_sum_one = F.normalize(target_sum_one, dim=1)
            propagated_similarity_two_one = torch.bmm(prop_pix_two.transpose(1, 2), targ_sum_one)

            # propagated_similarity_one_two = F.cosine_similarity(propagated_pixels_one,
            #                                                     target_sum_two, dim=1)
            # propagated_similarity_two_one = F.cosine_similarity(propagated_pixels_two,
            #                                                     target_sum_one, dim=1)


            loss_pixpro_one_two = - propagated_similarity_one_two.masked_select(positive_mask_one_two.unsqueeze(0)).mean()
            loss_pixpro_two_one = - propagated_similarity_two_one.masked_select(positive_mask_two_one.unsqueeze(0)).mean()

            pix_loss = (loss_pixpro_one_two + loss_pixpro_two_one) / 2

        # total loss

        loss = pix_loss * self.alpha + instance_loss

        log_data = {
            "loss": loss,
            "pixel_loss": pix_loss,
            "instance_loss": instance_loss,
        }

        ret = (log_data, positive_pixel_pairs) if return_positive_pairs else log_data
        return ret