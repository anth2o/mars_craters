import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import time
import gc
import pandas as pd
import numpy as np
from PIL import Image
import random
import sys
import math
import io
 
# yolo layer
 
class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
 
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.rescore = 0
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.
        self.nth_layer = 0
        self.seen = 0
        self.net_width = 0
        self.net_height = 0
 
    def get_mask_boxes(self, output):
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
 
        masked_anchors = torch.FloatTensor(masked_anchors).to(self.device)
        num_anchors = torch.IntTensor([len(self.anchor_mask)]).to(self.device)
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}
 
    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):
        nB = target.size(0)
        anchor_step = anchors.size(1)  # anchors[nA][anchor_step]
        noobj_mask = torch.ones(nB, nA, nH, nW)
        obj_mask = torch.zeros(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW, self.num_classes)
 
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0
 
        # it works faster on CPU than on GPU.
        anchors = anchors.to("cpu")
 
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5).to("cpu")
 
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * self.net_width, tbox[t][4] * self.net_height
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = (cur_ious > self.ignore_thresh).view(nA, nH, nW)
            noobj_mask[b][ignore_ix] = 0
 
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * self.net_width, tbox[t][4] * self.net_height
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)
 
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors), 1).t()
                _, best_n = torch.max(multi_bbox_ious(anchor_boxes, tmp_gt_boxes, x1y1x2y2=False), 0)
 
                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
 
                obj_mask[b][best_n][gj][gi] = 1
                noobj_mask[b][best_n][gj][gi] = 0
                coord_mask[b][best_n][gj][gi] = 2. - tbox[t][3] * tbox[t][4]
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi][int(tbox[t][0])] = 1
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.
 
                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1
 
        return nGT, nRecall, nRecall75, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls
 
    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        mask_tuple = self.get_mask_boxes(output)
        t0 = time.time()
        nB = output.data.size(0)  # batch size
        nA = mask_tuple['n'].item()  # num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        anchor_step = mask_tuple['a'].size(0) // nA
        anchors = mask_tuple['a'].view(nA, anchor_step).to(self.device)
        cls_anchor_dim = nB * nA * nH * nW
 
        output = output.view(nB, nA, (5 + nC), nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to(self.device)
        ix = torch.LongTensor(range(0, 5)).to(self.device)
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(self.device)
 
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1,
                                                                                                             cls_anchor_dim)  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()
 
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC).to(self.device)
 
        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim).to(
            self.device)
        anchor_w = anchors.index_select(1, ix[0]).repeat(nB, nH * nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(nB, nH * nW).view(cls_anchor_dim)
 
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()
 
        t2 = time.time()
        nGT, nRecall, nRecall75, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)
 
        tcls = tcls.view(cls_anchor_dim, nC).to(self.device)
 
        nProposals = int((conf > 0.25).sum())
 
        tcoord = tcoord.view(4, cls_anchor_dim).to(self.device)
        tconf = tconf.view(cls_anchor_dim).to(self.device)
 
        conf_mask = (obj_mask + noobj_mask).view(cls_anchor_dim).to(self.device)
        obj_mask = obj_mask.view(cls_anchor_dim).to(self.device)
        coord_mask = coord_mask.view(cls_anchor_dim).to(self.device)
 
        t3 = time.time()
        loss_coord = nn.MSELoss(size_average=False)(coord * coord_mask, tcoord * coord_mask) / nB
        loss_conf = nn.BCELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / nB
        loss_cls = nn.BCEWithLogitsLoss(size_average=False)(cls, tcls) / nB
        loss = loss_coord + loss_conf + loss_cls
 
        t4 = time.time()
 
        print(
            '%d: Layer(%03d) nGT %3d, nRC %3d, nRC75 %3d, nPP %3d, loss: box %6.3f, conf %6.3f, class %6.3f, total %7.3f'
            % (self.seen, self.nth_layer, nGT, nRecall, nRecall75, nProposals, loss_coord, loss_conf, loss_cls, loss))
        if math.isnan(loss.item()):
            print(conf, tconf)
            sys.exit(0)
        return loss
 
# regional layer
 
class RegionLayer(nn.Module):
    def __init__(self, num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(RegionLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        # self.anchors = torch.stack(torch.FloatTensor(anchors).split(self.anchor_step)).to(self.device)
        self.anchors = torch.FloatTensor(anchors).view(self.num_anchors, self.anchor_step).to(self.device)
        self.rescore = 1
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
 
    def build_targets(self, pred_boxes, target, nH, nW):
        nB = target.size(0)
        nA = self.num_anchors
        noobj_mask = torch.ones(nB, nA, nH, nW)
        obj_mask = torch.zeros(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)
 
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0  # number of ground truth
        nRecall = 0
        # it works faster on CPU than on GPU.
        anchors = self.anchors.to("cpu")
 
        if self.seen < 12800:
            tcoord[0].fill_(0.5)
            tcoord[1].fill_(0.5)
            coord_mask.fill_(0.01)
            # initial w, h == 0 means log(1)==0, s.t, anchor is equal to ground truth.
 
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5).to("cpu")
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gw = [i * nW for i in (tbox[t][1], tbox[t][3])]
                gy, gh = [i * nH for i in (tbox[t][2], tbox[t][4])]
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = (cur_ious > self.thresh).view(nA, nH, nW)
            noobj_mask[b][ignore_ix] = 0
 
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gw = [i * nW for i in (tbox[t][1], tbox[t][3])]
                gy, gh = [i * nH for i in (tbox[t][2], tbox[t][4])]
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)
 
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, 2), anchors), 1).t()
                tmp_ious = multi_bbox_ious(anchor_boxes, tmp_gt_boxes, x1y1x2y2=False)
                best_iou, best_n = torch.max(tmp_ious, 0)
 
                if self.anchor_step == 4:  # this part is not tested.
                    tmp_ious_mask = (tmp_ious == best_iou)
                    if tmp_ious_mask.sum() > 0:
                        gt_pos = torch.FloatTensor([gi, gj, gx, gy]).repeat(nA, 1).t()
                        an_pos = anchor_boxes[4:6]  # anchor_boxes are consisted of [0 0 aw ah ax ay]
                        dist = pow(((gt_pos[0] + an_pos[0]) - gt_pos[2]), 2) + pow(
                            ((gt_pos[1] + an_pos[1]) - gt_pos[3]), 2)
                        dist[1 - tmp_ious_mask] = 10000  # set the large number for the small ious
                        _, best_n = torch.min(dist, 0)
 
                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
 
                obj_mask[b][best_n][gj][gi] = 1
                noobj_mask[b][best_n][gj][gi] = 0
                coord_mask[b][best_n][gj][gi] = 2. - tbox[t][3] * tbox[t][4]
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi] = tbox[t][0]
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.
                if iou > 0.5:
                    nRecall += 1
 
        return nGT, nRecall, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls
 
    def get_mask_boxes(self, output):
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step).to(self.device)
        masked_anchors = self.anchors.view(-1)
        num_anchors = torch.IntTensor([self.num_anchors]).to(self.device)
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}
 
    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)  # batch size
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls_anchor_dim = nB * nA * nH * nW
 
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step).to(self.device)
 
        output = output.view(nB, nA, (5 + nC), nH, nW).to(self.device)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to(self.device)
        ix = torch.LongTensor(range(0, 5)).to(self.device)
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(self.device)
 
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1,
                                                                                                             cls_anchor_dim)  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()
 
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)
 
        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim).to(
            self.device)
        anchor_w = self.anchors.index_select(1, ix[0]).repeat(nB, nH * nW).view(cls_anchor_dim)
        anchor_h = self.anchors.index_select(1, ix[1]).repeat(nB, nH * nW).view(cls_anchor_dim)
 
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()
 
        t2 = time.time()
        nGT, nRecall, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), nH, nW)
 
        cls_mask = (obj_mask == 1)
        tcls = tcls[cls_mask].long().view(-1).to(self.device)
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).to(self.device)
        cls = cls[cls_mask].view(-1, nC)
 
        nProposals = int((conf > 0.25).sum())
 
        tcoord = tcoord.view(4, cls_anchor_dim).to(self.device)
        tconf = tconf.view(cls_anchor_dim).to(self.device)
 
        conf_mask = (self.object_scale * obj_mask + self.noobject_scale * noobj_mask).view(cls_anchor_dim).to(
            self.device)
        obj_mask = obj_mask.view(cls_anchor_dim).to(self.device)
        coord_mask = coord_mask.view(cls_anchor_dim).to(self.device)
 
        t3 = time.time()
        loss_coord = self.coord_scale * nn.MSELoss(size_average=False)(coord * coord_mask, tcoord * coord_mask) / nB
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / nB
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls) / nB
        loss = loss_coord + loss_conf + loss_cls
 
        t4 = time.time()
        print('%d: nGT %3d, nRC %3d, nPP %3d, loss: box %6.3f, conf %6.3f, class %6.3f, total %7.3f'
              % (self.seen, nGT, nRecall, nProposals, loss_coord, loss_conf, loss_cls, loss))
        if math.isnan(loss.item()):
            print(conf, tconf)
            sys.exit(0)
        return loss
 
 
# DARKNET
 
def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)
 
def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)
 
 
def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)
 
def parse_cfg(cfgfile):
    blocks = []
    fp = io.StringIO(cfgfile)
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()
 
    if block:
        blocks.append(block)
    fp.close()
    return blocks
 
def print_cfg(blocks):
    print('layer     filters    size              input                output')
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
            height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
            ind, 'avg', prev_width, prev_height, prev_filters, prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))
 
def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start
 
 
def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    # conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data))
    start = start + num_w
    return start
 
 
def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    # print("start: {}, num_w: {}, num_b: {}".format(start, num_w, num_b))
    # by ysyun, use .view_as()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]).view_as(conv_model.bias.data));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data));
    start = start + num_w
    return start
 
 
class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()
 
    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x
 
 
class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
 
    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, hs, W, ws).contiguous().view(B, C, H * hs, W * ws)
        return x
 
 
class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
 
    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, (H // hs) * (W // ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x
 
 
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
 
    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x
 
 
# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()
 
    def forward(self, x):
        return x
 
 
# support route shortcut and reorg
 
class Darknet(nn.Module):
    def net_name(self):
        names_list = ('region', 'yolo')
        name = names_list[0]
        for m in self.models:
            if isinstance(m, YoloLayer):
                name = names_list[1]
        return name
 
    def getLossLayers(self):
        loss_layers = []
        for m in self.models:
            if isinstance(m, RegionLayer) or isinstance(m, YoloLayer):
                loss_layers.append(m)
        return loss_layers
 
    def __init__(self, cfgfile, use_cuda=True):
        super(Darknet, self).__init__()
        self.use_cuda = use_cuda
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.loss_layers = self.getLossLayers()
 
        # self.width = int(self.blocks[0]['width'])
        # self.height = int(self.blocks[0]['height'])
 
        if len(self.loss_layers) > 0:
            last = len(self.loss_layers) - 1
            self.anchors = self.loss_layers[last].anchors
            self.num_anchors = self.loss_layers[last].num_anchors
            self.anchor_step = self.loss_layers[last].anchor_step
            self.num_classes = self.loss_layers[last].num_classes
 
        # default format : major=0, minor=1
        self.header = torch.IntTensor([0, 1, 0, 0])
        self.seen = 0
 
    def forward(self, x):
        ind = -2
        # self.loss_layers = None
        outputs = dict()
        out_boxes = dict()
        outno = 0
        for block in self.blocks:
            ind = ind + 1
 
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] in ['region', 'yolo']:
                boxes = self.models[ind].get_mask_boxes(x)
                out_boxes[outno] = boxes
                outno += 1
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x if outno == 0 else out_boxes
 
    def print_network(self):
        print_cfg(self.blocks)
 
    def create_network(self, blocks):
        models = nn.ModuleList()
 
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        ind = -2
        for block in blocks:
            ind += 1
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                self.width = int(block['width'])
                self.height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=False)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride / stride
                out_strides.append(prev_stride)
                # models.append(nn.Upsample(scale_factor=stride, mode='nearest'))
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                region_layer = RegionLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                region_layer.anchors = [float(i) for i in anchors]
                region_layer.num_classes = int(block['classes'])
                region_layer.num_anchors = int(block['num'])
                region_layer.anchor_step = len(region_layer.anchors) // region_layer.num_anchors
                region_layer.rescore = int(block['rescore'])
                region_layer.object_scale = float(block['object_scale'])
                region_layer.noobject_scale = float(block['noobject_scale'])
                region_layer.class_scale = float(block['class_scale'])
                region_layer.coord_scale = float(block['coord_scale'])
                region_layer.thresh = float(block['thresh'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(region_layer)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                try:
                    yolo_layer.rescore = int(block['rescore'])
                except:
                    pass
                yolo_layer.ignore_thresh = float(block['ignore_thresh'])
                yolo_layer.truth_thresh = float(block['truth_thresh'])
                yolo_layer.stride = prev_stride
                yolo_layer.nth_layer = ind
                yolo_layer.net_width = self.width
                yolo_layer.net_height = self.height
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))
 
        return models
 
    def load_binfile(self, weightfile):
        fp = open(weightfile, 'rb')
 
        version = np.fromfile(fp, count=3, dtype=np.int32)
        version = [int(i) for i in version]
        if version[0] * 10 + version[1] >= 2 and version[0] < 1000 and version[1] < 1000:
            seen = np.fromfile(fp, count=1, dtype=np.int64)
        else:
            seen = np.fromfile(fp, count=1, dtype=np.int32)
        self.header = torch.from_numpy(np.concatenate((version, seen), axis=0))
        self.seen = int(seen)
        body = np.fromfile(fp, dtype=np.float32)
        fp.close()
        return body
 
    def load_weights(self, weightfile):
        buf = self.load_binfile(weightfile)
 
        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
 
    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
 
        dirname = os.path.dirname(outfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
 
        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = np.array(self.header[0:3].numpy(), np.int32)
        header.tofile(fp)
        if (self.header[0] * 10 + self.header[1]) >= 2:
            seen = np.array(self.seen, np.int64)
        else:
            seen = np.array(self.seen, np.int32)
        seen.tofile(fp)
 
        ind = -1
        for blockId in range(1, cutoff + 1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
 
 
# UTILS
 
def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)
 
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea
 
def correct_yolo_boxes(boxes, im_w, im_h, net_w, net_h):
    im_w, im_h = float(im_w), float(im_h)
    net_w, net_h = float(net_w), float(net_h)
    if net_w / im_w < net_h / im_h:
        new_w = net_w
        new_h = (im_h * net_w) / im_w
    else:
        new_w = (im_w * net_h) / im_h
        new_h = net_h
 
    xo, xs = (net_w - new_w) / (2 * net_w), net_w / new_w
    yo, ys = (net_h - new_h) / (2 * net_h), net_h / new_h
    for i in range(len(boxes)):
        b = boxes[i]
        b[0] = (b[0] - xo) * xs
        b[1] = (b[1] - yo) * ys
        b[2] *= xs
        b[3] *= ys
    return
 
 
def letterbox_image(img, net_w, net_h):
    im_w, im_h = img.size
    if float(net_w) / float(im_w) < float(net_h) / float(im_h):
        new_w = net_w
        new_h = (im_h * net_w) // im_w
    else:
        new_w = (im_w * net_h) // im_h
        new_h = net_h
    resized = img.resize((new_w, new_h), Image.ANTIALIAS)
    lbImage = Image.new("RGB", (net_w, net_h), (127, 127, 127))
    lbImage.paste(resized, \
                  ((net_w - new_w) // 2, (net_h - new_h) // 2, \
                   (net_w + new_w) // 2, (net_h + new_h) // 2))
    return lbImage
 
 
def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0] - w1 / 2.0, box2[0] - w2 / 2.0)
        x2_max = max(box1[0] + w1 / 2.0, box2[0] + w2 / 2.0)
        y1_min = min(box1[1] - h1 / 2.0, box2[1] - h2 / 2.0)
        y2_max = max(box1[1] + h1 / 2.0, box2[1] + h2 / 2.0)
 
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0
 
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)
 
 
def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)
 
 
def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)
 
 
def get_region_boxes(output, netshape, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                     validation=False, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")
    anchors = anchors.to(device)
    anchor_step = anchors.size(0) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch * num_anchors * h * w
    if netshape[0] != 0:
        nw, nh = netshape
    else:
        nw, nh = w, h
 
    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
                                                                                                        cls_anchor_dim)
 
    grid_x = torch.linspace(0, w - 1, w).repeat(batch * num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(cls_anchor_dim).to(
        device)
    ix = torch.LongTensor(range(0, 2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(batch, h * w).view(cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(batch, h * w).view(cls_anchor_dim)
 
    xs, ys = output[0].sigmoid() + grid_x, output[1].sigmoid() + grid_y
    ws, hs = output[2].exp() * anchor_w.detach(), output[3].exp() * anchor_h.detach()
    det_confs = output[4].sigmoid()
 
    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5 + num_classes].transpose(0, 1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
 
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs, ys = convert2cpu(xs), convert2cpu(ys)
    ws, hs = convert2cpu(ws), convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
 
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    conf = det_conf * (cls_max_confs[ind] if not only_objectness else 1.0)
 
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / nw, bh / nh, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
 
    return all_boxes
 
 
def get_all_boxes(output, netshape, conf_thresh, num_classes, only_objectness=1, validation=False, use_cuda=True):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    tot = output[0]['x'].data.size(0)
    all_boxes = [[] for i in range(tot)]
    for i in range(len(output)):
        pred = output[i]['x'].data
 
        # find number of workers (.s.t, number of GPUS)
        nw = output[i]['n'].data.size(0)
        anchors = output[i]['a'].chunk(nw)[0]
        num_anchors = output[i]['n'].data[0].item()
 
        b = get_region_boxes(pred, netshape, conf_thresh, num_classes, anchors, num_anchors, \
                             only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)
        for t in range(tot):
            all_boxes[t] += b[t]
    return all_boxes
 
 
def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img
 
 
def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
 
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]
 
    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes
 
 
def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True):
    model.eval()
    t0 = time.time()
    img = image2torch(img)
    t1 = time.time()
 
    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    t2 = time.time()
 
    out_boxes = model(img)
    if model.net_name() == 'region':  # region_layer
        shape = (0, 0)
    else:
        shape = (model.width, model.height)
    boxes = get_all_boxes(out_boxes, shape, conf_thresh, model.num_classes, use_cuda=use_cuda)[0]
 
    t3 = time.time()
    boxes = nms(boxes, nms_thresh)
    t4 = time.time()
 
    return boxes
 
 
def image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy):
    scaled = img.resize((new_w, new_h))
    # find to be cropped area
    sx, sy = -dx if dx < 0 else 0, -dy if dy < 0 else 0
    ex, ey = new_w if sx + new_w <= net_w else net_w - sx, new_h if sy + new_h <= net_h else net_h - sy
    scaled = scaled.crop((sx, sy, ex, ey))
 
    # find the paste position
    sx, sy = dx if dx > 0 else 0, dy if dy > 0 else 0
    assert sx + scaled.width <= net_w and sy + scaled.height <= net_h
    new_img = Image.new("RGB", (net_w, net_h), (127, 127, 127))
    new_img.paste(scaled, (sx, sy))
    del scaled
    return new_img
 
 
def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
 
    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
 
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))
 
    im = im.convert('RGB')
    # constrain_image(im)
    return im
 
 
def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(2):
        return scale
    return 1. / scale
 
 
def random_distort_image(im, hue, saturation, exposure):
    dhue = np.random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    return distort_image(im, dhue, dsat, dexp)
 
 
def data_augmentation_crop(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width
 
    dw = int(ow * jitter)
    dh = int(oh * jitter)
 
    pleft = np.random.randint(-dw, dw)
    pright = np.random.randint(-dw, dw)
    ptop = np.random.randint(-dh, dh)
    pbot = np.random.randint(-dh, dh)
 
    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot
 
    sx = ow / float(swidth)
    sy = oh / float(sheight)
 
    flip = np.random.randint(2)
 
    cropbb = np.array([pleft, ptop, pleft + swidth - 1, ptop + sheight - 1])
    # following two lines are old method. out of image boundary is filled with black (0,0,0)
    # cropped = img.crop( cropbb )
    # sized = cropped.resize(shape)
 
    nw, nh = cropbb[2] - cropbb[0], cropbb[3] - cropbb[1]
    # get the real image part
    cropbb[0] = -min(cropbb[0], 0)
    cropbb[1] = -min(cropbb[1], 0)
    cropbb[2] = min(cropbb[2], ow)
    cropbb[3] = min(cropbb[3], oh)
    cropped = img.crop(cropbb)
 
    # calculate the position to paste
    bb = (pleft if pleft > 0 else 0, ptop if ptop > 0 else 0)
    new_img = Image.new("RGB", (nw, nh), (127, 127, 127))
    new_img.paste(cropped, bb)
 
    sized = new_img.resize(shape)
    del cropped, new_img
 
    dx = (float(pleft) / ow) * sx
    dy = (float(ptop) / oh) * sy
 
    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    # for compatibility to nocrop version (like original version)
    return img, flip, dx, dy, sx, sy
 
 
def data_augmentation_nocrop(img, shape, jitter, hue, sat, exp):
    net_w, net_h = shape
    img_w, img_h = img.width, img.height
 
    # determine the amount of scaling and cropping
    dw = jitter * img_w
    dh = jitter * img_h
 
    new_ar = (img_w + np.random.uniform(-dw, dw)) / (img_h + np.random.uniform(-dh, dh))
    # scale = np.random.uniform(0.25, 2)
    scale = 1.
 
    if (new_ar < 1):
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)
 
    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))
    sx, sy = new_w / net_w, new_h / net_h
 
    # apply scaling and shifting
    new_img = image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy)
 
    # randomly distort hsv space
    new_img = random_distort_image(new_img, hue, sat, exp)
 
    # randomly flip
    flip = np.random.randint(2)
    if flip:
        new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
 
    dx, dy = dx / net_w, dy / net_h
    return new_img, flip, dx, dy, sx, sy
 
 
# ------------
# MY-UTILS
 
def load_data_detection(im_raw, lab, shape, crop, jitter, hue, saturation, exposure):
    img = im_raw.convert('RGB')
    if crop:  # marvis version
        img, flip, dx, dy, sx, sy = data_augmentation_crop(img, shape, jitter, hue, saturation, exposure)
    else:  # original version
        img, flip, dx, dy, sx, sy = data_augmentation_nocrop(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection_(lab, crop, flip, -dx, -dy, sx, sy)
    return img, label
 
 
def lab_to_label(lab, im_size):
    lab = np.array(lab)
    xmin = np.maximum(lab[:, 1] - lab[:, 2], 1.0) / im_size
    xmax = np.minimum(lab[:, 1] + lab[:, 2], im_size - 1) / im_size
    ymin = np.maximum(lab[:, 0] - lab[:, 2], 1.0) / im_size
    ymax = np.minimum(lab[:, 0] + lab[:, 2], im_size - 1) / im_size
    x_ = (xmin + xmax) / 2
    y_ = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return np.stack([np.zeros(x_.shape), x_, y_, width, height], axis=1)
 
 
def fill_truth_detection_(lab, crop, flip, dx, dy, sx, sy):
    max_boxes = 50
    im_size = 224
    label = np.zeros((max_boxes, 5))
    if lab is None:
        return label
    bs = lab_to_label(lab, im_size)
 
    cc = 0
    for i in range(bs.shape[0]):
        x1 = bs[i][1] - bs[i][3] / 2
        y1 = bs[i][2] - bs[i][4] / 2
        x2 = bs[i][1] + bs[i][3] / 2
        y2 = bs[i][2] + bs[i][4] / 2
 
        x1 = min(0.999, max(0, x1 * sx - dx))
        y1 = min(0.999, max(0, y1 * sy - dy))
        x2 = min(0.999, max(0, x2 * sx - dx))
        y2 = min(0.999, max(0, y2 * sy - dy))
 
        bs[i][1] = (x1 + x2) / 2  # center x
        bs[i][2] = (y1 + y2) / 2  # center y
        bs[i][3] = (x2 - x1)  # width
        bs[i][4] = (y2 - y1)  # height
 
        if flip:
            bs[i][1] = 0.999 - bs[i][1]
 
            # when crop is applied, we should check the cropped width/height ratio
        if bs[i][3] < 0.002 or bs[i][4] < 0.002 or \
                (crop and (bs[i][3] / bs[i][4] > 20 or bs[i][4] / bs[i][3] > 20)):
            continue
        label[cc] = bs[i]
        cc += 1
        if cc >= 50:
            break
 
    label = np.reshape(label, (-1))
    return label
 
 
# ------------
 
def unique(y):
    ids = []
    for i in range(y.shape[0]):
        if len(y[i]) > 0:
            ids.append(i)
    return np.array(ids)
 
class Data(object):
    def __init__(self, X, y, shape=None, shuffle=True, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5,
                 transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
        self.X = X
        self.y = y
        self.idxs = unique(y)
        if shuffle:
            np.random.shuffle(self.idxs)
 
        self.nSamples = self.idxs.shape[0]
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
 
        self.crop = crop
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
 
    def __len__(self):
        return self.nSamples
 
    def get_different_scale(self):
        if self.seen < 4000 * 64:
            wh = 13 * 32  # 416
        elif self.seen < 8000 * 64:
            wh = (random.randint(0, 3) + 13) * 32  # 416, 480
        elif self.seen < 12000 * 64:
            wh = (random.randint(0, 5) + 12) * 32  # 384, ..., 544
        elif self.seen < 16000 * 64:
            wh = (random.randint(0, 7) + 11) * 32  # 352, ..., 576
        else:  # self.seen < 20000*64:
            wh = (random.randint(0, 9) + 10) * 32  # 320, ..., 608
        return (wh, wh)
 
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        id = self.idxs[index]
        raw_im = np.empty(self.X[id].shape + (3,))
        im_size = self.X[id].shape[0]
        raw_im[:, :, 0] = raw_im[:, :, 1] = raw_im[:, :, 2] = self.X[id]
        raw_im = Image.fromarray(np.uint8(raw_im))
        lab = self.y[id]
 
        if self.train:
 
            if index % 64 == 0:
                self.shape = self.get_different_scale()
            img, label = load_data_detection(raw_im, lab, self.shape, self.crop, self.jitter, self.hue, self.saturation,
                                             self.exposure)
            label = torch.from_numpy(label)
        else:
            img = raw_im.convert('RGB')
            if self.shape:
                img, org_w, org_h = letterbox_image(img, self.shape[0], self.shape[1]), img.width, img.height
 
            label = torch.zeros(50 * 5)
            try:
                # error on this part
                tmp = torch.from_numpy(lab_to_label(lab, im_size))
            except Exception:
                tmp = torch.zeros(1, 5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            if tsz > 50 * 5:
                label = tmp[0:50 * 5]
            elif tsz > 0:
                label[0:tsz] = tmp
 
        if self.transform is not None:
            img = self.transform(img)
 
        if self.target_transform is not None:
            label = self.target_transform(label)
 
        self.seen = self.seen + self.num_workers
        if self.train:
            return (img, label)
        else:
            return (img, label, org_w, org_h)
 
 
# global vars
use_cuda = torch.cuda.is_available()
cfgfile = '[net]\nbatch=8\nsubdivisions=16\nwidth=416\nheight=416\nchannels=3\nmomentum=0.9\ndecay=0.0005\nangle=0\nsaturation = 1.5\nexposure = 1.5\nhue=.1\nlearning_rate=0.001\nburn_in=1000\nmax_batches = 500200\npolicy=steps\nsteps=400000,450000\nscales=.1,.1\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=2\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=2\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=2\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=2\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=2\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\nactivation=leaky\n[shortcut]\nfrom=-3\nactivation=linear\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=1024\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=1024\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=1024\nactivation=leaky\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n[yolo]\nmask = 6,7,8\nanchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\nclasses=80\nnum=9\njitter=.3\nignore_thresh = .5\ntruth_thresh = 1\nrandom=1\n[route]\nlayers = -4\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[upsample]\nstride=2\n[route]\nlayers = -1, 61\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=512\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=512\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=512\nactivation=leaky\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n[yolo]\nmask = 3,4,5\nanchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\nclasses=80\nnum=9\njitter=.3\nignore_thresh = .5\ntruth_thresh = 1\nrandom=1\n[route]\nlayers = -4\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[upsample]\nstride=2\n[route]\nlayers = -1, 36\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=256\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=256\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n[convolutional]\nbatch_normalize=1\nsize=3\nstride=1\npad=1\nfilters=256\nactivation=leaky\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n[yolo]\nmask = 0,1,2\nanchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\nclasses=80\nnum=9\njitter=.3\nignore_thresh = .5\ntruth_thresh = 1\nrandom=1\n'
max_epochs = 60
keep_backup = 5
save_interval = 5  #
device = torch.device("cuda" if use_cuda else "cpu")
 
seed = int(time.time())
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)
 
backupdir = './backup'
ngpus = 1
num_workers = 1
 
batch_size = 8
max_batches = 500200
learning_rate = 0.001
momentum = 0.9
decay = 0.0005
steps = [float(step) for step in '400000,450000'.split(',')]
scales = [float(scale) for scale in '.1,.1'.split(',')]
xpath = './data/my/data_train.npy'
ypath = './data/my/labels_train.csv'
 
 
def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr
 
 
def train(epoch, X, y, model, loss_layers):
    global processed_batches
    init_width = model.width
    init_height = model.height
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
 
    train_loader = torch.utils.data.DataLoader(
        Data(X, y, shape=(init_width, init_height),
             shuffle=True,
             transform=transforms.Compose([
                 transforms.ToTensor(),
             ]),
             train=True,
             seen=model.seen,
             batch_size=batch_size,
             num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)
 
    processed_batches = model.seen // batch_size
    lr = adjust_learning_rate(optimizer, processed_batches)
    model.train()
 
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        org_loss = []
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol = l(output[i]['x'], target)
            org_loss.append(ol)
 
        sum(org_loss).backward()
 
        nn.utils.clip_grad_norm_(model.parameters(), 10000)
        optimizer.step()
 
        del data, target
        org_loss.clear()
        gc.collect()
 
    nsamples = len(train_loader.dataset)
    print('\n[%03d] training with %f samples/s' % (epoch, nsamples / (time.time() - t0)))
    return nsamples
 
 
def savemodel(epoch, model, nsamples, curmax=False):
    if curmax:
        print('save local maximum weights to %s/localmax.weights' % (backupdir))
    else:
        print('save weights to %s/%06d.weights' % (backupdir, epoch))
    model.seen = epoch * nsamples
    if curmax:
        model.save_weights('%s/localmax.weights' % (backupdir))
    else:
        model.save_weights('%s/%06d.weights' % (backupdir, epoch))
        old_wgts = '%s/%06d.weights' % (backupdir, epoch - keep_backup * save_interval)
        try:  # it avoids the unnecessary call to os.path.exists()
            os.remove(old_wgts)
        except OSError:
            pass
 
 
class ObjectDetector(object):
    def __init__(self, weightdir=None):
        model = Darknet(cfgfile, use_cuda)
        self.alr_trained = False
        if weightdir is not None:
            model.load_weights(weightdir)
            model.seen = 0
            self.alr_trained = True
        self.loss_layers = model.loss_layers
        for l in self.loss_layers:
            l.seen = model.seen
        self.model = model.to(device)
        self.weights_path = 'yolo_weights.pt'
 
    def fit(self, X, y):
        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                params += [{'params': [value], 'weight_decay': 0.0}]
            else:
                params += [{'params': [value], 'weight_decay': decay * batch_size}]
        global optimizer
        optimizer = optim.SGD(self.model.parameters(),
                              lr=learning_rate / batch_size, momentum=momentum,
                              dampening=0, weight_decay=decay * batch_size)
        try:
            for epoch in range(1, max_epochs + 1):
                nsamples = train(epoch, X, y, self.model, self.loss_layers)
                #if epoch % save_interval == 0:
                #    savemodel(epoch, self.model, nsamples)
                torch.save(self.model.state_dict(), self.weights_path)
            self.alr_trained = True
        except KeyboardInterrupt:
            print('\nExiting from training by interrupt!')
 
    def loadmodel(self, weightpath):
        self.model.load_weights(weightpath)
 
    def predict(self, X):
        assert self.alr_trained, 'You must train the model first!'
        nb_imgs = X.shape[0]
        labels = []
        for i in range(nb_imgs):
            im = grey_to_bgr(X[i])
            im = Image.fromarray(np.uint8(im))
            sized = letterbox_image(im, self.model.width, self.model.height)
            boxes = do_detect(self.model, sized, 0.5, 0.4, use_cuda)
            correct_yolo_boxes(boxes, im.width, im.height, self.model.width, self.model.height)
            label = []
            for box in boxes:
                label.append(label_to_lab(box, im))
            labels.append(label)
        u = np.empty(shape=(len(labels)), dtype=object)
        for i in range(u.shape[0]):
            u[i] = labels[i]
        return u
 
 
def grey_to_bgr(nparr):
    arr = np.empty(nparr.shape + (3,))
    arr[:, :, 0] = arr[:, :, 1] = arr[:, :, 2] = nparr
    return arr
 
 
def label_to_lab(label, img):
    return (label[5].item(), int(label[1].item() * img.height), int(label[0].item() * img.width),
            max(label[2].item() * img.width, label[3].item() * img.height) / 2)