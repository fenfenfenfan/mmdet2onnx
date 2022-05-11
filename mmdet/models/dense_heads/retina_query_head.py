# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from ..spase_heads import QueryInfer,RetinaNetHead_3x3,Head_3x3,\
                          RetinaNetHead_3x3_MergeBN,Head_3x3_MergeBN,\
                          get_box_scales,get_anchor_center_min_dis
from mmdet.core import build_prior_generator, multi_apply,images_to_levels
@HEADS.register_module()
class RetinaQueryHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 query_head_dict=dict(
                     query_threshold = 0.12,
                     query_head = [1,2],# 代表P3,P4
                     ),
                 query_anchor_generator=dict(
                     type='CenterAnchorGenerator',
                     base_size=[64, 128],
                     ratios=[1.0],
                     strides=[8, 16]),
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 loss_qry=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    alpha=0.25,
                    gamma=1.2,
                    loss_weight=10.0),  
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaQueryHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        # query part
        self.loss_qry = build_loss(loss_qry)
        self.query_head_dict = query_head_dict

        self.query_head = Head_3x3(256, 256, 4, 1)
            
        self.qInfer = QueryInfer(9, num_classes, query_head_dict["query_threshold"]) 
        self.query_anchor_generator = build_prior_generator(query_anchor_generator) 


    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def get_params(self):
        cls_weights = [x.weight for x in self.cls_convs] + [self.retina_cls.weight.data]
        cls_biases = [x.bias for x in self.cls_convs] + [self.retina_cls.bias.data]

        bbox_weights = [x.weight for x in self.reg_convs] + [self.retina_reg.weight.data]
        bbox_biases = [x.bias for x in self.reg_convs] + [self.retina_reg.bias.data]
        
        return cls_weights, cls_biases, bbox_weights, bbox_biases

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        qry_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        qry_pred = self.query_head(qry_feat)  
        # cls_score:(N,C1,H,W)
        # bbox_pred:(N,C2,H,W)
        return cls_score, bbox_pred, qry_pred

    def forward(self, feats):
        """
        feats:FPN结构输出,[(N,C,H,W),(N,C,2H,2W)...]
        输出：
            不同尺度的结果，每个结果里是所有图在同一尺度下的结果
        """
        if self.training:
            # return ([cls_score,cls_score...],[bbox_pred,bbox_pred...],[qry_pred,qry_pred...])
            return multi_apply(self.forward_single, feats)
        else:
            return self.test_forward(feats)

    @torch.no_grad()
    def get_query_gt(self, anchor_center, targets):
        # get_box_scales 等于 sqrt(h*w)，衡量gtbox大小
        target_box_scales = get_box_scales(targets)
            
        center_dis, minarg = get_anchor_center_min_dis((targets[:2]+targets[2:])/2, anchor_center)
        small_obj_target = torch.zeros_like(center_dis)
        
        if len(targets) != 0:
            min_small_target_scale = target_box_scales[minarg]
            small_obj_target[center_dis < min_small_target_scale * 1.0] = 1

        return small_obj_target

    def loss_single(self, 
                    cls_score, bbox_pred, qry_pred,
                    anchors, qry_centers,
                    labels, label_weights, 
                    bbox_targets, bbox_weights, 
                    num_total_samples):

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
            
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # query loss
        qry_loss = 0
        if qry_centers is not None:
            qry_targets = self.get_query_gt(qry_centers, bbox_targets)
            qry_x = qry_pred.flatten()
            qry_y = qry_targets.flatten()
            qry_loss = self.loss_qry(
                qry_x, 
                qry_y, 
                avg_factor=num_total_samples) 
        return loss_cls, loss_bbox, qry_loss
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds','qry_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             qry_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        print("featmap_sizes:",featmap_sizes)
        print("位置211，这里特征图大小需要从大到小")
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            #这一步把每个图的anchors都cat到了一起，[tensor,tensor]->tensor(size:Nx4)
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        # 把每张图的同一个尺度的anchors放一起
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        # 我需要做的是把每张图同一水平的center放一起？
        # query part
        query_head = self.query_head_dict["query_head"]
        # P3,P4的featmap_sizes
        featmap_sizes = featmap_sizes[query_head]
        # P3,P4的anchor数量
        num_level_centers = num_level_anchors[query_head]
        # centers list:[(a,2),(b,2)...]
        _, centers = self.query_anchor_generator.grid_priors(
            featmap_sizes, device=device)
        num_imgs = len(img_metas)
        # 每张图都分配centers
        centers_list = [centers for _ in range(num_imgs)]

        concat_centers_list = []
        for i in range(num_imgs):
            #这一步把每个图的centers都cat到了一起，[(a,2),(b,2)...]->tensor(size:(a+b)x2)
            concat_centers_list.append(torch.cat(centers_list[i]))
        # 把每张图的同一个尺度的centers放一起
        num_level = list(range(len(num_level_anchors)))
        centers_list = images_to_levels(concat_centers_list, num_level_centers)
        all_centers_list = [centers_list[i] \
                            if num_level.pop(0)==query_head[i] else None \
                                for i in range(len(query_head))]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores, bbox_preds, qry_preds,
            all_anchor_list, all_centers_list,
            labels_list, label_weights_list,
            bbox_targets_list, bbox_weights_list,
            num_total_samples=num_total_samples,)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


    def test_forward(self, feats):
        query_head_layers = self.query_head_dict["query_head"]

        num_imgs = len(feats[0].size()[0])
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        device = feats[0].device
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        # P4及以上部分的
        anchors_above  = multi_level_anchors[query_head_layers[-2]:]
        # P2,P3的anchors
        anchors_value  = [multi_level_anchors[i-1] for i in query_head_layers]

        # P4及以上部分的特征
        if len(feats)>(query_head_layers[-1]+1):
            feature_above = feats[query_head_layers[-2]:]
        # 要经过query头的特征 P3,P4
        feature_key = [feats[i] for i in query_head_layers]
        # 要与query头输出结合的特征 P2,P3
        feature_value = [feats[i-1] for i in query_head_layers]

        det_cls_above, det_delta_above = multi_apply(self.forward_single, feature_above)

        if not self.qInfer.initialized:
            cls_weights, cls_biases, bbox_weights, bbox_biases = self.get_params()
            qcls_weights, qcls_bias = self.query_head.get_params()
            params = [cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights, qcls_bias]
        else:
            params = None

        det_cls_query, det_bbox_query, query_anchors = self.qInfer.run_qinfer(params, feature_key, feature_value, anchors_value)
        for i in det_cls_above:
            print(i.size())
        for i in det_bbox_query:
            print(i.size())
        print("位置315，这里的特征图大小需要单调增减")
        return det_cls_above+det_cls_query, det_delta_above+det_bbox_query
