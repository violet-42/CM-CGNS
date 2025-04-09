import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import random
from transformers import AutoTokenizer, AdamW
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    # 对于需要精确匹配的任务（如人脸验证、文本相似度中的精确匹配），可能需要较小的 temp 来强调差异；而对于需要捕捉更广泛相似性的任务（如语义理解、推荐系统中的相关性推荐），较大的 temp 可能更合适。
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp  # 用于调整相似度度量的“锐度”


class kmeans_cluster(nn.Module):
    def __init__(self, k=128, lr=0.001):
        super().__init__()
        self.k = k
        self.sim = Similarity(temp=1.0)
        self.initialized = False
        self.global_step = 0
        self.optimization = "momentum"  # "adamw",kmeans", "momentum"
        self.lr = lr
        self.beta = 0.5
        self.alpha = 0.2
        self.hard_negative_weight = 3  # 调整参数？？？？2
        self.centroid = nn.Parameter(torch.zeros(k, 768))

    # 从给定的数据点(datapoints)中，根据它们与预定义聚类中心(centroid)的相似度，为每个数据点找到一个“硬负样本”（hard negative sample）
    def provide_hard_negative(self, datapoints: torch.Tensor, intra_cos_sim: torch.Tensor = None):
        D = self.centroid.data.shape[-1]
        if intra_cos_sim is None:
            intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))  # (256,768) (128,768)->(256,128)
        values, indices = torch.topk(intra_cos_sim, k=2, dim=-1)  # 相似度前二高的
        hard_neg_index = indices[:, 1].unsqueeze(-1).expand(-1, D)  # (256,768)
        hard_negative = torch.gather(self.centroid.data, dim=0, index=hard_neg_index)
        return hard_negative.detach(), indices[:, 1], values[:, 1]

    # 构建一个“假负样本掩码”（false negative mask），用于在对比学习任务中排除那些实际上是正样本但因计算或其他原因被错误标记为负样本的情况
    def mask_false_negative(self,
                            datapoints: torch.Tensor,
                            batch_cos_sim: torch.Tensor = None,  # cos(xi, xj) [bsz, bsz]
                            ):
        if batch_cos_sim is None:  # 没用上
            batch_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))  # 计算每个数据点与所有聚类中心的相似度

        dp_centroid_cos, dp_index, _ = self._clustering(datapoints)  # 基于数据点与聚类中心的相似度得到每个数据点对应的聚类中心的相似度dp_centroid_cos，以及它们归属的聚类索引dp_index
        dp_cluster, _ = self._intra_class_adjacency(dp_index)  # (bsz, bsz)  # 根据聚类索引构建一个表示数据点间是否属于同一聚类的邻接矩阵dp_cluster。这是一个(batch_size, batch_size)的矩阵，其中元素为1表示属于同一类别，0表示不属于同一类别。
        # dp_centroid_cos = dp_centroid_cos.expand_as(dp_cluster)  # (bsz, bsz)

        # false_negative_mask: {1:masked, 0:unmasked}
        # false_negative_mask = dp_cluster * (batch_cos_sim > dp_centroid_cos)  # 数据点间是否属于同一聚类，z1和z2之间的相似度(正样本)大于z1和z1对应的聚类中心
        false_negative_mask = dp_cluster
        return false_negative_mask  # 指示了哪些数据点对间的比较是潜在的假负样本情况

    def _clustering(self, datapoints: torch.Tensor):
        '''
        find the cluster belong to and corresponding centroid for each datapoint
        针对一批数据点执行聚类操作，找出每个数据点所属的聚类中心以及它们与对应聚类中心的余弦相似度
        return
        dp_centroid_cos: [bsz, 1], indicating that cosine similarity between datapoints to centroid to which they belong
        dp_index: [bsz, 1], indicating that the indices of cluster to which datapoints belong
        intra_cos_sim: [bsz, bsz], indicating that cosine similarity between datapoints and centroids
        '''
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_centroid_cos, dp_index = torch.max(intra_cos_sim, dim=-1, keepdim=True)  # (bsz, 1)  # 确定每个数据点的聚类归属
        return dp_centroid_cos, dp_index, intra_cos_sim

    def _intra_class_adjacency(self, dp_index: torch.Tensor):
        r'''
        dp_index: indicating the indices of cluster to which datapoints belong

        return:
        dp_cluster: [bsz, bsz], indicating that which datapoints belong to same cluster
        index_dp: [k, bsz], indicating that which datapoints belong to the clusters
        '''

        B, device = dp_index.shape[0], dp_index.device
        onehot_index = F.one_hot(dp_index.squeeze(-1), self.k)  # (bsz, k) 表示该数据点所属的聚类索引位
        index_dp = onehot_index.T  # (k, bsz) 转置得到聚类-数据点映射，每一行代表一个聚类，行中的1表示哪些数据点属于该聚类

        # adjacency matrix that dp_cluster[i][j]==1 if xi and xj belong to identical cluster
        dp_cluster = torch.matmul(onehot_index.float(), index_dp.float())  # 构建邻接矩阵，如果两个数据点属于同一个聚类，它们在dp_cluster中的对应位置就会是1，否则为0。这实际上构建了一个表示数据点间是否属于相同聚类的邻接矩阵
        # set dp_cluster[i][i] = 0
        dp_cluster.fill_diagonal_(0)  # 对角线元素设为0，这是因为一个数据点与自身属于同一聚类的信息并不需要在后续计算中体现，如计算相似度时避免自比较。
        return dp_cluster, index_dp  # 一个表示数据点间属于同一聚类的邻接矩阵(dp_cluster)，另一个表示每个聚类包含哪些数据点的索引映射(index_dp)

    def false_negative_loss(self,
                            datapoints: torch.Tensor,
                            batch_false_negative_mask: torch.Tensor,
                            batch_cos_sim: torch.Tensor = None,
                            batch_hard_negative: torch.Tensor = None,
                            reduction: str = "mean",
                            alpha: torch.Tensor = None,
                            beta=None,
                            ):
        """
        use BML loss for false negative examples. only implemented example-level now.

        batch_cos_sim: [bs, bs]
        batch_false_negative_mask: [bs, bs], {1:masked, 0:unmasked}
        batch_hard_negative: [bs, bs]
        """
        if batch_cos_sim is None:
            batch_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        batch_positive_sim = batch_cos_sim.diag().unsqueeze(-1).expand_as(
            batch_cos_sim)  # 对角线提取得到每个数据点与自身跨模态的相似度，作为正样本相似度，并广播到与batch_cos_sim相同形状
        if batch_hard_negative is None:
            batch_hard_negative, _, _ = self.provide_hard_negative(datapoints, batch_cos_sim)
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            # our goal:
            # cos(x, hard_neg) < cos(x, false_neg) < cos(x, pos)
            # cos(x, hard_neg)-cos(x, pos) < cos(x, false_neg)-cos(x, pos) < 0
            # -beta < cos(x, false_neg)-cos(x, pos) < -alpha
            # dp_hardneg_sim = self.sim(datapoints.unsqueeze(1), batch_hard_negative.unsqueeze(0))
            # batch_hardneg_sim = dp_hardneg_sim.diag().unsqueeze(-1).expand_as(batch_cos_sim)
            # beta = batch_hardneg_sim - batch_positive_sim
            beta = self.beta
        batch_delta = (batch_cos_sim - batch_positive_sim) * batch_false_negative_mask  # 计算了每个数据点与其他数据点相似度与自身正样本相似度的差异，并乘以假负样本掩码，这样仅对被认为是假负样本的部分计算损失。
        loss = self.BML_loss(batch_delta, alpha, beta)
        if reduction == "mean":
            return loss.mean()
        else:
            raise NotImplementedError

    @staticmethod
    def BML_loss(x, alpha, beta):
        """use for example-level BML loss"""
        return F.relu(x + alpha) + F.relu(-x - beta)

    def forward(self,
                datapoints: torch.Tensor,
                batch_cos_sim: torch.Tensor,
                ):
        B = datapoints.shape[0]
        device = datapoints.device
        datapoints = datapoints.clone().detach()
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))  # z1和质心的相似度(160,128)
        dp_index = torch.argmax(intra_cos_sim, dim=-1, keepdim=True)  # (bsz, 1)
        dp_cluster, index_dp = self._intra_class_adjacency(dp_index)

        dp_centroid = torch.gather(self.centroid, dim=0, index=dp_index.expand_as(datapoints))  # set the centroid corresponding to the datapoint
        # if self.model_args.kmean_debug:
        #     if not dist.is_initialized() or dist.get_rank() == 0:
        #         dp_hardneg, _, _ = self.provide_hard_negative(datapoints, intra_cos_sim)
        #         self.debug_stat(datapoints, dp_centroid, dp_hardneg, batch_cos_sim, dp_index.squeeze(-1), dp_cluster)

        self.update(datapoints, dp_centroid, index_dp)

    def optimized_centroid_init(self, centroid: torch.Tensor, temp=1.0):
        data = centroid.clone().detach()
        batch_cos_sim = self.sim(centroid.unsqueeze(1), centroid.unsqueeze(0)) * temp
        L = data.shape[0]
        assert self.k <= L
        self.centroid = nn.Parameter(data=torch.zeros_like(data[:self.k]))
        idx = list(range(data.shape[0]))  # 2048
        first_idx = random.randint(0, L - 1)  # 生成一个随机数
        self.centroid.data[0] = data[first_idx]  # 从候选质心的索引中随机选择第一个质心，并将其设为质心集合的第一个元素。
        last_idx = first_idx

        # heuristic initialize the centroid of Kmeans clustering
        for i in range(1, self.k):  # 对于剩下的每一个质心（共self.k-1个），执行以下步骤
            # set the last centroid in cos_sim to maxmimal, it will be ignored in the later centroid selection process.
            batch_cos_sim[:,last_idx] = 100  # 将上一次选中的质心在batch_cos_sim矩阵中的对应列设置为一个非常大的值（这里设置为100），这样它在接下来的最小值搜索中会被忽略，以避免重复选择。
            next_idx = torch.argmin(batch_cos_sim[last_idx])  # 从z2中选择的质心
            self.centroid.data[i] = data[next_idx]  # 找到与当前最后一个选择的质心余弦相似度最小的索引，即下一个最佳质心候选。
            last_idx = next_idx

        self.optimizer = AdamW(self.parameters(), lr=self.lr)  # used when self.optimization = adamw
        self.initialized = True  # 标签置为True

    def update(
            self,
            datapoints: torch.Tensor,
            dp_centroid: torch.Tensor = None,
            index_dp: torch.Tensor = None,  # [k, bsz]
    ):
        # update self.centroid in various ways
        if self.optimization == "adamw":
            loss_fct = nn.MSELoss()
            kmeans_loss = loss_fct(dp_centroid, datapoints)
            kmeans_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        elif self.optimization in ["kmeans", "momentum"]:  # 通过矩阵乘法实现对每个簇内所有数据点的特征求和，然后除以该簇数据点的数量（通过index_dp.sum(dim=-1)计算得到）来实现。虽然直接代码中没有显示除法操作，但基于K-means原理，这一步隐含了对每个簇内数据点数的归一化。
            data = torch.matmul(index_dp.float(), datapoints)
            updated_centroid = index_dp.sum(dim=-1).bool().unsqueeze(1).expand_as(self.centroid.data)
            data += self.centroid.data * updated_centroid.logical_not()
            if self.optimization == "kmeans":
                self.centroid.data = data
            elif self.optimization == "momentum":
                old_data = self.centroid.data
                self.centroid.data = self.lr * data + (1 - self.lr) * old_data  # 新的聚类中心是旧中心与根据当前数据计算出的新位置的加权平均，其中权重由学习率和动量参数决定。
        else:
            raise NotImplementedError("optimization %s not imlemented" % self.optimization)

    def case_study(self, datapoints: torch.tensor, input_ids: torch.tensor):
        # define your tokenizer for case studt
        tokenizer = AutoTokenizer.from_pretrained("../plm/roberta-base")
        sentence = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        B = datapoints.shape[0]
        datapoints = datapoints.clone().detach()
        cos_sim = self.sim(datapoints.unsqueeze(1), datapoints.unsqueeze(0))
        dp_centroid_cos, dp_index, intra_cos_sim = self._clustering(datapoints)
        _, hn_index, hn_sim = self.provide_hard_negative(datapoints, intra_cos_sim)

        cos_sim_l = cos_sim.tolist()
        dp_index_l = dp_index.squeeze(-1).tolist()
        dp_centroid_cos_l = dp_centroid_cos.squeeze(-1).tolist()
        hn_index_l = hn_index.tolist()
        hn_sim_l = hn_sim.tolist()
        import pandas as pd
        col = ["sentence", "cluster_idx", "sim_cent", "sim_hn"] + ["sim" + str(i) for i in range(B)]
        df = pd.DataFrame(columns=col)
        from tqdm import tqdm
        for i in tqdm(range(B)):
            one_col = [sentence[i], dp_index_l[i], dp_centroid_cos_l[i], hn_sim_l[i]] + cos_sim_l[i]
            df.loc[i] = one_col
        df.to_csv("case_study.csv")