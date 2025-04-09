from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import *


# No causal mask for sentence-wise decoder
class CrossModalityBertDecoder(BertModel):
    def __init__(self, config=None):
        if config is None:
            config = AutoConfig.from_pretrained(
                "/home/lihongxing/project/PRIOR-main/codes/prior/encoders/language/Bio_ClinicalBERT")
        config.num_hidden_layers = 8  # 隐藏层数
        config.is_decoder = True
        config.add_cross_attention = True  # 以便在模型中加入对来自其他模态（如图像）的编码器输出的交叉注意力机制。
        super().__init__(config, False)  # no pooling layer for sentence-wise decoder

    def forward(self, x, y):
        '''
        x: x input_queries
        y: image_embed
        '''
        # 只从返回的字典中选取'last_hidden_state'，这是解码器的最后一层输出，通常用于后续任务如生成
        return super().forward(inputs_embeds=x, encoder_hidden_states=y, return_dict=True)['last_hidden_state']

    # inputs_embeds (x): 这个参数直接提供了输入序列的嵌入表示，意味着输入文本已经被转换成了模型可以理解的向量形式。通常情况下，
    # 模型会通过一个嵌入层将input_ids映射到嵌入空间，但在这里我们直接给出了这些嵌入。这在你需要使用自定义嵌入或已经预计算好的嵌入时非常有用。

    # encoder_hidden_states (y): 提供给模型的来自于另一个编码器的隐藏状态。这个参数在编码器-解码器架构中特别重要，
    # 比如当你使用BERT作为解码器，接收来自另一个BERT（作为编码器）的输出来生成文本或者做其他条件生成任务时。这意味着模型不仅依赖于当前的输入嵌入，还依赖于之前编码器处理过的信息。