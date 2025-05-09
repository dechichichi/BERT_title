
# 定义损失函数
class Cross_entropy_loss(Layer):
    def forward(self, pred_logits, label, label_pad_mask):
        loss = paddle.nn.functional.cross_entropy(pred_logits, label, ignore_index=0, reduction='none')
        masked_loss = paddle.mean(loss * label_pad_mask, axis=0)
        return paddle.sum(masked_loss)