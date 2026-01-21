import torch
import torch.nn as nn

class myHybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_unit, embedding_dim=128, nhead=4):
        """
        input_size: 词表大小，对应输入整数的范围
        hidden_size: LSTM 隐藏状态维度
        num_layers: LSTM 层数
        output_unit: 输出类别数，通常与词表大小一致
        embedding_dim: 嵌入层输出的维度
        nhead: 多头注意力的头数
        """
        super(myHybridModel, self).__init__()
        # 嵌入层将输入整数映射到低维向量
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # 双向 LSTM，注意输出特征维度变为 hidden_size * 2
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True)
        # 多头注意力层，输入维度需与 LSTM 输出一致
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, 
                                               num_heads=nhead, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, output_unit)
        
    def forward(self, x):
        """
        x: (batch_size, seq_length) 的整数张量
        """
        # 嵌入后 shape: (batch_size, seq_length, embedding_dim)
        x = self.embedding(x)
        # 双向 LSTM 输出 shape: (batch_size, seq_length, hidden_size*2)
        lstm_out, _ = self.bilstm(x)
        # 使用自注意力层，query、key、value 都为 lstm_out
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # 取序列最后一个时间步的输出作为整体表示
        out = attn_out[:, -1, :]  # shape: (batch_size, hidden_size*2)
        out = self.dropout(out)
        out = self.fc(out)  # shape: (batch_size, output_unit)
        return out
